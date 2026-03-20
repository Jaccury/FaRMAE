import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import math

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.transcg_loader import TransCGDataset
from models.farmae import FaRMAE
from modules.losses import MaskedReconstructionLoss


def get_args_parser():
    parser = argparse.ArgumentParser('FaRMAE Stage II Pretraining', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int)
    # 论文中 Stage II 训练 200 个 Epoch
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=1.5e-4, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)

    # 核心 MAE 参数
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='失效引导的掩码比例')

    # 路径配置
    parser.add_argument('--data_root', default='G:/transcg/transcg', type=str)
    parser.add_argument('--output_dir', default='./checkpoints/stage2', type=str)

    # ⚠️ 必须传入 Stage I 训练好的模型权重路径！
    parser.add_argument('--stage1_ckpt', default='./checkpoints/stage1/farmae_stage1_epoch_100.pth', type=str,
                        help='Stage I 预训练权重路径')
    parser.add_argument('--device', default='cuda', type=str)
    return parser


def adjust_learning_rate(optimizer, epoch, args):
    """带 Warmup 的余弦退火学习率衰减"""
    warmup_epochs = 10
    if epoch < warmup_epochs:
        lr = args.lr * epoch / warmup_epochs
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def patchify(imgs, patch_size=16):
    """
    将图像 (B, 1, H, W) 切分为不重叠的 Patch 序列 (B, P, patch_size**2)
    这是为了和 Decoder 预测的输出对齐，以计算 MSE 损失。
    """
    p = patch_size
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
    # 使用 einsum 巧妙转换维度排列
    x = torch.einsum('nchpwq->nhwpqc', x)
    # 展平为 (B, h*w, p*p*1)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 1))
    return x


def main(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🚀 启动 FaRMAE Stage II 训练 (设备: {device})")

    # 1. 准备 DataLoader
    dataset_train = TransCGDataset(root_dir=args.data_root, split='train', img_size=224)
    data_loader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )

    # 2. 实例化模型
    model = FaRMAE().to(device)

    # 3. 🛡️ 核心步骤：加载 Stage I 训练好的编码器权重
    if os.path.exists(args.stage1_ckpt):
        print(f"🔄 正在加载 Stage I 预训练权重: {args.stage1_ckpt}")
        checkpoint = torch.load(args.stage1_ckpt, map_location='cpu')
        # strict=False 是因为 Stage I 的 Checkpoint 里包含 proj_head，
        # 而在 Stage II 我们主要需要的是 encoder 权重，decoder 是从头初始化的
        msg = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"✅ 权重加载完成! 忽略的缺失/多余键值情况: {msg}")
    else:
        print(f"⚠️ 警告: 未找到 Stage I 权重 {args.stage1_ckpt}，将从头开始训练！")

    # 4. 损失函数与优化器
    criterion = MaskedReconstructionLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()

    # 5. 训练循环
    for epoch in range(args.epochs):
        model.train()
        current_lr = adjust_learning_rate(optimizer, epoch, args)

        epoch_loss = 0.0
        pbar = tqdm(data_loader_train, desc=f"Epoch {epoch + 1}/{args.epochs} [LR: {current_lr:.6f}]")

        for step, batch in enumerate(pbar):
            rgb = batch['rgb'].to(device, non_blocking=True)
            # 论文指出在有监督数据集上，重构目标为真实深度图
            # 目前我们的 DataLoader 提取的是 'depth...-gt.png'
            depth_gt = batch['depth'].to(device, non_blocking=True)

            # 将目标深度图切分成 Patch 以匹配 Decoder 的输出维度
            target_depth_patches = patchify(depth_gt, patch_size=16)

            optimizer.zero_grad()

            with autocast():
                # 运行 Stage II 前向传播 (进行 75% 掩盖)
                pred_depth, mask = model.forward_stage2(rgb, depth_gt, mask_ratio=args.mask_ratio)

                # 计算仅在 Mask 位置的 MSE 重构损失
                loss = criterion(pred_depth, target_depth_patches, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix({'Rec_Loss': f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(data_loader_train)
        print(f"✅ Epoch {epoch + 1} 完成! 平均重构 Loss: {avg_loss:.4f}")

        # 6. 定期保存 Checkpoint
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.output_dir, f"farmae_stage2_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f"💾 Checkpoint 已保存至: {ckpt_path}")


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)