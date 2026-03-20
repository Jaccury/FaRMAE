import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import math

# 导入我们自己写的模块
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.transcg_loader import TransCGDataset
from models.farmae import FaRMAE
from modules.losses import FailureAwareAlignmentLoss


def get_args_parser():
    parser = argparse.ArgumentParser('FaRMAE Stage I Pretraining', add_help=False)
    # 论文默认 Batch Size 是 256，但受限于单卡显存，这里默认设为 16 或 32。
    # 如果显存不够（比如 8G 显存），请在此处将其下调为 8 或 4。
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=100, type=int, help='Stage I 预训练 100 个 epoch')  #
    parser.add_argument('--lr', default=1.5e-4, type=float, help='学习率 1.5e-4')  #
    parser.add_argument('--weight_decay', default=0.05, type=float, help='AdamW 权重衰减')

    # 路径配置
    parser.add_argument('--data_root', default='G:/transcg/transcg', type=str, help='数据集根目录')
    parser.add_argument('--output_dir', default='./checkpoints/stage1', type=str, help='模型保存路径')
    parser.add_argument('--device', default='cuda', type=str, help='cuda 或 cpu')
    return parser


def adjust_learning_rate(optimizer, epoch, args):
    """
    带 Warmup 的余弦退火学习率衰减
    论文指出：前 10 个 epoch 是 warmup
    """
    warmup_epochs = 10
    if epoch < warmup_epochs:
        lr = args.lr * epoch / warmup_epochs
    else:
        # Cosine decay
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🚀 启动 FaRMAE Stage I 训练 (设备: {device})")

    # 1. 准备 DataLoader
    # 这里加载我们重写好的字典返回形式的 Dataset
    dataset_train = TransCGDataset(root_dir=args.data_root, split='train', img_size=224)
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # 2. 实例化模型与损失函数
    model = FaRMAE().to(device)
    criterion = FailureAwareAlignmentLoss().to(device)

    # 3. 优化器 (AdamW)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 启用混合精度训练 (AMP) 极大节省显存并加速训练
    scaler = GradScaler()

    print(f"📦 模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f} M")

    # 4. 开始训练循环
    for epoch in range(args.epochs):
        model.train()

        # 动态调整学习率
        current_lr = adjust_learning_rate(optimizer, epoch, args)

        epoch_loss = 0.0
        pbar = tqdm(data_loader_train, desc=f"Epoch {epoch + 1}/{args.epochs} [LR: {current_lr:.6f}]")

        for step, batch in enumerate(pbar):
            # 完美解决你之前的报错：现在 batch 是一个字典，直接通过 key 提取
            rgb = batch['rgb'].to(device, non_blocking=True)
            depth_obs = batch['depth'].to(device, non_blocking=True)

            optimizer.zero_grad()

            # 开启混合精度前向传播
            with autocast():
                # Stage I 前向传播
                z_I, z_D, s_bar = model.forward_stage1(rgb, depth_obs)

                # 计算带失效感知权重的 InfoNCE 损失
                loss = criterion(z_I, z_D, s_bar)

            # 混合精度反向传播与优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            # 更新进度条
            pbar.set_postfix({'Align_Loss': f"{loss.item():.4f}"})

        avg_loss = epoch_loss / len(data_loader_train)
        print(f"✅ Epoch {epoch + 1} 完成! 平均 Loss: {avg_loss:.4f}")

        # 5. 定期保存 Checkpoint
        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.output_dir, f"farmae_stage1_epoch_{epoch + 1}.pth")
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