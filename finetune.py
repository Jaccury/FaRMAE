import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 1. 导入 TransCGDataset (因为你用的是 TransCG 数据集)
from datasets.transcg_loader import TransCGDataset
from models.farmae import FaRMAE


# 已经删除了刚才报错的 compute_metrics

# --------------------------------------------------------
# 微调模型：结合预训练 Encoder 和轻量级深度预测头
# --------------------------------------------------------
class FaRMAEDepthCompletion(nn.Module):
    def __init__(self, pretrained_farmae, embed_dim=768, img_size=224, patch_size=16):
        super().__init__()
        # 提取已经训练好的编码器
        self.rgb_encoder = pretrained_farmae.rgb_encoder
        self.depth_encoder = pretrained_farmae.depth_encoder

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size

        # 简单的多模态特征融合与上采样头 (轻量级任务头)
        # 输入是 RGB 和 Depth 特征拼接: 768 * 2 = 1536
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(embed_dim * 2, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 14 -> 28
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 28 -> 56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 56 -> 112
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 112 -> 224
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)  # 输出深度图
        )

    def forward(self, rgb, depth_obs):
        # 1. 提取特征 (B, P+1, D)
        feat_rgb = self.rgb_encoder(rgb, ids_keep=None)
        feat_depth = self.depth_encoder(depth_obs, ids_keep=None)

        # 2. 去除 CLS Token: (B, P, D)
        feat_rgb = feat_rgb[:, 1:, :]
        feat_depth = feat_depth[:, 1:, :]

        # 3. 维度转换 (B, P, D) -> (B, D, H/P, W/P)
        B, P, D = feat_rgb.shape
        feat_rgb_2d = feat_rgb.transpose(1, 2).reshape(B, D, self.grid_size, self.grid_size)
        feat_depth_2d = feat_depth.transpose(1, 2).reshape(B, D, self.grid_size, self.grid_size)

        # 4. 融合与解码
        fused = torch.cat([feat_rgb_2d, feat_depth_2d], dim=1)  # (B, 1536, 14, 14)
        fused = self.fusion_conv(fused)
        dense_depth = self.upsample(fused)  # (B, 1, 224, 224)

        return dense_depth


# --------------------------------------------------------
# 评估指标计算 (根据论文公式 10-14)
# --------------------------------------------------------
def eval_metrics(pred, target):
    """ 计算 RMSE, MAE, iRMSE, REL, delta1 """
    # 仅在 target 有效的区域计算 (target > 0)
    valid_mask = target > 0
    if valid_mask.sum() == 0:
        return 0, 0, 0, 0, 0

    p = pred[valid_mask]
    t = target[valid_mask]

    # 防止预测出负数或零导致除零错误
    p = torch.clamp(p, min=1e-3)

    rmse = torch.sqrt(torch.mean((p - t) ** 2))
    mae = torch.mean(torch.abs(p - t))
    irmse = torch.sqrt(torch.mean((1.0 / p - 1.0 / t) ** 2))
    rel = torch.mean(torch.abs(p - t) / t)

    max_ratio = torch.max(p / t, t / p)
    delta1 = (max_ratio < 1.25).float().mean()

    return rmse.item(), mae.item(), irmse.item(), rel.item(), delta1.item()


def main(args):
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🚀 启动 FaRMAE Downstream Finetuning (设备: {device})")

    # 2. 使用 TransCGDataset
    dataset_train = TransCGDataset(root_dir=args.data_root, split='train', img_size=224)
    data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 加载预训练的 FaRMAE 框架
    base_model = FaRMAE().to(device)
    if os.path.exists(args.pretrained_ckpt):
        print(f"🔄 正在加载预训练权重 (Stage II): {args.pretrained_ckpt}")
        base_model.load_state_dict(torch.load(args.pretrained_ckpt, map_location='cpu')['model_state_dict'],
                                   strict=False)

    # 包装成微调模型
    model = FaRMAEDepthCompletion(base_model).to(device)

    # 损失函数使用 Smooth L1 (Huber Loss) 或者 MSE
    criterion = nn.SmoothL1Loss()
    # 微调阶段通常使用较小的学习率
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        metrics = {'rmse': 0, 'mae': 0, 'irmse': 0, 'rel': 0, 'd1': 0}

        pbar = tqdm(data_loader_train, desc=f"Finetune Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            rgb = batch['rgb'].to(device)
            depth_obs = batch['depth'].to(device)
            depth_gt = batch['depth_gt'].to(device)

            optimizer.zero_grad()

            pred_depth = model(rgb, depth_obs)

            # 仅在 GT 有效处计算 Loss
            valid_mask = depth_gt > 0
            if valid_mask.sum() > 0:
                loss = criterion(pred_depth[valid_mask], depth_gt[valid_mask])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # 计算指标 [cite: 251-257]
                rmse, mae, irmse, rel, d1 = eval_metrics(pred_depth.detach(), depth_gt)
                metrics['rmse'] += rmse
                metrics['mae'] += mae
                metrics['irmse'] += irmse
                metrics['rel'] += rel
                metrics['d1'] += d1

                pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'RMSE': f"{rmse:.3f}", 'd1': f"{d1:.3f}"})

        # 打印当前 Epoch 的平均评估指标
        n_batches = len(data_loader_train)
        print(
            f"✅ Epoch {epoch + 1} 评估指标: RMSE={metrics['rmse'] / n_batches:.3f} | MAE={metrics['mae'] / n_batches:.3f} | REL={metrics['rel'] / n_batches:.3f} | δ1={metrics['d1'] / n_batches:.3f}")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"finetune_epoch_{epoch + 1}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=50, type=int)  # 论文中微调 50 个 Epoch [cite: 238]
    parser.add_argument('--lr', default=1e-4, type=float)
    # 默认路径直接改为你的 transcg 路径
    parser.add_argument('--data_root', default='G:/transcg/transcg', type=str)
    parser.add_argument('--output_dir', default='./checkpoints/finetune', type=str)
    # 指定刚刚训练好的 Stage II 权重
    parser.add_argument('--pretrained_ckpt', default='./checkpoints/stage2/farmae_stage2_epoch_200.pth', type=str)
    parser.add_argument('--device', default='cuda', type=str)

    args = parser.parse_args()
    main(args)