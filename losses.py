import torch
import torch.nn as nn
import torch.nn.functional as F


class FailureAwareAlignmentLoss(nn.Module):
    def __init__(self, temperature=0.07, lambda_w=1.0):
        """
        Stage I: 基于失效感知的跨模态对齐损失 (公式 5-7)
        temperature: InfoNCE 的温度超参数 (tau)
        lambda_w: 失效概率权重的缩放因子 (论文中默认为 1.0)
        """
        super().__init__()
        self.tau = temperature
        self.lambda_w = lambda_w

    def forward(self, z_I, z_D, s_bar):
        """
        z_I: RGB 的 Patch 特征，已经过投影头并做了 L2 归一化 (Eq. 5)，形状 (B, P, D)
        z_D: 深度图的 Patch 特征，已归一化，形状 (B, P, D)
        s_bar: Patch 级的失效概率分数，形状 (B, P)
        其中 B 是 Batch Size, P 是 Patch 数量 (如 196), D 是特征维度。
        """
        B, P, D = z_I.shape

        # 1. 计算对齐权重 w_p = 1 + \lambda * \bar{s}_p (Eq. 6)
        # shape: (B, P)
        w_p = 1.0 + self.lambda_w * s_bar

        # 2. 计算空间相似度矩阵
        # 使用 bmm (Batch Matrix-Matrix product) 计算每个 Batch 内 P 个 Patch 两两之间的内积
        # shape: (B, P, P)
        sim_matrix_I2D = torch.bmm(z_I, z_D.transpose(1, 2)) / self.tau

        # 矩阵转置即为 D 到 I 的相似度
        # ⚠️ 注意：这里转置后内存是不连续的，后续必须用 reshape 而不能用 view
        sim_matrix_D2I = sim_matrix_I2D.transpose(1, 2)

        # 3. 构造 InfoNCE 的标签
        # 对角线上的元素是正样本，所以目标 label 就是 0, 1, 2, ..., P-1
        # shape: (P,) -> 扩展为 (B*P,) 以适应 cross_entropy 接口
        labels = torch.arange(P, device=z_I.device).unsqueeze(0).expand(B, P).reshape(-1)

        # 4. 计算 I -> D 的损失 (Eq. 7)
        # ❗ 核心修复：使用 reshape 替代 view
        loss_I2D_raw = F.cross_entropy(sim_matrix_I2D.reshape(B * P, P), labels, reduction='none')
        loss_I2D_raw = loss_I2D_raw.reshape(B, P)

        # 乘以权重 w_p，并在 P 的维度上求平均 (分母为 \sum w_p)
        loss_I2D = (loss_I2D_raw * w_p).sum(dim=1) / w_p.sum(dim=1)

        # 5. 计算 D -> I 的损失 (对称项)
        # ❗ 核心修复：使用 reshape 替代 view
        loss_D2I_raw = F.cross_entropy(sim_matrix_D2I.reshape(B * P, P), labels, reduction='none')
        loss_D2I_raw = loss_D2I_raw.reshape(B, P)
        loss_D2I = (loss_D2I_raw * w_p).sum(dim=1) / w_p.sum(dim=1)

        # 6. 最终损失为双向损失的平均值 (在 Batch 维度再求均值)
        L_align = 0.5 * (loss_I2D.mean() + loss_D2I.mean())

        return L_align


class MaskedReconstructionLoss(nn.Module):
    def __init__(self):
        """
        Stage II: 结构化掩码重构损失 (公式 9)
        只在被 Mask 的位置计算 MSE。
        """
        super().__init__()

    def forward(self, pred_D, target_D, mask):
        """
        pred_D: Decoder 输出的深度图重构结果，形状 (B, P, D_patch)
        target_D: 真实的深度图目标 (D*)
        mask: 在 masking_strategies.py 中生成的 0/1 掩码矩阵，形状 (B, P)
              1 表示该 Patch 被掩码了，需要计算损失。
        """
        # pred_D 和 target_D 形状应为 (B, P, patch_size**2)
        # 计算所有位置的均方误差 (不求和/均值，保留维度)
        loss_all = (pred_D - target_D) ** 2

        # 在像素维度求平均，得到每个 Patch 的 MSE: (B, P)
        loss_patch = loss_all.mean(dim=-1)

        # 核心：只计算被 Mask 掉的位置 (mask == 1) 的损失
        loss_masked = (loss_patch * mask).sum() / (mask.sum() + 1e-6)

        return loss_masked


# ==========================================
# 测试代码
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing Losses on {device}")

    # --- 测试 Stage I 损失 ---
    B, P, D_dim = 2, 196, 256
    # 模拟投影并 L2 归一化后的特征
    z_I = F.normalize(torch.rand(B, P, D_dim).to(device), p=2, dim=-1)
    z_D = F.normalize(torch.rand(B, P, D_dim).to(device), p=2, dim=-1)

    # 模拟 Patch 级失效概率分数 s_bar (0 到 1 之间)
    s_bar = torch.rand(B, P).to(device)

    align_loss_fn = FailureAwareAlignmentLoss().to(device)
    L_align = align_loss_fn(z_I, z_D, s_bar)
    print(f"Stage I Alignment Loss: {L_align.item():.4f}")

    # --- 测试 Stage II 损失 ---
    patch_pixels = 16 * 16  # 256
    pred_D = torch.rand(B, P, patch_pixels).to(device)
    target_D = torch.rand(B, P, patch_pixels).to(device)

    # 模拟 mask，假设 75% 的位置是 1
    mask = (torch.rand(B, P) > 0.25).float().to(device)

    rec_loss_fn = MaskedReconstructionLoss().to(device)
    L_rec = rec_loss_fn(pred_D, target_D, mask)
    print(f"Stage II Reconstruction Loss: {L_rec.item():.4f}")