import torch
import torch.nn as nn
import torch.nn.functional as F


class FailureGuidedMasking(nn.Module):
    def __init__(self, patch_size=16, sigma=0.5):
        """
        基于失效概率的结构化掩码策略 (Eq. 8)
        patch_size: ViT 的 patch 尺寸，用于将像素级 Sf 池化到 Patch 级
        sigma: 温度系数，控制掩码的采样偏好。论文默认设定为 0.5
        """
        super().__init__()
        self.patch_size = patch_size
        self.sigma = sigma

    def forward(self, Sf, mask_ratio=0.75):
        """
        Sf: 第一步计算出来的像素级失效概率图，形状 (B, 1, H, W)
        mask_ratio: 掩盖比例，论文默认 0.75
        返回:
            ids_keep: 需要保留（输入给 Encoder）的 Patch 索引 (B, len_keep)
            ids_restore: 用于 Decoder 还原序列顺序的索引 (B, N)
            mask: 0/1 掩码矩阵，1 表示被 mask，0 表示可见 (B, N)
        """
        B, C, H, W = Sf.shape

        # 1. 空间池化：将像素级的 Sf (H, W) 降采样为 Patch 级的 \bar{s}_p (H/P, W/P)
        # 因为 Sf 已经是概率值，我们用平均池化是数学上最严谨的
        s_bar = F.avg_pool2d(Sf, kernel_size=self.patch_size, stride=self.patch_size)

        # 展平为一维的 Patch 序列 (B, N)，N = (H/P) * (W/P)
        s_bar = s_bar.view(B, -1)
        N = s_bar.shape[1]

        # 2. 计算掩码概率分布 P(mask_p) \propto exp(\bar{s}_p / \sigma) (Eq. 8)
        # 为了防止 exp 爆炸，可以先减去最大值（数值稳定性优化），这不改变 softmax 后的概率分布
        s_bar_stable = s_bar - s_bar.max(dim=1, keepdim=True)[0]
        weights = torch.exp(s_bar_stable / self.sigma)

        # 归一化为标准的概率分布 (和为1)
        probs = weights / weights.sum(dim=1, keepdim=True)

        # 3. 按概率无放回采样
        # 计算需要掩码的数量
        len_mask = int(N * mask_ratio)
        len_keep = N - len_mask

        # 使用多项式分布采样，选出概率最高的那些 Patch 的索引
        # shape: (B, len_mask)
        mask_ids = torch.multinomial(probs, num_samples=len_mask, replacement=False)

        # 4. 【工程核心点】生成 MAE 所需的 ids_keep 和 ids_restore
        # 创建一个全 0 的掩码张量，在被选中的 mask_ids 位置填入 1
        mask = torch.zeros([B, N], device=Sf.device)
        mask.scatter_(1, mask_ids, 1)

        # 我们巧妙地给 mask 矩阵加上一个极小的随机噪声 (0~0.1)
        # 这样：可见的 Patch 值在 0~0.1 之间，被掩码的 Patch 值在 1.0~1.1 之间
        # 对其进行 argsort，就能保证可见的 Patch 全排在前面，被 mask 的全排在后面！
        sort_noise = mask + torch.rand(B, N, device=Sf.device) * 0.1

        # ids_shuffle 包含了打乱后的索引，前 len_keep 个是 0 区域（可见），后面是 1 区域（掩码）
        ids_shuffle = torch.argsort(sort_noise, dim=1)

        # ids_restore 是 MAE Decoder 重组序列的钥匙
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # 截取前半部分作为送入 Encoder 的可见 Patch
        ids_keep = ids_shuffle[:, :len_keep]

        return ids_keep, ids_restore, mask


# ==========================================
# 测试代码 (可直接运行)
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on {device}")

    # 模拟前面模块输出的 Sf (B=2, 1, H=224, W=224)
    dummy_Sf = torch.rand(2, 1, 224, 224).to(device)

    # 手动让某个区域的失效概率极高，测试我们的算法是否会更倾向于 Mask 这里
    dummy_Sf[0, 0, 0:100, 0:100] = 1.0

    # 实例化模块
    fg_masking = FailureGuidedMasking(patch_size=16, sigma=0.5).to(device)

    # 前向传播 (以 75% 的概率 Mask)
    ids_keep, ids_restore, mask = fg_masking(dummy_Sf, mask_ratio=0.75)

    N = 224 // 16 * 224 // 16  # 14 * 14 = 196
    print(f"Total Patches (N): {N}")
    print(f"ids_keep shape: {ids_keep.shape} (Should be B, {int(N * 0.25)})")
    print(f"ids_restore shape: {ids_restore.shape} (Should be B, {N})")
    print(f"mask shape: {mask.shape} (Should be B, {N})")

    # 验证第一张图被 Mask 的 Token 数量
    mask_count = mask[0].sum().item()
    print(f"Actual Masked count in image 0: {mask_count} (Should be {int(N * 0.75)})")