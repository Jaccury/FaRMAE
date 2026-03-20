import torch
import torch.nn as nn
import torch.nn.functional as F

# 导入我们之前写好的各个模块
from .vit_backbone import ViTMAEEncoder
from .task_heads import ProjectionHead, MAEDecoder
from modules.failure_prior import FailurePrior
from modules.masking_strategies import FailureGuidedMasking


class FaRMAE(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 embed_dim=768,
                 proj_dim=256,
                 decoder_embed_dim=512,
                 mask_sigma=0.5):
        """
        FaRMAE (Failure-Aware Multimodal Representation Learning) 完整框架
        """
        super().__init__()
        self.patch_size = patch_size

        # --------------------------------------------------------
        # 1. 核心评估模块：失效概率先验与掩码策略
        # --------------------------------------------------------
        self.failure_prior = FailurePrior()
        self.masking_strategy = FailureGuidedMasking(patch_size=patch_size, sigma=mask_sigma)

        # --------------------------------------------------------
        # 2. 共享主干网络 (RGB 和 Depth 双分支编码器)
        # --------------------------------------------------------
        self.rgb_encoder = ViTMAEEncoder(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        self.depth_encoder = ViTMAEEncoder(img_size=img_size, patch_size=patch_size, in_chans=1, embed_dim=embed_dim)

        # --------------------------------------------------------
        # 3. Stage I 专用头：跨模态对齐投影头
        # --------------------------------------------------------
        self.rgb_proj = ProjectionHead(in_dim=embed_dim, out_dim=proj_dim)
        self.depth_proj = ProjectionHead(in_dim=embed_dim, out_dim=proj_dim)

        # --------------------------------------------------------
        # 4. Stage II 专用头：MAE 解码器
        # --------------------------------------------------------
        self.decoder = MAEDecoder(embed_dim=embed_dim, decoder_embed_dim=decoder_embed_dim,
                                  patch_size=patch_size, img_size=img_size, out_chans=1)

    def compute_Sf_and_sbar(self, rgb, depth_obs):
        """
        辅助函数：计算像素级 Sf 和 Patch 级 s_bar
        """
        # 1. 计算像素级失效概率图 Sf: (B, 1, H, W)
        Sf = self.failure_prior(rgb, depth_obs)

        # 2. 降采样为 Patch 级的 s_bar: (B, P)
        s_bar = F.avg_pool2d(Sf, kernel_size=self.patch_size, stride=self.patch_size)
        s_bar = s_bar.view(rgb.shape[0], -1)

        return Sf, s_bar

    def forward_stage1(self, rgb, depth_obs):
        """
        Stage I: 失效感知跨模态表示对齐
        输入:
            rgb: (B, 3, H, W)
            depth_obs: 观测深度图 (B, 1, H, W)
        输出:
            z_I: RGB 投影特征 (B, P, proj_dim)
            z_D: Depth 投影特征 (B, P, proj_dim)
            s_bar: Patch 级失效概率，用于计算 Weighted InfoNCE
        """
        # 计算失效概率
        _, s_bar = self.compute_Sf_and_sbar(rgb, depth_obs)

        # 提取特征 (此时全图输入，ids_keep=None)
        # 注意：Encoder 返回的是包含 CLS Token 的序列 (B, P+1, D)
        feat_I = self.rgb_encoder(rgb, ids_keep=None)
        feat_D = self.depth_encoder(depth_obs, ids_keep=None)

        # 剔除 CLS Token，仅对图像 Patch 投影 (B, P, D) -> (B, P, proj_dim)
        z_I = self.rgb_proj(feat_I[:, 1:, :])
        z_D = self.depth_proj(feat_D[:, 1:, :])

        return z_I, z_D, s_bar

    def forward_stage2(self, rgb, depth_obs, mask_ratio=0.75):
        """
        Stage II: 结构化掩码自编码 (深度重构)
        为了强迫模型学习几何先验，仅输入部分深度 Patch，重构完整深度图。
        输出:
            pred_depth: 预测的深度图 Patch 序列 (B, P, patch_size**2)
            mask: 0/1 掩码矩阵，用于计算 Loss (B, P)
        """
        # 1. 计算失效概率
        Sf, _ = self.compute_Sf_and_sbar(rgb, depth_obs)

        # 2. 获取失效引导的掩码索引
        ids_keep, ids_restore, mask = self.masking_strategy(Sf, mask_ratio=mask_ratio)

        # 3. 深度编码器仅处理保留的 Patch (可见区域)
        # feat_D_keep: (B, len_keep + 1, embed_dim)
        feat_D_keep = self.depth_encoder(depth_obs, ids_keep=ids_keep)

        # 4. 解码器利用 ids_restore 将打乱的序列复原，并填补 [MASK] 进行重构
        # pred_depth: (B, P, patch_size**2)
        pred_depth = self.decoder(feat_D_keep, ids_restore)

        return pred_depth, mask


# ==========================================
# 终极测试代码
# ==========================================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing Full FaRMAE Architecture on {device}")

    # 初始化模型
    model = FaRMAE().to(device)

    # 模拟输入数据
    B = 2
    dummy_rgb = torch.rand(B, 3, 224, 224).to(device)
    dummy_depth = torch.rand(B, 1, 224, 224).to(device)

    # --- 测试 Stage I ---
    print("\n[Running Stage I Forward...]")
    z_I, z_D, s_bar = model.forward_stage1(dummy_rgb, dummy_depth)
    print(f"z_I shape: {z_I.shape} (Should be B, 196, 256)")
    print(f"z_D shape: {z_D.shape} (Should be B, 196, 256)")
    print(f"s_bar shape: {s_bar.shape} (Should be B, 196)")

    # --- 测试 Stage II ---
    print("\n[Running Stage II Forward...]")
    pred_depth, mask = model.forward_stage2(dummy_rgb, dummy_depth, mask_ratio=0.75)
    print(f"pred_depth shape: {pred_depth.shape} (Should be B, 196, 256)")
    print(f"mask shape: {mask.shape} (Should be B, 196)")
    print("✅ All components successfully integrated!")