import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设你在 vit_backbone.py 中定义了 Block 类，这里我们需要复用它构建 Decoder
from .vit_backbone import Block


# --------------------------------------------------------
# Stage I: 跨模态对齐投影头 (公式 5)
# --------------------------------------------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=2048, out_dim=256):
        """
        轻量级投影头，将 ViT 提取的 Patch 特征映射到对比学习的共享潜空间。
        """
        super().__init__()
        # 典型的对比学习 MLP 结构 (Linear -> GELU -> Linear)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        """
        x: (B, P, in_dim) - ViT 输出的 Patch 级特征 (不含 CLS Token)
        """
        # 投影到共享维度
        z = self.mlp(x)
        # 根据 Eq. 5 强制进行 L2 归一化
        z_norm = F.normalize(z, p=2, dim=-1)
        return z_norm


# --------------------------------------------------------
# Stage II: 结构化 MAE 解码器 (公式 9 准备)
# --------------------------------------------------------
class MAEDecoder(nn.Module):
    def __init__(self, embed_dim=768, decoder_embed_dim=512, decoder_depth=8,
                 decoder_num_heads=16, patch_size=16, img_size=224, out_chans=1):
        """
        接收可见 Patch + Mask Tokens，重构损坏的深度图。
        out_chans: 默认为 1 (因为我们要重构的是单通道深度图)
        """
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size

        # 1. 维度转换：将 Encoder 的高维特征降维到 Decoder 维度
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # 2. [MASK] Token：可学习的掩码标记
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # 3. Decoder 的绝对位置编码 (包含 CLS Token 的位置)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim))

        # 4. Transformer 块构建
        self.blocks = nn.ModuleList([
            Block(dim=decoder_embed_dim, num_heads=decoder_num_heads, mlp_ratio=4.,
                  qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(decoder_depth)])
        self.norm = nn.LayerNorm(decoder_embed_dim)

        # 5. 像素级预测头：将每个 Patch 映射回原始像素空间 (16 * 16 * 1 = 256)
        self.pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * out_chans, bias=True)

        self._init_weights()

    def _init_weights(self):
        # 初始化位置编码和 MASK Token
        nn.init.normal_(self.decoder_pos_embed, std=.02)
        nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights_linear)

    def _init_weights_linear(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ids_restore):
        """
        x: Encoder 传来的可见 Patch 特征 (包含 CLS Token)，形状 (B, len_keep + 1, embed_dim)
        ids_restore: 从 masking_strategies 传来的用于恢复序列的索引 (B, N)
        """
        B = x.shape[0]

        # 1. 将 Encoder 特征映射到 Decoder 维度
        x = self.decoder_embed(x)

        # 2. 分离 CLS Token 和 图像 Patch
        cls_token = x[:, :1, :]
        x_keep = x[:, 1:, :]

        # 3. 补全序列：将可学习的 [MASK] Token 扩展到被遮盖的数量
        mask_tokens = self.mask_token.expand(B, self.num_patches - x_keep.shape[1], -1)

        # 此时序列的后半段全是 mask_tokens，这与原图空间位置是不对应的
        x_ = torch.cat([x_keep, mask_tokens], dim=1)

        # 4. ❗【核心工程操作】使用 ids_restore 将序列还原回原始的 2D 空间排列❗
        # ids_restore_expanded 形状: (B, N, decoder_embed_dim)
        ids_restore_expanded = ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[-1])
        x_ = torch.gather(x_, dim=1, index=ids_restore_expanded)

        # 5. 拼回 CLS Token，并加上 Decoder 的位置编码
        x = torch.cat([cls_token, x_], dim=1)
        x = x + self.decoder_pos_embed

        # 6. 通过 Transformer 块进行特征推理和重构
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # 7. 去除 CLS Token，仅对图像 Patch 预测像素
        x = self.pred(x[:, 1:, :])

        return x


# ==========================================
# 测试代码
# ==========================================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing Task Heads on {device}")

    B, P, embed_dim = 2, 196, 768

    # --- 测试 Stage I 投影头 ---
    proj_head = ProjectionHead(in_dim=embed_dim).to(device)
    dummy_encoder_out = torch.rand(B, P, embed_dim).to(device)
    z_norm = proj_head(dummy_encoder_out)

    print(f"Stage I Proj Output: {z_norm.shape} (Should be {B}, {P}, 256)")
    # 验证 L2 归一化 (每一行的 L2 范数应该等于 1)
    print(f"L2 Norm check: {torch.norm(z_norm[0, 0, :], p=2).item():.4f} (Should be ~1.0)")

    # --- 测试 Stage II MAE Decoder ---
    # 假设掩盖率 75%，保留的 patch 数为 49
    len_keep = 49
    # x 的形状是可见 Patch + CLS Token
    dummy_x_keep = torch.rand(B, len_keep + 1, embed_dim).to(device)
    # 模拟打乱和恢复序列用的索引
    dummy_ids_restore = torch.argsort(torch.rand(B, P).to(device), dim=1)

    decoder = MAEDecoder(embed_dim=embed_dim).to(device)
    pred_depth = decoder(dummy_x_keep, dummy_ids_restore)

    print(f"Stage II Decoder Output: {pred_depth.shape} (Should be {B}, {P}, 256)")