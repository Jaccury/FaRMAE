import torch
import torch.nn as nn


# --------------------------------------------------------
# 基础组件：Patch 嵌入、注意力机制、MLP
# --------------------------------------------------------
class PatchEmbed(nn.Module):
    """将 2D 图像转换为 1D 的 Patch 序列"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # 使用步长等于 patch_size 的卷积来实现无重叠的 Patch 切分
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, Embed_Dim, H/P, W/P) -> (B, Embed_Dim, N) -> (B, N, Embed_Dim)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# --------------------------------------------------------
# 核心 MAE 编码器 (支持可见 Patch 提取)
# --------------------------------------------------------
class ViTMAEEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm):
        """
        ViT-Base 架构，为 FaRMAE 定制，支持 ids_keep 掩码输入。
        """
        super().__init__()
        self.embed_dim = embed_dim

        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 2. CLS Token 与 绝对位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 位置编码包含 CLS token，所以长度是 num_patches + 1
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # 3. Transformer Blocks (ViT-Base 默认为 12 层)
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self._init_weights()

    def _init_weights(self):
        # 初始化位置编码
        nn.init.normal_(self.pos_embed, std=.02)
        nn.init.normal_(self.cls_token, std=.02)
        self.apply(self._init_weights_linear)

    def _init_weights_linear(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ids_keep=None):
        """
        x: 输入图像 (B, C, H, W)
        ids_keep: 从 masking_strategies 传来的可见 Patch 索引 (B, len_keep)。
                  如果是 Stage I，不需要掩码，传入 None 即可处理全图。
        """
        B = x.shape[0]

        # 1. 图像转 Patch (B, N, D)
        x = self.patch_embed(x)

        # 2. 加上位置编码 (注意：此时不能加 cls_token 的位置编码，只加图像 Patch 的)
        # pos_embed[:, 1:, :] 对应图像 Patch 的位置编码
        x = x + self.pos_embed[:, 1:, :]

        # 3. 【核心 MAE 逻辑】如果传入了 ids_keep，利用 gather 操作只提取可见的 Patch
        if ids_keep is not None:
            # ids_keep: (B, len_keep) -> (B, len_keep, D)
            ids_keep_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            x = torch.gather(x, dim=1, index=ids_keep_expanded)

        # 4. 拼接 CLS Token (CLS Token 也需要加上它的位置编码 pos_embed[:, :1, :])
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # 5. 送入 Transformer Block
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x


# ==========================================
# 测试代码
# ==========================================
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing ViTMAEEncoder on {device}")

    # 实例化一个处理 RGB 的编码器 (3通道输入)
    rgb_encoder = ViTMAEEncoder(in_chans=3).to(device)

    # 实例化一个处理 Depth 的编码器 (1通道输入)
    depth_encoder = ViTMAEEncoder(in_chans=1).to(device)

    dummy_rgb = torch.rand(2, 3, 224, 224).to(device)
    dummy_depth = torch.rand(2, 1, 224, 224).to(device)

    # --- 测试 Stage I: 无 Mask，处理全图 ---
    out_rgb_full = rgb_encoder(dummy_rgb, ids_keep=None)
    out_depth_full = depth_encoder(dummy_depth, ids_keep=None)

    # 期望形状: (B, 196 + 1, 768)
    print(f"Stage I Output (RGB Full): {out_rgb_full.shape}")
    print(f"Stage I Output (Depth Full): {out_depth_full.shape}")

    # --- 测试 Stage II: 带 Mask，只处理保留的 Patch ---
    # 模拟 75% 掩盖率，只保留 25% 的 Patch (196 * 0.25 = 49)
    len_keep = 49
    dummy_ids_keep = torch.argsort(torch.rand(2, 196).to(device), dim=1)[:, :len_keep]

    out_rgb_masked = rgb_encoder(dummy_rgb, ids_keep=dummy_ids_keep)

    # 期望形状: (B, 49 + 1, 768)
    print(f"Stage II Output (RGB Masked): {out_rgb_masked.shape}")