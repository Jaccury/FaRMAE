from __future__ import annotations
import math
import torch
import torch.nn as nn

def build_2d_sincos_pos_embed(h: int, w: int, dim: int, temperature: float = 10000.0) -> torch.Tensor:
    """(H*W, dim) fixed 2D sin-cos positional embeddings."""
    assert dim % 4 == 0
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    y = y.flatten().float()
    x = x.flatten().float()

    omega = torch.arange(dim // 4).float()
    omega = 1.0 / (temperature ** (omega / (dim // 4)))

    out_x = torch.einsum("n,d->nd", x, omega)
    out_y = torch.einsum("n,d->nd", y, omega)
    pos = torch.cat([torch.sin(out_x), torch.cos(out_x), torch.sin(out_y), torch.cos(out_y)], dim=1)
    return pos

class PatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B,embed,h,w)
        B, C, h, w = x.shape
        x = x.flatten(2).transpose(1,2)  # (B,N,C)
        return x, h, w

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
    def forward(self, x): 
        return self.fc2(self.act(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # self-attn
        y = self.norm1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + y
        # mlp
        y = self.mlp(self.norm2(x))
        x = x + y
        return x

class SimpleViTEncoder(nn.Module):
    def __init__(self, in_chans: int, patch_size: int, embed_dim: int, depth: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.register_buffer("pos_embed", torch.zeros(1, 1, embed_dim), persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        tokens, h, w = self.patch_embed(x)  # (B,N,C)
        B, N, C = tokens.shape
        if self.pos_embed.shape[1] != N:
            pos = build_2d_sincos_pos_embed(h, w, C).to(tokens.device, dtype=tokens.dtype)
            self.pos_embed = pos.unsqueeze(0)  # (1,N,C)
        tokens = tokens + self.pos_embed
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return tokens, h, w
