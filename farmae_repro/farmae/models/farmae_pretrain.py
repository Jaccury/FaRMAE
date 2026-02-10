from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import SimpleViTEncoder
from ..masking import patchify, unpatchify

class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MAEDecoder(nn.Module):
    def __init__(self, embed_dim: int, decoder_embed_dim: int, decoder_depth: int, decoder_num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        from .vit import Block, build_2d_sincos_pos_embed
        self.proj = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.blocks = nn.ModuleList([Block(decoder_embed_dim, decoder_num_heads, mlp_ratio) for _ in range(decoder_depth)])
        self.norm = nn.LayerNorm(decoder_embed_dim)
        self.pred = nn.Linear(decoder_embed_dim, 16*16*1)  # default patched later if patch_size != 16
        self.build_2d_sincos_pos_embed = build_2d_sincos_pos_embed
        self.register_buffer("pos_embed", torch.zeros(1, 1, decoder_embed_dim), persistent=False)

    def reset_pred_head(self, patch_size: int, out_chans: int = 1):
        self.pred = nn.Linear(self.pred.in_features, patch_size*patch_size*out_chans)

    def forward(self, visible_tokens: torch.Tensor, mask: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # visible_tokens: (B,N,C), mask: (B,N) bool
        B, N, C = visible_tokens.shape
        x = self.proj(visible_tokens)

        if self.pos_embed.shape[1] != N:
            pos = self.build_2d_sincos_pos_embed(h, w, x.shape[-1]).to(x.device, dtype=x.dtype)
            self.pos_embed = pos.unsqueeze(0)

        # insert mask tokens
        mask_token = self.mask_token.expand(B, N, -1)
        x_full = torch.where(mask.unsqueeze(-1), mask_token, x)
        x_full = x_full + self.pos_embed

        for blk in self.blocks:
            x_full = blk(x_full)
        x_full = self.norm(x_full)
        pred = self.pred(x_full)  # (B,N,ps*ps)
        return pred

class FaRMAE(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        proj_dim: int = 256,
        decoder_embed_dim: int = 256,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.rgb_encoder = SimpleViTEncoder(3, patch_size, embed_dim, depth, num_heads, mlp_ratio)
        self.dep_encoder = SimpleViTEncoder(1, patch_size, embed_dim, depth, num_heads, mlp_ratio)

        self.rgb_proj = ProjectionHead(embed_dim, proj_dim)
        self.dep_proj = ProjectionHead(embed_dim, proj_dim)

        self.decoder = MAEDecoder(embed_dim, decoder_embed_dim, decoder_depth, decoder_num_heads, mlp_ratio)
        self.decoder.reset_pred_head(patch_size, out_chans=1)

    def encode(self, rgb: torch.Tensor, depth_obs: torch.Tensor):
        z_i, h, w = self.rgb_encoder(rgb)
        z_d, _, _ = self.dep_encoder(depth_obs)
        return z_i, z_d, h, w

    def stage1_forward(self, rgb: torch.Tensor, depth_obs: torch.Tensor):
        z_i, z_d, h, w = self.encode(rgb, depth_obs)
        p_i = self.rgb_proj(z_i)
        p_d = self.dep_proj(z_d)
        return p_i, p_d, h, w

    def stage2_forward(self, rgb: torch.Tensor, depth_obs: torch.Tensor, mask: torch.Tensor):
        # In Stage II we reconstruct depth from the depth stream tokens.
        z_i, z_d, h, w = self.encode(rgb, depth_obs)
        pred = self.decoder(z_d, mask, h, w)  # patches
        return pred, h, w
