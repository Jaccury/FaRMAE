from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDepthCompletionHead(nn.Module):
    """A lightweight head for downstream depth completion (toy but runnable).

    It fuses RGB and depth tokens by concatenation and predicts a dense depth map via upsampling.
    """
    def __init__(self, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.fuse = nn.Linear(embed_dim * 2, embed_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, 1, 1),
        )

    def forward(self, z_i: torch.Tensor, z_d: torch.Tensor, h: int, w: int, H: int, W: int) -> torch.Tensor:
        # z_*: (B,N,C)
        x = torch.cat([z_i, z_d], dim=-1)
        x = self.fuse(x)  # (B,N,C)
        B, N, C = x.shape
        x = x.transpose(1,2).reshape(B, C, h, w)
        x = self.conv(x)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        return x
