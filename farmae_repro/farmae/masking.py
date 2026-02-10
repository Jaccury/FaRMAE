"""Structured masking biased by failure likelihood map S_f.

Stage II in the PDF replaces uniform random masking with a structured masking strategy
that assigns higher masking probability to degradation-prone regions (high S_f).
"""

from __future__ import annotations
import torch

def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    # x: (B,C,H,W) -> (B, N, patch_size*patch_size*C)
    B, C, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0
    h = H // patch_size
    w = W // patch_size
    x = x.reshape(B, C, h, patch_size, w, patch_size)
    x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, h*w, patch_size*patch_size*C)
    return x

def unpatchify(patches: torch.Tensor, patch_size: int, H: int, W: int, C: int) -> torch.Tensor:
    # patches: (B,N,ps*ps*C) -> (B,C,H,W)
    B, N, D = patches.shape
    h = H // patch_size
    w = W // patch_size
    assert N == h*w
    x = patches.reshape(B, h, w, patch_size, patch_size, C)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
    return x

@torch.no_grad()
def structured_mask(
    sf: torch.Tensor,
    patch_size: int,
    mask_ratio: float = 0.75,
    bias_strength: float = 2.0,
) -> torch.Tensor:
    """Return a per-patch binary mask, 1 means masked.

    sf: (B,1,H,W) in [0,1]
    bias_strength: controls how strongly we bias mask sampling toward high sf.
    """
    B, _, H, W = sf.shape
    # pool sf to patch grid
    sf_patch = torch.nn.functional.avg_pool2d(sf, kernel_size=patch_size, stride=patch_size)  # (B,1,h,w)
    sf_patch = sf_patch.flatten(1)  # (B,N)

    N = sf_patch.shape[1]
    num_mask = int(round(mask_ratio * N))

    # sampling weights: w_i ∝ exp(bias_strength * sf_i)
    weights = torch.exp(bias_strength * sf_patch)
    weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

    masks = torch.zeros((B, N), device=sf.device, dtype=torch.bool)
    for b in range(B):
        idx = torch.multinomial(weights[b], num_samples=num_mask, replacement=False)
        masks[b, idx] = True
    return masks
