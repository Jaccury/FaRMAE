"""Failure likelihood map S_f.

The PDF defines a spatial prior S_f ∈ [0,1]^{H×W} indicating regions prone to sensing degradation and
mentions that it can be derived from cues such as image gradients, texture sparsity, and depth discontinuities.

This implementation provides a deterministic, reproducible instantiation:
  g(p): normalized image gradient magnitude (Sobel)
  s(p): texture sparsity proxy (higher when local texture is weak) using local std
  d(p): normalized depth discontinuity magnitude (Sobel on observed depth)
  S_f = sigmoid(a*g + b*s + c*d)

All parameters are configurable in YAML.
"""

from __future__ import annotations
import numpy as np
import cv2
import torch

def _normalize01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn + eps)

def _local_std(gray: np.ndarray, ksize: int) -> np.ndarray:
    # E[x^2] - (E[x])^2
    blur = cv2.blur(gray, (ksize, ksize))
    blur2 = cv2.blur(gray * gray, (ksize, ksize))
    var = np.maximum(blur2 - blur * blur, 0.0)
    return np.sqrt(var)

@torch.no_grad()
def compute_failure_map(
    rgb: torch.Tensor,
    depth_obs: torch.Tensor,
    a: float = 2.0,
    b: float = 1.5,
    c: float = 2.0,
    texture_window: int = 9,
) -> torch.Tensor:
    """Compute S_f for a batch.

    Args:
        rgb: (B,3,H,W) in [0,1]
        depth_obs: (B,1,H,W) (can contain missing values / zeros)
    Returns:
        S_f: (B,1,H,W) in [0,1]
    """
    assert rgb.ndim == 4 and depth_obs.ndim == 4
    B, _, H, W = rgb.shape
    sf_list = []
    rgb_np = rgb.detach().cpu().numpy()
    d_np = depth_obs.detach().cpu().numpy()

    for i in range(B):
        img = np.transpose(rgb_np[i], (1,2,0))
        gray = cv2.cvtColor((img * 255.0).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

        # gradients (Sobel)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        g = np.sqrt(gx*gx + gy*gy)
        g = _normalize01(g)

        # texture sparsity proxy: high when local texture is weak (low std)
        std = _local_std(gray, max(3, int(texture_window) | 1))
        std = _normalize01(std)
        s = 1.0 - std

        # depth discontinuity on observed depth
        dep = d_np[i,0].astype(np.float32)
        dx = cv2.Sobel(dep, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(dep, cv2.CV_32F, 0, 1, ksize=3)
        dd = np.sqrt(dx*dx + dy*dy)
        dd = _normalize01(dd)

        logits = a * g + b * s + c * dd
        sf = 1.0 / (1.0 + np.exp(-logits))
        sf_list.append(sf[None, ...])

    sf = np.stack(sf_list, axis=0)  # (B,1,H,W)
    return torch.from_numpy(sf).to(rgb.device, dtype=rgb.dtype)
