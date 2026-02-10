from __future__ import annotations
import torch
import torch.nn.functional as F

def weighted_infonce(
    z_i: torch.Tensor,
    z_d: torch.Tensor,
    weights: torch.Tensor,
    tau: float = 0.2,
) -> torch.Tensor:
    """Patch-level InfoNCE with per-patch weights.

    Args:
        z_i, z_d: (B,N,C) normalized embeddings.
        weights: (B,N) nonnegative weights from S_f (pooled to patches).
    """
    B, N, C = z_i.shape
    z_i = F.normalize(z_i, dim=-1)
    z_d = F.normalize(z_d, dim=-1)

    # Flatten patches as instances
    zi = z_i.reshape(B*N, C)
    zd = z_d.reshape(B*N, C)

    logits = (zi @ zd.t()) / tau  # (BN, BN)
    labels = torch.arange(B*N, device=logits.device)

    # convert weights to per-instance weights
    w = weights.reshape(B*N)
    w = w / (w.mean() + 1e-8)  # stabilize

    loss = F.cross_entropy(logits, labels, reduction="none")
    loss = (loss * w).mean()
    return loss

def mae_reconstruction_loss(pred_patches: torch.Tensor, target_patches: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L1 loss on masked patches only.

    pred_patches, target_patches: (B,N,D)
    mask: (B,N) bool, True => masked
    """
    diff = (pred_patches - target_patches).abs().mean(dim=-1)  # (B,N)
    loss = diff[mask].mean()
    return loss
