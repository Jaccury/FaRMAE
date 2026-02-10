from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset

class SyntheticRGBDDataset(Dataset):
    """A simple synthetic RGB-D dataset for pipeline sanity checks.

    Produces:
      rgb: (3,H,W) in [0,1]
      depth_obs: (1,H,W) observed depth with structured 'failures'
      depth_gt: (1,H,W) ground truth depth (for downstream fine-tuning)
    """
    def __init__(self, length: int = 2000, image_size: int = 224, seed: int = 0):
        self.length = int(length)
        self.H = int(image_size)
        self.W = int(image_size)
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        H, W = self.H, self.W
        # simple smooth depth field + noise
        yy, xx = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing="ij")
        depth_gt = (0.5 + 0.5*np.sin(2*np.pi*xx) * np.cos(2*np.pi*yy)).astype(np.float32)
        depth_gt += 0.02 * self.rng.randn(H, W).astype(np.float32)
        depth_gt = np.clip(depth_gt, 0, 1)

        rgb = self.rng.rand(3, H, W).astype(np.float32)

        # simulate transparent-failure-like artifacts: missing stripes + boundary bias
        depth_obs = depth_gt.copy()
        # missing band
        if self.rng.rand() < 0.7:
            x0 = self.rng.randint(W//8, W - W//8)
            width = self.rng.randint(W//20, W//10)
            depth_obs[:, max(0, x0-width):min(W, x0+width)] = 0.0
        # biased region
        if self.rng.rand() < 0.7:
            y0 = self.rng.randint(H//8, H - H//8)
            rad = self.rng.randint(H//12, H//6)
            mask = (yy - y0/H)**2 + (xx - 0.5)**2 < (rad/H)**2
            depth_obs[mask] = np.clip(depth_obs[mask] + 0.12, 0, 1)

        return {
            "rgb": torch.from_numpy(rgb),
            "depth_obs": torch.from_numpy(depth_obs[None, ...]),
            "depth_gt": torch.from_numpy(depth_gt[None, ...]),
        }
