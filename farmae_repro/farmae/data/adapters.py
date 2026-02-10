"""Dataset adapters (placeholders).

Fill these to load ScanNet / NYUv2 / ClearGrasp according to your own directory layout.
The training scripts will work once these return dicts with keys:
  rgb: (3,H,W) float32 in [0,1]
  depth_obs: (1,H,W) float32 (observed depth)
  depth_gt: optional (1,H,W)
"""

from __future__ import annotations
from torch.utils.data import Dataset

class ScanNetAdapter(Dataset):
    def __init__(self, root: str, split: str = "train"):
        raise NotImplementedError("Implement ScanNet loading here.")

class NYUv2Adapter(Dataset):
    def __init__(self, root: str, split: str = "train"):
        raise NotImplementedError("Implement NYUv2 loading here.")

class ClearGraspAdapter(Dataset):
    def __init__(self, root: str, split: str = "train", synthetic: bool = True):
        raise NotImplementedError("Implement ClearGrasp loading here.")
