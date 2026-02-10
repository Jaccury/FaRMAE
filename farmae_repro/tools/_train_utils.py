from __future__ import annotations
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from farmae.utils import ensure_dir

def make_loader(dataset, batch_size: int, shuffle: bool = True, num_workers: int = 2):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)

def save_ckpt(path: str, model, optim, epoch: int, extra: dict | None = None):
    ensure_dir(os.path.dirname(path))
    payload = {
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)

def load_ckpt(path: str, model, optim=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=False)
    if optim is not None and "optim" in ckpt:
        optim.load_state_dict(ckpt["optim"])
    return ckpt
