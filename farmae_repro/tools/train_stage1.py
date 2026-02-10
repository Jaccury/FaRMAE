from __future__ import annotations
import argparse
import os
import torch
import torch.optim as optim
import torch.nn.functional as F

from farmae.utils import load_config, set_seed, get_device, ensure_dir
from farmae.data import SyntheticRGBDDataset
from farmae.failure_map import compute_failure_map
from farmae.losses import weighted_infonce
from farmae.models.farmae_pretrain import FaRMAE
from tools._train_utils import make_loader, save_ckpt, load_ckpt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "cuda"))

    # data
    dcfg = cfg["data"]
    if dcfg["name"] == "synthetic":
        ds = SyntheticRGBDDataset(length=dcfg["length"], image_size=dcfg["image_size"], seed=cfg.get("seed", 42))
    else:
        raise ValueError("Only synthetic dataset is wired by default. Implement adapters in farmae/data/adapters.py")

    loader = make_loader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True)

    # model
    mcfg = cfg["model"]
    model = FaRMAE(
        patch_size=mcfg["patch_size"],
        embed_dim=mcfg["embed_dim"],
        depth=mcfg["depth"],
        num_heads=mcfg["num_heads"],
        mlp_ratio=mcfg["mlp_ratio"],
        proj_dim=mcfg["proj_dim"],
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    start_epoch = 0
    if args.resume:
        ckpt = load_ckpt(args.resume, model, opt, map_location=device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1

    outdir = cfg.get("output_dir", "checkpoints")
    ensure_dir(outdir)

    fcfg = cfg["failure_map"]

    model.train()
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        running = 0.0
        for it, batch in enumerate(loader):
            rgb = batch["rgb"].to(device)
            depth_obs = batch["depth_obs"].to(device)

            sf = compute_failure_map(rgb, depth_obs, **fcfg)  # (B,1,H,W)
            # pool to patches for weighting
            ps = mcfg["patch_size"]
            w_patch = torch.nn.functional.avg_pool2d(sf, kernel_size=ps, stride=ps).flatten(1)  # (B,N)

            z_i, z_d, _, _ = model.stage1_forward(rgb, depth_obs)
            loss = weighted_infonce(z_i, z_d, w_patch, tau=cfg["train"]["tau"])

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item())
            if (it + 1) % cfg.get("log_every", 20) == 0:
                print(f"[Stage1][E{epoch:03d}][It{it+1:04d}] loss={running/(it+1):.4f}")

        save_ckpt(os.path.join(outdir, "stage1_last.pt"), model, opt, epoch)

if __name__ == "__main__":
    main()
