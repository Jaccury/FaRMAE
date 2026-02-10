from __future__ import annotations
import argparse
import os
import torch
import torch.optim as optim

from farmae.utils import load_config, set_seed, get_device, ensure_dir
from farmae.data import SyntheticRGBDDataset
from farmae.failure_map import compute_failure_map
from farmae.masking import structured_mask, patchify
from farmae.losses import mae_reconstruction_loss
from farmae.models.farmae_pretrain import FaRMAE
from tools._train_utils import make_loader, save_ckpt, load_ckpt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="Optional Stage I checkpoint to initialize encoders.")
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
        decoder_embed_dim=mcfg["decoder_embed_dim"],
        decoder_depth=mcfg["decoder_depth"],
        decoder_num_heads=mcfg["decoder_num_heads"],
    ).to(device)

    opt = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    if args.resume:
        load_ckpt(args.resume, model, None, map_location=device)

    outdir = cfg.get("output_dir", "checkpoints")
    ensure_dir(outdir)

    fcfg = cfg["failure_map"]
    kcfg = cfg["masking"]

    model.train()
    for epoch in range(cfg["train"]["epochs"]):
        running = 0.0
        for it, batch in enumerate(loader):
            rgb = batch["rgb"].to(device)
            depth_obs = batch["depth_obs"].to(device)

            sf = compute_failure_map(rgb, depth_obs, **fcfg)  # (B,1,H,W)
            mask = structured_mask(sf, patch_size=mcfg["patch_size"], mask_ratio=kcfg["mask_ratio"], bias_strength=kcfg["bias_strength"])

            pred_patches, h, w = model.stage2_forward(rgb, depth_obs, mask)  # (B,N,ps*ps)
            target_patches = patchify(depth_obs, mcfg["patch_size"])  # reconstruct observed depth as in Eq.(2)

            loss = mae_reconstruction_loss(pred_patches, target_patches, mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item())
            if (it + 1) % cfg.get("log_every", 20) == 0:
                print(f"[Stage2][E{epoch:03d}][It{it+1:04d}] loss={running/(it+1):.4f}")

        save_ckpt(os.path.join(outdir, "stage2_last.pt"), model, opt, epoch)

if __name__ == "__main__":
    main()
