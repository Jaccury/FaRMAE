from __future__ import annotations
import argparse
import os
import torch
import torch.optim as optim
import torch.nn.functional as F

from farmae.utils import load_config, set_seed, get_device, ensure_dir
from farmae.data import SyntheticRGBDDataset
from farmae.models.farmae_pretrain import FaRMAE
from farmae.models.depth_completion_head import SimpleDepthCompletionHead
from tools._train_utils import make_loader, save_ckpt, load_ckpt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="Stage II checkpoint for initialization.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = get_device(cfg.get("device", "cuda"))

    dcfg = cfg["data"]
    ds = SyntheticRGBDDataset(length=dcfg["length"], image_size=dcfg["image_size"], seed=cfg.get("seed", 42))
    loader = make_loader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True)

    mcfg = cfg["model"]
    backbone = FaRMAE(
        patch_size=mcfg["patch_size"],
        embed_dim=mcfg["embed_dim"],
        depth=mcfg["depth"],
        num_heads=mcfg["num_heads"],
        mlp_ratio=mcfg["mlp_ratio"],
    ).to(device)

    if args.resume:
        load_ckpt(args.resume, backbone, None, map_location=device)

    head = SimpleDepthCompletionHead(embed_dim=mcfg["embed_dim"], patch_size=mcfg["patch_size"]).to(device)

    params = list(backbone.parameters()) + list(head.parameters())
    opt = optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    outdir = cfg.get("output_dir", "checkpoints")
    ensure_dir(outdir)

    backbone.train()
    head.train()
    for epoch in range(cfg["train"]["epochs"]):
        running = 0.0
        for it, batch in enumerate(loader):
            rgb = batch["rgb"].to(device)
            depth_obs = batch["depth_obs"].to(device)
            depth_gt = batch["depth_gt"].to(device)

            z_i, z_d, h, w = backbone.encode(rgb, depth_obs)
            pred = head(z_i, z_d, h, w, H=rgb.shape[-2], W=rgb.shape[-1])

            loss = F.l1_loss(pred, depth_gt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item())
            if (it + 1) % cfg.get("log_every", 20) == 0:
                print(f"[Finetune][E{epoch:03d}][It{it+1:04d}] loss={running/(it+1):.4f}")

        save_ckpt(os.path.join(outdir, "finetune_dc_last.pt"), backbone, opt, epoch, extra={"head": head.state_dict()})

if __name__ == "__main__":
    main()
