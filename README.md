# FaRMAE / TO-ProgressiveMAE — runnable reproduction scaffold

This repository is a **faithful, executable implementation scaffold** of the two-stage failure-aware RGB–D pretraining
described in the paper *Failure-Aware Multimodal Representation Learning for Transparent-Object Depth Completion*.

The paper defines:
- **Stage I**: failure-aware cross-modal alignment with a spatial weight `w_p` derived from a failure likelihood map `S_f`.  
  (Eq. (1) in the PDF)  
- **Stage II**: structured masked autoencoding with **masking biased toward high `S_f`** and a masked reconstruction loss
  on depth. (Eq. (2) in the PDF)

This repo runs end-to-end on a **synthetic RGB–D dataset** (for sanity checks), and provides **dataset adapters** you can
fill for ScanNet / NYUv2 / ClearGrasp.

---

## 1. Installation

```bash
pip install -r requirements.txt
```

Tested with Python ≥ 3.9 and PyTorch ≥ 2.0.

---

## 2. Quick start (synthetic data)

### Stage I: failure-aware alignment (InfoNCE)

```bash
python -m tools.train_stage1 --config configs/stage1.yaml
```

### Stage II: structured MAE on depth

```bash
python -m tools.train_stage2 --config configs/stage2.yaml --resume checkpoints/stage1_last.pt
```

### Downstream: toy depth completion fine-tuning

```bash
python -m tools.finetune_depth_completion --config configs/finetune_dc.yaml --resume checkpoints/stage2_last.pt
```

Checkpoints are saved into `./checkpoints`.

---

## 3. Failure likelihood map S_f

The PDF states that `S_f` is in `[0,1]^{H×W}` and can be instantiated from visual+geometric cues such as **image gradient**,
**texture sparsity**, and **depth discontinuity**, while remaining agnostic to the exact estimator.

We provide a deterministic, reproducible instantiation:

- `g(p)`: normalized image gradient magnitude (Sobel on grayscale).
- `s(p)`: texture sparsity proxy (higher when local texture is weak) using local standard deviation.
- `d(p)`: normalized depth discontinuity magnitude (Sobel on observed depth).

Then:
`S_f = sigmoid(a*g + b*s + c*d)` with configurable `(a,b,c)`.

See: `farmae/failure_map.py`.

---

## 4. What you should still plug in (for a full reproduction)

This scaffold includes the core algorithmic pieces and training loops.
To match your released GitHub implementation exactly, you typically need to:
1. Implement **exact dataset preprocessing** and splits (ScanNet / NYUv2 / ClearGrasp).
2. Match your **encoder choice** (ViT size, patch size) and any data augmentations.
3. Match exact hyperparameters (LR, wd, epochs, mask ratio, resolution, etc.).
4. (If applicable) add any auxiliary losses mentioned in your full paper draft.

---

## 5. Repo structure

```
farmae/
  models/                 # encoders, projection heads, MAE decoder, downstream head
  data/                   # synthetic dataset + adapters (placeholders)
  failure_map.py          # S_f computation (reproducible)
  masking.py              # structured masking biased by S_f
  losses.py               # weighted InfoNCE + MAE reconstruction loss
tools/
  train_stage1.py
  train_stage2.py
  finetune_depth_completion.py
configs/
  stage1.yaml
  stage2.yaml
  finetune_dc.yaml
```

---

## 6. License

For your own paper/code release.
