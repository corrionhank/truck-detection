# Minimal training repo — bootstrap + hardware check

_A stripped-down setup to test whether **your hardware can train** the moving-echo Keypoint R-CNN, with
none of this repo's data / frontend / docs bloat. Copy this one file into a fresh folder and follow it._

---

## 0. The point

Before wiring up satellite data, answer one question: **does training the actual model architecture even
run correctly on your machine?** Detection models (Keypoint R-CNN) lean on ops (RoIAlign, anchor math) that
some GPU backends implement incorrectly. You can test this in ~2 minutes with **synthetic data — no dataset,
no downloads** — using the smoke test in §3.

## 1. The stack

**Just to test hardware** (the §3 smoke test uses random tensors):

```bash
pip install torch torchvision
```

**Full model loop** (once you plug in real data — §5) adds four packages:

| Package | Role |
|---|---|
| `torch` + `torchvision` | the model — Keypoint R-CNN lives in torchvision |
| `numpy` | arrays |
| `rasterio` | reads the imagery (8-band GeoTIFF pixels) |
| `pillow` | the 64×64 training chips (PNG) |
| `geopandas` | reads the labels (annotation GeoPackage → x/y) |

```bash
# fresh repo, isolated env, full stack
mkdir echo-train && cd echo-train
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchvision numpy rasterio pillow geopandas
```

That's the whole dependency list. **Not needed:** `pycocotools` (COCO JSON is hand-rolled), `opencv`
(only for rotation/scale augmentation), `flask` (only the web UI).

## 2. Files you need

Exactly one, to start: `train_smoke.py` (below). Everything else in this doc is prose.

## 3. The hardware smoke test

Trains a few steps of the real Keypoint R-CNN on synthetic data and reports whether the loss stays sane on
a given device. Save as `train_smoke.py`:

```python
#!/usr/bin/env python3
"""Minimal hardware smoke test: can this machine TRAIN a Keypoint R-CNN?
Synthetic data — no dataset, no downloads. Run: python3 train_smoke.py --device cpu|mps|cuda"""
import argparse, math, os, time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn

DIVERGE = 1e4   # a healthy loss here is < ~20; anything this big = numerically broken

def fake_batch(n, device):
    imgs, targets = [], []
    for _ in range(n):
        imgs.append(torch.rand(3, 256, 256, device=device))
        targets.append({
            "boxes": torch.tensor([[60., 60., 180., 180.]], device=device),
            "labels": torch.ones(1, dtype=torch.int64, device=device),
            "keypoints": torch.tensor([[[90., 90., 1.], [120., 120., 1.], [150., 150., 1.]]], device=device),
        })
    return imgs, targets

def main(device_name, iters, batch):
    dev = torch.device(device_name)
    print(f"torch {torch.__version__} | device {dev} | mps={torch.backends.mps.is_available()} | cuda={torch.cuda.is_available()}")
    model = keypointrcnn_resnet50_fpn(weights=None, weights_backbone=None,
                                      num_classes=2, num_keypoints=3,
                                      min_size=192, max_size=320).to(dev)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    broken, t0 = False, time.time()
    for i in range(iters):
        imgs, targets = fake_batch(batch, dev)
        loss = sum(model(imgs, targets).values())
        opt.zero_grad(); loss.backward(); opt.step()
        val = float(loss)
        ok = math.isfinite(val) and abs(val) < DIVERGE     # catch explosion, not just NaN
        broken = broken or not ok
        print(f"  iter {i+1}/{iters}  loss={val:12.3f}  ok={ok}")
    dt = (time.time() - t0) / iters
    verdict = "BROKEN — loss diverged/NaN; this device can't train the model as-is" if broken \
        else "OK — stable finite loss; training works"
    print(f"RESULT [{device_name}]: {verdict} | {dt:.2f} s/iter")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--batch", type=int, default=2)
    a = ap.parse_args()
    main(a.device, a.iters, a.batch)
```

Run it on each device your machine has:

```bash
python3 train_smoke.py --device cpu
python3 train_smoke.py --device mps     # Apple Silicon
python3 train_smoke.py --device cuda    # NVIDIA
```

## 4. Reading the result — and what this Mac actually does

A **healthy** run: loss starts ~10 and drifts *down*, every iter `ok=True`, ending `RESULT: OK`.
A **broken** device: loss explodes (10 → 1e8 → NaN) within 2–3 iters, `RESULT: BROKEN`.

Measured on this machine (Apple Silicon, torch 2.13):

| Device | Result | Speed | Notes |
|---|---|---|---|
| **CPU** | ✅ OK — stable (9.7 → 8.7) | ~3 s/iter | correct, but slow |
| **MPS** (Apple GPU) | ❌ **BROKEN** — 10 → 3×10⁸ → NaN by iter 3 | ~3–5 s/iter | *runs, isn't faster, and diverges* |

**The key lesson:** MPS **runs without erroring** and looks fine on iter 1 — then the loss silently explodes.
torchvision's detection ops are numerically unstable on the MPS backend (and `PYTORCH_ENABLE_MPS_FALLBACK=1`
doesn't save it — the bad ops don't fall back). So on Apple Silicon you **cannot train this model on the GPU
today**; CPU is the only correct path, and it's slow.

This is *the* reason to run the smoke test before investing in a data pipeline: a device can appear to work
(no crash, first loss looks normal) while being completely broken for training.

## 5. Your options for real training

- **CPU** — works, slow. Fine for small experiments (hundreds of vehicles, a handful of epochs).
- **NVIDIA CUDA GPU (cloud)** — the real fix for speed. Any Linux+NVIDIA box (Colab, a cloud VM, a lab
  machine); the smoke test should print `RESULT [cuda]: OK` and run ~10–50× faster than CPU. This is where
  to go once the model is worth training at scale.
- **MPS** — not reliable for this model. The only thing worth trying is a newer PyTorch nightly (MPS op
  coverage improves over releases); re-run the smoke test after upgrading. Don't build a workflow on it until
  `RESULT [mps]: OK`.

## 6. Plugging in real data (after the smoke test passes)

Once `train_smoke.py` reports OK on your device, replace synthetic data with real:

- **Labels:** `geopandas.read_file("annotations.gpkg", layer="Annotations")` → points tagged
  `vehicle_id` / `sequence` (1=blue, 2=red, 3=green) / `scene`.
- **Pixels:** `rasterio.open(f"{scene}.tif")` → read bands 6/4/2 (R/G/B), 2–98 % stretch, crop a 64×64 chip
  centered on each vehicle; the inverse affine maps the point's UTM coords → pixel.
- **Target per chip:** 3 keypoints `[x, y, v=2]` + a box from their extent + one class.

That's the entire data contract — see this repo's `docs/DATA.md` for the exact schema. The model-building
call is identical to the smoke test; you only swap `fake_batch()` for a real `Dataset`.
