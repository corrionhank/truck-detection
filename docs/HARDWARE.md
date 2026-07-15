# Hardware — can this machine train the model?

Detection models (Keypoint R-CNN) lean on ops (RoIAlign, anchor math) that some GPU backends implement
incorrectly — a device can **run without erroring and still be numerically broken for training**. Answer this
in ~2 minutes with synthetic data (no dataset, no downloads) *before* wiring up the real pipeline.

## The finding (why training is CPU-only)

Measured on this machine (Apple Silicon, torch 2.13):

| Device | Result | Speed | Notes |
|---|---|---|---|
| **CPU** | ✅ OK — stable loss (9.7 → 8.7) | ~3 s/iter | correct, but slow |
| **MPS** (Apple GPU) | ❌ **BROKEN** — 10 → 3×10⁸ → NaN by iter 3 | ~3–5 s/iter | runs, isn't faster, diverges |

**MPS looks fine on iter 1, then the loss silently explodes** — torchvision's detection ops are unstable on the
MPS backend, and `PYTORCH_ENABLE_MPS_FALLBACK=1` doesn't save it (the bad ops don't fall back). So on Apple
Silicon you **cannot train this model on the GPU today**; CPU is the only correct path. Real speed needs a
Linux + NVIDIA (CUDA) box — Colab, a cloud VM, or a lab machine — which runs ~10–50× faster; re-run the smoke
test there and expect `RESULT [cuda]: OK`. Retry MPS only after a newer PyTorch nightly, and only build on it
once it prints `OK`.

## The smoke test

Trains a few steps of the real Keypoint R-CNN on synthetic data and reports whether the loss stays sane. A
healthy run starts ~10 and drifts *down* (`ok=True` every iter); a broken device explodes to NaN within 2–3
iters. Save as `train_smoke.py`, run `python3 train_smoke.py --device cpu|mps|cuda`:

```python
#!/usr/bin/env python3
"""Minimal hardware smoke test: can this machine TRAIN a Keypoint R-CNN?
Synthetic data — no dataset, no downloads. Run: python3 train_smoke.py --device cpu|mps|cuda"""
import argparse, math, os, time
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn

DIVERGE = 1e4   # a healthy loss is < ~20; anything this big = numerically broken

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

## Dependencies

The smoke test needs only `torch` + `torchvision`. The full pipeline adds `numpy`, `rasterio` (reads the
8-band GeoTIFFs), `pillow` (64×64 chips), `geopandas` (reads the label GeoPackage). Not needed: `pycocotools`
(COCO JSON is hand-rolled), `opencv` (only for rotation/scale augmentation), `flask` (only the web console).
Once the smoke test passes, swap `fake_batch()` for a real `Dataset` — the model-building call is identical.
See [DATA.md](DATA.md) for the exact schema.
