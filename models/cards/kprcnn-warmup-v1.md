# Keypoint R-CNN (jitter+MV, LR warmup, 20-scene)

`kprcnn-warmup-v1` · created 2026-07-29 · weights `kprcnn-warmup-v1.pt`

## What this is
A **fresh** Keypoint R-CNN moving-echo detector following **Adamiak et al. 2025**, rebuilt from scratch on the
intact data pipeline (not derived from the archived training scripts). It replaces the archived experiments as
the current, understood baseline for the detector rebuild.

## Relative to the base data and the earlier models
- **Base data:** the same 339-vehicle / 8-scene corpus, exported to **64×64 COCO chips** by `export_coco.py`
  (see [DATA.md](../../docs/DATA.md) §6). Unchanged and upstream of this model.
- **`base-default` (baseline):** the zero-effort reference (default anchors, no augmentation). This model adds
  Adamiak's architecture choices + augmentation + small anchors on top of that starting point.
- **Benchmark to beat:** the archived `kprcnn-centralia-heldout` reached **full-scene F1 ≈ 0.50** (matching Van
  Etten 2024's PlanetScope truck F1 0.49) on the same held-out scene. This first pass is measured against that.

## Methodology (Adamiak architecture)
- **Model:** Keypoint R-CNN, `keypointrcnn_resnet50_fpn` (ResNet-50 + FPN).
- **Classes:** 2 (moving echo + background). **Keypoints:** 3 per
  vehicle, order blue → red → green.
- **Anchors (the high-leverage knob):** sizes `[4, 8, 16, 32, 48]`, ratios `[0.25, 0.5, 0.75, 1.0, 1.25]` —
  small anchors in Adamiak's spirit (their swept best was 4–48 px on 512² images). **Not swept here.** Sweep
  later via `--anchor-sizes` / `--aspect-ratios` (the registry stores them, so `model_registry` rebuilds the
  right graph). Input resized `min_size=192` / `max_size=320`.
- **Training:** Adam · ReduceLROnPlateau on validation loss · LR 1e-3 → 1e-5 · grad-clip 1.5 · composite
  torchvision loss · 16 epochs · batch 4 · cpu.
- **Augmentation:** rotate+flip+brightness+perspective (Adamiak).
- **Data:** trained on 624 vehicles across 20 scenes; held out 1 untrained scene(s) for testing (below).

## Deviations from Adamiak (our setup differs)
- **Finetuned from the COCO-pretrained backbone** (`weights="DEFAULT"`), not trained from scratch — our label
  count is far below their 3,236. Detection heads (RPN / box / keypoint) are trained fresh (custom
  anchors/classes/keypoints).
- **64×64 chips**, not their 512×512 images (kept our chip size; anchors may need a sweep because of it).
- **Leave-one-scene-out** split, not random 80/10/10 (a random split leaks same-scene cues and inflates).

## Results

Every number below is measured on scenes the model **never saw in training** (data-separated —
no leakage). *Centered* recall is the easy "recognise a centered echo" metric; *full* R/P/F1 is the
deployable sliding-window metric (threshold 0.3).

| held-out (untrained) scene | veh | centered | full R / P | full F1 |
|---|---:|---:|---:|---:|
| `Tacoma-Centralia_01_20260429` | 80 | 1.0 | 0.45 / 0.30 | **0.36** |

**Mean across held-out scenes: centered 1.0 · full-scene F1 0.36.**

## Not built yet (deliberately, for later)
Keypoint correction · the anchor sweep · threshold calibration · the geometry/physics filter · velocity.
See [REFINEMENT.md](../../docs/REFINEMENT.md).
