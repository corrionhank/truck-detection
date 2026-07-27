# Keypoint R-CNN V3 - Training & Inference Test

`kprcnn-3` · created 2026-07-20 · weights `kprcnn-3.pt`

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
- **Anchors (the high-leverage knob):** sizes `[8, 16, 32, 64, 128]`, ratios `[0.5, 1.0, 2.0]` —
  small anchors in Adamiak's spirit (their swept best was 4–48 px on 512² images). **Not swept here.** Sweep
  later via `--anchor-sizes` / `--aspect-ratios` (the registry stores them, so `model_registry` rebuilds the
  right graph). Input resized `min_size=192` / `max_size=320`.
- **Training:** Adam · ReduceLROnPlateau on validation loss · LR 1e-3 → 1e-5 · grad-clip 1.5 · composite
  torchvision loss · 12 epochs · batch 4 · cpu.
- **Augmentation:** rotate+flip+brightness+perspective (Adamiak).
- **Data:** trained on 260 vehicles across 9 scenes; held out 3 untrained
  scene(s) for testing (below). Training scenes: `Bellingham_01_20260425`, `Centralia_01_20260511`, `Ellensburg-Yakima_01_20260409`, `EllensburgPreferredTest_01_20260530`, `Stanwood_10_20260511`, `Tacoma-Centralia_01_20260429`, `Tacoma-Centralia_02_20260602`, `Yakima-Toppenish_01_20260503`, `blaine-bellingham_01_20260425`.

## Deviations from Adamiak (our setup differs)
- **Finetuned from the COCO-pretrained backbone** (`weights="DEFAULT"`), not trained from scratch — our label
  count is far below their 3,236. Detection heads (RPN / box / keypoint) are trained fresh (custom
  anchors/classes/keypoints).
- **64×64 chips**, not their 512×512 images (kept our chip size; anchors may need a sweep because of it).
- **Leave-one-scene-out** split, not random 80/10/10 (a random split leaks same-scene cues and inflates).

## Results — on untrained (held-out) scenes only

Every number below is measured on scenes the model **never saw in training** (data-separated — no leakage).
*Centered* recall is the easy "recognise a centered echo" metric; *full* R/P/F1 is the deployable sliding-window
metric (threshold 0.3, so threshold-dependent).

| held-out (untrained) scene | veh | centered | full R / P | full F1 |
|---|---:|---:|---:|---:|
| `Centralia_02_20260511` | 56 | 0.964 | 0.20 / 0.14 | **0.16** |
| `Ellensburg_01_20260504` | 13 | 0.923 | 0.15 / 0.09 | **0.11** |
| `auburn-snoqualmie_01_20260425` | 24 | 1.0 | 0.12 / 0.02 | **0.03** |

**Mean across held-out scenes: centered 0.962 · full-scene F1 0.103.** Cross-*corridor* scenes (a
region absent from training) are the honest generalization test; same-corridor held-out scenes measure
generalization to new traffic on a known road.

## Not built yet (deliberately, for later)
Keypoint correction · the anchor sweep · threshold calibration · the geometry/physics filter · velocity.
See [REFINEMENT.md](../../docs/REFINEMENT.md).
