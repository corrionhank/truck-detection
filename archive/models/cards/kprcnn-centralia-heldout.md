# Keypoint R-CNN (Centralia, small anchors)

`kprcnn-centralia-heldout` · created 2026-07-13 · weights `kprcnn_heldout_Tacoma-Centralia_01_20260429.pt`

## What this model is
The first model trained at real data scale (213 vehicles) with small anchors + rich augmentation. Built for
the held-out visualization; currently the backend default. **A throwaway, not a production model** — trained on
only 3 of the 4 Centralia scenes for 12 epochs.

## Methodology
- **Architecture:** Keypoint R-CNN (ResNet-50 + FPN), 1 class + 3 keypoints.
- **Anchors:** **small (8–128 px)** — the streaks are only 4–8 px, so default 32–512 px anchors are too large.
  (Adamiak 2025 swept anchors and landed on 4–48 px for the same signal.)
- **Augmentation:** **rich** — rotation ±25° + scale 0.85–1.20 + brightness, applied via one cv2 affine matrix
  to image *and* keypoints together (so they can't drift out of sync), on top of translation jitter + flips.
- **Data:** 213 vehicles, 3 scenes (Centralia_01, Centralia_02, Tacoma-Centralia_02).
- **Training:** 12 epochs · batch 4 · lr 0.005 · CPU.

## Results (held-out Tacoma-Centralia_01)
- Centered-chip recall: **79%** (74/94) · keypoint error ~1.2 px.
- Full-scene deployment (sliding window, thr 0.3): **40% recall / 68% precision**, F1 ~0.50.

## Findings
- **F1 ~0.50 matches Van Etten 2024's PlanetScope truck F1 (0.49)** — reproducing the state of the art at
  ~half the labels, on CPU.
- **Recognizing (79%) ≫ finding (40%).** A centered echo is easy; locating echoes across a raw scene is hard.
- **False positives = bright road / lane-paint** (not off-road hallucinations).
- **Misses concentrate in dense traffic** — the detection dedup (suppress within 96 m) merges neighbouring
  trucks. Both Adamiak and Van Etten report this exact failure; Van Etten used segmentation to sidestep it.
- **Low, flat confidences (~0.5)** → at threshold 0.5 it collapses to ~2% recall; needs thr ~0.3 or calibration.
- **Impossible geometries** occur on marginal detections (zig-zag / red not in the middle) — a physics filter
  would remove them (not yet built).

## Next ideas
Train a proper production model on all 339; add the geometry filter; fix the dedup radius for dense traffic;
consider Adamiak's keypoint-correction (snap to intensity peaks).
