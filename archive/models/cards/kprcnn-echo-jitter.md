# Keypoint R-CNN v1 (+ translation jitter)

`kprcnn-echo-jitter` · created 2026-07-10 · weights `keypoint_rcnn_echo_jitter.pt`

## What this model is
v0 + the one fix that made full-scene detection work: translation-jitter augmentation.

## Methodology
- **Architecture:** same as v0 — Keypoint R-CNN (ResNet-50 + FPN), 1 class + 3 keypoints.
- **Anchors:** default (32–512 px).
- **Augmentation:** **translation jitter ±20 px + horizontal/vertical flips.** Crops are pulled from the full
  raster with a random offset (via `JitteredSceneDataset`) rather than always centered, so the model sees the
  vehicle off-center.
- **Data:** 15 vehicles, 3 scenes (same as v0).
- **Training:** 30 epochs · batch 2 · lr 0.005 · CPU.

## Results
- Coarse-stride sliding window (Stanwood_08): **3/3 trucks** (vs 0/3 for v0).
- Detection confidence jumped from ~0.4 (v0) to ~0.9.

## Findings — what worked
- **Jitter fixed the overfit-to-center failure.** Teaching the model to detect an off-center vehicle is what
  turned a 0% full-scene detector into a working one. Augmentation > more epochs here.
- Still only 15 vehicles / 3 scenes → does **not** generalize across scenes (cross-scene recall swung 0–100%
  depending on which scene was held out). Data scale, not method, was the next bottleneck.

## Next ideas
More data (→ the 339-vehicle Centralia model) and small anchors for the tiny streaks.
