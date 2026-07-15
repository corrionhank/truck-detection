# Keypoint R-CNN v0 (overfit demo)

`kprcnn-echo-v0` · created 2026-07-10 · weights `keypoint_rcnn_echo.pt`

## What this model is
The first end-to-end pass — built to validate the pipeline (export → train → infer), not to work.

## Methodology
- **Architecture:** Keypoint R-CNN (ResNet-50 + FPN), COCO-pretrained, head re-sized to 1 class
  (`moving_echo`) + 3 keypoints (blue/red/green).
- **Anchors:** default (32–512 px).
- **Augmentation:** none.
- **Data:** 15 vehicles, 3 scenes (Bellingham_01, Centralia_02, Stanwood_08).
- **Training:** 40 epochs · batch 2 · lr 0.002 · CPU. Warmup + gradient clipping added to stop divergence.

## Results
- Chip-level (trained *and* evaluated on the same 15 chips): 0.18 px keypoint error — i.e. it memorised them.
- Coarse full-scene sliding window: **0 trucks found.**

## Findings — the mistake that taught us the most
- **Overfit to center.** Every training crop was pixel-centered on the vehicle, so the model learned "echo at
  dead center" and had *no translation tolerance*. On a real scene, where a truck is rarely dead-center in a
  window, it fired on nothing. This is why v1 adds translation jitter.
- **MPS diverges.** Apple-GPU (MPS) produced NaN losses within a couple of iterations → forced CPU. This is a
  torchvision-detection-on-MPS issue, not a data problem.

## Next ideas
Superseded by the jitter model (`kprcnn-echo-jitter`). Kept as the record of the overfit-to-center lesson.
