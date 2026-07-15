# Base (default anchors, no aug)

`base-default` · created 2026-07-14 · weights `base-default.pt`

## What this model is
_One-line purpose — edit me._

## Methodology
- **Architecture:** Keypoint R-CNN (ResNet-50 + FPN), COCO-pretrained, head → 1 class + 3 keypoints (B/R/G).
- **Anchors:** `default` (32-512 px).
- **Augmentation:** `none`.
- **Data:** 213 vehicles across 3 scene(s): Centralia_01_20260511, Centralia_02_20260511, Tacoma-Centralia_02_20260602.
- **Training:** 12 epochs · batch 4 · lr 0.005 · cpu.

## Results (held-out Tacoma-Centralia_01_20260429)
- Centered-chip recall: **0.968**  ·  keypoint error: 0.7 px
- Full-scene deployment: _run in the Inference tab and record here._

## Findings / what worked / what didn't
_Edit me — this is the point of the card._

## Next ideas
_Edit me._
