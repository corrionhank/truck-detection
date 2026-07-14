# Model — Keypoint R-CNN moving-echo detector (v0, overfit demo)

_Last updated: 2026-07-10._

> **⚠️ Superseded numbers.** Everything below is the **initial 15-vehicle** experiment (3 scenes). The
> working set is now **339 vehicles / 8 scenes** in `data/active/` (see [DATA.md](DATA.md)); paths have moved
> under `data/active/`. These results are kept as the historical pipeline-validation record and will be
> refreshed when the model is retrained on the full set.

First end-to-end pass of the Adamiak-style detector on the data in hand. This is the
pipeline-validation milestone from [CONTEXT.md](CONTEXT.md) §9 step 5 — **not** a
model that generalises. Read the data-reality note before quoting any number.

## Pipeline

```
Annotations-RGB.gpkg  +  imagery/all_geotiffs/*.tif
        │  src/export_coco.py   (join by `scene`, inverse affine → pixel space)
        ▼
data/coco/annotations.json  +  data/coco/images/*.png   (64×64 true-colour chips)
        │  src/train_keypoint_rcnn.py   (fine-tune from COCO-pretrained weights)
        ▼
weights/keypoint_rcnn_echo.pt
        │  src/infer_keypoints.py
        ▼
outputs/keypoint_predictions.png   (predicted crosses vs ground-truth circles)
```

## Model

- **Architecture:** Keypoint R-CNN, ResNet-50 + FPN (torchvision `keypointrcnn_resnet50_fpn`).
- **Transfer learning:** initialised from COCO person-keypoint weights, then the box head
  is re-sized to 1 object class (`moving_echo`) and the keypoint head to **3 keypoints**
  (blue → red → green, the SuperDove capture order). Adamiak trained from scratch on 3,236
  echoes; with 15 we fine-tune instead so the tiny set has a chance.
- **Input:** 64×64 true-colour chips (bands R6/G4/B2, 2–98 % per-scene stretch), one per vehicle.
- **Training:** CPU, SGD lr 0.002 + 1-epoch warmup + grad-clip 5.0, 40 epochs, batch 2.

## Results (overfit — trained and evaluated on the same 15 vehicles)

| metric | value |
|---|---|
| vehicles | 15 (3 scenes: Bellingham, Centralia, Stanwood) |
| detected (score > 0.3) | 15 / 15 |
| mean keypoint error | **0.18 px ≈ 0.5 m** at 3 m/px |
| median / max error | 0.16 / 0.46 px |
| train loss | 8.79 → 1.38 over 40 epochs |

These numbers only show the pipeline *fits* — export → train → infer → score all work and
the model places B/R/G keypoints on the echoes. They say **nothing about generalisation**
(no held-out set is meaningful at n=15). The box-classifier confidence caps ~0.45 because the
head is trained from scratch on sub-pixel objects, so inference accepts detections at 0.3.

## Full-scene inference & the translation-jitter fix

`src/detect_scene.py` slides the detector across a whole GeoTIFF (road pixels only),
dedupes across overlapping windows, and maps each detection back to a UTM coordinate
— the deployment path. Two model versions were tested on Stanwood_08 (3 labelled trucks):

| model | training crops | stride 40 (coarse) recall | note |
|---|---|---|---|
| v0 `keypoint_rcnn_echo` | 15 pixel-centered | **0 / 3** | only fires on dead-centre vehicles |
| v1 `keypoint_rcnn_echo_jitter` | 15 × ±20px jitter + flips | **3 / 3** | robust to off-centre; scores 0.9+ vs v0's 0.4 |

**Why v0 failed the scene test:** every v0 crop was pixel-centered on the vehicle, so
the model learned "echo at dead centre" and a coarse sliding window never centred a truck
tightly enough. `train_keypoint_rcnn_jitter.py` fixes this by cropping *jittered* windows
straight from the rasters (vehicle offset ±20px, random flips) — `detect_scene.py` at a
coarse stride then recovers all 3 labelled trucks.

**Precision is still unmeasurable, not necessarily bad.** On Stanwood_08 the jitter model
returns 18 detections at stride 40: 3 match labels, 15 elsewhere — but the scene was only
*partially* labelled (3 of its many vehicles). Eyeballing `outputs/<scene>_det_montage.png`,
most high-confidence "extras" (score ≥0.7) are clean color-separated echoes = real unlabelled
trucks; the low-confidence ones (≤0.5) sit on lane-paint/road texture = true false positives.
A ≥0.7 threshold keeps ~11 clean detections. **Dense labelling is required to score precision
honestly.**

## Reproduce

```bash
python3 src/export_coco.py                     # build the COCO dataset + chips
python3 src/train_keypoint_rcnn.py             # v0: ~6 min CPU -> weights/keypoint_rcnn_echo.pt
python3 src/infer_keypoints.py                 # chip-level eval -> outputs/keypoint_predictions.png
python3 src/train_keypoint_rcnn_jitter.py      # v1 w/ jitter: ~30 min CPU -> ..._jitter.pt
python3 src/detect_scene.py Stanwood_08_20260504 \
        --stride 40 --weights weights/keypoint_rcnn_echo_jitter.pt   # full-scene detection
```

## Notes & gotchas

- **Run on CPU.** MPS produces NaN losses with torchvision detection models (RoIAlign/anchor
  math); `--device cpu` is the default for this reason.
- **`scene` whitespace bug:** one annotation row had a leading space (`" Bellingham_01…"`).
  `export_coco.py` strips it — see [DATA.md](DATA.md).
- One vehicle had only 2 of 3 keypoints and is dropped at export (15 of 16 kept).

## What's needed to make this real

More labels — this is the binding constraint. 15 vehicles → hundreds/thousands, ideally
across more of the 15 scenes, with a genuine held-out test set. The signal is clearly visible
(see `outputs/coco_qa_montage.png`), so the payoff is there; the pipeline is now ready to
absorb the labels as they come.
