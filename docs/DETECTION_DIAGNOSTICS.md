# Detection-path diagnostics — `kprcnn-jitter-mv`

Raw measurements of the inference path: head configuration, discarded model outputs, matching method,
score calibration, per-scene threshold behaviour, recall ceiling, and box geometry.
Numbers only. Run 2026-07-29. Match radius throughout: **6 px = 18 m** (3.0 m/px GSD).

Scripts: `diag_run.py` (tasks 3–5), `diag_ceiling.py` (tasks 6, 7b).
Outputs: `outputs/diag_task4_calibration.csv`, `outputs/diag_task5_sweep.csv`,
`outputs/diag_task3_match.json`, `outputs/diag_ceiling.log`.

---

## 1. Head configuration

Built via `model_registry.build_model()` on the `kprcnn-jitter-mv` registry entry.

### `model.rpn`

| parameter | value | source |
|---|---|---|
| `_pre_nms_top_n` | `{'training': 2000, 'testing': 1000}` | torchvision default |
| `_post_nms_top_n` | `{'training': 2000, 'testing': 1000}` | torchvision default |
| `nms_thresh` | `0.7` | torchvision default |
| `fg_iou_thresh` | `0.7` | torchvision default |
| `bg_iou_thresh` | `0.3` | torchvision default |
| `batch_size_per_image` | `256` | torchvision default |
| `positive_fraction` | `0.5` | torchvision default |

### `model.roi_heads`

| parameter | value | source |
|---|---|---|
| `score_thresh` | `0.05` | torchvision default |
| `nms_thresh` | `0.5` | torchvision default |
| `detections_per_img` | `100` | torchvision default |
| `fg_iou_thresh` | `0.5` | torchvision default |
| `bg_iou_thresh` | `0.5` | torchvision default |
| `batch_size_per_image` | `512` | torchvision default |
| `positive_fraction` | `0.25` | torchvision default |

**Overrides in the repo: none of the above.** A repo-wide grep for all 14 parameter names across `src/` and
`backend/` returns one hit: `src/model_registry.py:63`, `rpn_anchor_generator=ag`. The only architecture
modifications are the custom `AnchorGenerator` (sizes 4/8/16/32/48, ratios 0.25–1.25) and the two replaced
predictor heads (`FastRCNNPredictor` 2 classes, `KeypointRCNNPredictor` 3 keypoints). All detection/sampling
hyperparameters are stock.

---

## 2. Model outputs consumed vs discarded (`src/detect_scene.py`)

| output key | read? | where |
|---|---|---|
| `out["scores"]` | yes | L133 `len()`, L135 `float(out["scores"][0])` |
| `out["keypoints"]` | yes | L138 `out["keypoints"][0].numpy()[:, :2]` |
| `out["keypoints_scores"]` | **no — 0 occurrences in file** | — |
| `out["boxes"]` | **no — 0 occurrences in file** | — |

Both indexed accesses use `[0]` only: every detection after the highest-scoring one in each window is
discarded before any threshold or dedup logic runs.

---

## 3. Matching asymmetry

_(pending — `diag_run.py` in progress)_

Method A (current, `detect_scene`): recall counts labels having ≥1 kept detection within 6 px; precision counts
kept detections having ≥1 label within 6 px. Neither side consumes its match, so a label may satisfy multiple
detections and a detection may satisfy multiple labels.

Method B (greedy bipartite): detections sorted score-descending; each takes its nearest unconsumed label within
6 px; each label consumed at most once.

Reported per labelled scene at threshold 0.3: labels within 18 m of >1 kept detection; kept detections within
18 m of >1 label; then P/R/F1 under both methods side by side.

---

## 4. Score calibration table

_(pending — `diag_run.py` in progress)_

`outputs/diag_task4_calibration.csv`, one row per kept detection across all labelled scenes, generated at
threshold **0.0** so the low-score tail is retained. Columns:

```
scene, score, kp_score_blue, kp_score_red, kp_score_green, dist_to_nearest_label_m, matched_bool
```

Kept-detection counts at threshold 0.0 observed so far (partial): `Tacoma-Centralia_01` 171 kept / 574 windows /
80 labels; `EllensburgPreferredTest_01` 322 kept / 1187 windows / 2 labels; `Stanwood_10` 49 kept / 160 windows /
6 labels; `Ellensburg_01` 15 kept / 54 windows / 13 labels.

---

## 5. Per-scene threshold sweep

_(pending — `diag_run.py` in progress)_

`outputs/diag_task5_sweep.csv`, thresholds 0.05→0.95 in 0.05 steps, per labelled scene: P, R, F1, raw kept count;
plus per-scene argmax-F1 threshold and per-scene median detection score.

---

## 6. Recall ceiling (RPN + window grid)

Scene `Tacoma-Centralia_01_20260429`: **574 windows, 80 labels**, top-1-per-window **disabled**, cross-window
dedup **disabled**, threshold **0**.

| configuration | raw detections | labels with ≥1 detection ≤18 m | fraction |
|---|---:|---:|---:|
| defaults (`score_thresh` 0.05, `nms` 0.5, `dpi` 100) | **931** | **80 / 80** | **1.000** |
| `score_thresh` 0.0 | _(pending)_ | | |
| `score_thresh` 0.0, NMS off, `dpi` 1000 | _(pending)_ | | |

For comparison, the same scene through the **normal** path (top-1 per window + 32 px dedup) yields **171** kept
detections at threshold 0.0, and at threshold 0.3 the deployed configuration reported recall 0.600 (48/80).

---

## 7. Box geometry and internal NMS

### 7a. Ground-truth box dimensions — `data/active/coco/annotations.json`

n = **1286** annotations; export 96 px (chip 64 + margin 16).

| | min | p25 | median | p75 | p90 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| width (px) | 6.05 | 8.74 | 11.69 | 15.24 | 17.02 | 24.26 | 12.04 |
| height (px) | 6.05 | 11.74 | 15.41 | 17.40 | 18.51 | 21.30 | 14.37 |
| area (px²) | 74.4 | | 164.8 | | | 282.0 | |

Fraction at the 4×4 minimum:

| test | count | fraction |
|---|---:|---:|
| width == 4.0 | 0 / 1286 | 0.0 % |
| height == 4.0 | 0 / 1286 | 0.0 % |
| both == 4×4 | 0 / 1286 | 0.0 % |
| either == 4.0 | 0 / 1286 | 0.0 % |

Smallest observed dimension 6.05 px. The 4.0 floor is applied in `train_detector.to_target()`
(`max(x1 - x0, 4.0)`), not in `export_coco.py`; exporter boxes are keypoint-extent + 3 px pad per side.

### 7b. Internal box NMS @ IoU 0.5 suppression

_(pending — `diag_ceiling.py` in progress)_

Same scene, measured as total raw detections across all 574 windows with `roi_heads.nms_thresh = 1.0`
(no suppression) minus the same with `nms_thresh = 0.5`, both at `score_thresh = 0.0`,
`detections_per_img = 1000`.

---

## Notes on measurement validity

`src/detect_scene.py:load_gt_reds()` applies the raster inverse-affine directly to GeoPackage coordinates with
no reprojection. For scenes whose raster CRS differs from the GeoPackage CRS — the two EPSG:32611 scenes
imported 2026-07-29 (`Tri-Cities-New_01`, `20260622_184442_88_2560_edit`) — those pixel coordinates are wrong.
The diagnostic scripts here reproject points into each raster's native CRS before the inverse affine
(`rasterio.warp.transform`), matching the corrected `export_coco.py` behaviour.

Corpus at time of run: **789 vehicles / 21 labelled scenes**.
