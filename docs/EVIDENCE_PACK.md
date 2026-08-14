# Evidence Pack — Midway Feasibility Report

Moving-echo truck detection, PlanetScope SuperDove (~3 m GSD). Generated 2026-08-08.

**MODEL_A** = `kprcnn-adamiak-v2` · **MODEL_B** = `kprcnn-warmup-v1`

Machine-readable outputs: `outputs/evidence_pack/results_by_scene_model_threshold.csv` (126 rows) · `outputs/evidence_pack/aggregate_by_model_threshold_group.csv` (9 rows)


---

## 1. Data landscape

### 1.1 Scenes

| scene_id | corridor | acquisition_date | in_training_set | annotated_vehicles | annotated_keypoints | image_dims |
|---|---|---|---|---:|---:|---|
| `Tri-Cities-New_01_20260611` | Tri-Cities | 2026-06-11 | B | 105 | 315 | 12295x8836 |
| `Tacoma-Centralia_02_20260602` | Centralia/I-5-south | 2026-06-02 | A+B | 89 | 267 | 3493x7396 |
| `Tacoma-Centralia_01_20260429` | Centralia/I-5-south | 2026-04-29 | — | 80 | 240 | 3888x7410 |
| `Centralia_01_20260511` | Centralia/I-5-south | 2026-05-11 | A+B | 56 | 168 | 1596x5393 |
| `Centralia_02_20260511` | Centralia/I-5-south | 2026-05-11 | A+B | 56 | 168 | 1855x7333 |
| `spokane-i90_01_20260622` | Spokane | 2026-06-22 | B | 55 | 165 | 10328x3959 |
| `blaine-bellingham_02_20260528` | blaine-bellingham/I-5-north | 2026-05-28 | B | 50 | 150 | 7342x9609 |
| `Yakima-Toppenish_04_20260620` | Yakima | 2026-06-20 | B | 43 | 129 | 7296x6704 |
| `auburn-snoqualmie_03_20260614` | auburn-snoqualmie/I-90 | 2026-06-14 | B | 42 | 126 | 8562x7531 |
| `Yakima-Toppenish_02_20260509` | Yakima | 2026-05-09 | B | 31 | 93 | 7296x6704 |
| `blaine-bellingham_03_20260504` | blaine-bellingham/I-5-north | 2026-05-04 | B | 29 | 87 | 6005x8891 |
| `Yakima-Toppenish_03_20260614` | Yakima | 2026-06-14 | B | 27 | 81 | 7296x6704 |
| `Yakima-Toppenish_01_20260503` | Yakima | 2026-05-03 | B | 25 | 75 | 7296x6704 |
| `auburn-snoqualmie_01_20260425` | auburn-snoqualmie/I-90 | 2026-04-25 | B | 24 | 72 | 8561x7486 |
| `polygon_01_20260503` | polygon/Kent-Auburn | 2026-05-03 | B | 19 | 57 | 2644x4565 |
| `blaine-bellingham_01_20260425` | blaine-bellingham/I-5-north | 2026-04-25 | A+B | 18 | 54 | 6384x9349 |
| `Ellensburg_01_20260504` | Ellensburg | 2026-05-04 | A+B | 13 | 39 | 734x777 |
| `Bellingham_01_20260425` | blaine-bellingham/I-5-north | 2026-04-25 | A+B | 11 | 33 | 4309x5750 |
| `Ellensburg-Yakima_01_20260409` | Ellensburg | 2026-04-09 | A+B | 8 | 24 | 1935x7454 |
| `Stanwood_10_20260511` | Stanwood | 2026-05-11 | A+B | 6 | 18 | 1200x2496 |
| `EllensburgPreferredTest_01_20260530` | Ellensburg | 2026-05-30 | A+B | 2 | 6 | 2427x1530 |
| `Tacoma-Centralia_03_20260527` | Centralia/I-5-south | 2026-05-27 | — | 0 | 0 | 3888x8428 |
| `auburn-snoqualmie_04_20260504` | auburn-snoqualmie/I-90 | 2026-05-04 | — | 0 | 0 | 8476x6914 |
| `spokane-pullman_01_20260511` | Spokane | 2026-05-11 | — | 0 | 0 | 1509x8444 |

### 1.2 Summary

| metric | value |
|---|---|
| total scenes | 24 |
| annotated scenes | 21 |
| total annotated vehicles | 789 |
| total annotated keypoints | 2367 |
| date range | 2026-04-09 → 2026-06-22 (74 days) |

| corridor | vehicles | share |
|---|---:|---:|
| Centralia/I-5-south | 281 | 35.6 % |
| Yakima | 126 | 16.0 % |
| blaine-bellingham/I-5-north | 108 | 13.7 % |
| Tri-Cities | 105 | 13.3 % |
| auburn-snoqualmie/I-90 | 66 | 8.4 % |
| Spokane | 55 | 7.0 % |
| Ellensburg | 23 | 2.9 % |
| polygon/Kent-Auburn | 19 | 2.4 % |
| Stanwood | 6 | 0.8 % |

### 1.3 Preprocessing spec

- **Source bands:** PSB.SD band 6 → R, band 4 → G, band 2 → B (1-indexed)
- **Stretch:** per-scene, per-band percentile stretch, 2nd–98th percentile over nonzero pixels; nodata (0) forced to 0
- **Output dtype:** uint8 (0–255), divided by 255 to float at model input
- **Chip size:** 64 px model chip; exported at 96 px (`chip 64 + 2 × margin 16`)
- **Chips generated:** 789 chip images, 1286 keypoint annotations (629 center + 657 neighbor)
- **Chip acceptance rule:** vehicle retained only if it has exactly 3 keypoints with `sequence` ∈ {1,2,3}; chip rejected if the export window falls outside the raster
- **Train/val split rule:** no held-out val *scenes*; val is a random chip subset of the training scenes, `n_val = max(8, 0.12 × n_train_chips)`, seeded permutation, disjoint from training chips

---

## 2. Model breakdown

| | **MODEL_A** `kprcnn-adamiak-v2` | **MODEL_B** `kprcnn-warmup-v1` |
|---|---|---|
| architecture / backbone | Keypoint R-CNN, ResNet-50 + FPN | Keypoint R-CNN, ResNet-50 + FPN |
| pretrained weights | COCO-pretrained backbone (`weights=DEFAULT`) | COCO-pretrained backbone (`weights=DEFAULT`) |
| num_keypoints | 3 | 3 |
| anchor sizes | 4, 8, 16, 32, 48 | 4, 8, 16, 32, 48 |
| aspect ratios | 0.25, 0.5, 0.75, 1.0, 1.25 | 0.25, 0.5, 0.75, 1.0, 1.25 |
| optimizer | Adam | Adam |
| initial LR | 1e-3 | 1e-3 |
| LR schedule | ReduceLROnPlateau (factor 0.3, patience 2, min_lr 1e-5); **no warmup** | linear warmup 1e-5→1e-3 over **300 iters**, then ReduceLROnPlateau (factor 0.3, patience 2, min_lr 1e-5) |
| epochs trained | 12 | 16 |
| batch size | 4 | 4 |
| gradient clipping | clip_grad_norm 1.5 | clip_grad_norm 1.5 |
| augmentations | rotate ±180°, H/V flip, brightness ×0.8–1.2, perspective ±5 px | rotate ±180°, H/V flip, brightness ×0.8–1.2, perspective ±5 px, **+ translation jitter ±16 px** |
| chip format | 64 px, single-vehicle target | 96 px → random 64 px crop, **multi-vehicle targets** |
| training scenes used | 9 (228 training chips) | 20 (624 training chips) |
| val scenes used | **none** — 31 chips sampled from training scenes | **none** — 85 chips sampled from training scenes |
| checkpoint selection | **last epoch** | **lowest val loss**, restored at end |
| best val metric / epoch | not recorded | val loss **4.4943** @ **epoch 16** |

**Differences and their intent**

- **LR warmup** (B only) — intended to fix the loss spike when fresh detection heads train at full 1e-3 from iteration 1, which had previously forced a permanent LR reduction to 1e-4.
- **Best-val checkpointing** (B only) — intended to fix shipping a final-epoch checkpoint worse than an earlier epoch.
- **Translation jitter** (B only) — intended to fix over-fitting to vehicle-centered chips, since sliding-window inference presents vehicles off-center.
- **Multi-vehicle chip targets** (B only) — intended to fix neighbor vehicles inside a chip being trained as background.
- **Training set size** (9 → 20 scenes, 228 → 624 chips) — intended to increase corridor coverage.
- **Epochs** (12 → 16) — intended to allow convergence given warmup.

---

## 3. Results landscape

### 3.1 Per-scene results

Full CSV: `outputs/evidence_pack/results_by_scene_model_threshold.csv` — 126 rows (21 scenes × 2 models × 3 thresholds).

Columns: `scene_id, corridor, model, threshold, annotated_count, detected_count, TP, FP, FN, precision, recall, f1, count_ratio, signed_count_error_pct`

<details><summary>Full table (126 rows)</summary>

| scene_id | corridor | model | thr | ann | det | TP | FP | FN | P | R | F1 | count_ratio | signed_err_% |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Bellingham_01_20260425` | blaine-bellingham/I-5-north | MODEL_A | 0.3 | 11 | 42 | 7 | 35 | 4 | 0.1667 | 0.6364 | 0.2642 | 3.8182 | 281.82 |
| `Bellingham_01_20260425` | blaine-bellingham/I-5-north | MODEL_A | 0.5 | 11 | 23 | 7 | 16 | 4 | 0.3043 | 0.6364 | 0.4118 | 2.0909 | 109.09 |
| `Bellingham_01_20260425` | blaine-bellingham/I-5-north | MODEL_A | 0.85 | 11 | 6 | 3 | 3 | 8 | 0.5 | 0.2727 | 0.3529 | 0.5455 | -45.45 |
| `Centralia_01_20260511` | Centralia/I-5-south | MODEL_A | 0.3 | 56 | 70 | 36 | 34 | 20 | 0.5143 | 0.6429 | 0.5714 | 1.25 | 25.0 |
| `Centralia_01_20260511` | Centralia/I-5-south | MODEL_A | 0.5 | 56 | 58 | 34 | 24 | 22 | 0.5862 | 0.6071 | 0.5965 | 1.0357 | 3.57 |
| `Centralia_01_20260511` | Centralia/I-5-south | MODEL_A | 0.85 | 56 | 22 | 18 | 4 | 38 | 0.8182 | 0.3214 | 0.4615 | 0.3929 | -60.71 |
| `Centralia_02_20260511` | Centralia/I-5-south | MODEL_A | 0.3 | 56 | 71 | 31 | 40 | 25 | 0.4366 | 0.5536 | 0.4882 | 1.2679 | 26.79 |
| `Centralia_02_20260511` | Centralia/I-5-south | MODEL_A | 0.5 | 56 | 55 | 29 | 26 | 27 | 0.5273 | 0.5179 | 0.5225 | 0.9821 | -1.79 |
| `Centralia_02_20260511` | Centralia/I-5-south | MODEL_A | 0.85 | 56 | 17 | 12 | 5 | 44 | 0.7059 | 0.2143 | 0.3288 | 0.3036 | -69.64 |
| `Ellensburg-Yakima_01_20260409` | Ellensburg | MODEL_A | 0.3 | 8 | 7 | 1 | 6 | 7 | 0.1429 | 0.125 | 0.1333 | 0.875 | -12.5 |
| `Ellensburg-Yakima_01_20260409` | Ellensburg | MODEL_A | 0.5 | 8 | 2 | 1 | 1 | 7 | 0.5 | 0.125 | 0.2 | 0.25 | -75.0 |
| `Ellensburg-Yakima_01_20260409` | Ellensburg | MODEL_A | 0.85 | 8 | 1 | 1 | 0 | 7 | 1.0 | 0.125 | 0.2222 | 0.125 | -87.5 |
| `EllensburgPreferredTest_01_20260530` | Ellensburg | MODEL_A | 0.3 | 2 | 45 | 1 | 44 | 1 | 0.0222 | 0.5 | 0.0426 | 22.5 | 2150.0 |
| `EllensburgPreferredTest_01_20260530` | Ellensburg | MODEL_A | 0.5 | 2 | 20 | 0 | 20 | 2 | 0.0 | 0.0 | 0 | 10.0 | 900.0 |
| `EllensburgPreferredTest_01_20260530` | Ellensburg | MODEL_A | 0.85 | 2 | 3 | 0 | 3 | 2 | 0.0 | 0.0 | 0 | 1.5 | 50.0 |
| `Ellensburg_01_20260504` | Ellensburg | MODEL_A | 0.3 | 13 | 11 | 8 | 3 | 5 | 0.7273 | 0.6154 | 0.6667 | 0.8462 | -15.38 |
| `Ellensburg_01_20260504` | Ellensburg | MODEL_A | 0.5 | 13 | 8 | 6 | 2 | 7 | 0.75 | 0.4615 | 0.5714 | 0.6154 | -38.46 |
| `Ellensburg_01_20260504` | Ellensburg | MODEL_A | 0.85 | 13 | 1 | 0 | 1 | 13 | 0.0 | 0.0 | 0 | 0.0769 | -92.31 |
| `Stanwood_10_20260511` | Stanwood | MODEL_A | 0.3 | 6 | 24 | 4 | 20 | 2 | 0.1667 | 0.6667 | 0.2667 | 4.0 | 300.0 |
| `Stanwood_10_20260511` | Stanwood | MODEL_A | 0.5 | 6 | 18 | 4 | 14 | 2 | 0.2222 | 0.6667 | 0.3333 | 3.0 | 200.0 |
| `Stanwood_10_20260511` | Stanwood | MODEL_A | 0.85 | 6 | 2 | 0 | 2 | 6 | 0.0 | 0.0 | 0 | 0.3333 | -66.67 |
| `Tacoma-Centralia_01_20260429` | Centralia/I-5-south | MODEL_A | 0.3 | 80 | 86 | 45 | 41 | 35 | 0.5233 | 0.5625 | 0.5422 | 1.075 | 7.5 |
| `Tacoma-Centralia_01_20260429` | Centralia/I-5-south | MODEL_A | 0.5 | 80 | 64 | 41 | 23 | 39 | 0.6406 | 0.5125 | 0.5694 | 0.8 | -20.0 |
| `Tacoma-Centralia_01_20260429` | Centralia/I-5-south | MODEL_A | 0.85 | 80 | 26 | 22 | 4 | 58 | 0.8462 | 0.275 | 0.4151 | 0.325 | -67.5 |
| `Tacoma-Centralia_02_20260602` | Centralia/I-5-south | MODEL_A | 0.3 | 89 | 105 | 60 | 45 | 29 | 0.5714 | 0.6742 | 0.6186 | 1.1798 | 17.98 |
| `Tacoma-Centralia_02_20260602` | Centralia/I-5-south | MODEL_A | 0.5 | 89 | 85 | 58 | 27 | 31 | 0.6824 | 0.6517 | 0.6667 | 0.9551 | -4.49 |
| `Tacoma-Centralia_02_20260602` | Centralia/I-5-south | MODEL_A | 0.85 | 89 | 36 | 28 | 8 | 61 | 0.7778 | 0.3146 | 0.448 | 0.4045 | -59.55 |
| `Tri-Cities-New_01_20260611` | Tri-Cities | MODEL_A | 0.3 | 105 | 116 | 65 | 51 | 40 | 0.5603 | 0.619 | 0.5882 | 1.1048 | 10.48 |
| `Tri-Cities-New_01_20260611` | Tri-Cities | MODEL_A | 0.5 | 105 | 75 | 56 | 19 | 49 | 0.7467 | 0.5333 | 0.6222 | 0.7143 | -28.57 |
| `Tri-Cities-New_01_20260611` | Tri-Cities | MODEL_A | 0.85 | 105 | 29 | 27 | 2 | 78 | 0.931 | 0.2571 | 0.403 | 0.2762 | -72.38 |
| `Yakima-Toppenish_01_20260503` | Yakima | MODEL_A | 0.3 | 25 | 69 | 9 | 60 | 16 | 0.1304 | 0.36 | 0.1915 | 2.76 | 176.0 |
| `Yakima-Toppenish_01_20260503` | Yakima | MODEL_A | 0.5 | 25 | 18 | 7 | 11 | 18 | 0.3889 | 0.28 | 0.3256 | 0.72 | -28.0 |
| `Yakima-Toppenish_01_20260503` | Yakima | MODEL_A | 0.85 | 25 | 3 | 3 | 0 | 22 | 1.0 | 0.12 | 0.2143 | 0.12 | -88.0 |
| `Yakima-Toppenish_02_20260509` | Yakima | MODEL_A | 0.3 | 31 | 95 | 7 | 88 | 24 | 0.0737 | 0.2258 | 0.1111 | 3.0645 | 206.45 |
| `Yakima-Toppenish_02_20260509` | Yakima | MODEL_A | 0.5 | 31 | 36 | 6 | 30 | 25 | 0.1667 | 0.1935 | 0.1791 | 1.1613 | 16.13 |
| `Yakima-Toppenish_02_20260509` | Yakima | MODEL_A | 0.85 | 31 | 5 | 4 | 1 | 27 | 0.8 | 0.129 | 0.2222 | 0.1613 | -83.87 |
| `Yakima-Toppenish_03_20260614` | Yakima | MODEL_A | 0.3 | 27 | 112 | 14 | 98 | 13 | 0.125 | 0.5185 | 0.2014 | 4.1481 | 314.81 |
| `Yakima-Toppenish_03_20260614` | Yakima | MODEL_A | 0.5 | 27 | 47 | 13 | 34 | 14 | 0.2766 | 0.4815 | 0.3514 | 1.7407 | 74.07 |
| `Yakima-Toppenish_03_20260614` | Yakima | MODEL_A | 0.85 | 27 | 6 | 5 | 1 | 22 | 0.8333 | 0.1852 | 0.303 | 0.2222 | -77.78 |
| `Yakima-Toppenish_04_20260620` | Yakima | MODEL_A | 0.3 | 43 | 95 | 20 | 75 | 23 | 0.2105 | 0.4651 | 0.2899 | 2.2093 | 120.93 |
| `Yakima-Toppenish_04_20260620` | Yakima | MODEL_A | 0.5 | 43 | 43 | 17 | 26 | 26 | 0.3953 | 0.3953 | 0.3953 | 1.0 | 0.0 |
| `Yakima-Toppenish_04_20260620` | Yakima | MODEL_A | 0.85 | 43 | 5 | 4 | 1 | 39 | 0.8 | 0.093 | 0.1667 | 0.1163 | -88.37 |
| `auburn-snoqualmie_01_20260425` | auburn-snoqualmie/I-90 | MODEL_A | 0.3 | 24 | 39 | 21 | 18 | 3 | 0.5385 | 0.875 | 0.6667 | 1.625 | 62.5 |
| `auburn-snoqualmie_01_20260425` | auburn-snoqualmie/I-90 | MODEL_A | 0.5 | 24 | 25 | 17 | 8 | 7 | 0.68 | 0.7083 | 0.6939 | 1.0417 | 4.17 |
| `auburn-snoqualmie_01_20260425` | auburn-snoqualmie/I-90 | MODEL_A | 0.85 | 24 | 3 | 3 | 0 | 21 | 1.0 | 0.125 | 0.2222 | 0.125 | -87.5 |
| `auburn-snoqualmie_03_20260614` | auburn-snoqualmie/I-90 | MODEL_A | 0.3 | 42 | 57 | 30 | 27 | 12 | 0.5263 | 0.7143 | 0.6061 | 1.3571 | 35.71 |
| `auburn-snoqualmie_03_20260614` | auburn-snoqualmie/I-90 | MODEL_A | 0.5 | 42 | 43 | 27 | 16 | 15 | 0.6279 | 0.6429 | 0.6353 | 1.0238 | 2.38 |
| `auburn-snoqualmie_03_20260614` | auburn-snoqualmie/I-90 | MODEL_A | 0.85 | 42 | 17 | 13 | 4 | 29 | 0.7647 | 0.3095 | 0.4407 | 0.4048 | -59.52 |
| `blaine-bellingham_01_20260425` | blaine-bellingham/I-5-north | MODEL_A | 0.3 | 18 | 28 | 7 | 21 | 11 | 0.25 | 0.3889 | 0.3043 | 1.5556 | 55.56 |
| `blaine-bellingham_01_20260425` | blaine-bellingham/I-5-north | MODEL_A | 0.5 | 18 | 12 | 3 | 9 | 15 | 0.25 | 0.1667 | 0.2 | 0.6667 | -33.33 |
| `blaine-bellingham_01_20260425` | blaine-bellingham/I-5-north | MODEL_A | 0.85 | 18 | 0 | 0 | 0 | 18 | 0.0 | 0.0 | 0 | 0.0 | -100.0 |
| `blaine-bellingham_02_20260528` | blaine-bellingham/I-5-north | MODEL_A | 0.3 | 50 | 33 | 16 | 17 | 34 | 0.4848 | 0.32 | 0.3855 | 0.66 | -34.0 |
| `blaine-bellingham_02_20260528` | blaine-bellingham/I-5-north | MODEL_A | 0.5 | 50 | 15 | 8 | 7 | 42 | 0.5333 | 0.16 | 0.2462 | 0.3 | -70.0 |
| `blaine-bellingham_02_20260528` | blaine-bellingham/I-5-north | MODEL_A | 0.85 | 50 | 4 | 2 | 2 | 48 | 0.5 | 0.04 | 0.0741 | 0.08 | -92.0 |
| `blaine-bellingham_03_20260504` | blaine-bellingham/I-5-north | MODEL_A | 0.3 | 29 | 41 | 15 | 26 | 14 | 0.3659 | 0.5172 | 0.4286 | 1.4138 | 41.38 |
| `blaine-bellingham_03_20260504` | blaine-bellingham/I-5-north | MODEL_A | 0.5 | 29 | 27 | 11 | 16 | 18 | 0.4074 | 0.3793 | 0.3929 | 0.931 | -6.9 |
| `blaine-bellingham_03_20260504` | blaine-bellingham/I-5-north | MODEL_A | 0.85 | 29 | 4 | 4 | 0 | 25 | 1.0 | 0.1379 | 0.2424 | 0.1379 | -86.21 |
| `polygon_01_20260503` | polygon/Kent-Auburn | MODEL_A | 0.3 | 19 | 18 | 6 | 12 | 13 | 0.3333 | 0.3158 | 0.3243 | 0.9474 | -5.26 |
| `polygon_01_20260503` | polygon/Kent-Auburn | MODEL_A | 0.5 | 19 | 11 | 4 | 7 | 15 | 0.3636 | 0.2105 | 0.2667 | 0.5789 | -42.11 |
| `polygon_01_20260503` | polygon/Kent-Auburn | MODEL_A | 0.85 | 19 | 3 | 3 | 0 | 16 | 1.0 | 0.1579 | 0.2727 | 0.1579 | -84.21 |
| `spokane-i90_01_20260622` | Spokane | MODEL_A | 0.3 | 55 | 54 | 15 | 39 | 40 | 0.2778 | 0.2727 | 0.2752 | 0.9818 | -1.82 |
| `spokane-i90_01_20260622` | Spokane | MODEL_A | 0.5 | 55 | 23 | 10 | 13 | 45 | 0.4348 | 0.1818 | 0.2564 | 0.4182 | -58.18 |
| `spokane-i90_01_20260622` | Spokane | MODEL_A | 0.85 | 55 | 4 | 3 | 1 | 52 | 0.75 | 0.0545 | 0.1017 | 0.0727 | -92.73 |
| `Bellingham_01_20260425` | blaine-bellingham/I-5-north | MODEL_B | 0.3 | 11 | 77 | 5 | 72 | 6 | 0.0649 | 0.4545 | 0.1136 | 7.0 | 600.0 |
| `Bellingham_01_20260425` | blaine-bellingham/I-5-north | MODEL_B | 0.5 | 11 | 44 | 3 | 41 | 8 | 0.0682 | 0.2727 | 0.1091 | 4.0 | 300.0 |
| `Bellingham_01_20260425` | blaine-bellingham/I-5-north | MODEL_B | 0.85 | 11 | 14 | 1 | 13 | 10 | 0.0714 | 0.0909 | 0.08 | 1.2727 | 27.27 |
| `Centralia_01_20260511` | Centralia/I-5-south | MODEL_B | 0.3 | 56 | 74 | 27 | 47 | 29 | 0.3649 | 0.4821 | 0.4154 | 1.3214 | 32.14 |
| `Centralia_01_20260511` | Centralia/I-5-south | MODEL_B | 0.5 | 56 | 69 | 24 | 45 | 32 | 0.3478 | 0.4286 | 0.384 | 1.2321 | 23.21 |
| `Centralia_01_20260511` | Centralia/I-5-south | MODEL_B | 0.85 | 56 | 30 | 14 | 16 | 42 | 0.4667 | 0.25 | 0.3256 | 0.5357 | -46.43 |
| `Centralia_02_20260511` | Centralia/I-5-south | MODEL_B | 0.3 | 56 | 91 | 33 | 58 | 23 | 0.3626 | 0.5893 | 0.449 | 1.625 | 62.5 |
| `Centralia_02_20260511` | Centralia/I-5-south | MODEL_B | 0.5 | 56 | 78 | 29 | 49 | 27 | 0.3718 | 0.5179 | 0.4328 | 1.3929 | 39.29 |
| `Centralia_02_20260511` | Centralia/I-5-south | MODEL_B | 0.85 | 56 | 30 | 17 | 13 | 39 | 0.5667 | 0.3036 | 0.3953 | 0.5357 | -46.43 |
| `Ellensburg-Yakima_01_20260409` | Ellensburg | MODEL_B | 0.3 | 8 | 38 | 7 | 31 | 1 | 0.1842 | 0.875 | 0.3043 | 4.75 | 375.0 |
| `Ellensburg-Yakima_01_20260409` | Ellensburg | MODEL_B | 0.5 | 8 | 17 | 5 | 12 | 3 | 0.2941 | 0.625 | 0.4 | 2.125 | 112.5 |
| `Ellensburg-Yakima_01_20260409` | Ellensburg | MODEL_B | 0.85 | 8 | 3 | 1 | 2 | 7 | 0.3333 | 0.125 | 0.1818 | 0.375 | -62.5 |
| `EllensburgPreferredTest_01_20260530` | Ellensburg | MODEL_B | 0.3 | 2 | 49 | 1 | 48 | 1 | 0.0204 | 0.5 | 0.0392 | 24.5 | 2350.0 |
| `EllensburgPreferredTest_01_20260530` | Ellensburg | MODEL_B | 0.5 | 2 | 28 | 1 | 27 | 1 | 0.0357 | 0.5 | 0.0667 | 14.0 | 1300.0 |
| `EllensburgPreferredTest_01_20260530` | Ellensburg | MODEL_B | 0.85 | 2 | 5 | 1 | 4 | 1 | 0.2 | 0.5 | 0.2857 | 2.5 | 150.0 |
| `Ellensburg_01_20260504` | Ellensburg | MODEL_B | 0.3 | 13 | 13 | 7 | 6 | 6 | 0.5385 | 0.5385 | 0.5385 | 1.0 | 0.0 |
| `Ellensburg_01_20260504` | Ellensburg | MODEL_B | 0.5 | 13 | 12 | 7 | 5 | 6 | 0.5833 | 0.5385 | 0.56 | 0.9231 | -7.69 |
| `Ellensburg_01_20260504` | Ellensburg | MODEL_B | 0.85 | 13 | 6 | 4 | 2 | 9 | 0.6667 | 0.3077 | 0.4211 | 0.4615 | -53.85 |
| `Stanwood_10_20260511` | Stanwood | MODEL_B | 0.3 | 6 | 38 | 4 | 34 | 2 | 0.1053 | 0.6667 | 0.1818 | 6.3333 | 533.33 |
| `Stanwood_10_20260511` | Stanwood | MODEL_B | 0.5 | 6 | 26 | 4 | 22 | 2 | 0.1538 | 0.6667 | 0.25 | 4.3333 | 333.33 |
| `Stanwood_10_20260511` | Stanwood | MODEL_B | 0.85 | 6 | 11 | 3 | 8 | 3 | 0.2727 | 0.5 | 0.3529 | 1.8333 | 83.33 |
| `Tacoma-Centralia_01_20260429` | Centralia/I-5-south | MODEL_B | 0.3 | 80 | 120 | 36 | 84 | 44 | 0.3 | 0.45 | 0.36 | 1.5 | 50.0 |
| `Tacoma-Centralia_01_20260429` | Centralia/I-5-south | MODEL_B | 0.5 | 80 | 101 | 35 | 66 | 45 | 0.3465 | 0.4375 | 0.3867 | 1.2625 | 26.25 |
| `Tacoma-Centralia_01_20260429` | Centralia/I-5-south | MODEL_B | 0.85 | 80 | 47 | 22 | 25 | 58 | 0.4681 | 0.275 | 0.3465 | 0.5875 | -41.25 |
| `Tacoma-Centralia_02_20260602` | Centralia/I-5-south | MODEL_B | 0.3 | 89 | 127 | 49 | 78 | 40 | 0.3858 | 0.5506 | 0.4537 | 1.427 | 42.7 |
| `Tacoma-Centralia_02_20260602` | Centralia/I-5-south | MODEL_B | 0.5 | 89 | 106 | 48 | 58 | 41 | 0.4528 | 0.5393 | 0.4923 | 1.191 | 19.1 |
| `Tacoma-Centralia_02_20260602` | Centralia/I-5-south | MODEL_B | 0.85 | 89 | 53 | 34 | 19 | 55 | 0.6415 | 0.382 | 0.4789 | 0.5955 | -40.45 |
| `Tri-Cities-New_01_20260611` | Tri-Cities | MODEL_B | 0.3 | 105 | 278 | 72 | 206 | 33 | 0.259 | 0.6857 | 0.376 | 2.6476 | 164.76 |
| `Tri-Cities-New_01_20260611` | Tri-Cities | MODEL_B | 0.5 | 105 | 204 | 62 | 142 | 43 | 0.3039 | 0.5905 | 0.4013 | 1.9429 | 94.29 |
| `Tri-Cities-New_01_20260611` | Tri-Cities | MODEL_B | 0.85 | 105 | 76 | 32 | 44 | 73 | 0.4211 | 0.3048 | 0.3536 | 0.7238 | -27.62 |
| `Yakima-Toppenish_01_20260503` | Yakima | MODEL_B | 0.3 | 25 | 155 | 16 | 139 | 9 | 0.1032 | 0.64 | 0.1778 | 6.2 | 520.0 |
| `Yakima-Toppenish_01_20260503` | Yakima | MODEL_B | 0.5 | 25 | 111 | 14 | 97 | 11 | 0.1261 | 0.56 | 0.2059 | 4.44 | 344.0 |
| `Yakima-Toppenish_01_20260503` | Yakima | MODEL_B | 0.85 | 25 | 32 | 4 | 28 | 21 | 0.125 | 0.16 | 0.1404 | 1.28 | 28.0 |
| `Yakima-Toppenish_02_20260509` | Yakima | MODEL_B | 0.3 | 31 | 104 | 20 | 84 | 11 | 0.1923 | 0.6452 | 0.2963 | 3.3548 | 235.48 |
| `Yakima-Toppenish_02_20260509` | Yakima | MODEL_B | 0.5 | 31 | 61 | 17 | 44 | 14 | 0.2787 | 0.5484 | 0.3696 | 1.9677 | 96.77 |
| `Yakima-Toppenish_02_20260509` | Yakima | MODEL_B | 0.85 | 31 | 17 | 9 | 8 | 22 | 0.5294 | 0.2903 | 0.375 | 0.5484 | -45.16 |
| `Yakima-Toppenish_03_20260614` | Yakima | MODEL_B | 0.3 | 27 | 143 | 18 | 125 | 9 | 0.1259 | 0.6667 | 0.2118 | 5.2963 | 429.63 |
| `Yakima-Toppenish_03_20260614` | Yakima | MODEL_B | 0.5 | 27 | 94 | 17 | 77 | 10 | 0.1809 | 0.6296 | 0.281 | 3.4815 | 248.15 |
| `Yakima-Toppenish_03_20260614` | Yakima | MODEL_B | 0.85 | 27 | 36 | 10 | 26 | 17 | 0.2778 | 0.3704 | 0.3175 | 1.3333 | 33.33 |
| `Yakima-Toppenish_04_20260620` | Yakima | MODEL_B | 0.3 | 43 | 141 | 29 | 112 | 14 | 0.2057 | 0.6744 | 0.3152 | 3.2791 | 227.91 |
| `Yakima-Toppenish_04_20260620` | Yakima | MODEL_B | 0.5 | 43 | 103 | 27 | 76 | 16 | 0.2621 | 0.6279 | 0.3699 | 2.3953 | 139.53 |
| `Yakima-Toppenish_04_20260620` | Yakima | MODEL_B | 0.85 | 43 | 44 | 18 | 26 | 25 | 0.4091 | 0.4186 | 0.4138 | 1.0233 | 2.33 |
| `auburn-snoqualmie_01_20260425` | auburn-snoqualmie/I-90 | MODEL_B | 0.3 | 24 | 89 | 14 | 75 | 10 | 0.1573 | 0.5833 | 0.2478 | 3.7083 | 270.83 |
| `auburn-snoqualmie_01_20260425` | auburn-snoqualmie/I-90 | MODEL_B | 0.5 | 24 | 63 | 14 | 49 | 10 | 0.2222 | 0.5833 | 0.3218 | 2.625 | 162.5 |
| `auburn-snoqualmie_01_20260425` | auburn-snoqualmie/I-90 | MODEL_B | 0.85 | 24 | 16 | 3 | 13 | 21 | 0.1875 | 0.125 | 0.15 | 0.6667 | -33.33 |
| `auburn-snoqualmie_03_20260614` | auburn-snoqualmie/I-90 | MODEL_B | 0.3 | 42 | 103 | 24 | 79 | 18 | 0.233 | 0.5714 | 0.331 | 2.4524 | 145.24 |
| `auburn-snoqualmie_03_20260614` | auburn-snoqualmie/I-90 | MODEL_B | 0.5 | 42 | 76 | 24 | 52 | 18 | 0.3158 | 0.5714 | 0.4068 | 1.8095 | 80.95 |
| `auburn-snoqualmie_03_20260614` | auburn-snoqualmie/I-90 | MODEL_B | 0.85 | 42 | 32 | 19 | 13 | 23 | 0.5938 | 0.4524 | 0.5135 | 0.7619 | -23.81 |
| `blaine-bellingham_01_20260425` | blaine-bellingham/I-5-north | MODEL_B | 0.3 | 18 | 113 | 16 | 97 | 2 | 0.1416 | 0.8889 | 0.2443 | 6.2778 | 527.78 |
| `blaine-bellingham_01_20260425` | blaine-bellingham/I-5-north | MODEL_B | 0.5 | 18 | 77 | 15 | 62 | 3 | 0.1948 | 0.8333 | 0.3158 | 4.2778 | 327.78 |
| `blaine-bellingham_01_20260425` | blaine-bellingham/I-5-north | MODEL_B | 0.85 | 18 | 24 | 7 | 17 | 11 | 0.2917 | 0.3889 | 0.3333 | 1.3333 | 33.33 |
| `blaine-bellingham_02_20260528` | blaine-bellingham/I-5-north | MODEL_B | 0.3 | 50 | 94 | 28 | 66 | 22 | 0.2979 | 0.56 | 0.3889 | 1.88 | 88.0 |
| `blaine-bellingham_02_20260528` | blaine-bellingham/I-5-north | MODEL_B | 0.5 | 50 | 68 | 24 | 44 | 26 | 0.3529 | 0.48 | 0.4068 | 1.36 | 36.0 |
| `blaine-bellingham_02_20260528` | blaine-bellingham/I-5-north | MODEL_B | 0.85 | 50 | 29 | 11 | 18 | 39 | 0.3793 | 0.22 | 0.2785 | 0.58 | -42.0 |
| `blaine-bellingham_03_20260504` | blaine-bellingham/I-5-north | MODEL_B | 0.3 | 29 | 147 | 20 | 127 | 9 | 0.1361 | 0.6897 | 0.2273 | 5.069 | 406.9 |
| `blaine-bellingham_03_20260504` | blaine-bellingham/I-5-north | MODEL_B | 0.5 | 29 | 102 | 16 | 86 | 13 | 0.1569 | 0.5517 | 0.2443 | 3.5172 | 251.72 |
| `blaine-bellingham_03_20260504` | blaine-bellingham/I-5-north | MODEL_B | 0.85 | 29 | 30 | 8 | 22 | 21 | 0.2667 | 0.2759 | 0.2712 | 1.0345 | 3.45 |
| `polygon_01_20260503` | polygon/Kent-Auburn | MODEL_B | 0.3 | 19 | 118 | 11 | 107 | 8 | 0.0932 | 0.5789 | 0.1606 | 6.2105 | 521.05 |
| `polygon_01_20260503` | polygon/Kent-Auburn | MODEL_B | 0.5 | 19 | 81 | 10 | 71 | 9 | 0.1235 | 0.5263 | 0.2 | 4.2632 | 326.32 |
| `polygon_01_20260503` | polygon/Kent-Auburn | MODEL_B | 0.85 | 19 | 30 | 6 | 24 | 13 | 0.2 | 0.3158 | 0.2449 | 1.5789 | 57.89 |
| `spokane-i90_01_20260622` | Spokane | MODEL_B | 0.3 | 55 | 140 | 35 | 105 | 20 | 0.25 | 0.6364 | 0.359 | 2.5455 | 154.55 |
| `spokane-i90_01_20260622` | Spokane | MODEL_B | 0.5 | 55 | 82 | 29 | 53 | 26 | 0.3537 | 0.5273 | 0.4234 | 1.4909 | 49.09 |
| `spokane-i90_01_20260622` | Spokane | MODEL_B | 0.85 | 55 | 29 | 18 | 11 | 37 | 0.6207 | 0.3273 | 0.4286 | 0.5273 | -47.27 |

</details>

**Matching / suppression settings — identical across both models** (verified field-by-field from both `manifest.json`):

| setting | value |
|---|---|
| match radius | 6 px = 18 m, on the red (sequence 2) keypoint |
| cross-window dedup | 32 px = 96 m, greedy by score |
| model-internal NMS | box IoU 0.5 (torchvision default) |
| model-internal score floor | 0.05 |
| max detections per window | 100 |
| sliding stride | 40 px |
| window acceptance | ≥ 15 % non-nodata |
| detection kept per window | top-1 |

### 3.2 Scene provenance

`group = repeat_visit` if max footprint IoU > 0.1 against that model's training set, else `transfer`.

| scene_id | max_IoU vs A_train | max_IoU vs B_train | days_since_nearest_A | days_since_nearest_B | group_A | group_B |
|---|---:|---:|---:|---:|---|---|
| `Bellingham_01_20260425` | 1.0000 | 1.0000 | 0 | 0 | repeat_visit | repeat_visit |
| `Centralia_01_20260511` | 1.0000 | 1.0000 | 0 | 0 | repeat_visit | repeat_visit |
| `Centralia_02_20260511` | 1.0000 | 1.0000 | 0 | 0 | repeat_visit | repeat_visit |
| `Ellensburg-Yakima_01_20260409` | 1.0000 | 1.0000 | 0 | 0 | repeat_visit | repeat_visit |
| `EllensburgPreferredTest_01_20260530` | 1.0000 | 1.0000 | 0 | 0 | repeat_visit | repeat_visit |
| `Ellensburg_01_20260504` | 1.0000 | 1.0000 | 0 | 0 | repeat_visit | repeat_visit |
| `Stanwood_10_20260511` | 1.0000 | 1.0000 | 0 | 0 | repeat_visit | repeat_visit |
| `Tacoma-Centralia_02_20260602` | 1.0000 | 1.0000 | 0 | 0 | repeat_visit | repeat_visit |
| `blaine-bellingham_01_20260425` | 1.0000 | 1.0000 | 0 | 0 | repeat_visit | repeat_visit |
| `Tacoma-Centralia_01_20260429` | 0.9652 | 0.9652 | 12 | 12 | repeat_visit | repeat_visit |
| `blaine-bellingham_03_20260504` | 0.8950 | 1.0000 | 9 | 0 | repeat_visit | repeat_visit |
| `blaine-bellingham_02_20260528` | 0.7004 | 1.0000 | 33 | 0 | repeat_visit | repeat_visit |
| `Tri-Cities-New_01_20260611` | 0.0000 | 1.0000 | — | 0 | transfer | repeat_visit |
| `Yakima-Toppenish_01_20260503` | 0.0000 | 1.0000 | — | 0 | transfer | repeat_visit |
| `Yakima-Toppenish_02_20260509` | 0.0000 | 1.0000 | — | 0 | transfer | repeat_visit |
| `Yakima-Toppenish_03_20260614` | 0.0000 | 1.0000 | — | 0 | transfer | repeat_visit |
| `Yakima-Toppenish_04_20260620` | 0.0000 | 1.0000 | — | 0 | transfer | repeat_visit |
| `auburn-snoqualmie_01_20260425` | 0.0000 | 1.0000 | — | 0 | transfer | repeat_visit |
| `auburn-snoqualmie_03_20260614` | 0.0000 | 1.0000 | — | 0 | transfer | repeat_visit |
| `polygon_01_20260503` | 0.0000 | 1.0000 | — | 0 | transfer | repeat_visit |
| `spokane-i90_01_20260622` | 0.0000 | 1.0000 | — | 0 | transfer | repeat_visit |

### 3.3 Aggregates by model × threshold × group

| model | thr | group | n_scenes | macro_P | macro_R | macro_F1 | micro_P | micro_R | micro_F1 | median_F1 | IQR_F1 | SD_F1 | n_count_block | n_excl_<5 | median_count_ratio | IQR_count_ratio |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MODEL_A | 0.3 | repeat_visit | 12 | 0.3643 | 0.5169 | 0.3927 | 0.4103 | 0.5526 | 0.4709 | 0.4071 | 0.2834 | 0.1956 | 11 | 1 | 1.25 | 0.5097 |
| MODEL_A | 0.3 | transfer | 9 | 0.3084 | 0.4851 | 0.3616 | 0.2855 | 0.504 | 0.3645 | 0.2899 | 0.3868 | 0.2048 | 9 | 0 | 1.625 | 1.6552 |
| MODEL_A | 0.5 | repeat_visit | 12 | 0.4503 | 0.4071 | 0.3926 | 0.522 | 0.4833 | 0.5019 | 0.4023 | 0.3353 | 0.2024 | 11 | 1 | 0.931 | 0.3679 |
| MODEL_A | 0.5 | transfer | 9 | 0.4534 | 0.403 | 0.414 | 0.4891 | 0.4232 | 0.4538 | 0.3514 | 0.3555 | 0.1886 | 9 | 0 | 1.0 | 0.3274 |
| MODEL_A | 0.85 | repeat_visit | 12 | 0.5123 | 0.1417 | 0.2121 | 0.7377 | 0.2153 | 0.3333 | 0.2323 | 0.3684 | 0.1889 | 11 | 1 | 0.3036 | 0.2606 |
| MODEL_A | 0.85 | transfer | 9 | 0.8754 | 0.159 | 0.2607 | 0.8667 | 0.1752 | 0.2915 | 0.2222 | 0.0887 | 0.1083 | 9 | 0 | 0.1579 | 0.1022 |
| MODEL_B | 0.3 | repeat_visit | 21 | 0.2153 | 0.6156 | 0.2948 | 0.2096 | 0.5982 | 0.3104 | 0.3043 | 0.1642 | 0.124 | 20 | 1 | 3.317 | 3.706 |
| MODEL_B | 0.5 | repeat_visit | 21 | 0.2631 | 0.5503 | 0.3347 | 0.2651 | 0.5387 | 0.3554 | 0.3699 | 0.1568 | 0.1217 | 20 | 1 | 2.0463 | 2.2532 |
| MODEL_B | 0.85 | repeat_visit | 21 | 0.3804 | 0.304 | 0.3185 | 0.4074 | 0.3067 | 0.35 | 0.3333 | 0.1241 | 0.1127 | 20 | 1 | 0.6952 | 0.7293 |

**Count-block exclusions (< 5 annotated vehicles):** 1 scene excluded from every MODEL_A `repeat_visit` group and every MODEL_B group — `EllensburgPreferredTest_01_20260530` (2 vehicles). 0 excluded from MODEL_A `transfer` groups.

**MODEL_B has zero `transfer` scenes** — all 21 evaluated scenes have footprint IoU > 0.1 against its training set.


---

## 4. Flags

1. **Console/metric threshold mismatch.** Backend `/api/detect` defaults to `thresh=0.5` (`backend/server.py:190`) and the frontend slider defaults to 0.5 (`frontend/src/App.tsx:715`), while all registry-recorded F1 values were computed at 0.3.

2. **Registry metrics vs this evidence pack use different codepaths.** Registry `metrics` for both models were produced by `train_detector.eval_full_scene` at training time; all numbers in §3 were produced by `src/collect_inference.py`. §3 is internally consistent; registry values are not directly comparable to it.

3. **Post-processing was identical across both models in §3** — verified field-by-field from both `manifest.json` (match radius, dedup radius, stride, min_valid, top-1-per-window, and all torchvision ROI defaults match).

4. **Partial labeling across all scenes.** Labels mark only clear, well-formed truck echoes; blurry, malformed, over-bright and very small echoes are deliberately unlabeled. FP counts therefore include real unlabeled vehicles, so precision and F1 are biased low and positive `signed_count_error_pct` biased high.

5. **MODEL_B has no transfer-group scenes**, so no spatial-generalization statistic can be computed for it from this data. MODEL_A transfer statistics (9 scenes) have no MODEL_B counterpart.

6. **No scene is a transfer scene for both models simultaneously.** `Tacoma-Centralia_01` is the only scene excluded from both training sets, and it is `repeat_visit` for both (IoU 0.9652).

7. **Training-set composition differs substantially** (9 vs 20 scenes; 228 vs 624 chips; Centralia share 78 % vs 23 %). Group-level comparisons between MODEL_A and MODEL_B are confounded by this.

8. **Custom TP/FP definition.** Matching uses red-keypoint distance ≤ 18 m, not box IoU, and is asymmetric (neither side consumes its match, so one label may validate multiple detections). Values are not comparable to COCO mAP or to published figures using IoU-based matching.

9. **Chip format differs between the two models' training data** (MODEL_A: 64 px single-vehicle; MODEL_B: 96 px multi-vehicle with jitter crop). Evaluation input was identical (64 px windows) for both.

10. **GeoPackage CRS mis-stamp.** `Annotations-RGB.gpkg` is stamped EPSG:32610, but `Tri-Cities-New_01` and `spokane-i90_01` rasters are UTM 11. Evaluation code reprojects points into each raster's native CRS before the inverse affine; any downstream consumer reading those coordinates as literal EPSG:32610 will place them incorrectly.

11. **`days_since_nearest_*` is 0 for scenes in the model's own training set** (self-match, IoU 1.0); it is informative only for held-out scenes.

