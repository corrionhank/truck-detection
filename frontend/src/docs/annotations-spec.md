# Annotations — Data Spec

The source of truth for the hand-labeled truck **"moving echo"** keypoints. A moving
vehicle leaves a **Blue → Red → Green** streak because SuperDove captures those bands a
fraction of a second apart; each vehicle is labeled as 3 points, one per band.

## File

- **`Annotations-RGB.gpkg`** — a GeoPackage (SQLite under the hood), layer **`Annotations`**.
- CRS **EPSG:32610** (UTM 10N, metres).
- Geometry type **Point** — one point per band, per vehicle.
- When QGIS has it open, `-wal` / `-shm` sidecars appear. Read with **geopandas / OGR**
  (which merges them), not raw SQLite.

## Fields

| field | type | meaning |
|---|---|---|
| `fid` | int (auto) | OGR primary key — not a data column |
| `vehicle_id` | int | groups the 3 points belonging to one vehicle |
| `sequence` | int | capture order: **1 = Blue, 2 = Red, 3 = Green** |
| `scene` | text | GeoTIFF filename **without extension** — links point → raster |
| `geometry` | Point | location in EPSG:32610 metres |

## Semantics

- A **complete** vehicle has all three sequences `{1, 2, 3}`. Incomplete vehicles are skipped downstream.
- `scene` ties every point to its GeoTIFF, whose **affine transform** converts the point's
  metres → pixel coordinates.
- The order matters: sequence 1 → 2 → 3 traces the vehicle's motion (Blue then Red then Green).

## Current contents

| metric | value |
|---|---|
| keypoints | 1,017 |
| vehicles | 339 |
| **usable (complete)** | **339** |
| scenes | 8 |

~91% is the Centralia / south-I-5 corridor; the rest spans Ellensburg (I-90), Bellingham and Stanwood
(north I-5). Per-scene breakdown in `docs/DATA.md` §6.

## Known data quirks (handled in export)

- **Incomplete vehicle** — any `vehicle_id` without all three sequences `{1,2,3}` is counted and skipped.
- **Whitespace scene** — a `scene` with a stray leading space (`" Bellingham_01…"`) is `.strip()`-ed before
  the filename lookup, or that vehicle fails to match its raster.

## Export → COCO keypoints

1. Group points by `vehicle_id`, sort each group by `sequence`.
2. Convert map coords → **full-scene pixel coords** via the scene GeoTIFF's inverse affine
   (`~src.transform`).
3. Emit **one COCO annotation per vehicle**: `keypoints` = 9 numbers `[x, y, v]` × 3 for
   Blue / Red / Green (visibility `v = 2`), plus a tight `bbox` (+ a few px pad, since
   Keypoint R-CNN needs boxes). One category, `moving_echo`.
4. Skip incomplete vehicles.

## Invariants

> **Never reproject the rasters** — the inter-band offset *is* the velocity signal; resampling would smear it.

- Model input = bands **6 / 4 / 2** (R / G / B), percentile-stretched (2–98%) to 8-bit.
