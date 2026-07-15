 # Data reference — annotations, exports, and where every piece lives

_The full data path: what the annotation file holds, how it joins to imagery, what gets exported, what the
model actually consumes, and which file each piece lives in. Verified against the real files and
`src/export_coco.py`._

## 1. The annotation file (the input labels)

`data/active/Annotations-RGB.gpkg`, table `Annotations` — a GeoPackage of points. No pixels, imagery, or
colours are stored; each row is a real-world coordinate plus tags.

| Field | Type | Notes |
|---|---|---|
| `geometry` | Point | x, y in **EPSG:32610** (UTM 10N, metres) |
| `vehicle_id` | int | groups the 3 points of one vehicle (unique **within a scene**) |
| `sequence` | int | **1 = blue, 2 = red, 3 = green** (capture order) |
| `scene` | text | must exactly match the GeoTIFF filename stem (no extension) |
| `fid` | int | auto row id, unused by the pipeline |

**A vehicle = 3 rows** sharing `vehicle_id`, ordered by `sequence` (1→blue, 2→red, 3→green). Rows without
exactly 3 points are incomplete and dropped at export.

## 2. The join (annotation → training data)

The annotation is only a pointer; the matching GeoTIFF supplies every pixel. `src/export_coco.py`:

1. Match each point to its GeoTIFF by the `scene` **text field** (never spatial extent).
2. Apply the GeoTIFF's inverse affine to convert the UTM coordinate → pixel `(col, row)`.
3. Group by `vehicle_id`, order by `sequence`; crop a chip and write COCO keypoints.

The coordinate is the anchor; the GeoTIFF supplies the pixels.

### Rules that must hold
- **Match by `scene`, never spatial extent.** Overlapping scenes (e.g. the 10 Stanwood dates) cover the
  same ground, so an extent join leaks labels across dates. (Confirmed: a raw GEOS spatial join reassigns
  Stanwood_08's labels to Stanwood_01.)
- **`scene` must exactly match the GeoTIFF stem**, e.g. `Stanwood_08_20260504` — not a Planet scene id.
  Trim whitespace (a stray leading space once split a phantom scene).
- **No blank `scene`** — unmatchable, dropped.
- **Never reproject the GeoTIFFs** — the inter-band offset is the velocity signal; resampling smears it.

## 3. What gets exported (one vehicle = a chip + 3 points)

**The image** — `data/active/coco/images/<scene>__v<id>.png`
- A **64×64×3 uint8** RGB thumbnail (values `0–255`), cropped centered on the vehicle.
- RGB = SuperDove **bands 6/4/2 (red/green/blue)**, each **2–98% percentile-stretched to 8-bit**.

**The label** — one record per chip in `data/active/coco/annotations.json`:
```json
{"keypoints": [28.1, 29.04, 2,  32.0, 31.94, 2,  36.33, 35.21, 2],
 "bbox": [25.1, 26.04, 14.23, 12.17], "category_id": 1}
```
- **3 keypoints = `[x, y, v]×3`** in **chip pixel coordinates** (sub-pixel floats), order blue→red→green;
  `v=2` = labeled & visible.
- One class `moving_echo`; `bbox` auto-derived from the 3 points (+3 px pad).

Current export: **339 chips / 339 records** across 8 scenes (one per complete vehicle).

## 4. What the model consumes — and what it ignores

**Consumed:** the 64×64×3 image (→ floats `0–1`) + the 3 keypoints + box + class. That's all.

**Available but NOT used** (decided in code, stored in neither file):
- **5 of 8 bands** — coastal-blue, green-I, yellow, red-edge, NIR are discarded; only 6/4/2 survive.
- **16-bit reflectance precision** — flattened to 8-bit by the per-scene 2–98% stretch.
- **World coordinates / EPSG:32610** — used only to place the crop, then dropped; the model never sees geography.
- **UDM2 cloud mask** (`*_udm2_clip.tif`) — not applied (and absent from this repo).
- **Road mask** (OSM buffer) — an optional export-time filter, not baked into labels.

## 5. File map

Data is organized into a hot **`data/active/`** working set and a **`data/cold/`** archive.

| Data | File(s) | Format |
|---|---|---|
| 64×64 RGB chip (model input) | `data/active/coco/images/<scene>__v<id>.png` | 64×64×3 uint8 PNG |
| keypoints + bbox + class (target) | `data/active/coco/annotations.json` | COCO-keypoints JSON |
| 8-band 16-bit reflectance (labeled scenes) | `data/active/imagery/<scene>.tif` | 8-band uint16 GeoTIFF |
| world-coordinate annotation points | `data/active/Annotations-RGB.gpkg`, table `Annotations` | GeoPackage |
| unlabeled scenes (archive) | `data/cold/imagery/<scene>.tif` | 8-band uint16 GeoTIFF |
| trained models | `weights/<id>.pt` (indexed by [`models/registry.json`](../models/registry.json)) | PyTorch state dict |

The join that ties imagery + labels together is `src/export_coco.py`.

## 6. Current corpus (snapshot)

**339 vehicles / 1,017 keypoints across 8 labeled scenes** (up ~21× from an initial 16). Format-validated:
exports cleanly to COCO (339/339, 0 dropped), every vehicle a complete B/R/G triple, EPSG:32610.

| Scene | Corridor | Vehicles |
|---|---|---:|
| Tacoma-Centralia_02_20260602 | south I-5 | 101 |
| Tacoma-Centralia_01_20260429 | south I-5 | 94 |
| Centralia_01_20260511 | south I-5 | 58 |
| Centralia_02_20260511 | south I-5 | 54 |
| Ellensburg_01_20260504 | I-90 | 13 |
| Bellingham_01_20260425 | north I-5 | 11 |
| Stanwood_10_20260511 | north I-5 | 6 |
| EllensburgPreferredTest_01_20260530 | I-90 (test) | 2 |
| **Total** | | **339** |

**The diversity caveat:** ~91 % (307/339) is one region — greater Centralia / Tacoma south-I-5 (4 scenes of
essentially the same corridor). The rest spans Ellensburg (I-90), Bellingham + Stanwood (north I-5). Volume is
strong; diversity is concentrated, so the next labels are worth more on **new corridors** than more Centralia.
**10 unlabeled scenes** (Seattle + 9 more Stanwood dates) are archived in `data/cold/`; the corpus is frozen
until the Planet quota resets (see [CONTEXT.md](CONTEXT.md)).

## Notes
- **All imagery (`data/active/imagery/`, `data/cold/`), `data/active/coco/`, and `weights/` are gitignored**
  — on disk, not on GitHub. The **only** data file tracked in git is `data/active/Annotations-RGB.gpkg`
  (the labels). A fresh clone has the labels but must be given the imagery, then **regenerates** the chips
  by running `export_coco.py`.
