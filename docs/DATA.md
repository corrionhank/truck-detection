# Data reference — annotations, exports, and where every piece lives

_The full data path: what the annotation file holds, how it joins to imagery, what gets exported, what the
model actually consumes, and which file each piece lives in. Verified against the real files and
`src/export_coco.py`._

## 1. The annotation file (the input labels)

`Annotations-RGB.gpkg`, table `Annotations` — a GeoPackage of points. No pixels, imagery, or colours are
stored; each row is a real-world coordinate plus tags.

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

**The image** — `data/coco/images/<scene>__v<id>.png`
- A **64×64×3 uint8** RGB thumbnail (values `0–255`), cropped centered on the vehicle.
- RGB = SuperDove **bands 6/4/2 (red/green/blue)**, each **2–98% percentile-stretched to 8-bit**.

**The label** — one record per chip in `data/coco/annotations.json`:
```json
{"keypoints": [28.1, 29.04, 2,  32.0, 31.94, 2,  36.33, 35.21, 2],
 "bbox": [25.1, 26.04, 14.23, 12.17], "category_id": 1}
```
- **3 keypoints = `[x, y, v]×3`** in **chip pixel coordinates** (sub-pixel floats), order blue→red→green;
  `v=2` = labeled & visible.
- One class `moving_echo`; `bbox` auto-derived from the 3 points (+3 px pad).

Current export: **15 chips / 15 records** (one per complete vehicle).

## 4. What the model consumes — and what it ignores

**Consumed:** the 64×64×3 image (→ floats `0–1`) + the 3 keypoints + box + class. That's all.

**Available but NOT used** (decided in code, stored in neither file):
- **5 of 8 bands** — coastal-blue, green-I, yellow, red-edge, NIR are discarded; only 6/4/2 survive.
- **16-bit reflectance precision** — flattened to 8-bit by the per-scene 2–98% stretch.
- **World coordinates / EPSG:32610** — used only to place the crop, then dropped; the model never sees geography.
- **UDM2 cloud mask** (`*_udm2_clip.tif`) — not applied (and absent from this repo).
- **Road mask** (OSM buffer) — an optional export-time filter, not baked into labels.

## 5. File map

| Data | File(s) | Format |
|---|---|---|
| 64×64 RGB chip (model input) | `data/coco/images/<scene>__v<id>.png` | 64×64×3 uint8 PNG |
| keypoints + bbox + class (target) | `data/coco/annotations.json` | COCO-keypoints JSON |
| original 8-band, 16-bit reflectance | `imagery/all_geotiffs/<scene>.tif` | 8-band uint16 GeoTIFF |
| world-coordinate annotation points | `Annotations-RGB.gpkg`, table `Annotations` | GeoPackage |
| trained model | `weights/keypoint_rcnn_echo.pt`, `..._jitter.pt` | PyTorch state dict |

The join that ties imagery + labels together is `src/export_coco.py`.

## Notes
- **`data/coco/`, `imagery/`, and `weights/` are gitignored** — on disk, not on GitHub. Only the labels
  (`Annotations-RGB.gpkg`) and `src/` code are committed, so a fresh clone **regenerates** the chips by
  running `export_coco.py` rather than downloading them.
- For the external labeling-tool version of this contract (parity checks, acceptance tests), see
  [LABELING_TOOL_CONTRACT.md](LABELING_TOOL_CONTRACT.md).
