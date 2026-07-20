# Export request — to the data aggregation / clipping / annotation tool

_From the **training & inference** repo (`truck-detection`) to the **Satellite Data Tooling Hub** (ordering,
clipping, annotation, export). This is what your export must produce so we can train and run inference on it.
The format is defined in [DATA_EXCHANGE.md](DATA_EXCHANGE.md); this is the implementation ask + the per-function
responsibilities. Everything here is checkable — our importer validates it and refuses a bad bundle._

## The one-liner

You order + clip SuperDove imagery and annotate the moving-echo vehicles. Hand us an **exchange bundle** (a
folder or `.zip`: `manifest.json` + `imagery/*.tif` + `annotations.gpkg`). We import it, regenerate training
chips, and it's immediately trainable + runnable. This is the manual handoff and the spec for the eventual live
app-to-app pipeline — same contract either way.

## Must-have checklist

- [ ] **Clip** each scene to an **8-band uint16 Surface-Reflectance GeoTIFF**, **EPSG:32610, native grid,
      never reprojected/resampled**.
- [ ] **Name** each file `<Location>_<NN>_<YYYYMMDD>.tif` — that **stem is the join key** (e.g.
      `Centralia_03_20260601`).
- [ ] **Annotate** each moving echo as **3 points** (blue → red → green) in a **GeoPackage**, layer
      `Annotations`, Point geometry, EPSG:32610, fields `vehicle_id` / `sequence` (1,2,3) / `scene`.
- [ ] The `scene` field on **every** annotation row **exactly equals** the GeoTIFF stem.
- [ ] Emit a **`manifest.json`** (format `trg-echo-exchange` v1) listing the imagery and the annotation
      **counts** (vehicles + keypoints).
- [ ] **Verify** before sending: run our importer's `--dry-run` and confirm `handshake: OK`.

If those hold, import is mechanical. The rest of this doc is the *why* and the *how* per function.

---

## Why the constraints exist (so they don't get "optimized away")

- **The signal is sub-pixel motion.** A truck is 2–6 px; we detect the **colour-separated blue→red→green
  streak** it leaves because bands are captured a fraction of a second apart. **Resampling/reprojecting the
  raster smears that streak and destroys the signal** — this is the single most important constraint.
- **The join is by name, not geography.** We match a label to its image by the `scene` **text field**, never by
  spatial extent (overlapping scenes on the same corridor would cross-contaminate). A wrong `scene` string
  silently drops every label for that scene.

## Per-function responsibilities

### Ordering
- Product: PlanetScope `PSScene`, bundle `analytic_8b_sr_udm2`, instrument **PSB.SD (SuperDove)** — 8-band
  Surface Reflectance. **Reproject OFF.** (Mind the EDU quota: billed per intersecting scene, not by area.)
- Capture, per scene, the sidecar metadata (`*_metadata.json`) — we need it later for velocity (see below).

### Clipping
- Clip to the corridor, but keep the raster on its **native grid in EPSG:32610** — do **not** resample, warp,
  or change resolution. Keep all **8 bands, uint16**. Nodata = 0 outside the clip is fine.
- Output one GeoTIFF per scene, named `<Location>_<NN>_<YYYYMMDD>.tif`.

### Annotation
Label on the **GeoTIFF**, styled true-colour (bands R6/G4/B2). Emit a **GeoPackage** (not GeoJSON — GeoJSON
assumes WGS84 and will silently misread 32610 metres):

| Field | Type | Rule |
|---|---|---|
| `vehicle_id` | int | groups a vehicle's 3 points; **unique within a scene** |
| `sequence` | int | **1 = blue, 2 = red, 3 = green** — place point 1 on the blue blob, 2 on red, 3 on green |
| `scene` | text | **exactly** the GeoTIFF stem |

Geometry **Point** (not MultiPoint), CRS **EPSG:32610**, **exactly 3 points per vehicle**. If you label in
pixel space (`L.CRS.Simple`) and apply the affine at save (recommended), watch the fidelity foot-guns:

1. **Y-flip.** Raster pixel space has **row 0 at the top, increasing downward**; a math canvas has y increasing
   upward. Read `row` as pixels-**from-top**, or every keypoint comes out vertically mirrored. In the save
   transform `y = f + e·row`, the **`e` term is negative** — that minus sign is the most important line.
2. **Native-resolution canvas.** 1 canvas pixel = 1 GeoTIFF pixel — no fit-scaling, or `(col,row)` no longer
   maps through the affine.
3. **Keep `(col,row)` as floats** — integer snapping shows up as a systematic offset.

### Export (assemble the bundle)
- Layout: `manifest.json` + `imagery/<scene>.tif` (the scenes in this batch) + `annotations.gpkg`.
- `manifest.json` (see [DATA_EXCHANGE.md](DATA_EXCHANGE.md) for the full schema):
  ```json
  {
    "format": "trg-echo-exchange", "version": 1,
    "created": "2026-07-20T14:30:00Z", "source": "satellite-data-tooling-hub", "crs": "EPSG:32610",
    "imagery": [{ "scene": "Centralia_03_20260601", "file": "imagery/Centralia_03_20260601.tif" }],
    "annotations": { "file": "annotations.gpkg", "layer": "Annotations", "vehicles": 42, "keypoints": 126 }
  }
  ```
- The **`vehicles`/`keypoints` counts must match the file** — that's the handshake; a mismatch (truncated or
  wrong export) is rejected before it can pollute training.
- Ship as a folder or a `.zip`. Imagery-only (no `annotations`) is valid for inference-only scenes;
  annotations-only (no `imagery`, referencing scenes we already have) is valid for label updates.

## The #1 risk: the `scene` string

Our exporter looks up `imagery/{scene}.tif` by the **short stem**, not a Planet scene id or the full Planet
filename:

```
scene = "Centralia_03_20260601"           ✅
NOT     "20260601_193044_12_2461"          ❌  (Planet scene id)
NOT     "..._3B_AnalyticMS_SR_8b_clip"     ❌  (full Planet stem)
```

If your app stores Planet-named files, either write the short stem into `scene`, or tell us the naming and
we'll add an alias. Either way, **send us your actual filenames before the first bulk export.**

## Acceptance handshake (prove parity before bulk work)

1. **Match the reference format.** We can emit a golden bundle from our data —
   `python3 src/export_bundle.py --out data/exports/demo --scenes <scene> --zip`. Diff your export's layout +
   `manifest.json` against it.
2. **Round-trip one scene.** Export one bundle, we drop it in `data/inbox/` and run
   `python3 src/import_data.py --dry-run` — it must print **`handshake: OK`** and resolve every scene. Then we
   scale up.

## Forward-looking (nice-to-have, for velocity)

Not needed to train the detector, but needed to turn pixel displacement into speed — cheap to include while you
have the scenes in hand: **sub-second acquisition timestamp**, **`satellite_id` / `strip_id`**, and a real
**off-nadir / view angle** (the GeoTIFF's satellite zenith/azimuth read `0`). Fold these into the manifest (or a
per-scene sidecar) whenever convenient.
