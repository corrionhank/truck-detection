# Data exchange contract — aggregation/annotation tool → training & inference

The interface between the **Satellite Data Tooling Hub** (data aggregation, clipping, annotation) and this
**training/inference** repo. The hub *exports* an **exchange bundle**; this repo *imports* it with
`src/import_data.py`. One format, validated on both sides — the manual stand-in for, and the spec for, the
eventual live app-to-app pipeline.

---

## The bundle

A folder **or** a `.zip` with exactly this layout:

```
<bundle>/
  manifest.json                     # the contract (below) — REQUIRED
  imagery/
    <scene>.tif                     # clipped SuperDove GeoTIFFs (0+; omit for annotations-only updates)
  annotations.gpkg                  # one GeoPackage, layer "Annotations" (omit for imagery-only drops)
```

`<scene>` is the **GeoTIFF filename stem** and the join key — e.g. `Centralia_03_20260601`. Drop the whole
folder (or the `.zip`) into `data/inbox/` and run the importer.

## `manifest.json`

```json
{
  "format": "trg-echo-exchange",
  "version": 1,
  "created": "2026-07-20T14:30:00Z",
  "source": "satellite-data-tooling-hub",
  "crs": "EPSG:32610",
  "imagery": [
    { "scene": "Centralia_03_20260601", "file": "imagery/Centralia_03_20260601.tif" }
  ],
  "annotations": {
    "file": "annotations.gpkg",
    "layer": "Annotations",
    "vehicles": 42,
    "keypoints": 126
  },
  "notes": "optional free text"
}
```

| Field | Required | Meaning |
|---|---|---|
| `format` | ✅ | must be `"trg-echo-exchange"` |
| `version` | ✅ | schema version (currently `1`) |
| `created` | ✅ | ISO-8601 timestamp |
| `source` | – | who produced it (provenance) |
| `crs` | ✅ | must be `"EPSG:32610"` |
| `imagery[]` | – | `{scene, file}` per GeoTIFF (relative path). Omit/`[]` for annotations-only. |
| `annotations` | – | `{file, layer, vehicles, keypoints}` — **counts are the handshake**. Omit for imagery-only. |
| `notes` | – | free text |

## The `annotations.gpkg` layer (unchanged from the internal schema)

Layer **`Annotations`**, geometry **Point**, CRS **EPSG:32610**, one row per keypoint:

| Field | Type | Rule |
|---|---|---|
| `vehicle_id` | int | groups a vehicle's 3 points; **unique within a scene** |
| `sequence` | int | **1 = blue, 2 = red, 3 = green** (capture order) |
| `scene` | text | **exactly** the GeoTIFF stem — the one field that silently breaks the join |

A vehicle = exactly 3 points (sequences 1,2,3); incomplete vehicles are dropped at import.

## Rules both sides enforce

1. **`scene` must resolve to a GeoTIFF** — either shipped in the same bundle or already imported. The importer
   reports orphaned scenes and skips them; it never drops labels silently.
2. **EPSG:32610 everywhere.** Annotation *points* are reprojected to 32610 if needed (safe). A **raster** in the
   wrong CRS is flagged, never reprojected — resampling would smear the moving-echo signal.
3. **The manifest counts must match the file** (`vehicles` / `keypoints`). A mismatch fails the handshake so a
   truncated or wrong export is caught before it pollutes the training set.
4. **Merge is replace-by-scene** on import: a bundle is authoritative for the scenes it covers; the active gpkg
   is backed up first, so nothing is ever lost.

## Producing a bundle (the hub side)

Implement the layout + `manifest.json` above. For a **reference implementation / golden fixture** to match,
this repo can emit a valid bundle from its own data:

```bash
python3 src/export_bundle.py --out data/exports/demo --scenes Centralia_01_20260511 --zip
```

`src/export_bundle.py` is the executable spec — diff your hub export against its output.

## Consuming a bundle (this repo)

```bash
# drop the bundle folder or .zip into data/inbox/, then
python3 src/import_data.py --dry-run    # unpack + validate the manifest handshake + the join, change nothing
python3 src/import_data.py              # ingest imagery, merge labels (replace-by-scene), rebuild COCO chips
```

The importer unpacks any `.zip`, validates each `manifest.json` (format/CRS/declared-vs-actual counts, missing
files), then ingests. Loose `.tif` + `.gpkg` without a manifest still work (filename-inferred) — the manifest
just makes the handshake explicit. See [`../data/inbox/README.md`](../data/inbox/README.md) and
[DATA.md](DATA.md).
