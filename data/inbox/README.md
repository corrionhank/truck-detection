# data/inbox/ — drop zone for new data

Drop new data here, then run the importer. This is the manual stand-in for the eventual
app-to-app pipeline: everything you drop is validated against the join contract and moved
into the working set (`data/active/`), ready for training and inference.

## Preferred: an exchange bundle

A **bundle** (folder or `.zip`) with `manifest.json` + `imagery/*.tif` + `annotations.gpkg`, as defined in
[`../../docs/DATA_EXCHANGE.md`](../../docs/DATA_EXCHANGE.md). The importer unpacks it, validates the manifest
**handshake** (declared vs. actual counts, missing files, CRS) and refuses a mismatched export. This is what the
aggregation/annotation tool should produce — the ask is in
[`../../docs/EXPORT_REQUEST.md`](../../docs/EXPORT_REQUEST.md). Loose files (below) still work without a manifest.

## What to drop

- **Raw imagery** — PlanetScope SuperDove GeoTIFFs (`<Location>_<NN>_<YYYYMMDD>.tif`), 8-band
  uint16 Surface Reflectance, **EPSG:32610, native grid (never reprojected)**. Needed for both
  inference and (with annotations) training.
- **Annotation sets** — a GeoPackage with a layer `Annotations`: Point geometry, EPSG:32610,
  fields `vehicle_id` (int), `sequence` (int **1=blue, 2=red, 3=green**), `scene` (text = the
  GeoTIFF filename stem, e.g. `Centralia_01_20260511`). A vehicle = exactly 3 points.

You can drop just imagery (inference-only), just an annotation set (if its imagery is already
imported), or both together. Subfolders are fine — the importer scans recursively.

## Then run

```bash
python3 src/import_data.py --dry-run   # validate + report, change nothing
python3 src/import_data.py             # ingest: imagery -> active/imagery, merge labels, rebuild chips
```

The importer:
1. Validates each annotation set (CRS, sequence 1/2/3, complete B/R/G triples) and checks every
   `scene` resolves to a GeoTIFF — **the one field that silently breaks the join**.
2. Moves imagery into `data/active/imagery/` and merges labels into
   `data/active/Annotations-RGB.gpkg` (**replace-by-scene**; the gpkg is backed up first).
3. Regenerates the 64×64 COCO chips (`export_coco.py`).
4. Moves processed annotation sets to `_processed/<timestamp>/` for provenance.

After that: the new scenes appear in the console's Inference tab, and
`python3 src/train_detector.py --held <scene>` trains on the updated set.

## The #1 gotcha

`scene` must **exactly** match the GeoTIFF stem — not a Planet scene id, not the full Planet
filename. If it doesn't resolve, the importer reports the orphaned scenes and skips them rather
than dropping labels silently.

_Everything in this folder is gitignored except this README._
