# Annotation Data Spec

_What the annotation file holds, what the pipeline pulls from elsewhere, and how they join. Reference for the export step._

## What each annotation row holds

The annotation layer is a GeoPackage of points (`Annotations-RGB.gpkg`, table `Annotations`).

| Field | Type | Notes |
|-------|------|-------|
| geometry | Point | x, y in EPSG:32610 (UTM zone 10N, meters) |
| vehicle_id | int | groups the 3 points of one vehicle |
| sequence | int | 1 = blue, 2 = red, 3 = green (capture order) |
| scene | text | must exactly match the GeoTIFF filename (no extension) |
| fid | int | auto row id, not used by the pipeline |

No pixels, no imagery, no colors are stored. Each row is a real-world coordinate plus tags.

## A vehicle = 3 rows

One vehicle is three rows sharing the same `vehicle_id`, ordered by `sequence`:
- sequence 1 -> blue keypoint
- sequence 2 -> red keypoint
- sequence 3 -> green keypoint

Rows without exactly 3 points are incomplete and should be dropped at export.

## Pulled from the GeoTIFF at export (NOT in the annotation)

The annotation is only a pointer. The matching scene GeoTIFF supplies everything visual:

- **Affine transform** — converts each annotation coordinate to a pixel row and column. This is the bridge from UTM meters to pixel space. Without it, labels cannot be placed on the image.
- **CRS** — confirms both are EPSG:32610 so coordinates line up.
- **Band values** — the 8-band reflectance pixels. This is the actual training input.
- **Image size and bounds** — width, height, extent. Used to chip scenes into tiles and to know which points fall in which tile.

## From code, not either file

These are preprocessing decisions, defined in the pipeline, not stored in the annotation or the GeoTIFF header:

- Band pick and order: bands 2, 6, 4 = blue, red, green.
- Stretch or scaling from 16-bit reflectance to model input.
- Optional: UDM2 cloud mask (separate `_udm2_clip.tif` per scene) to drop cloudy pixels.
- Optional: road mask (OSM buffer) to constrain labels or detections to the road.

## The join

The export is a join between the annotation and its GeoTIFF:

1. Match each point to its GeoTIFF by the `scene` field (text match, not spatial extent).
2. Use the GeoTIFF affine transform to convert the point coordinate into a pixel row/col.
3. Pair that pixel location with the band values from the same GeoTIFF.
4. Group by `vehicle_id`, order by `sequence`, write COCO keypoints plus image chips.

The coordinate is the anchor. The GeoTIFF supplies the pixels.

## Rules that must hold

- **Match by `scene`, never by spatial extent.** Overlapping scenes cover the same ground, so an extent-based join leaks one scene's labels onto another. The `scene` field prevents this. A point labeled `Stanwood_07` only ever becomes a label on `Stanwood_07`.
- **`scene` values must exactly match GeoTIFF filenames.** It is a text match. Trim whitespace (a stray leading space breaks the lookup).
- **No blank `scene` values.** A point with no scene cannot be matched to a raster and must be dropped or it errors.
- **Do not reproject the GeoTIFFs.** The band offset is the signal; resampling smears it.
