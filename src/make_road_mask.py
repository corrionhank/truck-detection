#!/usr/bin/env python3
"""
Lesson 2 / Pipeline steps 1-2 — turn OSM road lines into a label mask aligned to a scene.

What it does:
  1. Read the scene's georeferencing (CRS, bounds, affine transform, pixel grid).
  2. Fetch OSM drivable roads for that footprint (osmnx, in lat/lon).
  3. Reproject the road centerlines to the scene CRS (EPSG:32610, metres) and BUFFER
     them by a fixed half-width -> a road region (polygons).
  4. Rasterize that region onto the scene's exact pixel grid using the affine transform
     -> a binary mask (road=1 / not-road=0) that lines up pixel-for-pixel with the image.
  5. Write the mask GeoTIFF + a QA overlay PNG so you can eyeball the alignment.

This is the label half of the U-Net pipeline. The buffer half-width is the knob that sets
the "precision ceiling" (see docs/MODELING.md, road segmentation): too wide and the model just learns the buffer,
too tight and it misses pavement. Tune it by looking at the overlay.

Run:
    python3 src/make_road_mask.py                                   # Seattle, 15 m buffer
    python3 src/make_road_mask.py imagery/all_geotiffs/Stanwood_06_20260502.tif 12
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from PIL import Image
from shapely.ops import unary_union
import osmnx as ox

from inspect_scene import stretch, RED, GREEN, BLUE  # reuse Lesson 1's helpers

REPO = Path(__file__).resolve().parent.parent


def build_mask(scene_path, buffer_m=15.0):
    path = Path(scene_path)
    if not path.is_absolute():
        path = REPO / path

    with rasterio.open(path) as src:
        H, W = src.height, src.width
        transform = src.transform          # the pixel<->map affine
        crs = src.crs                      # EPSG:32610
        # OSM is queried in lat/lon, so convert the scene bounds 32610 -> 4326.
        west, south, east, north = transform_bounds(crs, "EPSG:4326", *src.bounds)
        rgb = np.dstack([stretch(src.read(RED)), stretch(src.read(GREEN)), stretch(src.read(BLUE))])
        valid = src.read(BLUE) > 0         # the corridor footprint (non-nodata)

    print(f"scene        {path.name}  ({W}x{H} px, {src_res(transform)} m/px)")
    print(f"bbox lon/lat W{west:.4f} S{south:.4f} E{east:.4f} N{north:.4f}")

    # --- 2. fetch OSM drivable roads as a routable graph (edges = centerlines) ---
    G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type="drive")
    edges = ox.graph_to_gdfs(G, nodes=False)            # GeoDataFrame, EPSG:4326
    print(f"OSM edges    {len(edges)} road segments fetched")

    # --- 3. reproject to scene CRS (metres) and buffer the centerlines ---
    edges_m = edges.to_crs(crs)
    road_region = unary_union(edges_m.geometry.buffer(buffer_m).tolist())

    # --- 4. rasterize onto the scene's exact grid via the affine transform ---
    mask = rasterize(
        [(road_region, 1)],
        out_shape=(H, W),
        transform=transform,      # <-- this is what aligns geometry to pixels
        fill=0,
        dtype="uint8",
    )

    # Restrict the road label to where real imagery exists. Outside the corridor is
    # nodata; labelling black pixels "road" would just teach the model the footprint.
    osm_cover = float(mask.mean())                      # raw OSM coverage of whole chip
    mask = (mask * valid.astype(np.uint8)).astype(np.uint8)

    # --- stats: how the OSM mask relates to the scene's own corridor footprint ---
    road_frac = float(mask.mean())
    valid_frac = float(valid.mean())
    road_in_valid = float((mask[valid] == 1).mean()) if valid.any() else 0.0
    print(f"buffer       {buffer_m:g} m/side")
    print(f"OSM raw      {osm_cover:.2%} of chip (all OSM roads, before clip-to-valid)")
    print(f"road label   {road_frac:.2%} of chip | valid footprint {valid_frac:.2%}")
    print(f"within real imagery: {road_in_valid:.1%} labelled road, {1 - road_in_valid:.1%} not-road")
    if road_in_valid > 0.97:
        print("  ^ ~all valid pixels are 'road' -> buffer ~= the clip footprint = near-degenerate.")
        print("    Drop the buffer (e.g. 10-12 m) so shoulders/verges become not-road, OR use OSM polygons.")

    # --- 5. write mask GeoTIFF (aligned) + QA overlay PNG ---
    out_dir = REPO / "outputs"; out_dir.mkdir(exist_ok=True)
    mask_tif = out_dir / f"{path.stem}_roadmask.tif"
    with rasterio.open(
        mask_tif, "w", driver="GTiff", height=H, width=W, count=1,
        dtype="uint8", crs=crs, transform=transform, nodata=0,
    ) as dst:
        dst.write(mask, 1)

    overlay = rgb.copy()
    red = np.array([255, 0, 0], dtype=np.float32)
    overlay[mask == 1] = (0.45 * rgb[mask == 1] + 0.55 * red).astype(np.uint8)
    overlay_png = out_dir / f"{path.stem}_maskoverlay.png"
    Image.fromarray(overlay).save(overlay_png)

    print(f"wrote        {mask_tif.relative_to(REPO)}")
    print(f"wrote        {overlay_png.relative_to(REPO)}  (eyeball road alignment here)")


def src_res(transform):
    return f"{abs(transform.a):.0f}"


if __name__ == "__main__":
    scene = sys.argv[1] if len(sys.argv) > 1 else "data/cold/imagery/Seattle_01_20260502.tif"
    buf = float(sys.argv[2]) if len(sys.argv) > 2 else 15.0
    build_mask(scene, buf)
