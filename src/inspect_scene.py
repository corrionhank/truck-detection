#!/usr/bin/env python3
"""
Lesson 1 — read a SuperDove scene and render its true-colour view in code.

Goal: prove you can open the GeoTIFF, read its georeferencing, and produce the
same R6/G4/B2 image QGIS shows -- without QGIS. Everything downstream (OSM road
masks, training chips, COCO export) is built on exactly these primitives:
    - rasterio to open the raster and read bands
    - src.transform (the affine) to map pixel <-> map coordinates
    - a percentile stretch to turn 16-bit reflectance into an 8-bit picture

Run (from anywhere -- paths are resolved relative to the repo, not your cwd):
    python3 src/inspect_scene.py
    python3 src/inspect_scene.py imagery/all_geotiffs/Stanwood_06_20260502.tif
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

# Repo root = parent of this file's folder. Lets the script run from any cwd.
REPO = Path(__file__).resolve().parent.parent

# Planet PSB.SD band order -> true colour. rasterio bands are 1-based.
#   1 Coastal Blue, 2 Blue, 3 Green I, 4 Green, 5 Yellow, 6 Red, 7 Red Edge, 8 NIR
RED, GREEN, BLUE = 6, 4, 2


def stretch(band, lo_pct=2, hi_pct=98):
    """2-98% percentile stretch over valid (nonzero) pixels -> uint8 0-255.

    nodata is 0 outside the road corridor; we exclude it so the padding doesn't
    drag the contrast, and we keep it black in the output.
    """
    valid = band[band > 0]
    if valid.size == 0:
        return np.zeros(band.shape, dtype=np.uint8)
    p_lo, p_hi = np.percentile(valid, [lo_pct, hi_pct])
    scaled = np.clip((band.astype(np.float32) - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1)
    scaled[band == 0] = 0  # nodata stays black
    return (scaled * 255).astype(np.uint8)


def main(scene_path):
    path = Path(scene_path)
    if not path.is_absolute():
        path = REPO / path

    with rasterio.open(path) as src:
        # --- What is this file? (the metadata you'd otherwise read in QGIS) ---
        print(f"file         {path.name}")
        print(f"size         {src.width} x {src.height} px, {src.count} bands, {src.dtypes[0]}")
        print(f"CRS          {src.crs}")
        print(f"resolution   {src.res[0]:.2f} x {src.res[1]:.2f} map-units/px (metres, EPSG:32610)")
        print(f"bounds       {src.bounds}")
        print(f"nodata       {src.nodata}")
        # The affine transform: (col,row) -> (x,y) in map units. The bridge to OSM.
        print(f"transform    {tuple(round(v, 2) for v in src.transform[:6])}")

        # --- Read the three true-colour bands and stretch each one ---
        r = stretch(src.read(RED))
        g = stretch(src.read(GREEN))
        b = stretch(src.read(BLUE))

    rgb = np.dstack([r, g, b])
    valid_frac = (rgb.sum(axis=2) > 0).mean()
    print(f"valid pixels  {valid_frac:.1%}  (the rest is nodata corridor padding)")

    out_dir = REPO / "outputs"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{path.stem}_truecolor.png"
    Image.fromarray(rgb).save(out)
    print(f"wrote         {out.relative_to(REPO)}")


if __name__ == "__main__":
    default = "data/cold/imagery/Seattle_01_20260502.tif"
    main(sys.argv[1] if len(sys.argv) > 1 else default)
