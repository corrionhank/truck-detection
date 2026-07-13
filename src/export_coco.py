#!/usr/bin/env python3
"""
Export the QGIS annotations + SuperDove GeoTIFFs to a COCO-keypoints dataset.

This is the join described in docs/ANNOTATION_SPEC.md, made concrete:
    annotation point (UTM metres) --[inverse affine of its scene]--> pixel col/row
    grouped by vehicle_id, ordered by sequence (1 blue, 2 red, 3 green)
    -> one COCO annotation of 3 keypoints, on a per-vehicle image chip.

The annotation GeoPackage stores only coordinates + tags; every pixel comes from
the matching scene GeoTIFF (matched by the `scene` text field, never by extent).

Outputs (under data/coco/):
    images/<scene>__v<vehicle_id>.png   true-colour (R6/G4/B2) chip per vehicle
    annotations.json                    COCO keypoints (1 category: moving_echo)

Run:  python3 src/export_coco.py [--chip 64]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
GPKG = REPO / "Annotations-RGB.gpkg"
GEOTIFF_DIR = REPO / "imagery" / "all_geotiffs"
OUT_DIR = REPO / "data" / "coco"

# Planet PSB.SD band order -> true colour; also the capture order blue->red->green.
RED, GREEN, BLUE = 6, 4, 2
# sequence field: 1 = blue, 2 = red, 3 = green (capture order).
SEQ_NAMES = {1: "blue", 2: "red", 3: "green"}


def stretch_params(band, lo_pct=2, hi_pct=98):
    """Percentile stretch bounds over valid (nonzero) pixels; nodata is 0."""
    valid = band[band > 0]
    if valid.size == 0:
        return 0.0, 1.0
    p_lo, p_hi = np.percentile(valid, [lo_pct, hi_pct])
    return float(p_lo), float(max(p_hi - p_lo, 1e-6))


def apply_stretch(band, p_lo, span):
    scaled = np.clip((band.astype(np.float32) - p_lo) / span, 0, 1)
    scaled[band == 0] = 0  # keep nodata black
    return (scaled * 255).astype(np.uint8)


def load_vehicles():
    """Return {scene: {vehicle_id: [(seq, x, y), ...]}} for complete vehicles."""
    gdf = gpd.read_file(GPKG, layer="Annotations")
    gdf["scene"] = gdf["scene"].astype(str).str.strip()  # fix the whitespace bug
    gdf = gdf[gdf["scene"].str.len() > 0]                 # drop blank scenes

    by_scene = defaultdict(lambda: defaultdict(list))
    for _, r in gdf.iterrows():
        by_scene[r["scene"]][int(r["vehicle_id"])].append(
            (int(r["sequence"]), r.geometry.x, r.geometry.y)
        )

    # Keep only vehicles with exactly the 3 sequences {1,2,3}.
    kept, dropped = defaultdict(dict), 0
    for scene, vehicles in by_scene.items():
        for vid, pts in vehicles.items():
            if sorted(s for s, _, _ in pts) == [1, 2, 3]:
                kept[scene][vid] = sorted(pts)
            else:
                dropped += 1
    return kept, dropped


def main(chip):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img_dir = OUT_DIR / "images"
    img_dir.mkdir(exist_ok=True)
    half = chip // 2

    by_scene, dropped = load_vehicles()

    images, annotations = [], []
    img_id = ann_id = 0
    n_vehicles = 0

    for scene in sorted(by_scene):
        tif = GEOTIFF_DIR / f"{scene}.tif"
        if not tif.exists():
            print(f"  ! no GeoTIFF for scene {scene!r} -- skipping")
            continue
        with rasterio.open(tif) as src:
            r = src.read(RED); g = src.read(GREEN); b = src.read(BLUE)
            rp = stretch_params(r); gp = stretch_params(g); bp = stretch_params(b)
            rgb = np.dstack([apply_stretch(r, *rp),
                             apply_stretch(g, *gp),
                             apply_stretch(b, *bp)])
            W, H = src.width, src.height
            inv = ~src.transform

            for vid, pts in sorted(by_scene[scene].items()):
                # pixel coords per keypoint, in blue/red/green order
                px = [(inv * (x, y)) for _, x, y in pts]  # (col,row) floats
                cols = [c for c, _ in px]; rows = [ro for _, ro in px]
                cx, cy = float(np.mean(cols)), float(np.mean(rows))

                # chip window clamped inside the scene
                x0 = int(round(cx)) - half
                y0 = int(round(cy)) - half
                x0 = max(0, min(x0, W - chip))
                y0 = max(0, min(y0, H - chip))
                crop = rgb[y0:y0 + chip, x0:x0 + chip]
                if crop.shape[:2] != (chip, chip):
                    continue  # scene smaller than a chip (shouldn't happen)

                # keypoints relative to chip origin; v=2 means labelled+visible
                kp = []
                for c, ro in px:
                    kp += [round(c - x0, 2), round(ro - y0, 2), 2]
                kxs = kp[0::3]; kys = kp[1::3]
                pad = 3
                bx0 = max(0.0, min(kxs) - pad); by0 = max(0.0, min(kys) - pad)
                bx1 = min(float(chip), max(kxs) + pad); by1 = min(float(chip), max(kys) + pad)
                bw, bh = bx1 - bx0, by1 - by0

                fname = f"{scene}__v{vid}.png"
                Image.fromarray(crop).save(img_dir / fname)
                img_id += 1
                images.append({"id": img_id, "file_name": fname,
                               "width": chip, "height": chip, "scene": scene})
                ann_id += 1
                annotations.append({
                    "id": ann_id, "image_id": img_id, "category_id": 1,
                    "keypoints": kp, "num_keypoints": 3,
                    "bbox": [round(bx0, 2), round(by0, 2), round(bw, 2), round(bh, 2)],
                    "area": round(bw * bh, 2), "iscrowd": 0,
                    "vehicle_id": vid,
                })
                n_vehicles += 1

    coco = {
        "info": {"description": "SuperDove moving-echo keypoints (blue->red->green)",
                 "chip_px": chip, "gsd_m": 3.0},
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": 1, "name": "moving_echo", "supercategory": "vehicle",
            "keypoints": ["blue", "red", "green"],
            "skeleton": [[1, 2], [2, 3]],
        }],
    }
    out = OUT_DIR / "annotations.json"
    out.write_text(json.dumps(coco, indent=2))

    print(f"scenes with labels : {len(by_scene)}")
    print(f"vehicles exported  : {n_vehicles}  (incomplete dropped: {dropped})")
    print(f"chips              : {img_dir.relative_to(REPO)}/  ({chip}x{chip} px)")
    print(f"coco               : {out.relative_to(REPO)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", type=int, default=64, help="chip size in pixels")
    main(ap.parse_args().chip)
