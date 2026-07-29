#!/usr/bin/env python3
"""
Export the QGIS annotations + SuperDove GeoTIFFs to a COCO-keypoints dataset.

This is the join described in docs/DATA.md, made concrete:
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
from rasterio.warp import transform as warp_points

REPO = Path(__file__).resolve().parent.parent
GPKG = REPO / "data" / "active" / "Annotations-RGB.gpkg"
GEOTIFF_DIR = REPO / "data" / "active" / "imagery"
OUT_DIR = REPO / "data" / "active" / "coco"

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
    """Return ({scene: {vehicle_id: [(seq, x, y), ...]}}, dropped, crs) for complete vehicles.
    Coordinates stay in the GeoPackage's CRS; main() reprojects them per scene."""
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
    return kept, dropped, gdf.crs


def _annotation(px, x0, y0, export, ann_id, img_id, vid, is_center):
    """One COCO keypoint annotation for a vehicle, in export-window pixel space."""
    kp = []
    for c, ro in px:
        kp += [round(c - x0, 2), round(ro - y0, 2), 2]         # v=2 = labelled + visible
    kxs = kp[0::3]; kys = kp[1::3]
    pad = 3
    bx0 = max(0.0, min(kxs) - pad); by0 = max(0.0, min(kys) - pad)
    bx1 = min(float(export), max(kxs) + pad); by1 = min(float(export), max(kys) + pad)
    bw, bh = bx1 - bx0, by1 - by0
    return {
        "id": ann_id, "image_id": img_id, "category_id": 1,
        "keypoints": kp, "num_keypoints": 3,
        "bbox": [round(bx0, 2), round(by0, 2), round(bw, 2), round(bh, 2)],
        "area": round(bw * bh, 2), "iscrowd": 0,
        "vehicle_id": vid, "center": is_center,
    }


def main(chip, margin, out_dir, single):
    export = chip + 2 * margin          # exported image size; the model still crops `chip` from it at train time
    half_exp = export // 2
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(exist_ok=True)

    by_scene, dropped, src_crs = load_vehicles()

    images, annotations = [], []
    img_id = ann_id = 0
    n_vehicles = n_neighbors = n_reprojected = 0

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

            # The join only requires that a scene's points agree with ITS OWN raster — not that
            # the whole project sits in one UTM zone (WA straddles 10/11, and imagery from
            # anywhere is equally usable). So the raster stays in its native CRS — reprojecting
            # it would resample and smear the 1-3 px echo, which is the entire signal — and the
            # labels move instead. Reprojecting points is exact arithmetic on coordinates, and
            # past this line everything is pixel space, so CRS is gone from the pipeline.
            scene_pts = by_scene[scene]
            if src_crs is not None and src.crs is not None and src_crs != src.crs:
                flat = [(vid, seq, x, y) for vid, v in scene_pts.items() for seq, x, y in v]
                xs, ys = warp_points(src_crs, src.crs, [f[2] for f in flat], [f[3] for f in flat])
                moved = defaultdict(list)
                for (vid, seq, _, _), x, y in zip(flat, xs, ys):
                    moved[vid].append((seq, x, y))
                scene_pts = {vid: sorted(v) for vid, v in moved.items()}
                n_reprojected += 1

            # pixel keypoints per vehicle (blue/red/green order) — reused for the neighbour lookup
            veh_px = {vid: [(inv * (x, y)) for _, x, y in pts]
                      for vid, pts in scene_pts.items()}

            for vid in sorted(by_scene[scene]):
                px = veh_px[vid]
                cols = [c for c, _ in px]; rows = [ro for _, ro in px]
                cx, cy = float(np.mean(cols)), float(np.mean(rows))

                # export window (size `export`), centered on the vehicle, clamped inside the scene
                x0 = int(round(cx)) - half_exp
                y0 = int(round(cy)) - half_exp
                x0 = max(0, min(x0, W - export))
                y0 = max(0, min(y0, H - export))
                crop = rgb[y0:y0 + export, x0:x0 + export]
                if crop.shape[:2] != (export, export):
                    continue  # scene smaller than the export window (shouldn't happen)

                fname = f"{scene}__v{vid}.png"
                Image.fromarray(crop).save(img_dir / fname)
                img_id += 1
                images.append({"id": img_id, "file_name": fname,
                               "width": export, "height": export,
                               "chip_px": chip, "margin_px": margin, "scene": scene})

                # center vehicle first; unless --single, also every OTHER vehicle fully inside the
                # window, so a neighbour's echo is a labelled positive rather than trained-as-background.
                members = [vid]
                if not single:
                    for w in sorted(by_scene[scene]):
                        if w != vid and all(x0 <= c < x0 + export and y0 <= ro < y0 + export
                                            for c, ro in veh_px[w]):
                            members.append(w)
                for w in members:
                    ann_id += 1
                    annotations.append(_annotation(veh_px[w], x0, y0, export, ann_id, img_id, w, w == vid))
                n_vehicles += 1
                n_neighbors += len(members) - 1

    coco = {
        "info": {"description": "SuperDove moving-echo keypoints (blue->red->green)",
                 "chip_px": chip, "margin_px": margin, "export_px": export, "gsd_m": 3.0},
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": 1, "name": "moving_echo", "supercategory": "vehicle",
            "keypoints": ["blue", "red", "green"],
            "skeleton": [[1, 2], [2, 3]],
        }],
    }
    out = out_dir / "annotations.json"
    out.write_text(json.dumps(coco, indent=2))

    mode = "single-vehicle" if single else f"multi-vehicle (+{n_neighbors} neighbour annotations)"
    print(f"scenes with labels : {len(by_scene)}"
          + (f"  ({n_reprojected} in a different CRS — points reprojected, rasters untouched)"
             if n_reprojected else ""))
    print(f"chips (1/vehicle)  : {n_vehicles}  (incomplete dropped: {dropped})")
    print(f"export             : {export}x{export} px (chip {chip} + 2*margin {margin}) · {mode}")
    print(f"chips dir          : {img_dir}")
    print(f"coco               : {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chip", type=int, default=64, help="model chip size (cropped from the export at train time)")
    ap.add_argument("--margin", type=int, default=16,
                    help="padding each side; export = chip + 2*margin (for train-time jitter). "
                         "--margin 0 --single reproduces the legacy 64px single-vehicle output byte-for-byte")
    ap.add_argument("--out", default=None, help="output dir (default data/active/coco)")
    ap.add_argument("--single", action="store_true",
                    help="one annotation per chip (legacy); default emits multi-vehicle targets")
    a = ap.parse_args()
    main(a.chip, a.margin, Path(a.out) if a.out else OUT_DIR, a.single)
