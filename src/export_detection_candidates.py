#!/usr/bin/env python3
"""
Export the models' unmatched detections as a review layer for the Annotation Studio.

These are the detections that landed nowhere near a labelled vehicle: the model's own
false positives. Reviewing them turns negative annotation from "hunt for look-alikes"
into "confirm or reject this specific point", and it targets the exact failure mode.

Two things this file is careful about:

  1. CRS. `red_utm_x/y` in the detection records are in each scene's NATIVE CRS (they
     come from `scene_transform * pixel`), which is EPSG:32611 for the eastern scenes.
     They are reprojected to EPSG:32610 here, genuinely, because the studio's layers are
     32610. Writing native values under a 32610 label is the mis-stamp bug that cost the
     studio a release; the round-trip check at the end of this script proves we did not
     repeat it.
  2. Duplicates. Both models flag many of the same places. Rows are deduplicated ACROSS
     models within a scene, so a reviewer never sees the same spot twice, and the `model`
     field records who flagged it (one model or both).

Run:  python3 src/export_detection_candidates.py [--min-score 0.3]
Writes data/exports/detection-candidates.gpkg (layer: DetectionCandidates)
"""
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_points
from shapely.geometry import Point

REPO = Path(__file__).resolve().parent.parent
DET_DIR = REPO / "analysis" / "truck-detection-report" / "data"
GEO = REPO / "data" / "active" / "imagery"
OUT = REPO / "data" / "exports" / "detection-candidates.gpkg"
TARGET_CRS = "EPSG:32610"

# The best-known operating point per model (see docs/KEYPOINT_GATE.md). A false positive
# that survives its model's gate is one that still costs precision at the tuned config,
# so it is worth more as a training negative than one the gate already removes.
TUNED_GATE = {"kprcnn-adamiak-v2": 8.0, "kprcnn-warmup-v1": 3.5}
SOURCES = {"kprcnn-adamiak-v2": "detections_adamiak-v2.csv",
           "kprcnn-warmup-v1": "detections_warmup-v1.csv"}
DEDUP_PX = 8.0          # 24 m; matches the tuned inference config


def main(a):
    frames = []
    for model_id, fname in SOURCES.items():
        d = pd.read_csv(DET_DIR / fname)
        d = d[(d.labelled == 1) & (d.matched_bool == 0) & (d.score >= a.min_score)].copy()
        d["model_id"] = model_id
        d["survives_gate"] = d.kp_score_mean > TUNED_GATE[model_id]
        frames.append(d)
    det = pd.concat(frames, ignore_index=True)

    rows = []
    for scene, g in det.groupby("scene"):
        tif = GEO / f"{scene}.tif"
        if not tif.exists():
            print(f"  ! no GeoTIFF for {scene!r}, skipping {len(g)} rows")
            continue
        with rasterio.open(tif) as src:
            native_crs, tr = src.crs, src.transform

        g = g.sort_values("score", ascending=False)
        xs, ys = g.kp_red_x.to_numpy(), g.kp_red_y.to_numpy()
        kept = []                                    # (index into g, set of models)
        kx, ky = [], []
        for i in range(len(g)):
            hit = None
            if kx:
                dx = np.asarray(kx) - xs[i]; dy = np.asarray(ky) - ys[i]
                j = int(np.argmin(dx * dx + dy * dy))
                if (dx[j] * dx[j] + dy[j] * dy[j]) < DEDUP_PX * DEDUP_PX:
                    hit = j
            if hit is not None:                      # same place, already have it
                kept[hit][1].add(g.model_id.iloc[i])
                kept[hit][2].append(g.survives_gate.iloc[i])
                continue
            kx.append(xs[i]); ky.append(ys[i])
            kept.append([i, {g.model_id.iloc[i]}, [g.survives_gate.iloc[i]]])

        idx = [k[0] for k in kept]
        sub = g.iloc[idx]
        # native easting/northing -> EPSG:32610 (no-op for zone 10, real work for zone 11)
        ex, ny = warp_points(native_crs, TARGET_CRS,
                             sub.red_utm_x.tolist(), sub.red_utm_y.tolist())
        for (k, (_, models, gates)), x32610, y32610 in zip(enumerate(kept), ex, ny):
            r = sub.iloc[k]
            rows.append({
                "scene": scene,
                "model": "both" if len(models) > 1 else next(iter(models)),
                "score": round(float(r.score), 5),
                "kp_score_mean": (round(float(r.kp_score_mean), 4)
                                  if pd.notna(r.kp_score_mean) else None),
                "survives_gate": bool(any(gates)),
                "px_col": round(float(r.kp_red_x), 2),
                "px_row": round(float(r.kp_red_y), 2),
                "dist_to_label_m": (round(float(r.dist_to_nearest_label_m), 1)
                                    if pd.notna(r.dist_to_nearest_label_m) else None),
                "streak_len_m": (round(float(r.streak_len_m), 1)
                                 if pd.notna(r.streak_len_m) else None),
                "native_crs": str(native_crs),
                "geometry": Point(x32610, y32610),
            })

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=TARGET_CRS)
    # rank: gate survivors first (they cost precision at the tuned operating point), then score
    gdf = gdf.sort_values(["survives_gate", "score"], ascending=[False, False]).reset_index(drop=True)
    gdf.insert(0, "det_id", np.arange(1, len(gdf) + 1))
    gdf.insert(1, "rank", np.arange(1, len(gdf) + 1))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(OUT, layer="DetectionCandidates", driver="GPKG")

    # ---- verification: reproject back to native, inverse affine, compare to px_col/px_row
    worst = 0.0
    for scene, g in gdf.groupby("scene"):
        with rasterio.open(GEO / f"{scene}.tif") as src:
            bx, by = warp_points(TARGET_CRS, src.crs,
                                 g.geometry.x.tolist(), g.geometry.y.tolist())
            inv = ~src.transform
            for x, y, pc, pr in zip(bx, by, g.px_col, g.px_row):
                c, ro = inv * (x, y)
                worst = max(worst, abs(c - pc), abs(ro - pr))

    print(f"\nwrote {OUT}  layer 'DetectionCandidates'  crs {gdf.crs}")
    print(f"  rows                 : {len(gdf)}")
    print(f"  scenes               : {gdf.scene.nunique()}")
    print(f"  flagged by both      : {int((gdf.model == 'both').sum())}")
    print(f"  survive tuned gate   : {int(gdf.survives_gate.sum())}  <- highest value")
    print(f"  zone-11 rows         : {int((gdf.native_crs == 'EPSG:32611').sum())}")
    print(f"  round-trip max error : {worst:.4f} px  (must be ~0; proves the CRS is genuine)")
    print("\n  by model:"); print(gdf.model.value_counts().to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--min-score", type=float, default=0.3)
    main(p.parse_args())
