#!/usr/bin/env python3
"""
Local backend for the Truck Detection web console — DATA-ONLY.

Serves the real annotation dataset to the React console:
  GET  /api/dataset   -> true annotation counts read from the GeoPackage
  GET  /api/scenes    -> every GeoTIFF scene available, with its labelled-vehicle count

The modeling half of this backend (model registry, /api/detect, /api/models*, /outputs)
was moved to archive/ during the deliberate modeling rebuild — see archive/README.md.
Re-add those endpoints here when the new training/inference pipeline is ready.

Run: python3 backend/server.py (8787). The Vite dev server proxies /api here.
"""
from pathlib import Path

import geopandas as gpd
from flask import Flask, jsonify

REPO = Path(__file__).resolve().parent.parent
GPKG = REPO / "data" / "active" / "Annotations-RGB.gpkg"
GEOTIFF_DIR = REPO / "data" / "active" / "imagery"

app = Flask(__name__)


def dataset_stats():
    """Real counts straight from the annotation GeoPackage."""
    g = gpd.read_file(GPKG, layer="Annotations")
    g["scene"] = g["scene"].astype(str).str.strip()
    g = g[g["scene"].str.len() > 0]
    per_scene = {}
    for s in sorted(g["scene"].unique()):
        sub = g[g["scene"] == s]
        per_scene[s] = {"vehicles": int(sub["vehicle_id"].nunique()),
                        "echoes": int(len(sub))}
    return {
        "vehicles": int(g["vehicle_id"].nunique()),
        "echoes": int(len(g)),            # keypoints; 3 per vehicle (blue/red/green)
        "scenes_labelled": len(per_scene),
        "per_scene": per_scene,
    }


def all_scenes():
    """Every scene GeoTIFF, tagged with its labelled-vehicle count (0 if none)."""
    labelled = dataset_stats()["per_scene"]
    scenes = []
    for tif in sorted(GEOTIFF_DIR.glob("*.tif")):
        name = tif.stem
        scenes.append({"name": name, "vehicles": labelled.get(name, {}).get("vehicles", 0)})
    return scenes


@app.get("/api/dataset")
def api_dataset():
    return jsonify(dataset_stats())


@app.get("/api/scenes")
def api_scenes():
    return jsonify({"scenes": all_scenes()})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    app.run(host="127.0.0.1", port=args.port, debug=False)
