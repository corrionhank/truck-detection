#!/usr/bin/env python3
"""
Reference exporter — package scenes from the active working set into an exchange bundle.

This is the executable spec for the format defined in docs/DATA_EXCHANGE.md: the bundle the
aggregation/annotation tool (Satellite Data Tooling Hub) should produce. It doubles as a
golden fixture to hand the hub's developer, and as a round-trip test for import_data.py
(export a bundle here, import it back).

Emits:
  <out>/manifest.json          the exchange contract (format 'trg-echo-exchange', v1)
  <out>/imagery/<scene>.tif     the requested scenes' GeoTIFFs
  <out>/annotations.gpkg        their annotations (layer 'Annotations'), if any

Run:  python3 src/export_bundle.py --out data/exports/demo --scenes Centralia_01_20260511 --zip
      python3 src/export_bundle.py --out data/exports/all           # every active scene, folder form
"""
import argparse
import datetime
import json
import shutil
import zipfile
from pathlib import Path

import geopandas as gpd

REPO = Path(__file__).resolve().parent.parent
GPKG = REPO / "data" / "active" / "Annotations-RGB.gpkg"
IMAGERY = REPO / "data" / "active" / "imagery"
LAYER = "Annotations"
EPSG = 32610


def main(a):
    out = Path(a.out)
    if not out.is_absolute():
        out = REPO / out
    scenes = ([s.strip() for s in a.scenes.split(",") if s.strip()]
              if a.scenes else sorted(p.stem for p in IMAGERY.glob("*.tif")))
    if not scenes:
        raise SystemExit("no scenes to export (none given and none in active/imagery)")

    if out.exists():
        shutil.rmtree(out)
    (out / "imagery").mkdir(parents=True)

    imagery = []
    for s in scenes:
        tif = IMAGERY / f"{s}.tif"
        if not tif.exists():
            print(f"  ! no imagery for {s} — skipping (annotations for it, if any, still export)")
            continue
        shutil.copy2(tif, out / "imagery" / tif.name)
        imagery.append({"scene": s, "file": f"imagery/{tif.name}"})

    annotations = None
    if GPKG.exists():
        g = gpd.read_file(GPKG, layer=LAYER)
        g["scene"] = g["scene"].astype(str).str.strip()
        sub = g[g["scene"].isin(scenes)]
        if len(sub):
            if "fid" in sub.columns:
                sub = sub.drop(columns=["fid"])
            sub = gpd.GeoDataFrame(sub, geometry="geometry", crs=EPSG)
            sub.to_file(out / "annotations.gpkg", layer=LAYER, driver="GPKG")
            annotations = {"file": "annotations.gpkg", "layer": LAYER,
                           "vehicles": int(sub["vehicle_id"].nunique()), "keypoints": int(len(sub))}

    manifest = {
        "format": "trg-echo-exchange", "version": 1,
        "created": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "truck-detection/export_bundle.py", "crs": "EPSG:32610",
        "imagery": imagery,
    }
    if annotations:
        manifest["annotations"] = annotations
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    v = annotations["vehicles"] if annotations else 0
    print(f"bundle: {len(imagery)} scene(s), {v} vehicles -> {out.relative_to(REPO)}/")

    if a.zip:
        zpath = out.with_suffix(".zip")
        if zpath.exists():
            zpath.unlink()
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(out.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(out))
        shutil.rmtree(out)
        print(f"zipped -> {zpath.relative_to(REPO)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output bundle dir (e.g. data/exports/demo)")
    ap.add_argument("--scenes", default=None, help="comma-separated scenes (default: all in active/imagery)")
    ap.add_argument("--zip", action="store_true", help="zip the bundle and remove the folder")
    main(ap.parse_args())
