#!/usr/bin/env python3
"""
Import dropped imagery + annotation sets into the working set.

Drop new GeoTIFFs and/or annotation GeoPackages into `data/inbox/`, then run this. It
validates the join contract, moves imagery into `data/active/imagery/`, merges annotations
into `data/active/Annotations-RGB.gpkg` (backing it up first), and regenerates the COCO
chips — so the new data is immediately ready for training (`train_detector.py`) and
inference (the console). This is the manual stand-in for the eventual app-to-app pipeline;
when that exists, point it at the same contract and this script becomes optional.

Contract a valid annotation set must satisfy (the same join `export_coco.py` does):
  - layer `Annotations` (or the file's only layer), Point geometry, EPSG:32610.
  - fields: `vehicle_id` (int), `sequence` (int in {1,2,3}), `scene` (text = GeoTIFF stem).
  - a vehicle = exactly 3 points (sequences 1,2,3); incomplete vehicles are dropped.
  - every `scene` must resolve to a GeoTIFF (in the inbox or already in active/imagery).

Merge policy: **replace-by-scene** (default) — a dropped set is authoritative for the scenes
it covers; existing rows for those scenes are replaced (so re-dropping a corrected set for a
scene just updates it). `--append` concatenates instead (only safe for brand-new scenes — it
can collide vehicle_ids on a scene that already exists). The active gpkg is backed up before
any change, so nothing is ever lost.

Run:  python3 src/import_data.py             # import everything in data/inbox/
      python3 src/import_data.py --dry-run   # validate + report only, change nothing
      python3 src/import_data.py --append    # add annotations instead of replacing per scene
      python3 src/import_data.py --no-export # skip regenerating COCO chips
"""
import argparse
import datetime
import shutil
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio

REPO = Path(__file__).resolve().parent.parent
INBOX = REPO / "data" / "inbox"
IMAGERY = REPO / "data" / "active" / "imagery"
GPKG = REPO / "data" / "active" / "Annotations-RGB.gpkg"
LAYER = "Annotations"
EPSG = 32610


def read_layer(path):
    """Read the annotation layer (prefer 'Annotations', else the file's only/first layer)."""
    import fiona
    layers = fiona.listlayers(path)
    layer = LAYER if LAYER in layers else layers[0]
    return gpd.read_file(path, layer=layer), layer


def validate(gdf):
    """Return (clean GeoDataFrame or None, issues[]). Reprojects POINTS to EPSG:32610 if
    needed (safe — only rasters must never be reprojected)."""
    issues = []
    need = {"vehicle_id", "sequence", "scene", "geometry"}
    missing = need - set(gdf.columns)
    if missing:
        return None, [f"missing required columns {sorted(missing)}"]

    if gdf.crs is None:
        issues.append("no CRS on file; assuming EPSG:32610")
        gdf = gdf.set_crs(EPSG)
    elif gdf.crs.to_epsg() != EPSG:
        issues.append(f"reprojected points EPSG:{gdf.crs.to_epsg()} -> {EPSG}")
        gdf = gdf.to_crs(EPSG)

    gdf = gdf.copy()
    if not (gdf.geom_type == "Point").all():
        n = int((gdf.geom_type != "Point").sum())
        issues.append(f"{n} non-Point geometries dropped")
        gdf = gdf[gdf.geom_type == "Point"]

    gdf["scene"] = gdf["scene"].astype(str).str.strip()          # fix the whitespace/blank bugs
    gdf = gdf[gdf["scene"].str.len() > 0]
    gdf["sequence"] = gdf["sequence"].astype(int)
    gdf["vehicle_id"] = gdf["vehicle_id"].astype(int)

    bad = ~gdf["sequence"].isin([1, 2, 3])
    if bad.any():
        issues.append(f"{int(bad.sum())} rows with sequence not in 1/2/3 dropped")
        gdf = gdf[~bad]

    keep = []                                                    # complete B/R/G triples only
    for (sc, vid), grp in gdf.groupby(["scene", "vehicle_id"]):
        if sorted(grp["sequence"].tolist()) == [1, 2, 3]:
            keep.append(grp)
        else:
            issues.append(f"incomplete vehicle {sc}/{vid} (sequences {sorted(grp['sequence'])}) dropped")
    clean = gpd.GeoDataFrame(pd.concat(keep), crs=EPSG) if keep else None
    return clean, issues


def main(a):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tifs = sorted(p for p in INBOX.rglob("*.tif") if "_processed" not in p.parts)
    gpkgs = sorted(p for p in INBOX.rglob("*.gpkg") if "_processed" not in p.parts)
    if not tifs and not gpkgs:
        print(f"inbox empty. Drop .tif imagery and/or .gpkg annotation sets into "
              f"{INBOX.relative_to(REPO)}/ and re-run. See {INBOX.relative_to(REPO)}/README.md.")
        return
    print(f"inbox: {len(tifs)} imagery file(s), {len(gpkgs)} annotation set(s)")

    inbox_scenes = {p.stem: p for p in tifs}
    existing = {p.stem for p in IMAGERY.glob("*.tif")}
    available = set(inbox_scenes) | existing

    # --- validate annotation sets + join ---
    valid_sets = []
    for gp in gpkgs:
        gdf, layer = read_layer(gp)
        clean, issues = validate(gdf)
        print(f"\n{gp.relative_to(REPO)}  [layer '{layer}']")
        for msg in issues:
            print(f"  - {msg}")
        if clean is None or len(clean) == 0:
            print("  ! no valid annotations — skipping this set")
            continue
        orphans = sorted(s for s in clean["scene"].unique() if s not in available)
        if orphans:
            print(f"  ! scenes with NO matching GeoTIFF (drop the .tif too, or fix the name): {orphans}")
            clean = clean[~clean["scene"].isin(orphans)]
        for s in sorted(clean["scene"].unique()):
            print(f"  scene {s}: {clean[clean['scene'] == s]['vehicle_id'].nunique()} vehicles")
        if len(clean):
            valid_sets.append(clean)

    # --- imagery CRS sanity (warn only; never reproject a raster) ---
    for stem, p in inbox_scenes.items():
        with rasterio.open(p) as src:
            epsg = src.crs.to_epsg() if src.crs else None
        if epsg != EPSG:
            print(f"\n  ! imagery {stem}: CRS EPSG:{epsg} != {EPSG} — the pipeline assumes {EPSG}; "
                  "the point->pixel join will be wrong. Re-export this scene in EPSG:32610.")

    if a.dry_run:
        print("\n[dry-run] validated only — nothing changed.")
        return

    # --- ingest imagery (move into active/imagery) ---
    IMAGERY.mkdir(parents=True, exist_ok=True)
    moved = 0
    for p in inbox_scenes.values():
        dest = IMAGERY / p.name
        if dest.exists():
            print(f"  imagery {p.name}: already in active/imagery — leaving your inbox copy in place (skip)")
            continue
        shutil.move(str(p), str(dest))
        moved += 1
    if inbox_scenes:
        print(f"\nimagery: {moved} moved into {IMAGERY.relative_to(REPO)}/")

    # --- merge annotations (backup first) ---
    if valid_sets:
        new = gpd.GeoDataFrame(pd.concat(valid_sets, ignore_index=True), crs=EPSG)
        if GPKG.exists():
            base = gpd.read_file(GPKG, layer=LAYER)
            base["scene"] = base["scene"].astype(str).str.strip()
            bdir = REPO / "data" / "active" / "_backups"
            bdir.mkdir(exist_ok=True)
            shutil.copy2(GPKG, bdir / f"Annotations-RGB.{stamp}.gpkg")
            print(f"backed up gpkg -> {(bdir / f'Annotations-RGB.{stamp}.gpkg').relative_to(REPO)}")
            if a.append:
                clash = sorted(set(new["scene"].unique()) & set(base["scene"].unique()))
                if clash:
                    print(f"  ! --append with scenes already present {clash}: vehicle_ids may collide; "
                          "replace-by-scene (the default) is safer for existing scenes")
                merged = pd.concat([base, new], ignore_index=True)
            else:
                new_scenes = set(new["scene"].unique())
                for s in sorted(new_scenes):
                    old_n = base[base["scene"] == s]["vehicle_id"].nunique()
                    if old_n:
                        print(f"  scene {s}: replaced {old_n} existing vehicles")
                merged = pd.concat([base[~base["scene"].isin(new_scenes)], new], ignore_index=True)
            GPKG.unlink()
        else:
            merged = new
        merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=EPSG)
        if "fid" in merged.columns:
            merged = merged.drop(columns=["fid"])
        merged.to_file(GPKG, layer=LAYER, driver="GPKG")
        print(f"annotations: now {merged['vehicle_id'].nunique()} vehicles / {len(merged)} points across "
              f"{merged['scene'].nunique()} scenes -> {GPKG.relative_to(REPO)}")

        pdir = INBOX / "_processed" / stamp                      # keep the raw sets for provenance
        pdir.mkdir(parents=True, exist_ok=True)
        for gp in gpkgs:
            if gp.exists():
                shutil.move(str(gp), str(pdir / gp.name))

    # --- regenerate chips (only if labels changed) ---
    if valid_sets and not a.no_export:
        print("\nregenerating COCO chips (export_coco.py) ...")
        subprocess.run([sys.executable, str(REPO / "src" / "export_coco.py")], check=False)

    print("\ndone.")
    if valid_sets:
        print("  train:  python3 src/train_detector.py --id <id> --name \"<name>\" --held <scene>")
    print("  infer:  the console Inference tab now lists the imported scene(s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", dest="dry_run", action="store_true", help="validate + report, change nothing")
    ap.add_argument("--append", action="store_true", help="concatenate annotations instead of replacing per scene")
    ap.add_argument("--no-export", dest="no_export", action="store_true", help="skip regenerating COCO chips")
    main(ap.parse_args())
