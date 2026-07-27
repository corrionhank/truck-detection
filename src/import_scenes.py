#!/usr/bin/env python3
"""
Import imagery for INFERENCE only — no annotations, no training changes.

Drop GeoTIFFs (loose, a folder, or a .zip / inference bundle) into `data/inbox-scenes/`, then
run this. Each scene is moved into `data/active/imagery/` so it shows up in the console's
Inference tab, ready to run any trained model against. **It never touches the annotation set or
the training data** — existing models are untouched, and the new scenes read "UNSEEN" for every
model (nothing to leak).

Difference from `import_data.py` (the training importer): **CRS is not required to be EPSG:32610.**
Inference doesn't join labels to pixels, so a scene in any UTM zone works — detections map back in
the tif's own CRS. (The training importer must reject non-32610 scenes because the point->pixel
join would be wrong; here there's no join.) The imagery must still be the same **8-band SuperDove
Surface-Reflectance** product the models expect (bands 6/4/2 -> R/G/B).

Run:  python3 src/import_scenes.py             # import everything in data/inbox-scenes/
      python3 src/import_scenes.py --dry-run   # validate + report, change nothing
"""
import argparse
import datetime
import shutil
import zipfile
from pathlib import Path

import rasterio

REPO = Path(__file__).resolve().parent.parent
INBOX = REPO / "data" / "inbox-scenes"
IMAGERY = REPO / "data" / "active" / "imagery"


def main(a):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    INBOX.mkdir(parents=True, exist_ok=True)

    for z in INBOX.glob("*.zip"):                          # unpack any bundles/zips of imagery
        dest = INBOX / "_unpacked" / z.stem
        if dest.exists():
            shutil.rmtree(dest)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        print(f"unpacked {z.name} -> _unpacked/{z.stem}/")

    tifs = sorted(p for p in INBOX.rglob("*.tif") if "_processed" not in p.parts)
    if not tifs:
        print(f"no imagery in {INBOX.relative_to(REPO)}/. Drop GeoTIFFs (or a .zip of them) there and re-run. "
              f"See {INBOX.relative_to(REPO)}/README.md.")
        return
    print(f"{len(tifs)} GeoTIFF(s) to import for inference")

    ready = []
    for p in tifs:
        try:
            with rasterio.open(p) as src:
                epsg = src.crs.to_epsg() if src.crs else None
                bands = src.count
        except Exception as e:
            print(f"  ! {p.name}: not a readable raster ({e}) — skipping")
            continue
        if bands < 6:
            print(f"  ! {p.stem}: only {bands} bands — expected 8-band SuperDove (needs bands 6/4/2); skipping")
            continue
        crs_note = f"EPSG:{epsg}" if epsg else "no CRS"
        extra = "" if epsg == 32610 else "  (non-32610 — fine for inference, not for training)"
        print(f"  {p.stem}: {bands} bands, {crs_note}{extra}")
        ready.append(p)

    if a.dry_run:
        print("\n[dry-run] validated only — nothing moved.")
        return
    if not ready:
        print("\nnothing valid to import.")
        return

    IMAGERY.mkdir(parents=True, exist_ok=True)
    moved = 0
    for p in ready:
        dest = IMAGERY / p.name
        if dest.exists():
            print(f"  {p.name}: already in active/imagery — skipping (leaving your inbox copy)")
            continue
        shutil.move(str(p), str(dest))
        moved += 1

    for z in list(INBOX.glob("*.zip")):                    # file processed zips for provenance
        (INBOX / "_processed" / stamp).mkdir(parents=True, exist_ok=True)
        shutil.move(str(z), str(INBOX / "_processed" / stamp / z.name))
    if (INBOX / "_unpacked").exists():
        shutil.rmtree(INBOX / "_unpacked")

    print(f"\n{moved} scene(s) -> data/active/imagery/  (annotations + training untouched)")
    print("  run them in the console's Inference tab — they'll show 'UNSEEN' for every trained model.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", dest="dry_run", action="store_true", help="validate + report, change nothing")
    main(ap.parse_args())
