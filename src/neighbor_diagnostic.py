#!/usr/bin/env python3
"""
Neighbor diagnostic — how often a chip contains a SECOND vehicle (trained as background).

Export writes one vehicle per chip, centered. A neighbor vehicle whose echo falls inside the
crop is currently an unlabeled positive → trained as background, which can teach the model to
suppress real echoes (a plausible cause of dense-scene under-firing).

Padded-chip jitter changes the exposure window, so count neighbors in TWO bands by centroid
distance from the chip's center vehicle:
  - <= 16 px : inside EVERY jittered 64px crop (always-covered region)
  - 16-48 px : inside SOME epochs' crops (jitter reach; old fixed crop reached only 32 px)

Report-only. High counts (expect Centralia densest) mean multi-vehicle chip targets should ship
inside the Fix 2 export rewrite. Also dumps per-scene acquisition date.

Run:  python3 src/neighbor_diagnostic.py
"""
import datetime
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio

REPO = Path(__file__).resolve().parent.parent
GPKG = REPO / "data" / "active" / "Annotations-RGB.gpkg"
GEO = REPO / "data" / "active" / "imagery"
BAND1, BAND2 = 16.0, 48.0   # px


def scene_date(s):
    d = s.rsplit("_", 1)[1]
    return datetime.date(int(d[:4]), int(d[4:6]), int(d[6:8]))


def main():
    g = gpd.read_file(GPKG, layer="Annotations")
    g["scene"] = g["scene"].astype(str).str.strip()
    by = defaultdict(lambda: defaultdict(list))
    for _, r in g.iterrows():
        by[r["scene"]][int(r["vehicle_id"])].append((r.geometry.x, r.geometry.y))

    print(f"{'scene':36} {'date':>10} {'veh':>4} {'≤16px':>7} {'16-48px':>8}  (chips with a neighbor)")
    tot_v = tot_b1 = tot_b2 = 0
    rows = []
    for scene in sorted(by):
        tif = GEO / f"{scene}.tif"
        if not tif.exists():
            continue
        with rasterio.open(tif) as src:
            inv = ~src.transform
        cents = {}
        for vid, pts in by[scene].items():
            if len(pts) != 3:
                continue
            cx, cy = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])
            cents[vid] = np.array(inv * (cx, cy))
        vids = list(cents)
        b1 = b2 = 0
        for vi in vids:
            d = [np.hypot(*(cents[vj] - cents[vi])) for vj in vids if vj != vi]
            if any(x <= BAND1 for x in d):
                b1 += 1
            if any(BAND1 < x <= BAND2 for x in d):
                b2 += 1
        n = len(vids)
        tot_v += n; tot_b1 += b1; tot_b2 += b2
        rows.append((scene, n, b1, b2))
        print(f"{scene:36} {scene_date(scene)!s:>10} {n:>4} "
              f"{b1:>4} {100*b1/max(n,1):>4.0f}% {b2:>4} {100*b2/max(n,1):>3.0f}%")

    print(f"\n{'TOTAL':36} {'':>10} {tot_v:>4} "
          f"{tot_b1:>4} {100*tot_b1/max(tot_v,1):>4.0f}% {tot_b2:>4} {100*tot_b2/max(tot_v,1):>3.0f}%")
    print("\ninterpretation: '≤16px' chips carry a neighbor in EVERY crop (worst — always background-labeled);")
    print("'16-48px' appear in SOME jittered epochs. If either is high on the dense scenes, multi-vehicle")
    print("chip targets should ship inside the Fix 2 export rewrite. Report only — nothing changed.")

    print("\nacquisition-date span:")
    ds = sorted(scene_date(s) for s, *_ in rows)
    print(f"  {ds[0]} → {ds[-1]}  ({(ds[-1]-ds[0]).days} days, all {ds[0].strftime('%b')}–{ds[-1].strftime('%b')}) "
          f"— one spring window, a clutter-diversity limitation.")


if __name__ == "__main__":
    main()
