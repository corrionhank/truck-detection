# Training data request — ML repo → data/labeling agent

_From the ML/training repo (`truck-detection`) to the agent managing imagery + the annotation tool
(`satellite-aoi-data-aggregator`). What the detector needs, in what format, and — the part that actually
gates progress — **how much and how diverse.** Grounded in a real leave-one-scene-out cross-validation
just run on the current labels._

## TL;DR

Two things gate training. One is solved, one is the real ask:

1. **Format** — already specified byte-for-byte in [LABELING_TOOL_CONTRACT.md](LABELING_TOOL_CONTRACT.md)
   and [DATA.md](DATA.md). Recap below; nothing new.
2. **Volume + diversity of labels** — this is the binding constraint, and we can now prove it with numbers.
   **We need ~10× more labeled vehicles, spread across more distinct corridors, densely labeled.**

## Why volume/diversity is the ask (the evidence)

A 3-fold leave-one-scene-out CV on the current 16 vehicles (train on 2 scenes, test on the held-out third):

| Trained on | Held-out (unseen) scene | Recall | Keypoint error |
|---|---|---|---|
| Centralia + Stanwood (8 veh) | Bellingham | **0/7 (0%)** | — |
| Bellingham + Stanwood (10 veh) | Centralia | 5/5 (100%) | 11.3 px |
| Bellingham + Centralia (12 veh) | Stanwood | 3/3 (100%) | 0.9 px |

Read: the model **can** generalize to an unseen scene (fold 3), but the result **swings 0%→100% on a
~4-vehicle difference** and collapses to 0% the moment the largest scene (Bellingham) leaves training.
That instability is the signature of being far below the data threshold. Localization (the part velocity
needs) only sharpens from 11 px → 1 px across our tiny data range. **More, more-diverse labels is the
single highest-value input you can give training.**

## Current state (from this repo)

- **15 scenes on disk**, but **5 distinct corridors**, and **10 of 15 are the same Stanwood I-5 corridor on
  different dates** (overlapping ground — see the scene-overlap note in [DATA.md](DATA.md)).
- **Only 3 scenes labeled:** Bellingham_01 (7 veh), Centralia_02 (6), Stanwood_08 (3) = **16 vehicles / 47
  points**. 12 scenes have **zero** labels, including 3 untouched distinct corridors (Seattle, Ellensburg,
  the other Centralia pass).

## Requirement 1 — Format (recap; full detail in the contract)

Emit a **GeoPackage** point layer named `Annotations`, one row per keypoint:

- `vehicle_id` (int, unique within a scene) · `sequence` (int **1=blue, 2=red, 3=green**, capture order) ·
  `scene` (text) · geometry `Point` in **EPSG:32610**.
- **One vehicle = exactly 3 points** (sequences 1,2,3). Incomplete triples are dropped at export.
- **`scene` must exactly match the GeoTIFF filename stem** the exporter looks up (e.g.
  `Stanwood_08_20260504`, not a Planet scene id). This is the one field that silently breaks the join —
  **send us your actual filenames so we can align the exporter.** See LABELING_TOOL_CONTRACT.md §"The #1 risk".

## Requirement 2 — Volume & diversity (the real ask)

**Density — label every echo, not a sample.** In each scene, digitize **every** visible moving echo, not a
handful. Partial labeling (what we have now) makes precision unmeasurable — we literally can't tell a false
positive from a real-but-unlabeled truck — and biases training toward whatever subset got picked.

**Diversity — new corridors beat new dates.** More Stanwood dates add temporal variety but cover ground the
model already generalizes to (fold 3). Spatial diversity across **distinct corridors/geographies** is what
breaks the single-scene dominance the CV exposed. Prioritize breadth of location.

**Targets** (Phase-1, to get off the steep part of the curve — not production scale):

| Metric | Now | Phase-1 target |
|---|---|---|
| Labeled vehicles | 16 | **~150–250** |
| Scenes with labels | 3 | **≥ 6–8** |
| Distinct corridors labeled | 3 | **≥ 4–5** |
| Max share from any one scene | ~44% | **≤ 30%** (no single scene dominates) |
| Vehicles per labeled scene | 3–7 | **≥ 15–20** |

(For reference, the method we're replicating — Adamiak 2025 — trained on **3,236** echoes. Thousands is the
eventual production target; the table above is just the near-term milestone that makes the model *stable*
and the eval *honest*.)

## Requirement 3 — Imagery access + naming

Training reads pixels **only** from the SR GeoTIFFs (the annotation supplies location/order/scene only).
So we need:

- The **exact GeoTIFFs** each label references — **8-band uint16 SR, native grid, never reprojected**
  (resampling smears the inter-band offset that *is* the signal).
- A settled **scene-naming convention** shared by labels and imagery (Requirement 1). Whatever the app calls
  its files, the `scene` string must resolve to the tif the exporter opens.

## Requirement 4 — Per-scene metadata (forward-looking, for velocity)

Not needed for the detector; needed to turn pixel displacement into **speed**. If cheap to include alongside:

- **Sub-second acquisition timestamp** (the `*_metadata.json` has it; the tif tag is minute-resolution).
- **`satellite_id` / `strip_id`** — to confirm the inter-band timing constant applies per bird.
- Real **off-nadir / view angle** (the tif's satellite zenith/azimuth read `0` placeholder).

Nice-to-have now; flag so it's captured while scenes are being handled, not reconstructed later.

## Priority order for you

1. **Now (frozen 16-scene corpus):** densely label **every echo** in the **3 unlabeled distinct corridors
   first** — `Seattle_01`, `Ellensburg_01`, `Centralia_01` — for spatial diversity; then bring
   Bellingham_01 / Centralia_02 / Stanwood_08 to full density; then a few more Stanwood dates for temporal
   variety. This alone could take us from 16 to ~100–200 vehicles.
2. **When Planet quota resets:** prioritize **new distinct freight corridors** over more dates of existing
   ones. Aim for ≥ 6–8 corridors total.

## Handoff / acceptance

Before bulk labeling, run the parity handshake in
[LABELING_TOOL_CONTRACT.md](LABELING_TOOL_CONTRACT.md) §"Acceptance handshake": label one already-labeled
scene, we diff GeoPackages (geometry sub-pixel, identical ids/scene, CRS=32610), and we run
`export_coco.py` unchanged on your output to confirm valid COCO + a correctly-centered chip. Then scale up.
