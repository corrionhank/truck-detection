# Truck Detection — Repo Seed & Context

_Last updated: 2026-06-24_

**TL;DR** — This repo is the **machine-learning half** of a WSU Transportation Research Group
feasibility study: detect **trucks** and other vehicles (and, ideally, estimate their velocity) in single-capture
**PlanetScope SuperDove (~3 m)** satellite imagery over Washington freight corridors, plus the QGIS
labeling pipeline to train it. The physical signal is the **"moving echo"** — a colour-separated streak
left by a moving vehicle because SuperDove captures its bands a fraction of a second apart.

> Portable context for this **standalone repo**. It assumes **no access to the parent tool**. Copy the
> GeoTIFF set (see [§4 Data](#4-data-on-hand)) into `data/`, then start from
> [§9 Suggested first run-through](#9-suggested-first-run-through).

---

## 1. Background & goal
WSU TRG feasibility study: can **PlanetScope SuperDove (~3 m)** imagery estimate **truck/vehicle traffic
on Washington State freight corridors**, to supplement WSDOT permanent ground counters? This repo owns
the **ML half** — vehicle detection + velocity estimation in single-capture SuperDove imagery over
highways, and the labeling workflow that feeds it.

## 2. The physical signal — "moving echoes"
SuperDove images its 8 spectral bands from slightly different rows of the focal plane, so each band is
captured a **fraction of a second apart** as the satellite moves. Consequences:

- **Static ground** is co-registered → all bands line up.
- A **moving vehicle** sits at a different ground position in each band → it smears into a short
  **colour-separated streak**. The temporal capture order is **Blue → Red → Green** (focal-plane readout
  order, **not** band-number order), so a moving car appears as a blue, then red, then green blob in a row.
- The **streak length/direction ∝ velocity.** The relative inter-band pixel displacement is directly
  measurable; converting it to absolute speed needs the **fixed inter-band time gap** — a sensor constant
  that is **NOT in per-scene metadata**. Get it from Planet docs or the Adamiak paper.

**Reference method — Adamiak et al. 2025** (_Int. J. Applied Earth Obs. Geoinformation_ 142:104707):
- **Keypoint R-CNN**, **3 keypoints per vehicle** (one per band: B, R, G). Detect + track, estimate
  velocity from the inter-band displacement.
- Trained on **manual QGIS annotations**, masked to OSM road centerline + 30 m buffer, **one class**
  ("moving echo"), exported as **COCO keypoints**.
- Note: GPS is a poor velocity ground truth (it underestimates high speeds).

## 3. The ML task (concrete)
- **Input:** an 8-band SuperDove Surface-Reflectance raster (or an RGB / streak-derived rendering of it),
  restricted to the road corridor.
- **Output:** per-vehicle detections — at minimum a point/box; ideally **3 keypoints (B, R, G)** to recover
  heading + speed.
- **Model to replicate (current target):** Keypoint R-CNN / keypoint detector, COCO-keypoint training format
  (Adamiak — each truck's moving echo as 3 keypoints B/R/G). _First **gate on the signal-visibility check**
  (can the echo even be seen in SR at 3 m?) before investing in annotation + training. Full approach/model
  catalog in [APPROACHES.md](APPROACHES.md)._
- **Eval:** per-vehicle counts and (where possible) speed, validated against WSDOT counter aggregates at
  co-located sites. Supervision is realistically **weak/aggregate** (counts over a window) unless you
  hand-label keypoints — design metrics accordingly.

## 4. Data on hand
**15 scenes**, each a clipped strip of a T-1 freight corridor near I-5/I-90. In the parent repo they live
in `imagery/all_geotiffs/` — copy that folder into this repo's `data/`.

| Location          | scenes | dates                   |
|-------------------|:------:|-------------------------|
| Stanwood I-5      | 10     | 2026-04-19 … 2026-05-11 |
| Centralia I-5     | 2      | 2026-05-11 (two passes) |
| Bellingham I-5    | 1      | 2026-04-25              |
| Seattle I-5       | 1      | 2026-05-02              |
| Ellensburg I-90   | 1      | 2026-05-04              |

- **Naming:** `<Location>_<NN>_<YYYYMMDD>.tif` (NN = chronological per location). True-colour PNG previews
  of each also exist (`all_previews/`) for quick browsing.
- **Format:** PlanetScope `PSScene`, bundle `analytic_8b_sr_udm2`, instrument **PSB.SD (SuperDove)**.
  **8-band, uint16, Surface Reflectance, ~3.0 m/px (3.4 m GSD), EPSG:32610.** Nodata = 0 outside the
  corridor (clips are thin — e.g. Bellingham is a 4309×5750 bbox but only ~368k valid pixels ≈ 3.31 km²).
- **UDM2 masks removed.** The per-scene `*_udm2_clip.tif` usable-data/cloud masks that shipped with the raw
  downloads were deleted in cleanup — not needed for manual labeling on hand-picked clear scenes. They
  re-ship with any future Planet order if automated cloud/QA filtering is wanted later
  ([APPROACHES.md](APPROACHES.md) §5).
- **Band order (Planet PSB.SD standard):** 1 Coastal Blue · **2 Blue** · 3 Green I · **4 Green** · 5 Yellow ·
  **6 Red** · 7 Red Edge · 8 NIR. **True colour = R:6, G:4, B:2** (verified). Red-edge / NIR are available
  if useful.
- **Reproject is OFF on every order, on purpose.** Resampling would smear the inter-band offset, which
  *is* the velocity signal. Keep it off; **never re-grid these rasters.**

## 5. QGIS labeling workflow
1. **Load** a `*.tif`. Layer Properties → Symbology → Render type **Multiband color** → **Red = Band 6,
   Green = Band 4, Blue = Band 2** → Contrast enhancement **Stretch to MinMax**, **Cumulative count cut
   2–98%**. (QGIS defaults to bands 1/2/3 and looks wrong until you set this.)
2. Optionally **mask to the road** (OSM highway centerline + ~30 m buffer) so labeling stays on-road.
3. Create a **point/keypoint label layer** (GeoPackage). Per Adamiak: one class "moving echo", **3 keypoints
   per vehicle in capture order B → R → G**. Digitize the streak, not a single dot.
4. **Export to COCO keypoints** for training.

> Not yet built: an auto-generated QGIS keypoint-layer template + default RGB style + COCO exporter. The
> parent project planned these but never built them — **standing up these helpers is a sensible first task
> for this repo.**

## 6. Open questions / risks (read before modeling)
1. **Signal visibility / SR-vs-basic.** Does the inter-band echo survive in the SR product at 3 m? A
   morphology (white top-hat) test on the Bellingham SR scene found bright candidates with **near-zero
   inter-band offset (max ≈ 0.4 px ≈ 1.2 m)**. **Do not over-read this as "SR kills the echo"** — Adamiak
   used an orthorectified product (Ortho-Analytic 3B) and still saw strong echoes, and the null is equally
   explained by (a) a low-traffic scene (Bellingham) and (b) a brightness detector that finds lane paint,
   not colour-separated movers. Cheaper free checks first (high-traffic Seattle/Stanwood scene + a Red−Blue
   band-difference view); order the basic product only if those still show nothing. Full reasoning in
   [APPROACHES.md](APPROACHES.md) §3.
2. **Sub-resolution vehicles.** At 3 m/px a car ≈ 1 px, a truck ≈ 2–6 px. The **streak (multi-pixel) is
   more detectable than the vehicle** — target the echo pattern, not the car.
3. **Traffic / scene selection.** The baseline scene had little apparent moving traffic. Prefer
   high-traffic scenes/times and scenes co-located with counters.
4. **Absolute speed** needs the fixed inter-band dt (sensor constant; not in scene metadata).

## 7. Ground truth
WSDOT **permanent counters (149 sites)** give counts/AADT at points — aggregate, not per-vehicle. The 10
T-1 AOIs were placed near these. Use them for count validation; per-vehicle speed has no clean reference
(GPS is poor). Expect **weak/aggregate supervision** unless you hand-label.

## 8. Getting more data (constraint)
Imagery comes from a Planet **EDU "Education & Research Basic"** plan: a hard **3,000 km²/month** cap,
billed a **100 km² minimum per intersecting scene** regardless of clip size (so cost ≈ scenes × 100 km²,
not area). The account is **currently over quota (3,266 / 3,000 as of 2026-06-22)** → **no new scenes until
the cycle resets.** Plus 30-day data latency; SuperDove only (no SkySat / sub-metre). **The 15 scenes on
hand are the working set for now.**

## 9. Suggested first run-through
1. Copy `all_geotiffs/` (15 tifs + README) into this repo's `data/`.
2. QGIS: render one Stanwood scene true-colour (R6/G4/B2); eyeball the road for colour streaks at full zoom.
3. Hand-label a handful of clear moving echoes (3 keypoints each) → export COCO → sanity-check the format.
4. Decide **SR vs basic** by inspecting whether any scene shows real inter-band streaks; if not, plan one
   basic-product order (after quota resets).
5. Stand up a minimal Keypoint R-CNN training loop on the toy labels to validate the end-to-end pipeline
   before scaling annotation.

## 10. Reference
- **Approaches & model replication detail:** [APPROACHES.md](APPROACHES.md) — signal-visibility analysis
  (the gate), Adamiak Keypoint R-CNN (truck moving-echo detection), baselines, and optional corridor masking.
- **Method:** Adamiak et al. 2025, IJAEOG 142:104707 — "moving echoes", keypoint detection, SuperDove
  velocity.
- **Origin tool** (AOI building + Planet ordering + viewing): `wsu-trg-satellite-freight-feasibility`
  (local parent repo).
- **Stakeholders:** WSU TRG — Jake Wagner, Eric Jessup.
