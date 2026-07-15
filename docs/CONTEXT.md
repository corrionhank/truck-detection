# Context — project, signal, data, constraints

The *what and why* of this repo. For how the system is built see [ARCHITECTURE.md](ARCHITECTURE.md); for the
model reference see [MODELING.md](MODELING.md); for the data schema see [DATA.md](DATA.md).

---

## Goal

WSU Transportation Research Group feasibility study: can **PlanetScope SuperDove (~3 m)** imagery estimate
**truck/vehicle traffic on Washington freight corridors**, to supplement WSDOT permanent ground counters?
This repo owns the **ML half** — vehicle detection (and, ideally, velocity) in single-capture SuperDove
imagery over highways, plus the labeling workflow that feeds it.

## The signal — "moving echoes"

SuperDove images its 8 spectral bands from slightly different rows of the focal plane, so each band is
captured a **fraction of a second apart** as the satellite moves:

- **Static ground** is co-registered → all bands line up.
- A **moving vehicle** sits at a different ground position in each band → it smears into a short
  **colour-separated streak**. The temporal order is **blue → red → green** (focal-plane readout order, *not*
  band-number order), so a moving vehicle appears as a blue, then red, then green blob in a row.
- **Streak length/direction ∝ velocity.** The inter-band pixel displacement is directly measurable; converting
  it to absolute speed needs the inter-band time gap **Δt**, which is treated as **per-scene** — derived from the
  satellite's ephemeris velocity, `Δt = (v_sat / (w_bands · d_GSD))⁻¹` (Adamiak 2025) — **not** a fixed constant.
  (Van Etten's ~800 ms green–blue delta is only an order-of-magnitude cross-check.)

At 3 m/px a car ≈ 1 px and a truck ≈ 2–6 px, so the **streak is more detectable than the vehicle** — we target
the echo pattern, not the car. The reference method is Adamiak et al. 2025 (Keypoint R-CNN, 3 keypoints per
vehicle); full detail in [MODELING.md](MODELING.md).

## Data on hand

**18 scenes total**, each a clipped strip of a freight corridor near I-5 / I-90.

- **8 labeled** (`data/active/imagery/`) — the working set, 339 labeled vehicles (see [DATA.md](DATA.md) §6).
- **10 unlabeled** archived (`data/cold/imagery/`) — Seattle (I-5) + 9 more Stanwood dates, for future labeling.

**Format** — PlanetScope `PSScene`, bundle `analytic_8b_sr_udm2`, instrument **PSB.SD (SuperDove)**: 8-band,
uint16, Surface Reflectance, ~3.0 m/px, **EPSG:32610**, nodata = 0 outside the corridor. Naming:
`<Location>_<NN>_<YYYYMMDD>.tif`.

- **Band order (PSB.SD):** 1 Coastal Blue · **2 Blue** · 3 Green I · **4 Green** · 5 Yellow · **6 Red** ·
  7 Red Edge · 8 NIR. **True colour = R:6, G:4, B:2** (verified). Red-edge / NIR are available if useful.
- **Never re-grid these rasters.** Reproject is off on every order on purpose — resampling would smear the
  inter-band offset that *is* the velocity signal.
- **UDM2 cloud masks** were dropped (not needed for manual labeling on hand-picked clear scenes); they re-ship
  with any future Planet order if automated QA is wanted.

## Constraints

- **Imagery is frozen.** The Planet **EDU "Education & Research Basic"** plan has a hard **3,000 km²/month** cap
  billed at a **100 km² minimum per intersecting scene**; the account is over quota, so **no new scenes until
  the cycle resets** (plus 30-day latency; SuperDove only, no sub-metre SkySat).
- **Ground truth is weak/aggregate.** WSDOT **permanent counters (149 sites)** give counts/AADT at points, not
  per-vehicle; the AOIs were placed near them for count validation. Per-vehicle speed has no clean reference
  (GPS underestimates high speeds). Design metrics for aggregate supervision.
- **Training is CPU-only** — see [HARDWARE.md](HARDWARE.md).

## Open questions (research)

Mostly answerable from **Planet docs** (Imagery Product Spec, PSB.SD tech specs) and **Adamiak 2025**, not from
per-scene metadata:

- **Velocity timing — settled approach (per-scene ephemeris).** Δt is derived **per scene** from the satellite's
  ephemeris velocity, not a universal constant: `Δt = (v_sat / (w_bands · d_GSD))⁻¹` — `v_sat` from Planet's
  ephemeris service per scene, `w_bands` = pixels across a band's width, `d_GSD` = ground sampling distance
  (Adamiak 2025, Eq. 3–4). Then `velocity = mean(B→R, R→G keypoint distances) / Δt`. Still to do per scene:
  fetch `v_sat` from ephemeris and confirm `w_bands`. Van Etten's ~800 ms is only an order-of-magnitude check.
- **Sensor fleet consistency.** Are all SuperDove birds identical in band layout & inter-band timing, or does
  the constant vary per satellite? Map each scene → `satellite_id` to confirm it applies uniformly.
- **SR processing.** Does SR/ortho perform explicit **inter-band co-registration**? (Confirmed from tags: SR =
  6Sv2.1 atmospheric/radiometric correction — the geometric/ortho steps and any band-to-band registration
  accuracy still need Planet docs.) A **basic (L1B) product order** when quota resets would quantify SR-vs-basic
  echo strength.
- **Spectral.** Do red-edge (7) / NIR (8) / yellow (5) add echo signal beyond the true-colour trio? (Adamiak
  found no relationship between keypoint difficulty and spectral channel — room to test.)
- **Per-scene metadata.** Recover the sidecars (`*_metadata.json`) from the parent repo for **`satellite_id`**
  and a real **off-nadir/view angle** (the GeoTIFF's satellite zenith/azimuth read `0` placeholder).

## Reference

- **Method:** Adamiak et al. 2025, IJAEOG 142:104707 — moving echoes, keypoint detection, SuperDove velocity
  (used Ortho-Analytic 3B, not raw). Van Etten 2024 — PlanetScope vehicle segmentation + counts.
- **Origin tool:** `wsu-trg-satellite-freight-feasibility` (local parent repo — AOI building, Planet ordering,
  viewing) — also holds the per-scene metadata sidecars.
- **Stakeholders:** WSU TRG — Jake Wagner, Eric Jessup.
