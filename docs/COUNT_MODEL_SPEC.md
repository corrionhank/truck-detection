# Detector spec → count model

Everything the detector does, precisely, and what its output physically represents — written for building the
downstream model that turns detections into daily/annual counts comparable to WSDOT permanent counters.

Code references are file:line against the working tree at 2026-07-29. Units are stated everywhere.

---

# Part A — the detector, exactly

## A1. Source data

| Property | Value |
|---|---|
| Product | PlanetScope `PSScene`, bundle `analytic_8b_sr_udm2`, instrument **PSB.SD (SuperDove)** |
| Bands | 8, uint16, Surface Reflectance (6Sv2.1 atmospheric correction) |
| GSD | **3.0 m/px** (verified on all 21 scenes) |
| CRS | native per scene — EPSG:32610 (19 scenes) or 32611 (Tri-Cities, Spokane). **Never resampled.** |
| Nodata | 0, outside the corridor clip |
| Band order | 1 Coastal Blue · **2 Blue** · 3 Green I · **4 Green** · 5 Yellow · **6 Red** · 7 Red Edge · 8 NIR |

**The signal.** SuperDove reads its bands off different rows of the focal plane, so each band is captured a
fraction of a second later as the satellite advances. Static ground co-registers; a moving vehicle sits at a
different ground position in each band and smears into a **blue → red → green** streak (focal-plane readout
order, *not* band-number order). At 3 m/px a car ≈ 1 px and a truck ≈ 2–6 px, so **the streak is more detectable
than the vehicle**. The model targets the streak.

## A2. Preprocessing — identical at train and inference

`export_coco.py:40-52`, reused by `detect_scene.py:38-43`. Exact sequence, per band ∈ {6, 4, 2} → {R, G, B}:

```
valid   = band[band > 0]                          # nodata is 0
p_lo    = percentile(valid, 2)
p_hi    = percentile(valid, 98)
span    = max(p_hi - p_lo, 1e-6)
scaled  = clip((band - p_lo) / span, 0, 1)
scaled[band == 0] = 0                             # nodata stays black
uint8   = scaled * 255
model input = uint8 / 255                         # float32 in [0, 1]
```

**Consequence that matters for counting:** the stretch is computed **per scene**, over that scene's own
percentiles. Radiometry is therefore **not absolute across scenes**, so a confidence score of 0.5 does not
mean the same thing in two different scenes. Any fixed global threshold is not a fixed operating point.

**Discarded and never seen by the model:** 5 of 8 bands (coastal-blue, green-I, yellow, red-edge, NIR), the
16-bit precision, all geographic coordinates (used to place the crop, then dropped), the UDM2 cloud mask, and
any road mask.

## A3. Model graph

`model_registry.py:46-68`. torchvision `keypointrcnn_resnet50_fpn`:

| Component | Value |
|---|---|
| Backbone | ResNet-50 + FPN, COCO-pretrained, finetuned |
| Box head | `FastRCNNPredictor`, **2 classes** (0 = background, 1 = moving_echo) |
| Keypoint head | `KeypointRCNNPredictor`, **3 keypoints** (blue, red, green) |
| Anchors | sizes **(4, 8, 16, 32, 48)** px, ratios **(0.25, 0.5, 0.75, 1.0, 1.25)** — one size per FPN level, all ratios at each |
| Internal resize | `min_size=192`, `max_size=320` |

**The resize is load-bearing.** A 64 px chip is upscaled **3×** to 192 px before the backbone sees it, so anchors
act in 192-space: the nominal 4–48 px anchors correspond to **12–144 px** there, and a 4–8 px real streak
presents as 12–24 px. Any anchor sweep must reason in 192-space, not in native GSD.

Anchor count and ratio count both change the RPN head's shape, so the arch stored at train time must match at
load time. The registry stores it per model; `model_registry.build_model()` reconstructs it.

## A4. Training input pipeline (per sample)

1. **Export** (`export_coco.py`): 96×96 uint8 PNG = chip 64 + 2 × margin 16, centered on the vehicle's keypoint
   centroid, window clamped inside the scene.
2. **Augment on the 96, before cropping** (`train_detector.py:81-107`): h-flip p=0.5 · v-flip p=0.5 ·
   rotation `U(-180°, 180°)` about center · perspective warp, corner jitter `U(-5, 5)` px · brightness
   `× U(0.8, 1.2)`. Border mode `BORDER_REFLECT_101`. Running before the crop means rotation pulls **real
   neighbouring pixels** into the corners rather than reflected padding.
3. **Random 64×64 crop** — the translation jitter, **±16 px**, resampled every epoch. Crop origin is derived
   from where the keypoints sit, clamped to `[margin - jitter, margin + jitter]` ∩ `[0, S - CHIP]`, so it can
   never cut a keypoint and never needs to resample.
4. **Target** (`to_target`): keypoints clipped to `[1, CHIP-2]`; box = keypoint bbox + 3 px pad, min 4×4 px;
   label = 1; visibility flag v = 2. Multi-vehicle: **every** vehicle fully inside the window is a labelled
   positive, not background.
5. `repeat = 3` → three independent augmented draws per vehicle per epoch.

## A5. Optimization

Adam · **LR warmup** lr/100 → lr over 300 iters (`warmup_lr`) → `ReduceLROnPlateau(mode=min, factor=0.3,
patience=2, min_lr=1e-5)` on validation loss · grad-clip **1.5** (L2 norm) · batch 4 · non-finite batches
skipped · CPU (MPS diverges to NaN on this model and is probed for and rejected at startup).

Loss = torchvision's composite **sum**: RPN objectness + RPN box regression + ROI classification + ROI box
regression + keypoint loss (cross-entropy over the keypoint heatmap).

Validation = random **12 %** of training-scene chips, used **only** for the LR scheduler and best-epoch
selection. Held-out scenes are never touched during training. Best-val weights are restored at the end.

## A6. Inference — the deployment path

`detect_scene.py:102-206`. **This is the function whose output feeds the count model.**

```
rgb   = build_rgb(scene)                    # same band/stretch pipeline as training
valid = rgb.sum(axis=2) > 0                 # validity mask

for y0 in range(0, H-64+1, stride):         # stride default 40 px
  for x0 in range(0, W-64+1, stride):
    if valid[y0:y0+64, x0:x0+64].mean() >= min_valid:   # default 0.15
      evaluate window

per window: keep ONLY out["scores"][0] and out["keypoints"][0]     # <-- top-1 (detect_scene.py:135)
            discard if score < thresh
            kp_scene = kp_window + [x0, y0]

dedup:  sort by score desc; greedily keep a detection only if its RED keypoint
        is > CHIP/2 = 32 px from every already-kept red keypoint
```

**Every constant, in ground units at 3 m GSD:**

| Constant | px | metres | Where |
|---|---:|---:|---|
| Window | 64 | **192 m** | `CHIP` |
| Stride | 40 | **120 m** | `--stride` |
| Dedup suppression radius | 32 | **96 m** | `CHIP/2`, `detect_scene.py:146` |
| GT match radius | 6 | **18 m** | `detect_scene.py:166` |
| Min valid fraction per window | — | ≥15 % of window | `min_valid` |

**Output record, per detection:**

```json
{"score": 0.83,
 "keypoints_px": [[x_blue, y_blue], [x_red, y_red], [x_green, y_green]],
 "red_utm": [easting, northing]}
```

Only the **red (middle) keypoint** is georeferenced, via `transform * (px, py)`, in the **scene's native CRS**.
The blue and green keypoints stay in scene-pixel coordinates — but they are the velocity signal, so a count
model should georeference all three, or at minimum keep the pixel displacements.

Scene-level: `{scene, stride, thresh, count, detections[], gt{...}}` plus a montage and preview PNG.

## A7. How the reported metrics are computed

`detect_scene.py:160-172`. Both use the **red keypoint only**, matched within **6 px = 18 m**:

- `recall` = (# labelled vehicles with ≥1 detection within 18 m) / (# labelled)
- `precision` = (# kept detections within 18 m of ≥1 label) / (# kept detections)
- `f1` = harmonic mean

Note the asymmetry: a single detection can satisfy several labels and vice versa; this is a proximity match,
not a bipartite assignment. At high density that inflates both slightly.

---

# Part B — what a detection physically represents

## B1. The core conversion: density, not flow

This is the central point for the count model.

- A **permanent counter measures flux** — vehicles crossing a fixed point over time. Units veh/h → AADT.
- A **satellite scene measures density** — vehicles present on a length of road at a single instant. Units veh/km.

They are not the same quantity and are linked by the fundamental relation of traffic flow:

> **q = k · v**  —  flow [veh/h] = density [veh/km] × space-mean speed [km/h]

The detector supplies **both terms**: density from the count over observed road length, and speed from the echo
geometry itself. That is what makes a satellite snapshot convertible to a counter-comparable number at all.

## B2. The estimator

```
N_corr = N_det × (precision / recall)          # bias-correct the raw count
k      = N_corr / L_obs                        # veh per km of observed road
q      = k × v̄                                 # veh per hour
```

with `L_obs` in km and `v̄` the space-mean speed in km/h. Do this **per direction** where the road geometry
allows separating them, because expansion factors are directional.

> **Superseded for counting by B5 (2026-10-08).** `precision / recall` is measured against an easy-echo label
> subset (C1, C2) and has unknown sign of error. B5 replaces it with a detection share fitted directly against
> permanent-counter volumes, which absorbs C1, C2 and C7 at the densities sampled.

Worked, with the numbers we actually have for `Tacoma-Centralia_01`:

| Term | Value | Source |
|---|---|---|
| N_det | 86 (adamiak-v2 @ 0.3) | registry |
| precision / recall | 0.523 / 0.562 = **0.93** | registry |
| N_corr | 80 | |
| valid footprint | 4.5 km² | measured |
| bbox long axis | 22.2 km | measured |
| L_obs | **needs road-centerline intersection** | not yet computed |

If the corridor runs the long axis, `L_obs ≈ 22 km` and `k ≈ 3.6 veh/km`; at an assumed 100 km/h that is
**q ≈ 364 veh/h** across the full cross-section. Treat every number after `N_corr` as provisional until
`L_obs` is measured properly — the buffer width implied by area/extent varies from 136 m (Centralia_01) to
927 m (polygon_01), so the long-axis assumption is not safe generally.

## B3. Measuring L_obs properly

Intersect the **valid-data footprint** (not the bounding box — validity runs 1.1 %–41 % of the raster) with a
road centerline, then take length per direction:

```
footprint = polygonize(rgb.sum(2) > 0) in scene CRS
roads     = WSDOT centerlines, reprojected to scene CRS      # vector op, reproject freely
L_obs     = length(roads ∩ footprint) restricted to the classes you count
```

Restrict to the functional classes the counter covers, or the denominator includes frontage roads and ramps the
detector also fires on.

## B4. Measuring v̄ — two tiers

**Tier 0 (available today):** posted speed limit from the WSDOT speed-limit layer, joined to the road segment.
Biases high in congestion, but congestion is largely invisible anyway (see C5).

**Tier 1 (the real answer — built 2026-10-08, see B5):** the echo *is* a speed measurement.

```
Δt = (w_bands · d_GSD) / v_sat                      # Adamiak 2025 Eq. 3-4; seconds
v  = mean(‖B→R‖, ‖R→G‖) · d_GSD / Δt                # m/s
```

`d_GSD` = 3.0 m. `v_sat` = satellite ground-track velocity, from Planet's ephemeris service **per scene**.
`w_bands` = pixels across a band's width on the focal plane. Neither is in the GeoTIFF; both need the per-scene
`*_metadata.json` sidecars (in the parent `wsu-trg-satellite-freight-feasibility` repo) plus Planet docs.
Van Etten's ~800 ms green–blue delta is an order-of-magnitude cross-check only, not a constant to adopt.

This gives **per-vehicle speed**, so `v̄` is a genuine space-mean speed over the detections rather than an
assumption — and speed is itself a deliverable WSDOT would value.

**As built, Δt is fitted, not derived.** The ephemeris route above was never needed: the permanent counters
report speed, so Δt is solved from them (B5, step 2). Fitted blue→green offset: **1.17–1.21 s** for every source,
hand labels included (1.20 s). Van Etten's ~0.8 s is ruled out — it would put free-flow traffic at ~100 mph.
The equation above remains the physical cross-check if the sidecars are ever recovered.

## B5. Density method: scene time × the 15-minute count

How a capture is set against a counter, pair by pair, so that the pairs can later be fitted into a general
conversion from detections to vehicles per unit time (and, through Part D, AADT). Implemented in
`corridor-counts/scripts/calibrate.py` (run by every harvest; writes `frontend/public/data/calibration.json`),
with counter bins from `corridor-counts/scripts/counters.py`.

**The procedure.**

1. **How much time the scene covers.** Road length in the capture ÷ speed = how long a vehicle takes to cross
   the captured stretch. That is the slice of time the scene stands for.
2. **How many detections are in the scene** — on that road, in that direction.
3. **The real count.** The counter's 15-minute count for the same direction around the capture.
4. **The vehicle estimate.** `(15 min, V₁₅ vehicles) × (scene time ÷ 15 min)` = the vehicles the counter says
   should be within the scene over that time range — the counter's estimate of the number of points the scene holds.
5. **Save the pair** (detections, vehicle estimate), for every capture × counter × direction.
6. **Fit the collection** into a generalisable conversion from detections to vehicles per unit time.

Steps 1–4 are the measurement; one pair is noisy on its own (step 6 gives the floor). The conversion is a
property of the collection, and its form is fitted, not assumed.

### Inputs as chosen — `corridor-counts/scripts/density.py` (2026-10-08, roads from TIGER 2026-10-12)

The pairs are built by `scripts/density.py`, run by every harvest after `calibrate.py`, which writes one row
per capture × source × direction to `frontend/public/data/density.json` and shows them in the study report's
**Density method** table. The inputs the procedure leaves open are fixed as:

| Input | Chosen | Detail |
|---|---|---|
| Road `r*` | **The longest road in the scene** | Census TIGER/Line primary and secondary roads (MTFCC S1100/S1200; no ramps or local streets) inside the capture footprint, which is the AOI clipped to the capture. Grouped by route (TIGER route type + number: "I- 90", "US Hwy 97 N", "State Hwy 16" → I-90, US-97, SR-16); a road with 80% of its length within 8 m of another's (a street name over a route, a concurrent route) is merged into it. Users are told to draw one road, or several of similar length, so the longest stands for the scene. |
| Road length `L` | **r*'s centreline in the scene** | TIGER draws a divided highway as one line per direction: the share of r*'s length with a parallel line of the same road 4–300 m to the side is counted once, `L = total × (1 − divided share / 2)`. Checked against WSDOT section lengths: 58 of 68 usable captures within 5%; the rest are concurrent routes WSDOT files under another number (I-182 for US 12 in Tri-Cities), where TIGER's length is the physical road. |
| Speed `v` | **Median streak speed** | Per direction, the median of `streak / Δt × 3.6` over r*'s detections (Δt: the source's fitted band delay, `calibration.json`), keeping 8–160 km/h; the mean is recorded beside it. The counters' speed (harmonic mean of the bin's speed classes) when the direction has fewer than 3 detections. |
| Counters | **Every counter near r*, averaged** | Counter sites inside the footprint within 150 m of r*'s lines, this direction. Matched by distance, not route number, because of concurrent routes. |
| 15-minute bin | **The bin the capture falls in** | Pacific time; a counter enters only if that bin is Accepted and not a glitch. |
| Detections `n` | **On r*, by local direction** | The source's detections within 60 m of r*; each joins the direction its compass heading agrees with at r*'s nearest point, so a road that bends keeps two directions. |

Then, per row: scene time `T = 60 · L / v`; real count `λ = V₁₅ × T / 15 min`; real trucks `= λ × τ`, with `τ`
the WSDOT 2024 truck share km-weighted over the traffic sections along r*; and the ratio `n / (λ τ)`. The
direction labels (NB/SB or EB/WB) come from r*'s counters, or from its overall orientation. A row without a real
count says why. The road tables are cached per footprint (`scripts/.density-cache.json`), so TIGER is re-read
only for new captures.

**How it relates to `calibrate.py`.** `calibrate.py` (steps 1–2 below, and the Results table) keeps a
narrower definition — only the counter's own traffic stream (route sections within 3 km whose AADT is within
25% of the site section's), the counter's speed, and a 30-minute overlap-weighted window — because its job is
to fit the band delay Δt without leaning on the detections' own speed. `density.py` uses that Δt and builds the
pairs with the inputs above.

**What the chosen inputs give (density.json, 2026-10-12).** 229 rows, 66 with a real count (the rest are
mostly captures after the counter data ends, 2026-05-27).

| Source | Pairs | Detections ÷ real trucks | ÷ real count (all vehicles) | Dispersion `φ` |
|---|---|---|---|---|
| hand labels | 18 | 25% | 2.7% | 4.4 |
| adamiak-v2 | 18 | 27% | 2.9% | 9.3 |
| warmup-v1 | 18 | 36% | 3.8% | 4.4 |
| nojitter-v1 | 8 | 33% | 4.0% | 5.1 |
| nojitter-v1 @ 24 px | 4 | 37% | 5.4% | 1.2 |

On the counter's own stretch (`calibrate.py`) the same sources find 43–55% of trucks with `φ` ≈ 1. On the
longest road, with every nearby counter averaged, the share is lower and the pairs do not agree within counting
noise (`φ` 4–9 except the 4-pair 24 px source). The two directions of one capture can differ by 2× (Blaine–
Bellingham I-5: 19% NB, 40% SB). Likely causes, not yet checked: traffic changes along a 20–35 km road between
interchanges, and counter speed bins and truck shares that are assumed (see Assumptions).

### Terms (one direction of travel, one capture, one counter)

| Symbol | Meaning | Units | From |
|---|---|---|---|
| `L` | road length of the stretch inside the capture footprint | km | WSDOT section centrelines ∩ footprint |
| `n` | detections on the stretch heading in this direction | veh | `detections.json` |
| `s̄` | mean blue→green streak of those detections | m | keypoints |
| `Δt` | blue→green band time offset | s | fitted, step 2 |
| `v̄` | space-mean speed, `s̄ / Δt` (or the counter's, when fitting) | km/h | |
| `T` | residence time, `L / v̄` | min | |
| `V₁₅` | counter volume in a 15-minute bin, this direction | veh / 15 min | `continuousSpeed15min.parquet` |
| `τ` | truck share of the stretch (WSDOT 2024 annual, km-weighted) | — | `segments.geojson` |
| `λ` | the counter's vehicle estimate for the scene, `V₁₅ × T / 15 min` | veh | step 3 |
| `p` | share of vehicles (or `p_τ`: of trucks) the source detects — the first-pass conversion | — | fitted, step 5 |

### 1. The stretch and the direction (`calibrate.py`; `density.py` uses the whole road — see above)

- **Stretch:** sections of the counter's route within 3 km of the site whose AADT is within 25% of the site
  section's — one traffic stream, no major interchange between — counted only inside the footprint (`L ≥ 0.5 km`).
- **Direction:** each detection joins whichever of the counter's two directions (NB/SB or EB/WB) is nearer its
  **compass heading** (true north, from the keypoints; the collector's `bearing_deg` is a math angle and is not
  used — see the CC CHANGELOG, 2026-10-08).

### 2. Speed: fit Δt

A snapshot's mean speed over the vehicles on a stretch is the **space-mean speed** — the same quantity a
counter's **harmonic mean** of spot speeds estimates. So for each pair `Δt_i = s̄_i / v_counter,i`, pooled
detection-weighted across pairs:

```
Δt̂ = Σ nᵢ s̄ᵢ / Σ nᵢ v_counter,i          # s
v̂  = s̄ / Δt̂                              # m/s, per vehicle or per direction
```

### 3. Scene time → the counter's vehicle estimate for the scene

```
T  = L / v̄                                   # minutes the scene covers: the time to cross the stretch
λ  = V₁₅ × (T / 15 min)                      # the 15-min count × the fraction of it the scene covers
   = q × L / v̄ = k_counter × L               # identical to q = k·v (B1), with q = 4·V₁₅ veh/h
```

`λ` is what the counter says the scene should hold; `n` is what the source found in it. `λ` is a **best
estimate at the granularity of the data** — a 15-minute rate spread over a few minutes of scene time — not an
observed count of the vehicles in the image; finer counter data (per-vehicle or 1-minute) would tighten it. Fitting, `v̄` is the
counter's speed (so the comparison does not lean on a Δt being fitted from the same pairs); applying the
conversion where there is no counter, it is the streak speed `v̂`.

The counter side is not a single bin: `V₁₅` is the **overlap-weighted mean of the bins covering a window of
length `max(30 min, T)` centred on the capture** (the two bins either side of the capture's are kept for this),
and a pair is used only if every bin it touches is Accepted and not a glitch. Speeds in the window are combined
as a volume-weighted harmonic mean. With `T` ≈ 1–11 min in the data, the 30-minute floor is what usually binds.

### 4. Save every pair

With the chosen inputs, the pairs are the rows of `density.json` (above). `calibrate.py` keeps its own set for
the Δt fit: each usable capture × counter × direction is one row of `calibration.json` → `pairs`: the stretch (sections,
`L`), `n`, mean heading and streak, the scene time `T`, the counter window and the bins it used, `q`, the
counter's speed, `k_counter`, `τ`, and the per-pair ratio and Δt. Every harvest recomputes all of them from the
current detections and counter data, so a new capture over a counter, or new counter data from WSDOT, adds pairs
without anything being run by hand. A pair enters a fit only if every counter bin it touches is Accepted (not a
glitch) and it holds at least 3 detections.

### 5. Fit the collection

The general conversion is `n = f(λ)` (or `f(τ λ)` against trucks), fitted over all saved pairs. Its form is
open; the first pass is the simplest model with a physical reading.

With random arrivals and a stationary stream over the window, the number of vehicles actually on the stretch is
**Poisson(λ)**. Each one is detected independently with probability `p` (a thinning), so

```
N ~ Poisson(λ)        n | N ~ Binomial(N, p)        ⇒   n ~ Poisson(p λ)
trucks only:                                            n ~ Poisson(p_τ τ λ)
```

The maximum-likelihood share across `m` direction-pairs is the ratio of totals:

```
p̂   = Σ nᵢ / Σ λᵢ
p̂_τ = Σ nᵢ / Σ τᵢ λᵢ
```

**Goodness of fit** is the Pearson dispersion `φ = Σ (nᵢ − p̂ λᵢ)² / (p̂ λᵢ) / (m − 1)`. `φ ≈ 1` means the
scatter between sites is Poisson counting noise and one `p` fits all of them; `φ > 1` means real site-to-site
variation the model does not explain, and the distribution should become a negative binomial (Gamma-Poisson)
with that extra variance.

**What could change the form.** Proportional `f` holds only while the detector's share does not depend on
density. C3–C5 say it will at high density (dedup ceiling, window saturation, stopped traffic leaving no echo), so
as pairs accumulate over a wider range of `λ`, a concave `f` (share falling with density) is the expected next
candidate, chosen by the same dispersion test and by leave-one-site-out error.

### 6. Use the fitted conversion: a capture → vehicles per unit time

With the first-pass (proportional) conversion; a refitted `f` replaces the first line with `λ̂ = f⁻¹(n)`.

```
λ̂    = n / (p̂_τ τ)                        # vehicles in the scene over its time range (all classes)
V̂₁₅  = λ̂ × 15 min / T̂,   T̂ = L / v̂       # the 15-minute count it implies
q̂    = 4 V̂₁₅ = λ̂ v̂ / L                    # veh/h
AADT ≈ q̂ × f_hour × f_dow × f_season      # Part D, factors from the nearest counter
```

Uncertainty comes first from `n` itself — use the exact (Garwood) Poisson interval on `n` and carry it through:

| n | 95% interval on n | relative |
|---|---|---|
| 5 | 1.6 – 11.7 | −68% / +133% |
| 10 | 4.8 – 18.4 | −52% / +84% |
| 23 | 14.6 – 34.5 | −37% / +50% |
| 100 | 81.4 – 121.6 | −19% / +22% |

A single capture of a short stretch cannot beat that floor; more captures, longer stretches and both directions
pooled are how the interval narrows.

### Worked: I-5 NB at Tumwater (R097S), Tacoma-Centralia_01, 2026-04-29 12:41 PT, nojitter-v1

| Step | Value |
|---|---|
| Stretch | 5 sections, `L` = 12.57 km inside the footprint |
| Detections NB | `n` = 23, `s̄` = 33.97 m, mean heading 18° |
| Counter window | 12:26–12:56 (bins 12:15, 12:30, 12:45): `q` = 2,366 veh/h → `V₁₅` = 591.5; `v` = 111.2 km/h |
| Scene time | `T` = 12.57 / 111.2 h = **6.8 min**, 45% of a 15-min bin |
| Vehicle estimate | `λ` = 591.5 × 0.452 = **267 vehicles** in the scene over that time; trucks at `τ` = 15.3%: 40.9 |
| The saved pair | (23 detected, 267 estimated) → 23 / 40.9 = 56% of trucks (all pairs pooled: `p̂_τ` = 0.536) |
| Speed | `v̂` = 33.97 m / 1.171 s = **104 km/h** vs 111 counter (left out: 102, −8%) |
| Flow, this site left out of the fit | **2,318 veh/h** vs 2,366 counted (−2%) |

### Results (calibration.json, 2026-10-08)

68 usable direction-pairs at 14 sites; counter data runs through **2026-05-27**, so no capture after that pairs.

| Source | Δt̂ (s) | per-pair range | `p̂` all | `φ` all | `p̂_τ` trucks | `φ` trucks | left-out \|v err\| | left-out \|q err\| via `τ` |
|---|---|---|---|---|---|---|---|---|
| hand labels | 1.202 | 0.89–1.45 | 0.053 | 1.71 | 0.478 | 1.16 | 10.5% | 40.8% |
| adamiak-v2 | 1.091 | 0.93–1.34 | 0.066 | 1.78 | 0.547 | 0.98 | 9.3% | 32.1% |
| warmup-v1 | 1.198 | 1.02–1.57 | 0.045 | 3.65 | 0.453 | 1.48 | 7.7% | 43.3% |
| nojitter-v1 | 1.171 | 0.97–1.43 | 0.054 | 2.32 | 0.536 | 1.04 | 13.7% | 47.4% |
| nojitter-v1 @ 24 px | 1.209 | 1.11–1.36 | 0.051 | 1.73 | 0.427 | 0.91 | 8.4% | 25.8% |

"Left-out" errors are leave-one-site-out medians: `Δt` and `p` refitted without the site, then the site
predicted — what a road with no counter would see.

**Reading it (first-pass fit; the form is not settled).**
- **Δt is a sensor constant** and every source agrees on it (1.09–1.21 s).
- **As a share of all vehicles, the pairs do not agree** (`φ` 1.7–3.7): the share found depends on the site.
  **As a share of trucks, they do** (`φ` 0.9–1.2 for every source but warmup-v1): sources find **about half the
  trucks**, and what is left between pairs is counting noise. So far the detector counts trucks, and `τ` is what
  carries a truck count to total traffic.
- **The flow error is the Poisson floor**, not the model: pairs hold 3–23 detections, and the table in step 6 puts
  a single pair at ±40–100% before any modelling error. More pairs — and pairs over a wider range of density —
  are what settle the form of `f`.

### Assumptions to keep in view

- Counter speed-bin edges are **assumed** (WSDOT has not confirmed them; `counters.py` `EDGES`), and the
  harmonic mean is sensitive to the open-ended bottom bin in congestion.
- `τ` is **annual**, not midday; `p_τ` absorbs the difference only while it is similar between sites.
- Stationarity over the window and no on/off-ramp exchange inside the stretch (the AADT tolerance is the guard).
- `p` is fitted over counter densities of **3.7–48 veh/km per direction**. It does not transfer outside that
  range, because C3–C5 are nonlinear: the current operating point (nojitter-v1: dedup 8 px = 24 m, all
  detections per window) lowers but does not remove the density ceiling, and congested traffic leaves no echo.

---

# Part C — the bias ledger

Every term below corrupts a count. Sign and magnitude given where known. **The nonlinear ones are the
dangerous ones**, because a calibration fitted at one density will not transfer to another.

## C1. Precision under-measured — count biased LOW ✗

Labels mark **only clear, well-formed echoes** by policy; blurry, malformed, over-bright and very small ones are
deliberately skipped. A real-but-unlabelled vehicle scores as a false positive. So measured precision is a
**lower bound**, `precision/recall` is too small, and `N_corr` under-counts. Confirmed visually on jitter-mv:
most of its "false positives" were real echoes.

## C2. Recall over-measured — count biased HIGH ✗

Recall is measured against that same easy subset. Recall on *all* real vehicles — including the faint, small and
malformed ones never labelled — is necessarily **lower** than the reported figure.

**C1 and C2 push in opposite directions and neither is quantified.** This is the single strongest argument for
densely labelling one scene: it collapses both unknowns at once. Without it, the correction factor
`precision/recall` has unknown sign of error, which means the count model's calibration constant absorbs an
unknown bias.

## C3. Dedup radius — a hard density ceiling, NONLINEAR ✗✗

Suppression at **96 m** means two detections cannot coexist closer than that, in any direction:

> **k_max = 1 / 0.096 km = 10.4 veh/km**, across the entire cross-section

It is isotropic in 2D, so opposing carriageways of a divided highway (median ≪ 96 m) share the ceiling — both
directions together cannot exceed 10.4 veh/km. At 100 km/h that caps observable flow at **≈ 1,040 veh/h for the
whole corridor**. Trucks at a 2 s headway sit ~55 m apart and are **routinely suppressed**.

TC_01's labelled density is ~3.6 veh/km, about 35 % of the ceiling on average — but traffic is not uniform, and
platoons hit it locally. This mechanism is the quantitative explanation for the failure already recorded in the
registry: *"misses concentrated in DENSE traffic (dedup merges neighbours)."*

## C4. Top-1 per window — compounds C3, NONLINEAR ✗✗

`detect_scene.py:135` keeps only `scores[0]` per window. A 192 m window containing three trucks yields **one**
detection. Windows overlap (stride 120 m < window 192 m) so some redundancy exists, but the effect is the same
direction: **recall falls as density rises**, so counts **saturate**.

Together C3 and C4 mean the pipeline has a **density ceiling**, and any calibration fitted on low-density scenes
will over-predict on high-density ones. **Both should be fixed before fitting a calibration constant** — they
are inference-side changes requiring no retraining.

## C5. Only moving vehicles echo — NONLINEAR ✗✗

A stationary vehicle produces no colour separation and is invisible. Congested and queued traffic therefore
vanishes precisely when density is highest and flow is lowest. The `q = k·v` relation is still formally correct
(v → 0), but the *measurement* of k collapses too, so the estimator fails in exactly the regime where flow
departs from free-flow.

Practical consequence: **restrict the count model to free-flow conditions**, and detect/exclude congested scenes
rather than letting them enter the fit.

## C6. Per-scene stretch — threshold is not an operating point ✗

From A2: the 2–98 % stretch is per scene, so confidence distributions are scene-relative. A fixed 0.3 (or 0.5,
or 0.55) is a different operating point in each scene. For counting, calibrate the threshold per scene — or
adopt a stretch-invariant score.

Also note three inconsistent thresholds currently in play: metrics computed at **0.3**, console renders at
**0.5** (`server.py:162`, `App.tsx:679`), measured optimum **0.55**.

## C7. Class mixing ✗

Labels bias toward large, fast, well-formed echoes — i.e. trucks — and the confidence score therefore doubles as
a crude size/quality proxy. But nothing enforces truck-only. For a **truck** AADT you need the geometry filter
(streak length, keypoint spacing, collinearity, B→R→G ordering) to separate trucks from cars; for **total**
vehicle AADT you need the opposite — to recover the faint car echoes currently discarded.

Decide which deliverable you are building **before** calibrating; they need different operating points.

---

# Part D — instant → AADT

Frame a scene as a **short-duration count of zero duration**, and use the expansion machinery WSDOT already
runs for short counts (FHWA Traffic Monitoring Guide):

```
AADT ≈ q_observed × f_hour × f_dow × f_season × f_axle
```

| Factor | Corrects for | Source |
|---|---|---|
| `f_hour` | capture is ~10:30 local (sun-synchronous) — always the same time of day | nearest permanent counter's hourly profile |
| `f_dow` | day of week of the capture | same counter, day-of-week profile |
| `f_season` | month of capture | same counter, monthly profile |
| `f_axle` | only if converting axle counts ↔ vehicle counts | counter class scheme |

**Two structural constraints to state in the write-up:**

1. **Every capture is at the same solar time.** A sun-synchronous orbit means you never sample the night, the
   AM peak, or the PM peak. `f_hour` is doing enormous work and is entirely borrowed from ground data — the
   satellite contributes no independent information about the diurnal curve.
2. **All 21 scenes fall in Apr 9 – Jun 22**, a 72-day spring window. `f_season` is therefore untested outside
   spring, and the corpus cannot validate it.

Neither is fatal — this is the normal situation for short-duration counts — but both belong in the limitations
section, and both argue for anchoring on counter-derived factors rather than fitting them from imagery.

---

# Part E — validation design

149 WSDOT permanent counters exist and the AOIs were deliberately placed near them. **Validate in two separate
steps, never one**, or you confound detection error with expansion error:

**Step 1 — detection + density→flow.** For each scene, find counters inside the valid footprint. Pull the
counter's **raw volume in the hour of capture** (not its AADT). Compare to satellite `q`. This tests A6, B2, B3,
B4 and the whole of Part C, with zero expansion assumptions. *Built as B5, at 15-minute rather than hourly
resolution, with leave-one-site-out validation.*

**Step 2 — expansion.** Take the Step 1 output and apply Part D. Compare to the counter's published AADT. This
tests only the factors.

The capture timestamp needed for Step 1 is in the per-scene `*_metadata.json` sidecars — currently in the parent
repo, not here. **Recover those first**; without a capture time, Step 1 is not possible and only the much
weaker AADT-vs-AADT comparison remains.

**What "success" should mean.** Per `PROJECT_STATE.md`, the deliverable is *aggregate counts that are stably
proportional to reality* — a **stable ratio**, statistically calibratable per corridor. Not per-vehicle
accuracy. So the metric that matters is the **variance of (satellite q / counter q) across scenes**, not its
mean: a consistent 0.4× is useful and calibratable; a ratio wandering between 0.2× and 0.9× is not, regardless
of whether it averages to 1.

---

# Part F — what must be built

Ordered by whether the count model can proceed without it.

**Blocking:**
1. **Road-centerline intersection → `L_obs` per scene, per direction.** Without it there is no density.
2. **Capture timestamps** from the metadata sidecars. Without them there is no Step 1 validation.

**Blocking a *trustworthy* calibration (all inference-side, no retraining):**
3. **Lift top-1-per-window** (C4) and **shrink the dedup radius** from 96 m to ~15–20 m (C3). These remove the
   nonlinear density ceiling. Fitting a calibration before this bakes the saturation into the constant.
4. **Per-scene threshold calibration** (C6).

**Needed for a defensible number:**
5. **Densely label one scene** (C1, C2) — collapses both correction-factor unknowns.
6. **Geometry filter** (C7) — decides trucks vs all vehicles, which the deliverable requires.

**Upgrades:**
7. **Per-scene Δt** → measured speed instead of posted limits (B4).
8. **Free-flow gating** (C5).

---

_Sources: `src/export_coco.py`, `src/train_detector.py`, `src/model_registry.py`, `src/detect_scene.py`,
`models/registry.json`, `docs/CONTEXT.md`, `docs/PROJECT_STATE.md`. Scene geometry measured 2026-07-29 over all
21 labelled scenes._
