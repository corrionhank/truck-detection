# TODO

Open items for the **truck-detection** project. Companion to [CONTEXT.md](CONTEXT.md) (project/data) and
[APPROACHES.md](APPROACHES.md) (models). `[ ]` = open, `[x]` = done. Priority: **P0** blocks progress,
**P1** important, **P2** nice-to-have.

---

## 1. Truck-detection build (CURRENT FOCUS)

Detect trucks via their moving echoes. See [APPROACHES.md](APPROACHES.md) §1–§2. Scripts in `src/`.
- [x] Read scene + true-colour (R6/G4/B2) render (`src/inspect_scene.py`).
- [x] **P0** **Signal-visibility check** — RESOLVED. 47 keypoints hand-labeled across Bellingham / Centralia /
      Stanwood **SR** scenes show clear B→R→G streaks, and a Keypoint R-CNN trained on them detects real echoes.
      **SR does not kill the signal**; the earlier Bellingham morphology null was a brightness detector finding
      static clutter, not the absence of movers. (The Red−Blue band-difference view was never needed.)
- [ ] **P1** Confirm on more scenes / a rigorous sample, and (optional) still queue a **basic-product order**
      when quota resets to quantify SR-vs-basic echo strength (ties to §2d).
- [x] **P2** QGIS keypoint annotation → COCO export — DONE. 47 keypoints (16 vehicles, 3 scenes) in
      `Annotations-RGB.gpkg`; exporter is `src/export_coco.py`. Schema in [DATA.md](DATA.md).
- [x] **P2** Keypoint R-CNN replication ([APPROACHES.md](APPROACHES.md) §2.1) — DONE. Fine-tuned from
      COCO-pretrained (`src/train_keypoint_rcnn.py`); jitter-augmented v1 fixes overfit-to-center
      (`src/train_keypoint_rcnn_jitter.py`). Results + reproduce in [MODEL.md](MODEL.md). **Train on CPU** —
      MPS produces NaN losses with torchvision detection models. (Anchor sweep not yet done.)

---

## 2. Imagery & sensor characterization (PRIORITY)

Understand exactly what the Planet product is — several of these gate the velocity method and the SR-vs-basic
decision. Most answers live in **Planet docs** (Imagery Product Spec PDF, SuperDove/PSB.SD tech specs) and
**Adamiak et al. 2025**, *not* in the per-scene metadata.

### 2a. Bands & spectral
- [ ] **P1** Exact center wavelengths + bandwidths (FWHM) for all 8 PSB.SD bands (confirm vs CONTEXT §4).
- [ ] **P1** Full **temporal capture order** across all 8 bands (CONTEXT records B→R→G for the 3 used —
      get the complete focal-plane readout sequence, not band-number order).
- [ ] **P1** Which bands are actually **usable** for truck/echo detection: true-colour trio (6/4/2); do
      red-edge (7) / NIR (8) / yellow (5) / coastal-blue (1) add echo signal?
- [ ] **P2** SR **reflectance scaling** (e.g. DN ÷ 10000 = reflectance?), valid range, nodata = 0 convention.
- [ ] **P2** Is this product **Sentinel-2 harmonized** SR or standard SR? (affects radiometry/comparability.)

### 2b. Timing & geometry — the velocity constants (needed for absolute speed)
- [ ] **P0** **Exact inter-band time offset** (focal-plane readout delay between bands). THE constant for
      converting pixel displacement → speed. Not in scene metadata; from Planet docs / Adamiak. ← highest value
- [ ] **P1** Per-band **exposure / integration time**, and total scene acquisition duration.
- [ ] **P1** Satellite **ground-track speed** and **distance travelled between band exposures**.
- [ ] **P1** **Focal-plane layout**: which detector rows host which bands, the physical reason for the
      offset, and the resulting inter-band parallax geometry.
- [ ] **P1** Assemble the conversion: pixel displacement → metres → m/s (combine dt + GSD + geometry).
- [ ] **P2** Reconcile **3.4 m GSD vs 3.0 m/px** grid — what resampling produced the 3.0 m product grid?

### 2c. Sensor fleet consistency
- [ ] **P1** Are all SuperDove (PSB.SD) satellites **identical** in band layout & inter-band timing, or do
      these constants vary per bird?
- [ ] **P1** Map each of our 15 scenes → **satellite ID** (in metadata) → confirm the timing constant applies
      uniformly across the set.

### 2d. Product processing — the SR-vs-basic crux (gates the echo method)
See [APPROACHES.md](APPROACHES.md) §1 and the `sr-signal-reframing` note.
- [ ] **P0** Does SR / ortho processing perform explicit **inter-band co-registration** (band-to-band
      alignment)? This is the single thing that would damp the moving-echo signal. ← critical
- [ ] **P1** Document the processing chain raw → `basic_analytic` → `ortho_analytic` → `ortho_analytic_SR`:
      what each step corrects (geometric vs radiometric). _Confirmed from tags: SR = 6Sv2.1 atmospheric
      (radiometric) correction, sr_version 2.1 — consistent with "SR is radiometric"; the geometric/ortho
      steps and any band-to-band co-registration still need Planet docs._
- [ ] **P1** Documented **band-to-band registration accuracy** (sub-pixel?) and **geolocation accuracy**
      (CE90/RMSE) of the SR product.
- [ ] **P1** Confirm the **basic (L1B, uncorrected) product** is orderable on the EDU plan + its asset name,
      for the eventual comparison order (after quota resets).
- [ ] **P2** Clarify **Adamiak's exact product** ("Ortho-Analytic 3B") vs ours (`3B_AnalyticMS_SR_8b`):
      does their use of a non-SR ortho vs our atmospherically-corrected SR matter for the echo?

### 2e. Per-scene metadata
**Already embedded in each GeoTIFF** (confirmed 2026-06-24 via `rasterio` `src.tags()`): acquisition
`TIFFTAG_DATETIME` (Seattle = `2026:05:02 19:43:43`), **solar zenith/azimuth** (32.3° / 169.7° for Seattle),
the full **atmospheric-correction** block (algo 6Sv2.1, `sr_version` 2.1, AOT/ozone/water-vapor), and band
names/order. So sun angles + acquisition time + SR params do **not** need the sidecars.
- [ ] **P1** Still need the **sidecars** (`*_metadata.json` / `*_metadata_clip.xml`) from the **parent repo**
      (`wsu-trg-satellite-freight-feasibility/imagery/<AOI>/<order>/PSScene/`) for **satellite_id** and a
      **real off-nadir / view angle** — the GeoTIFF's satellite zenith/azimuth read `0` (placeholder, unusable).
- [ ] **P2** Confirm no sidecar field carries inter-band timing (expected absent).
- [ ] **P2** Per scene, once recovered: off-nadir/view angle (parallax of elevated & moving objects); sun
      angles are already in the tags (shadow streaks can confound echoes).

---

## 3. Data & ops
- [x] Git initialized and pushed to **github.com/corrionhank/truck-detection** (data/imagery/weights gitignored).
- [ ] **P2** Pin environment: `requirements.txt` (rasterio, geopandas, shapely, osmnx, torch, torchvision).
- [ ] **P1** Track Planet **quota reset date**; plan the basic-product comparison order for then.

---

## Appendix A. (Demoted) Road-segmentation pipeline

Optional corridor-masking helper, not the focus — see [APPROACHES.md](APPROACHES.md) Appendix A. Largely
redundant since the clips are already corridor-masked.
- [x] OSM lines → buffer → rasterize aligned mask + QA overlay (`src/make_road_mask.py`, Seattle, verified).
- [ ] **P2** If ever needed: batch masks for all 15 scenes; tune buffer / use OSM polygons; tile; train U-Net.

---

## Sources to consult
- **Planet docs:** Imagery Product Specification (PDF), SuperDove / PSB.SD technical specs, SR &
  Sentinel-2-harmonization docs, UDM2 spec, Orders/Data API asset names.
- **Adamiak et al. 2025**, IJAEOG 142:104707 — method + likely source of the inter-band dt constant.
- **Per-scene metadata** — in the parent repo (deleted from this one); see §2e.
- **GeoTIFF embedded tags** — inspect with `rasterio` `src.tags()` / `src.tags(bidx)` for anything useful.
