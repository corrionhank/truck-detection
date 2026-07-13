# Truck Detection — Approaches & Models

_Companion to [CONTEXT.md](CONTEXT.md) (project seed/status/data). Last updated: 2026-06-24._

**Purpose** — catalog the candidate approaches and models for detecting **trucks** (and other movers) in
SuperDove imagery via the moving-echo signal, with enough detail to replicate, plus the methodology debates
that should shape which one you trust. CONTEXT.md is the *what/why*; this is the *how* of the modeling.

> **Focus (2026-06-24):** Detect **trucks** — the freight-relevant and most-detectable movers (a truck's
> 2–6 px echo is a bigger signal than a ~1 px car) — in single-capture SuperDove imagery via their
> **moving echoes** (colour-separated B→R→G streaks). The **immediate gating step is the signal-visibility
> check (§1)**: confirm the echo is even visible in the SR product at 3 m before investing in annotation +
> training. Then replicate the **Adamiak Keypoint R-CNN (§2)**. Road segmentation — a brief earlier detour —
> is **demoted to an optional corridor-masking helper (Appendix A)**.

---

## 0. The problem (one task, one gate)

**Task.** Per-truck detection in single-capture SuperDove SR imagery over freight corridors — at minimum a
point/box + count; ideally **3 keypoints (B, R, G)** per truck to recover heading + speed from the inter-band
displacement. Trucks are the target class; the echo detects any mover, but trucks leave the largest streak.

**The gate.** Everything downstream depends on one unanswered question: **is the moving echo actually visible
in our SR product at 3 m?** Do not trust any null detection result, and do not scale annotation, until §1 is
settled.

Corridor location is essentially free here — the 15 clips are already masked to the road (nodata outside the
OSM buffer), so "where to look" is solved; the work is "what's moving." (Optional refinement: Appendix A.)

---

## 1. Signal visibility & SR-vs-basic (THE GATING STEP — do this first)

The negative morphology baseline has been over-read as "SR co-registration kills the echo." Before spending
a scarce quota order on the basic product, weigh the cheaper, more likely explanations.

**Evidence the echo survives correction.** Adamiak did **not** use a raw product — they used
**Ortho-Analytic (3B)**, which is radiometrically corrected and DEM-orthorectified, and their Figure 2
shows strong, obvious rainbows. The physics holds: orthorectification aligns the *static ground* to map
coordinates, but a moving vehicle physically occupied a *different ground position* when each band fired, so
it stays displaced regardless of registration quality. **SR adds atmospheric correction, which is
radiometric, not geometric** — it shouldn't move bands relative to each other. The *only* way SR damps the
echo is if it performs explicit **inter-band co-registration** (aligning bands to each other). Worth
testing — but a hypothesis, not the likely first cause.

**Two cheaper confounds to rule out first:**
1. **Scene choice.** The baseline ran on **Bellingham I-5**, a thin lower-traffic corridor on one date — it
   may genuinely contain ~zero moving trucks. A quiet scene yields a null with nothing wrong with the method.
2. **Wrong detector.** White top-hat finds bright objects → the ~65 near-zero-offset candidates are almost
   certainly lane markings/structures. That tests whether *bright things move* (they don't; paint is static),
   not whether *trucks echo*.

**Free checks on data already in hand (do these before ordering anything):**
- Re-run on the **highest-traffic scene** (Seattle, or a busy Stanwood date) — not Bellingham.
- **Swap the detector** to a **Red−Blue band difference** along the lanes and look for the streak directly.

**Decision rule:** if a high-traffic scene with proper band-differencing *still* shows nothing, the SR
co-registration hypothesis gets real weight → justify a **basic-product order when quota resets**. Queue it
as the *second* test, not the first. Until then: **conclude nothing about SR.**

---

## 2. Truck / moving-echo detection models

### 2.1 Adamiak et al. 2025 — Keypoint R-CNN (PRIMARY TARGET)

Full replication detail. Source: Adamiak et al. 2025, *Int. J. Applied Earth Obs. Geoinformation*
142:104707. This is the method to replicate **once §1 confirms the echo is visible** in our product.

**Architecture (paper §4.2)**
- **Model:** Keypoint R-CNN (built on Mask R-CNN).
- **Backbone:** ResNet-50 + FPN.
- **Init:** random weights, **trained from scratch** (no pretrained weights) — they had 3,236 echoes.
- **Classes:** 2 — "moving echoes" + background.
- **Keypoints:** **3 per vehicle**, linked in temporal order **Blue → Red → Green**.
- **Selection rationale:** chose Keypoint R-CNN over alternatives (e.g. CenterNet) purely for
  **accessibility/reproducibility** — it's curated in the official PyTorch/torchvision repo. Not a
  performance claim.

**RPN / anchors — the part they actually tuned (most important for tiny vehicles)**
- Custom anchor generator; entire backbone optimized during training.
- **Anchor sizes tested:** 4, 8, 16, 32, 48 px.
- **Aspect ratios tested:** 0.25, 0.5, 0.75, 1.0, 1.25.
- Small anchors are what make 1–2 px objects detectable at medium resolution — the single most important
  knob for our case.

**Optimizer / schedule / loss (§4.2)**
- Optimizer **Adam**; scheduler **ReduceLROnPlateau** (on val loss); LR **1e-3 → 1e-5**; **grad clip 1.5**.
- Composite loss = classification + bbox regression + keypoint placement + objectness + RPN box regression.
  Built into torchvision's model — returned as a loss dict; you don't assemble it.

**Training setup (§4.2.2)**
- **PyTorch Lightning**; inputs resized to **512×512**.
- Augmentation: random rotations, H/V flips, brightness, perspective.
- Hardware: NVIDIA 3090Ti / 1080 / A4000; ~**2.5 h** per model.
- **Data split 80/10/10** → 502 train / 77 val / 147 test images, from **3,236 echoes in 726 clipped images**.

**Hyperparameter process:** ~**120 runs**, mostly sweeping the **anchor generator**, scored by **RMSE on a
fixed val subset**; pick the best anchor set, **lock it**, then run all cross-validation folds. → *Sweep
anchors first, lock the winner, then train for real.*

**Keypoint correction (§4.3, Algorithm 1)** — applied twice: to clean training labels before training, and
to refine predictions after inference:
1. Per band, find local intensity peaks with **scikit-image `h_maxima`** (peak must exceed neighbors by
   **h = 0.02** over a 3×3 ≈ 11.1 m neighborhood).
2. Build a **k-d tree** of peak coords.
3. **Snap** each keypoint to the nearest peak within **2 cells (~7.4 m)**, else keep the original.

**Evaluation (§4.2.3)**
- Primary: **mAP** over OKS thresholds 0.5–0.95 (step 0.05), using a **customized OKS** whose per-keypoint
  constant reflects the pixel distance among the 3 connected keypoints (handles the larger gaps fast
  vehicles produce); scale = 1, no visibility param. TP requires keypoint score **> 0.7**.
- Secondary: **RMSE on trajectory length** (≈ speed proxy).

**torchvision replication skeleton**
```python
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator

# 2 classes (bg + moving echo), 3 keypoints/object
model = keypointrcnn_resnet50_fpn(weights=None, num_classes=2, num_keypoints=3)

# Adamiak's tuned anchors — small sizes are the key for tiny vehicles
anchor_sizes  = ((4,), (8,), (16,), (32,), (48,))
aspect_ratios = ((0.25, 0.5, 0.75, 1.0, 1.25),) * len(anchor_sizes)
model.rpn.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
# then: Adam, ReduceLROnPlateau (1e-3→1e-5), grad clip 1.5, 512×512 inputs, aug list above.
```

**One deliberate deviation for us:** Adamiak trained from scratch on 3,236 echoes. With far fewer labels,
**start with `weights="DEFAULT"` (COCO-pretrained Keypoint R-CNN) and finetune** — a pretrained backbone
learns from small data far better. Switch to from-scratch only later to match them exactly. Replication
order: (1) get it training at all on pretrained weights, (2) sweep anchor sizes, (3) lock best, train real.

### 2.2 Baselines & alternatives (truck detection)

- **Morphology — white top-hat (tried; misleading null).** Found ~65 bright candidates on Bellingham SR
  with near-zero inter-band offset. **This does not test the method** — top-hat finds *bright* blobs (lane
  paint, structures), but the echo signal is *colour-separated displacement*, not brightness. It measured
  static clutter and correctly found it static. See §1.
- **Red−Blue band-difference offset view (signal visualization, not a model).** Subtract the Blue band from
  the Red band along the lanes and look for the displaced streak directly — this keys on the *actual* signal
  and is the right way to confirm echoes exist before annotating or modeling. **This is the §1 gating check —
  build it as a QA tool first.**
- **YOLO for small-object truck detection (alt to keypoints).** Possible for plain detection/counting, but
  it doesn't natively recover the B/R/G keypoints needed for velocity. Consider only if the goal collapses
  to counts, not speed.

---

## 3. Training-data pipeline (annotation → COCO)

For the keypoint model. QGIS step-by-step lives in [CONTEXT.md](CONTEXT.md) §5; the model-relevant essentials:

- **Annotate on the GeoTIFF, never the PNG.** PNGs have no CRS — keypoints would land in arbitrary image
  space, disconnected from the ground, road layers won't overlay, and you can't cleanly convert to COCO
  pixel coords. PNGs are also 8-bit display data. Load the `*_SR_8b_clip.tif`, style R6/G4/B2 + stretch to
  get the same true-colour view *with* georeferencing.
- **Point/keypoint layer (GeoPackage):** geometry **Point**; fields **`vehicle_id`** (groups the 3 points of
  one truck) and **`sequence`/`band`** (1/2/3 = Blue/Red/Green capture order). That grouping + order is
  exactly what the COCO keypoint converter consumes.
- **Match CRS = EPSG:32610** for project + layer + rasters. CRS mismatch is the most common silent breakage.
- **Map coords → pixel coords** for COCO via the raster's **affine transform** when chipping to 512.
- **One class** for echoes ("moving echo"); digitize the **streak**, not a single dot.
- **Caveat:** at 3 m, true-colour barely shows echoes. If a styled scene looks like flat grey road, that's
  the §1 signal-visibility problem — use the Red−Blue view to find what to annotate, not more QGIS fiddling.

---

## 4. Data note — UDM2 masks (are they needed?)

**Short answer: not for current work.** UDM2 is Planet's per-pixel **Usable Data Mask** (clear / snow /
shadow / haze-light / haze-heavy / cloud / confidence + an unusable-data band). Its value is **automated
quality/cloud filtering at scale**. For this project's near-term work — **manual labeling on a handful of
hand-picked, visibly clear scenes** — you can eyeball cloud/haze, so UDM2 adds little. That's why the masks
were deleted in the cleanup. They **re-ship with any future Planet order**, so if you later automate QA
(e.g. drop cloudy pixels before batch inference, or weight training chips by usability) you can recover them
for free at order time. No action needed now.

---

## Appendix A. (Optional, demoted) Corridor masking — road segmentation

> **Demoted 2026-06-24.** Road segmentation was a brief warm-up milestone; the project refocused on truck
> detection. It's kept here because a road mask can *optionally* tighten "where to look" — but it's **largely
> redundant**: the 15 clips are already corridor-masked (Seattle is only ~5.7% valid pixels), so the nodata
> footprint already restricts search to the road. `src/make_road_mask.py` builds an OSM-derived mask if needed.

**If you do want a learned road mask** (e.g. for tighter pavement extent, or on future unclipped scenes):
- **Model:** U-Net + ResNet34 (ImageNet) via `segmentation-models-pytorch`; binary semantic seg (road vs
  not-road), 3-channel R6/G4/B2 input. Chosen over YOLO11-seg (instance seg, rough on thin roads). Fallback:
  DeepLabv3+ (same library).
- **Recipe:** Dice+BCE loss, Adam 1e-3, ReduceLROnPlateau, 512/640 chips, albumentations (flip/rotate/slight
  brightness), pretrained encoder + finetune (not from scratch), eval mIoU/Dice.
- **Label precision ceiling:** a fixed-width OSM buffer mask only teaches the model that width, not true
  pavement edges. For real extent, use OSM road polygons or hand-correct buffer masks in QGIS.
- **Pipeline:** OSM lines (osmnx/QuickOSM) → buffer → `rasterio.features.rasterize` onto the scene grid via
  the affine transform → mask; **intersect with the valid footprint** so nodata isn't labelled road; tile →
  train. (Steps 1–2 are implemented and verified in `src/make_road_mask.py`.)

---

## 5. References & decisions log
- **Adamiak et al. 2025**, IJAEOG 142:104707 — moving-echo keypoint detection, SuperDove velocity. Used
  **Ortho-Analytic 3B** (corrected+orthorectified), not raw.
- **Origin tool:** `wsu-trg-satellite-freight-feasibility` (local parent repo — AOI building, Planet
  ordering, viewing).
- **2026-06-24 decision:** **refocused the project on truck detection**; the moving-echo / Adamiak
  Keypoint R-CNN track is the primary target, gated on the §1 signal-visibility check; road segmentation
  demoted to optional corridor masking (Appendix A).
- **2026-06-22 decisions (superseded):** road segmentation via U-Net+ResNet34 chosen over YOLO11-seg as a
  first milestone; Adamiak deferred; UDM2 masks dropped; SR-vs-basic treated as the *second* test after free
  scene/detector checks.
