# Project State — truck detection

The durable "where things stand" reference. Consolidates the prior working docs (operational companion,
best-result breakdown, cross-scene plan). **The active next-model plan lives in
[NEXT_MODEL_PLAN.md](NEXT_MODEL_PLAN.md);** this doc is the stable context around it. Updated 2026-07-26.

---

## 1. What this is
WSDOT feasibility study: estimate **freight-truck counts** on WA corridors from PlanetScope SuperDove (~3 m).
Signal = the **moving echo** — bands captured a fraction of a second apart smear a moving vehicle into a
**blue→red→green** streak. Model = Keypoint R-CNN (`keypointrcnn_resnet50_fpn`), 2 classes, 3 keypoints, finetuned
from the COCO backbone. Deliverable = **aggregate counts that are stably proportional to reality** (statistically
calibratable), not perfect per-vehicle detection.

## 2. Current data & model state
- **Data on disk: 629 vehicles / 19 labeled scenes** (import done + gpkg committed `94128aa`). All captures fall
  in one spring window (**Apr 9 – Jun 20**) — a clutter-diversity limitation for the write-up.
- **Active model = `kprcnn-adamiak-all`, which is BROKEN** (§5). The console is currently serving the worst model
  in the registry; consider pointing `active` at `adamiak-v2` until the fresh model is measured.
- **Typical scores:** centered-chip recall ~0.97 (recognise a handed echo) vs full-scene F1 ~0.47 (sliding-window,
  the deployable number, thr 0.3). The gap is the translation-tolerance story.
- **Benchmark:** full-scene F1 ~0.50 ≈ Van Etten 2024 (0.49) and Adamiak mAP (~0.53). At 3 m a vehicle is 1–3 px —
  ~0.5 is respectable, not weak.

## 3. The diagnosis — why cross-scene fails
Failure splits by corridor, and the axis is **background clutter, not terrain type** (early "forested vs arid" read
was wrong — forested auburn/Bellingham work; agricultural Yakima/Ellensburg flood false positives on field texture):

| corridor | mean F1 | failure mode |
|---|---|---|
| auburn-snoqualmie | ~0.63 | works |
| Centralia / Tacoma-Centralia | ~0.54 | works |
| blaine-bellingham | ~0.33 | recall, misses trucks |
| Yakima | ~0.18 | precision, FP flood |
| Ellensburg | ~0.11 | both collapse |

Two contributing causes, not separated: **coverage** (failing corridors have thin training data) and **clutter**
(agricultural backgrounds mimic echoes). A gated 10-min look at *where* Yakima's FPs land (off-road → road masking
leads; on/near road → hard negatives + geometry filters lead) decides the clutter fix.

**Spatial-overlap finding (measured, footprint IoU not bounds):** every multi-scene corridor is the **same road
re-captured on different dates** (Yakima 99–100%, TC_01↔_02 97.5%, auburn 59%; Centralia_01↔_02 only 21%, same day;
`polygon_01` 0% vs everything). So name-based leave-one-scene-out never produced spatial hold-outs — but the
overlapping scores sit *at or below* the clean ones, so it's **mislabeling, not inflation.** Two generalizations:
- **temporal** = later captures of a *known* corridor → the WSDOT per-corridor-calibration deployment (not a leak);
- **spatial** = transfer to an *unlabeled* corridor → needs the whole corridor held out.
Report both, labeled. Multi-date same-footprint data is valuable (new echoes, seasonal variation) — it just doesn't
add spatial *diversity* (N dates of one place ≈ 1 place for transfer).

## 4. Pipeline & repo state (from a full code read)
**The pipeline is clean — no hidden bug.** Train and inference reproduce each other exactly: bands 6/4/2, per-scene
2–98 % stretch, /255, 64 px chips; deployed models load their custom anchors via `model_registry` (not torchvision
defaults). Earlier metrics were not spoofed by a train/inference mismatch.

| area | state | detail |
|---|---|---|
| train/inference match | CLEAN | bands, normalization, scaling, chip size identical |
| deployed anchors | OK | custom `[4,8,16,32,48]` load via registry; only the raw `detect_scene` CLI can fall back to wrong defaults |
| top-1 per window | BUG | `detect_scene.py:135` keeps only `scores[0]`/`keypoints[0]` → recall loss in dense areas |
| console vs metric threshold | MISMATCH | console/backend default **0.5** (`server.py:162`, `App.tsx:679`); all F1 computed at **0.3** |
| postprocessing | THIN | only `score≥thresh` + a 32 px (~96 m) red-keypoint dedup; no geometry/bearing/speed checks |
| road masking | UNWIRED | `make_road_mask.py` exists, imported by nothing; needs live OSM |
| translation jitter | ABSENT | `train_detector.py:76`; chips are vehicle-centered → over-learns echo-at-center |

**Anchor caveat:** chips upscale 64→192 (~3×) before detection, so anchors act in 192-space (a 4–8 px streak ≈
12–24 px, matched by the *mid* anchors). The upscale already does size-matching → lower sweep urgency; any sweep must
reason in 192-space. On auburn, 8–128 beat 4–48 (0.71 vs 0.61), so the inherited set isn't proven optimal.

**Where things run:** all local, `src/train_detector.py` on **CPU** (MPS diverges → falls back; CUDA is the real
unlock, see `HARDWARE.md`), reads `data/active/coco/`, writes `weights/<id>.pt` + `models/registry.json`. Per-epoch
checkpoint/resume; background runs get killed ~27 min and resume.

## 5. Model lineage (registry, 8 models)
| model | anchors | aug | lesson |
|---|---|---|---|
| `echo-v0` | default | none, centered | center-overfit: **0/3** full-scene |
| `echo-jitter` | default | **+jitter** | jitter fixes it: **3/3**, conf 0.4→0.9 |
| `base-default` | default | none | zero-effort floor |
| `centralia-heldout` | **small 8–128** | **rich: rotate+scale+photo+jitter+flip** | **best generalizer, 0.71 clean on unseen I-90** |
| `adamiak-v1/v2` | 4–48 | no jitter | v2 = best *measured* all-rounder (~0.47) |
| `kprcnn-3` | 8–128 | no jitter | anchor experiment, poor (0.10) |
| `adamiak-all` | 4–48 | no jitter | **ACTIVE + BROKEN: 0.06 on its own trained scenes** |

**Why `adamiak-all` is broken:** trained on all scenes with **no held-out set**, so the LR-scheduler's val subset
overlapped training → val loss never plateaued → LR never annealed → under-converged (under-fires dense, over-fires
clutter). The lesson driving the next run: **always `--held`**, and the best generalizer used the archived recipe
**with jitter+scale** — which the current rebuild dropped.

## 6. Best result (the clean proof-of-concept)
`centralia-heldout` → `auburn-snoqualmie_03` (**unseen, cross-corridor**): **F1 0.706**, P 0.70 / R 0.71, 43 detected
vs 42 real. Highest leakage-free full-scene F1 on record; from the archived jitter+scale recipe + small anchors.
Caveat: the near-perfect count is error cancellation (~30 TP + 13 FP ≈ 42), not 42 perfect hits.

## 7. Correctness landmines
- Join labels↔imagery by the **`scene` text field, never spatial extent**.
- **Any CRS, training or inference** — the join needs each scene's points to agree with *its own* raster, not one
  zone project-wide (WA spans UTM 10/11). Never resample/reproject a **raster** (smears the echo); reproject
  **points** freely (exact). Watch for a **mis-stamped** CRS — right coordinates, wrong label — which passes a
  CRS-equality check; `import_data.align_to_imagery` tests it functionally and re-stamps losslessly.
- Only `data/active/Annotations-RGB.gpkg` is git-tracked; imagery/weights/coco/outputs gitignored.
- Anchors must match at load (registry handles it).
- Name-based split leakage guard **misses spatial overlap** (§3) — a real spatial guard is needed.
- Don't conflate centered-chip recall (~0.97) with full-scene F1 (~0.47).

## 8. Refinement landscape (deferred levers)
The active pre-training plan is [NEXT_MODEL_PLAN.md](NEXT_MODEL_PLAN.md) (jitter + the split). Everything else is
parked, by stage:

**Post-inference (no retrain, do *after* the next model scores, one at a time, per-corridor):** lift top-1-per-window
· threshold calibration + fix 0.5/0.3 · geometry/physics filters (collinearity, B-R-G spacing, color order, bearing,
speed) · dedup radius 96 m→~15–20 m · road masking at inference · keypoint correction (low gain).

**Later retrain:** hard negatives from the Yakima/Ellensburg FPs (the density-independent precision fix) · more
distinct corridors (the real ceiling) · anchor sweep. *Rejected:* offline augmentation copies; capping/dropping
Centralia.

**Imbalance strategy (can't equalise traffic density):** balance *attention* not density (scene-balanced sampling,
non-destructive) · hard negatives (free, unlimited, density-independent) · lean on invariant echo physics · more
scenes per weak corridor · per-corridor count calibration at deployment.

## 9. Longer-horizon (beyond the next run)
Velocity (per-scene acquisition Δt → speed; needs hub timestamps/off-nadir) · count calibration for WSDOT (stable
proportional count > per-vehicle F1) · segmentation alternative (Van Etten, dense packing) · use more than 3/8 bands ·
parametric keypoint head / geometry-consistency loss (collinearity by construction).

## 10. Doc map
- **[NEXT_MODEL_PLAN.md](NEXT_MODEL_PLAN.md)** — the active pre-training fixes (jitter, diagnostic, split, guard).
- `docs/` reference: `ARCHITECTURE`, `CONTEXT`, `DATA`, `DATA_EXCHANGE` (bundle contract), `EXPORT_REQUEST` (ask to
  the annotation hub), `HARDWARE`, `MODELING` (reference method), `REFINEMENT` (historical playbook).
- `models/registry.json` + `models/cards/` — per-model methodology + results.
- `docs/archive/` — superseded drafts (Detection_Refinement_Roadmap).
- External handoff: `Truck_Detection_Full_Context.docx` (the narrative handoff, kept in Downloads).

_Guiding principles: report per-corridor never aggregate · augmenting positives can't fix precision (only
negatives/masking/geometry) · only new corridors add information · inference-only fixes after training · change one
thing at a time and measure it · scores are agreement with a "clear truck-ish echo" policy, not ground truth._
