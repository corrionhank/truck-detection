# `kprcnn-jitter-mv` — what changed, why, and what we learned

A briefing on the most recent training run. Written to be read aloud / summarized to a research lead.
Model id `kprcnn-jitter-mv`, trained 2026-07-26. Prior best: `kprcnn-adamiak-v2`.

---

## 1. One-paragraph summary

We rebuilt the training data and the chip pipeline, reintroduced **translation jitter** (an augmentation the
earlier rebuild had dropped), and added **multi-vehicle chip labels**. The dataset grew from 538 to **629 labeled
vehicles** and — more importantly — went from **78% concentrated in one corridor to a roughly even spread across
four**. The resulting model **detects more real trucks** (recall 0.60 vs 0.56) but scores **lower on F1**
(0.49 vs 0.54) because it also produces more detections that our partial labels count as false positives. The
headline number understates it for three measurable reasons (below), so the honest conclusion is: **the changes
worked mechanically, the experiment was confounded, and the metric is not currently trustworthy.**

---

## 2. What was changed, and why

### 2.1 Translation jitter (the main change)

**The problem.** Training chips are cut *centered* on each labeled vehicle. Every training example therefore had
the echo at dead center. But at inference we slide a 64 px window across the scene on a fixed grid, so a real
truck lands wherever the grid happens to fall — a corner, an edge, anywhere. The model had learned an implicit
false rule: *"a truck echo appears in the middle of the window."*

**The evidence this mattered.** Our own model registry documents it: an early model trained on centered crops
found **0 of 3** trucks in a full-scene sweep; the same model with ±20 px jitter found **3 of 3**, with confidence
rising from ~0.4 to ~0.9. Separately, our best-generalizing model to date (`centralia-heldout`, F1 0.71 on an
unseen corridor) was trained with a recipe that *included* jitter — and the current rebuild had explicitly dropped
it.

**What we implemented.** Chips are now exported at **96 px** (64 px model chip + 16 px margin per side). At each
epoch the trainer takes a **random 64 px crop** from the 96 px chip, so the vehicle appears off-center on **real
pixels** (not mirrored padding), differently every epoch. The crop offset is derived from where the keypoints
actually sit, so it can never cut a keypoint. Augmentation (rotation/flips/brightness/perspective) now runs on the
96 px chip *before* cropping, so rotation pulls in real scene content instead of reflected filler.

**Safety check:** running the exporter with `--margin 0` reproduces the old 64 px chips **byte-for-byte**, proving
the crop geometry was not silently altered.

### 2.2 Multi-vehicle chip labels

**The problem.** The exporter wrote **one vehicle per chip**. If a second truck fell inside the same 64 px window,
it was unlabeled — and therefore trained as *background*. We were actively teaching the model to suppress real
echoes in dense traffic.

**We measured it before fixing it.** A diagnostic counted neighbors per chip: **16% of chips contain a second
vehicle in every crop; 39% within jitter reach.** On the dense Centralia scenes it reaches **31% and 61%**. Sparse
corridors (Yakima, Ellensburg) were near zero — confirming this is specifically a dense-traffic problem.

**What we implemented.** Every vehicle whose echo falls inside the export window is now labeled. This added
**404 neighbor annotations** on top of the 629 center annotations.

### 2.3 Cleaned and rebalanced dataset

The annotation set was re-exported from the upstream tool with corrections and three newly labeled scenes
(`Yakima-Toppenish_04`, `blaine-bellingham_03`, `polygon_01`), taking the corpus from **538 vehicles / 16 scenes
to 629 / 19**.

**Verification before training.** The importer validated a declared-vs-actual count handshake (629 vehicles /
1887 keypoints), confirmed every annotation resolved to its GeoTIFF, and enforced EPSG:32610. Because the labels
had been *edited upstream* — the exact situation where a coordinate flip can silently corrupt everything — we also
rendered 36 random chips across 6 corridors with the keypoints drawn on top and confirmed visually that they land
on the echoes. No misregistration.

**The rebalancing is the underrated part.** Training-set composition:

| corridor | `adamiak-v2` | `jitter-mv` |
|---|---:|---:|
| Centralia / I-5 south | **201 (78%)** | 145 (31%) |
| Yakima (arid) | **0 (0%)** | 126 (27%) |
| blaine-bellingham / I-5 north | 29 (11%) | 108 (23%) |
| auburn-snoqualmie / I-90 | **0 (0%)** | 66 (14%) |
| Ellensburg (arid) | 23 (9%) | 21 (5%) |
| **total** | **259 veh / 9 scenes** | **466 veh / 14 scenes** |

`adamiak-v2` was effectively a **Centralia specialist** — 78% of its training signal from one corridor, with
*zero* Yakima and *zero* auburn. `jitter-mv` is a **generalist** spread across four corridors. This matters for
interpreting the results (§4.1).

### 2.4 Split integrity fix (spatial-overlap guard)

We discovered that **every multi-scene corridor in this dataset is the same ground re-photographed on different
dates** — measured on actual valid-data footprints, not bounding boxes: Yakima scenes overlap 99–100%,
Tacoma-Centralia_01 vs _02 at 97.5%.

Our leave-one-scene-out splitting had only ever checked scene **names**, so a "held-out" scene could be the same
physical road the model trained on. We added a **footprint-overlap guard** that reprojects each held-out scene's
footprint against every training scene and labels the result:

- **temporal** = later capture of a corridor the model has seen → this is the *deployment* condition (WSDOT wants
  to monitor known corridors over time), **not** a leak, but it is not evidence of transfer either;
- **spatial** = a genuinely new place → the only valid generalization claim.

Important nuance for the lead: this is a **labeling/interpretation correction, not a discovery of inflated
numbers.** Empirically the overlapping scores sit at or *below* the clean ones, so past results were not inflated —
we simply lost the right to call them "generalization."

---

## 3. Results

### 3.1 Head-to-head (the only directly comparable scene)

`Tacoma-Centralia_01`, held out by **both** models, threshold 0.3:

| | recall | precision | **F1** | detections (80 real trucks) |
|---|---:|---:|---:|---:|
| `adamiak-v2` | 0.562 | 0.523 | **0.542** | 86 |
| `jitter-mv` | **0.600** | 0.414 | **0.490** | 116 |

Threshold sweep (best achievable F1 on that scene): `adamiak-v2` **0.590** at 0.55; `jitter-mv` **0.527** at 0.55.
Calibration lifts both but does not change the ordering.

### 3.2 Other measurements

- `jitter-mv` on `Centralia_01` (temporal): **F1 0.564** (R 0.70 / P 0.47) — its strongest scene.
- Centered-chip recall **0.997**, keypoint error **0.84–1.16 px** — echo *recognition* and *localization* are
  essentially solved; the entire performance gap lives in full-scene sweeping.
- Two of the four held-out scenes had only **2 and 6 vehicles** — statistically meaningless. The registry's
  "mean held-out F1 0.312" is therefore **not a usable number**; use per-scene results only.

---

## 4. Findings

### 4.1 The head-to-head is biased *against* the new model

The one comparable scene is **Centralia**. `adamiak-v2` spent **78%** of its training capacity on Centralia;
`jitter-mv` spent **31%** while also covering Yakima, auburn, and blaine-bellingham. So we are comparing a
specialist on its home turf against a generalist. **Losing by 0.05 F1 on Centralia while carrying three additional
corridors is a materially better result than the raw number suggests.** We do not yet have the measurement that
would show this — a fair test needs a corridor `adamiak-v2` never saw.

### 4.2 Precision is a lower bound, and it penalizes the new model harder

Our scenes are **partially labeled** — we mark clear, well-formed truck echoes and deliberately skip blurry,
malformed, over-bright, and very small ones. Manual inspection of `jitter-mv`'s "false positives" found **most are
real echoes** — small vehicles or unlabeled trucks — not model errors. Because `jitter-mv` detects more, it is
punished more by these label gaps. Its true precision is meaningfully higher than 0.414.

### 4.3 The confidence score behaves as a quality filter

Because we label only clean echoes, the model's confidence effectively ranks *echo quality*. Raising the threshold
preferentially removes small vehicles and noise while keeping large, well-formed truck echoes — which is what we
observe. This is useful but leaky: it also discards faint real trucks, and the right cutoff drifts between
corridors. The clean equivalent is an explicit **echo-size / geometry filter** (§6).

### 4.4 The changes did what they were designed to do

Jitter and multi-vehicle labeling were intended to make the model **stop suppressing detections**. It now detects
**35% more objects** (116 vs 86) and finds **more real trucks** (48 vs 46 of 80). That is the intended mechanical
effect. The cost — more low-quality detections surviving — is a **post-processing** problem, not a training
failure.

---

## 5. What went wrong (honestly)

**Experimental design**
- **Three variables changed at once** — jitter, multi-vehicle labels, *and* learning rate. We cannot attribute the
  recall gain or the precision loss to any single one.
- **The learning rate is an unintended confound.** The fresh detection heads spiked at the original 1e-3 (loss
  9 → 528 in two iterations), so we dropped to 1e-4. A lower LR can under-train the model, which alone could
  explain looser, less discriminative detections.
- **Two held-out scenes were too small to inform anything** (2 and 6 vehicles), and the split produced **no usable
  spatial generalization measurement** — both meaningful hold-outs were temporal.

**Training dynamics**
- **The learning-rate schedule never fired.** LR stayed at 1e-4 for all 12 epochs. The scheduler needs three
  consecutive non-improving epochs, and the validation set is only 55 chips — noisy enough (swings of ±0.6) that
  it kept registering accidental new "bests." The model therefore never got its low-LR refinement phase.
- **We saved the wrong epoch.** Best validation was epoch 11 (4.715); we kept epoch 12 (4.955). There is no
  best-checkpoint saving.
- **The model was likely under-trained, not over-trained.** Training loss was still falling at epoch 12 and
  validation was still tracking it. (This corrects an earlier read of "mild overfitting" — the epoch-12 uptick is
  smaller than mid-run noise spikes that recovered.)

**Process**
- The trainer **auto-selected the Apple GPU (MPS)** despite a documented divergence problem; its safety probe
  returned a false pass. Caught at the smoke-test stage and forced to CPU, but the probe cannot be trusted.

---

## 6. What we recommend next

1. **De-confound the learning rate** — retrain with LR *warm-up* (ramp into 1e-3) instead of a flat 1e-4. This
   avoids the initial spike without under-training, and gives a clean read on jitter itself.
2. **Fix the validation signal** — a larger, quieter validation set plus **best-epoch checkpointing** and real
   early stopping, so the LR schedule actually anneals and we keep the best model.
3. **Densely label one held-out scene** — mark *every* echo on one scene so precision can be measured honestly.
   This is the single highest-value action: it plausibly reverses the current ranking.
4. **Add a non-deep-learning echo-size / geometry filter** — keep only truck-sized, well-formed (collinear,
   correct blue→red→green order) echoes. This converts the new model's extra recall into precision, and it lifts
   the old model too. No retraining required.
5. **Measure spatial generalization properly** — hold out an entire corridor (all its dates), which is the only
   split that supports a transfer claim.
6. **Acquire imagery in new distinct corridors** (arid/agricultural and urban), which is the actual ceiling —
   more dates of existing corridors add little.

---

## 7. Talking points for the lead

- We reintroduced an augmentation the pipeline had lost, and rebuilt the dataset from **78% one-corridor to an
  even four-corridor spread**.
- The new model **finds more real trucks**; it scores lower on F1 only because it also surfaces echoes our
  labeling deliberately skips.
- **The headline metric is currently untrustworthy** — scenes are partially labeled, so precision is a lower
  bound, and the one comparable test scene structurally favors the older, Centralia-specialized model.
- We found and fixed a **split-integrity flaw**: every corridor in our data is the same road re-photographed, so
  past "held-out" numbers measured *temporal* consistency, not *spatial* transfer. Past numbers were not inflated —
  they were mislabeled.
- The clear next step is **measurement, not modeling**: densely label one scene, then apply a physics-based
  size/geometry filter. Both are cheap and both are likely to move the number more than further training.

---

_Sources: `models/registry.json`, training log `outputs/train_kprcnn-jitter-mv.log`, loss curve
`outputs/jitter-mv_loss_curve.png`, the threshold sweep on Tacoma-Centralia_01, the neighbor diagnostic
(`src/neighbor_diagnostic.py`), and the label-registration check (`src/verify_labels.py`). Companion:
[MODEL_COMPARISON.md](MODEL_COMPARISON.md), [PROJECT_STATE.md](PROJECT_STATE.md)._
