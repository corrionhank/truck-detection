# Next approach — what to do before training another model

Written 2026-08-09, after sweeping both trained models across the full confidence range and compounding the
post-processing levers. **The headline finding reverses the prior conclusion of this project**, so read §1 before
acting on anything in the older docs.

Models referenced: **A** = `kprcnn-adamiak-v2`, **B** = `kprcnn-warmup-v1`.

---

## 1. The finding: inference configuration was the bottleneck, not the model

Both models were re-scored across a grid of post-processing settings — threshold, duplicate-suppression radius,
top-1-per-window, and a gate on the per-keypoint confidence scores that `detect_scene.py` currently discards.
No re-inference was required: every detection is stored with its score, so this is a filter over records on disk.

| model | configuration | precision | recall | F1 | count ratio |
|---|---|---:|---:|---:|---:|
| adamiak-v2 | deployed (top-1, 32 px dedup, thr 0.5) | 0.507 | 0.455 | 0.480 | 0.90 |
| adamiak-v2 | **tuned** (all dets, 8 px, thr 0.55, kp > 3) | 0.548 | 0.508 | **0.527** | 0.93 |
| warmup-v1 | deployed (top-1, 32 px dedup, thr 0.5) | 0.265 | 0.539 | 0.355 | 2.03 |
| warmup-v1 | **tuned** (all dets, 8 px, thr 0.40, kp > 3) | 0.541 | 0.545 | **0.543** | **1.01** |

Post-processing alone is worth **+0.047 F1** for model A and **+0.188 F1** for model B.

**Tuned, model B beats model A** — on precision and recall simultaneously, at a count ratio of 1.01 (parity), and
above Van Etten 2024's PlanetScope truck F1 benchmark of 0.49.

### What this means

Every prior statement in this project that model A is the better model was **an artifact of inference
configuration**, not a property of the models. Model B was being run at a badly wrong operating point:

- threshold 0.5, when its optimum is far lower once the other levers are fixed;
- **top-1-per-window** discarding every detection but the highest-scoring one in each 64 px window;
- a **96 m duplicate-suppression radius**, which merges genuinely adjacent trucks on a freeway;
- the **per-keypoint confidence scores discarded entirely** — they separate real from spurious detections better
  than the box score does (AUC 0.891 vs 0.771).

Model B's training changes — translation jitter, multi-vehicle chip targets, LR warmup — were real improvements.
They were invisible because they were measured through a broken inference path.

### Compounding the levers (model B)

| configuration | P | R | F1 | ratio |
|---|---:|---:|---:|---:|
| deployed: top-1, dedup 32 px, thr 0.5 | 0.265 | 0.539 | 0.355 | 2.03 |
| + threshold → 0.65 | 0.324 | 0.483 | 0.388 | 1.49 |
| + keypoint gate > 2 | 0.492 | 0.436 | 0.462 | 0.89 |
| + top-1 removed | 0.489 | 0.451 | 0.469 | 0.92 |
| + dedup 32 → 16 px | 0.490 | 0.487 | 0.489 | 0.99 |
| + dedup 16 → 8 px | 0.496 | 0.513 | 0.505 | 1.03 |
| grid optimum (thr 0.40, 8 px, kp > 3) | 0.541 | 0.545 | 0.543 | 1.01 |

No single lever dominates. The keypoint gate is the largest single step (+0.074), but the gains compound and the
dedup radius matters more than expected once top-1 is removed.

---

## 2. Priority order

### Step 1 — Fix the inference path. Do not train anything first.

A few hours of work in `src/detect_scene.py`, worth more than the last three training generations combined:

1. **Keep every detection above threshold per window**, not just `scores[0]` (line ~135).
2. **Shrink the duplicate-suppression radius** from 32 px (96 m) to ~8 px (24 m).
3. **Gate on `keypoints_scores`** — currently computed and thrown away. `mean(kp_score) > 3` in the tuned config.
4. **Set the threshold per model** from the sweep rather than defaulting to 0.5.

Expose each as a parameter with the current behaviour as the default, so existing numbers stay reproducible.

> **Matching correctness note.** The evaluation's non-consuming matching is only safe because the dedup radius
> (32 px) exceeds twice the match radius (6 px). **Dropping dedup to 8 px breaks that guarantee** — two kept
> detections can then both fall within 6 px of one label and be double-counted. Make the matching explicitly
> consuming (greedy, score-descending, each label used once) *before* measuring any reduced-dedup configuration.

### Step 2 — Re-baseline every model at the fixed configuration

Every number in `models/registry.json`, `docs/EVIDENCE_PACK.md` and the R report was measured through the broken
path. `adamiak-all`, `jitter-mv` and `centralia-heldout` may all rank differently. Any training decision taken
against the old numbers was optimising the wrong target.

### Step 3 — Close the two measurement gaps, before more modelling

Both make further training guesswork:

- **Densely label one scene.** Precision is currently a *lower bound* — only clear, well-formed echoes were
  labelled, and manual inspection confirmed many counted false positives are real unlabelled or small vehicles.
  The true precision is unknown. This is the highest-value non-modelling action available.
- **Hold out a whole corridor.** There is no clean transfer measurement anywhere in the project. The only scene
  withheld from both models (`Tacoma-Centralia_01`) overlaps trained ground at **0.965 footprint IoU** — a repeat
  visit, not a transfer test.

### Step 4 — The training change worth making: hard negatives

Every training chip is centred on a vehicle, so the negative distribution the model sees is "road surface and
roadside within ±32 px of a truck." It has never seen field interior, rooftops or bare soil labelled as
background.

This is a negative **diversity** gap, not a negative **existence** gap — torchvision already samples hundreds of
in-chip negatives per image (RPN 256 anchors at 0.5 positive fraction; ROI 512 at 0.25). The fix is background
chips sampled from the same scenes, **weighted across all corridors** — the residual over-detection is global,
not arid-specific (model B exceeds count parity on every corridor, including Centralia at 1.26× and
polygon/Kent-Auburn at 4.26×).

### Step 5 — Fix scene-separated validation

Validation is currently a random chip subset drawn from the *training* scenes
(`train_detector.py`: `nval = max(8, 0.12 * n_train_chips)`), so best-val checkpointing selects for fitting
training-scene background. Split validation by scene instead. This only becomes meaningful once done.

---

## 3. Suggested split for the next run

No scene is currently out-of-sample for both models, which is why no transfer claim is possible. To fix that:

| role | scenes | vehicles | rationale |
|---|---|---:|---|
| **test (transfer)** | `Tri-Cities-New_01` | 105 | arid, single scene, zero footprint overlap with anything else; largest clean transfer test available |
| **validation** | `auburn-snoqualmie_01` + `_03` | 66 | distinct corridor, zero overlap with the remainder, large enough for a stable LR signal |
| **train** | everything else, incl. all Yakima and Spokane | 618 | keeps arid terrain in training so the Tri-Cities result measures transfer, not novelty |

Cost: 171 of 789 vehicles (22 %) held out — the price of the project's first honest transfer measurement.

---

## 4. What not to do

- **Do not raise the threshold to 0.9–0.95 to suppress over-counting.** It replaces over-counting with severe
  under-counting: at 0.95 model B finds 33 % of labelled trucks and model A about 5 %. Both score worse than at
  0.3. Count parity is reached at 0.775 (model B) and 0.45 (model A) under the *deployed* config, and near 0.4–0.55
  under the tuned one.
- **Do not chase F1 through more training runs alone.** Three generations have now failed to beat model A through
  training changes, while a few hours of post-processing work beat it comfortably. The recall ceiling is already
  100 % — with top-1 and dedup disabled at threshold 0, every one of the 80 labelled trucks on
  `Tacoma-Centralia_01` has a detection within 18 m. The model finds the trucks; the pipeline discards them.
- **Do not read low variance as quality on its own.** Per-scene F1 standard deviation keeps falling all the way
  to threshold 0.95, purely because a model detecting almost nothing is consistently bad.

---

## 5. Caveats on the tuned numbers

1. **The tuned thresholds were fitted on the same 21 scenes they are scored on.** They are descriptive of this
   dataset, not validated operating points. Validate the configuration on scenes not used to choose it before
   shipping, or you have tuned to the dataset rather than the problem.
2. **Precision remains a lower bound**, so the true F1 of both tuned configurations is higher than shown — and the
   real optimum probably sits at a slightly *lower* threshold than identified, since fewer of the filtered
   detections are genuinely wrong.
3. **The two models' training sets differ substantially** (9 vs 20 scenes; 78 % vs 23 % Centralia), so the
   head-to-head remains confounded with corridor coverage even after the configuration is equalised.
4. **Matching is custom** — red-keypoint distance within 18 m, not box IoU — so none of these values is comparable
   to published mAP figures.

---

_Derived from `outputs/inference_collect/{kprcnn-adamiak-v2,kprcnn-warmup-v1}/detections.csv` and the R project in
`analysis/truck-detection-report/`. Related: [EVIDENCE_PACK.md](EVIDENCE_PACK.md) (per-scene results and column
definitions), [KEYPOINT_GATE.md](KEYPOINT_GATE.md) (the keypoint-score mechanism in depth),
[NEXT_RUN_REVIEW.md](NEXT_RUN_REVIEW.md) (verification of the pipeline claims)._
