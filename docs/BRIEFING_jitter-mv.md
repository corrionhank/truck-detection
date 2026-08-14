# Briefing — `kprcnn-jitter-mv`, the latest training round

**Date:** 2026-07-26 · **Model id:** `kprcnn-jitter-mv` · **Commit:** `750b1f7`
**Audience:** research lead. Everything below is measured, not estimated; sources are
`models/registry.json`, `docs/MODEL_COMPARISON.md`, `docs/PROJECT_STATE.md`, and the training logs.

---

## 0. The 60-second version

We rebuilt the training data and re-introduced two mechanisms the current detector was missing:
**translation jitter** and **multi-vehicle chip targets**. We also grew the labeled corpus from
**339 vehicles / 8 scenes → 629 vehicles / 19 scenes** and, critically, spread it across corridors —
the training set went from **91 % one corridor** to a **31 % max share**.

The result behaved exactly as the mechanism predicted: the new model is **recall-rich** — it finds
**more real trucks** (recall 0.60 vs 0.56 on the shared test scene) — but its measured precision drops,
so the headline F1 came in slightly **lower** (0.49 vs 0.54).

The honest read is that **this is not a clean loss**. Two things contaminate the comparison: (1) our
scenes are **partially labeled**, so most of the "false positives" are visually confirmed **real,
unlabeled vehicle echoes**, meaning precision is a *lower bound* and the recall-rich model is punished
hardest; and (2) we had to **drop the learning rate 10×** mid-build to stop a loss blow-up, which is an
independent confound. So `adamiak-v2` remains the best **validated** model, while jitter-mv's recall
gain is very likely a **real** improvement that our current metric cannot credit.

**The actionable conclusion:** the remaining gap is not a training problem — it's a
**labeling-density problem** and a **post-processing problem**. The next two moves are (a) densely
label one test scene so precision is measurable, and (b) add a non-deep-learning
**size/geometry filter** to convert the extra recall into truck-specific precision.

---

## 1. Where we were before this round

| | State entering this round |
|---|---|
| Corpus | 538 vehicles / 16 scenes |
| Best validated model | `kprcnn-adamiak-v2` — mean held-out F1 0.467, best scene 0.542 |
| Model serving the console | `kprcnn-adamiak-all` — **broken** |
| Known-best result ever | `kprcnn-centralia-heldout` → auburn-snoqualmie_03, **F1 0.706** (clean, cross-corridor) |
| Literature benchmark | Van Etten 2024 PlanetScope truck **F1 0.49**; Adamiak et al. 2025 mAP ~0.53 |

**Three problems were diagnosed and are the reason this round exists:**

1. **The active deployment model was broken.** `kprcnn-adamiak-all` was trained on *all* scenes with
   **no held-out set**, so the LR scheduler's validation subset overlapped training → validation loss
   never plateaued → the learning rate never annealed → the model under-converged. It scores **0.06 on
   the very scenes it trained on.** Lesson encoded as a rule: **always hold scenes out.**
2. **Translation jitter had been silently dropped.** The single best-performing model on record
   (`centralia-heldout`, F1 0.706) used a rich augmentation recipe *including* jitter. When we rebuilt
   the detector to Adamiak spec, jitter did not carry over — every training chip was perfectly
   vehicle-centered. Historically this exact failure was catastrophic: our very first model
   (`echo-v0`, centered chips only) scored **0/3** on a full scene, and adding **±20 px jitter**
   (`echo-jitter`) recovered **3/3** with confidence rising ~0.4 → ~0.9. We had re-created the original
   bug.
3. **Neighbor vehicles were being trained as background.** The exporter wrote one vehicle per chip, so
   a *second* real vehicle inside the same 64 px window was an unlabeled negative — we were actively
   teaching the model to suppress real echoes in dense traffic, which is precisely where recall was
   failing.

---

## 2. The dataset work

### 2.1 Import and clean-up: 538/16 → 629/19

Imported the `trg-echo-exchange` bundle. Three new scenes (`Yakima-Toppenish_04`,
`blaine-bellingham_03`, `polygon_01`), and — importantly — **all 16 existing scenes were replaced with
upstream-corrected labels**, not merely appended to. The GeoPackage is committed to git, so 629/19 is
now the reproducible record (a clean checkout previously reverted to 538/16).

**Label integrity was verified, not assumed.** Because the annotations were corrected upstream, a
coordinate flip would have passed every count-based check we had. We built `src/verify_labels.py` and
eyeballed **36 chips across 6 corridors including the new scenes** — echoes centered, **0 flags**.

### 2.2 The distribution fix — the headline data change

This is the change with the longest-term value. The corpus is no longer one corridor with a few
satellites of data attached to it.

**Full corpus (629 vehicles / 19 scenes):**

| Corridor | Scenes | Vehicles | Share |
|---|---:|---:|---:|
| Centralia / Tacoma-Centralia (south I-5) | 4 | 281 | 44.7 % |
| Yakima-Toppenish (eastern, agricultural) | 4 | 126 | 20.0 % |
| blaine-bellingham + Bellingham (north I-5) | 4 | 108 | 17.2 % |
| auburn-snoqualmie (I-90) | 2 | 66 | 10.5 % |
| Ellensburg + Ellensburg-Yakima (I-90) | 3 | 23 | 3.7 % |
| polygon_01 (Kent/Auburn AOI) | 1 | 19 | 3.0 % |
| Stanwood_10 (north I-5) | 1 | 6 | 1.0 % |
| **Total** | **19** | **629** | |

**Within the training set actually used for this model (466 vehicles):**

| Corridor | Vehicles | Share |
|---|---:|---:|
| Centralia (TC_02 + C_02) | 145 | 31 % |
| Yakima (all 4) | 126 | 27 % |
| blaine-bellingham (+ Bellingham_01) | 108 | 23 % |
| auburn-snoqualmie (both) | 66 | 14 % |
| Ellensburg | 21 | 5 % |

**Compare to the 339-vehicle corpus we started the year with: 307 of 339 vehicles — 91 % — were the
single greater-Centralia / south-I-5 corridor.** Maximum corridor share is now **31 %**. That is the
difference between a model that has memorized one stretch of I-5 and a model with genuine background
diversity to learn from.

### 2.3 Two data limitations we found and are stating openly

- **Every multi-scene corridor is the same footprint re-captured on a different date.** Measured on
  *valid-data footprint IoU*, not bounding boxes: Yakima_01 vs _03/_04 = **99.5 % / 100 %**;
  Tacoma-Centralia_01 vs _02 = **97.5 %**; auburn_01 vs _03 = **59 %**; Centralia_01 vs _02 = **21 %**
  (same day); polygon_01 vs everything = **0 %**.
  **Consequence:** our name-based leave-one-scene-out split has *never actually produced a spatial
  hold-out.* We now distinguish two different generalizations and report both, labeled:
  - **Temporal** — a later capture of a corridor the model has seen. This is *not* a leak; it is
    literally the WSDOT deployment condition (per-corridor calibration on a known corridor).
  - **Spatial** — transfer to an entirely unlabeled corridor. Requires holding out the whole corridor.

  Important nuance for the write-up: the overlapping scores sit **at or below** the clean ones, so this
  was **mislabeling of what we measured, not inflation of the numbers.**
- **All 19 captures fall in a single spring window (Apr 9 – Jun 20, a 72-day span).** Our clutter
  exposure is one season's crop stage and sun-angle range. A stated limitation.

---

## 3. What we built — the four changes

### Change 1 — Translation jitter, re-introduced properly (the headline)

**The problem.** Every training chip was cropped dead-center on the vehicle, so the model learned
"echo at the exact center of the window." At inference we slide a window across a raw scene, where
echoes land *anywhere* in the window. This is the single cleanest explanation for our persistent gap
between **centered-chip recall ≈ 0.97–1.00** (recognize a hand-delivered echo — nearly perfect) and
**full-scene F1 ≈ 0.47** (find echoes yourself — mediocre).

**The implementation** (deliberately better than the 2026-07-10 version):

- `export_coco.py` gained `--margin` (default 16): chips are exported at **96×96** (`64 + 2×16`)
  instead of 64×64.
- At training time, `train_detector.py` random-crops a **64×64** window out of the 96 px chip
  **fresh every epoch**, subtracting the offset from the keypoints. Jitter is **±16 px**.
- The crop origin is **derived from where the keypoints actually sit**, clamped to the legal window, so
  it is mathematically incapable of cutting a keypoint and never needs to resample and retry.
  Edge-clamped vehicles narrow their own jitter range automatically.
- Augmentation runs on the **96 px** chip *before* the crop, so rotation pulls **real neighboring
  pixels** into the corners rather than zero-padding — this is the improvement over the old approach.
- `jitter=0` is an exact center crop, i.e. a built-in **ablation control**.

**Safety check on the rewrite:** we hashed the entire pre-change chip directory first and required
`--margin 0 --single` to reproduce the legacy 64 px output **byte-for-byte**. It does.

### Change 2 — Multi-vehicle chip targets

**We measured before we fixed.** `src/neighbor_diagnostic.py` counted neighbor vehicles per chip in two
bands (≤16 px = present in *every* crop; 16–48 px = present in *some* epochs once jitter is on):

| | ≤16 px (every crop) | 16–48 px (some epochs) |
|---|---:|---:|
| Overall | **16 %** of chips | **39 %** of chips |
| Dense Centralia scenes | 18–31 % | 52–61 % |
| Sparse corridors (Yakima / Ellensburg) | ~0–8 % | low |

Roughly a third of Tacoma-Centralia_01's chips carry a neighbor inside the always-visible band. That is
a substantial volume of **real trucks being trained as background**, concentrated in exactly the dense
traffic where our recall was worst.

**The fix:** the exporter now labels **every** vehicle in the window, not just the center one. The
corpus went from 629 targets to **1,033 targets — 629 center + 404 neighbor annotations.**

### Change 3 — A spatial-overlap guard on the split

The existing leakage guard compared **scene names only** — it cannot see that `Tacoma-Centralia_01` and
`_02` are 97.5 % the same ground. `train_detector.py` now reprojects **valid-data footprints** and, for
each held-out scene, **warns and labels the split `temporal` or `spatial`**, recording that label into
the registry entry. It warns rather than hard-rejects, because temporal hold-outs are legitimate and
deployment-relevant — they just must not be *called* generalization.

Also added: `--exclude` (drop a scene from training without holding it out — used for `polygon_01`) and
`--jitter` (expose the jitter magnitude, enabling the ablation).

### Change 4 — Tooling built alongside

`eval_matrix.py` (model × scene evaluation matrix), `src/neighbor_diagnostic.py`, `src/verify_labels.py`
(label-on-echo visual overlay), `src/import_scenes.py` (imagery-only ingest for inference in any CRS).
The web console gained a **Docs tab** that serves `docs/*.md` directly, so this briefing and the
supporting docs are readable in the app.

---

## 4. How the model was trained

| | `adamiak-v2` (prior best) | **`jitter-mv` (new)** |
|---|---|---|
| Architecture | Keypoint R-CNN, ResNet-50 + FPN | *same* |
| Anchors | 4/8/16/32/48 px, ratios 0.25–1.25 | *same* |
| Backbone | COCO-pretrained, finetuned | *same* |
| Input resize | min 192 / max 320 | *same* |
| **Training data** | **228 veh / 9 scenes** | **466 veh / 14 scenes** |
| **Chips** | 64 px, single-vehicle | **96 px → 64 px random crop, multi-vehicle** |
| **Translation jitter** | **none** | **±16 px, resampled every epoch** |
| **Neighbor targets** | no (neighbor = background) | **yes (+404 annotations)** |
| **Learning rate** | 1e-3 | **1e-4** ← *confound, see §6* |
| Epochs / batch / optimizer | 12 / 4 / Adam + ReduceLROnPlateau, grad-clip 1.5 | *same* |
| Device | CPU | CPU |
| Held-out split | 3 scenes, name-based | **4 scenes, footprint-guarded + labeled temporal/spatial** |

**The split, explicitly:** 466 vehicles across 14 scenes in the training pool (411 used for gradient
updates, 55 carved off as the LR-scheduler validation subset), **144 vehicles across 4 scenes held out
and never trained on**, and `polygon_01` (19 vehicles) excluded from both.

**Pipeline settings identical across both models** (so the comparison is clean on this axis): SuperDove
bands 6/4/2, per-scene 2–98 % percentile stretch, ÷255, and labels joined to imagery by the **`scene`
text field, never spatial extent** (a spatial join demonstrably leaks labels between overlapping scenes).

---

## 5. Results

Every number is measured on scenes the model **never saw in training**. Threshold 0.3.

### 5.1 Head-to-head — `Tacoma-Centralia_01`, the only directly comparable scene

Both models held this scene out, so this is the apples-to-apples read. 80 labeled vehicles.

| | recall | precision | **F1** | centered recall | detections (vs 80 labeled) |
|---|---:|---:|---:|---:|---:|
| `adamiak-v2` | 0.562 | 0.523 | **0.542** | 1.000 | 86 |
| **`jitter-mv`** | **0.600** | 0.414 | **0.490** | 0.988 | **116** |

**Read this as:** the new model **found more real trucks** (0.600 vs 0.562 recall) and fired far more
often overall (116 vs 86 detections). Against our partial labels, those extra detections score as false
positives, so F1 dips even though recall rose.

### 5.2 `jitter-mv`'s full held-out record

| Held-out scene | split type | veh | centered | recall | precision | **F1** |
|---|---|---:|---:|---:|---:|---:|
| `Centralia_01_20260511` | temporal | 56 | 1.000 | 0.696 | 0.474 | **0.564** |
| `Tacoma-Centralia_01_20260429` | temporal | 80 | 0.988 | 0.600 | 0.414 | **0.490** |
| `Stanwood_10_20260511` | spatial | 6 | 1.000 | 0.667 | 0.095 | 0.167 |
| `EllensburgPreferredTest_01_20260530` | spatial | 2 | 1.000 | 1.000 | 0.014 | 0.027 |

**Mean centered-chip recall 0.997 · keypoint error 0.84–1.16 px.**

⚠️ **Do not quote the mean F1 of 0.312.** It is arithmetically dragged down by the 6-vehicle and
2-vehicle scenes, where a handful of detections swings precision from 0.10 to 0.01. Those two scenes are
statistical noise, not evidence. **Quote per-scene: 0.564 and 0.490.**

The consequence: our two spatial hold-outs this run were both too small to be informative, so **this run
produced a strong temporal read and essentially no spatial read.** That was a known trade-off going in —
we chose to keep all Yakima and auburn data *in* training. A leave-one-corridor-out run is the fix.

### 5.3 Threshold sweep on `Tacoma-Centralia_01`

| model | best F1 | at threshold |
|---|---:|---:|
| `adamiak-v2` | **0.590** | 0.55 |
| `jitter-mv` | 0.527 | 0.55 |

Calibration helps both meaningfully (v2: 0.542 → 0.590). v2 leads at every operating point — but this
sweep is scored against the same partial labels, so it inherits the caveat in §6.

### 5.4 Context

- Van Etten 2024, PlanetScope trucks: **F1 0.49**. Adamiak et al. 2025: **mAP ~0.53**.
- Our best clean cross-corridor result on record: **F1 0.706** (`centralia-heldout` → auburn_03).
- At 3 m ground sample distance a vehicle is **1–3 pixels**. ~0.5 F1 is a respectable number in this
  regime, not a weak one.

---

## 6. Honest assessment — successes, failures, and the two confounds

### What succeeded

1. **The mechanisms did exactly what they were designed to do.** Jitter + neighbor targets shifted the
   model **recall-rich**: more detections, more real trucks found, higher recall on the shared scene.
   The causal chain we predicted is the causal chain we observed.
2. **The dataset is substantially better** — 85 % more vehicles than the 339 corpus, corridor
   concentration cut from 91 % to 31 %, labels upstream-corrected and visually verified.
3. **Centered-chip recall is effectively solved: 0.997.** The model recognizes an echo when handed one.
   **The entire remaining performance gap is in the sliding window** — translation tolerance and
   precision — not echo recognition. That is a much better-defined problem than "the model is bad."
4. **We can now tell temporal from spatial generalization**, and we caught a measurement-framing error
   that had silently affected every prior number.
5. **Methodology discipline held** — byte-identical export control, keypoint-derived crop bounds, an
   ablation switch built in, split types recorded in the registry.

### What failed or fell short

1. **Headline F1 did not beat `adamiak-v2`** (0.490 vs 0.542 on the shared scene). On the raw metric,
   this round did not win.
2. **We shipped a slightly overfit checkpoint.** Validation loss bottomed at **epoch 11 (4.715)** and
   ticked back up at **epoch 12 (4.955)** — and we had no best-validation checkpointing, so the
   registered weights are epoch 12. Small, avoidable, and now on the fix list.
3. **The spatial-generalization question went unanswered** this run (both spatial hold-outs were 2 and
   6 vehicles).
4. **The deployment model in the console is still the broken `adamiak-all`.** Should be repointed.

### The two confounds — say these out loud before anyone quotes the F1

1. **Partial labels make precision a lower bound.** Our labeling policy is deliberately
   *only clear, well-formed echoes* — blurry, malformed, over-bright, and very small echoes are left
   out on purpose. So a scene is **not exhaustively labeled**, and a model that detects a real but
   unlabeled small vehicle is **penalized as if it hallucinated.** We inspected jitter-mv's "false
   positives" visually and **most are real echoes** — unlabeled or small vehicles. Because F1 is fed by
   precision, **this metric punishes the recall-rich model hardest.** jitter-mv is not cleanly worse; it
   is being scored by a ruler that cannot see what it found.
2. **The learning rate is a genuine confound.** We did not choose 1e-4 as a design decision — the fresh
   detection heads **blew up at 1e-3** (loss went **9 → 528** in the smoke test; 1e-3 oscillated, 1e-4
   was smooth), so we dropped it 10×. A lower LR can **under-train**, producing a less discriminative
   model that fires more loosely. So an unknown portion of the precision drop is LR, **not** jitter or
   neighbor targets. Two variables moved at once, which violates our own change-one-thing rule.

**Therefore:** `adamiak-v2` stays the best **validated** model. jitter-mv's recall gain is very likely
**real**. The experiment is **inconclusive on F1 by construction**, and we know precisely why.

---

## 7. What we learned (the transferable findings)

1. **Confidence score ≈ echo quality.** Because we only label clear, well-formed echoes, the model's
   confidence learns "clear echo," which means **raising the threshold acts as a crude truck-vs-car
   filter.** The clean version of that is an explicit non-deep-learning geometry gate — and it is the
   single highest-leverage thing left to build.
2. **Threshold can't rescue precision here.** The extra detections are *confident* — they are real
   echoes. Only a **size/geometry** gate separates target trucks from cars and clutter.
3. **Failure splits by background clutter, not terrain.** Our earlier "forested vs arid" read was wrong:
   forested auburn and Bellingham work fine, while agricultural Yakima and Ellensburg flood with false
   positives on field texture.

   | corridor | mean F1 | failure mode |
   |---|---:|---|
   | auburn-snoqualmie | ~0.63 | works |
   | Centralia / Tacoma-Centralia | ~0.54 | works |
   | blaine-bellingham | ~0.33 | recall — misses trucks |
   | Yakima | ~0.18 | precision — false-positive flood |
   | Ellensburg | ~0.11 | both collapse |
4. **N dates of one place ≈ 1 place** for spatial transfer. Multi-date repeat coverage is genuinely
   valuable (new echoes, seasonal variation) but adds **no spatial diversity**. Only new corridors do.
5. **Augmenting positives cannot fix precision.** Only hard negatives, masking, or geometry can. Jitter
   and neighbor targets were always recall-side levers — they delivered on recall, and it was never
   reasonable to expect them to fix precision.
6. **Always hold scenes out**, even for a deployment model — the `adamiak-all` failure traces entirely
   to a validation subset that overlapped training, which stalled the LR scheduler.

---

## 8. What we do next, in priority order

1. **De-confound the learning rate.** Retrain the same recipe with **LR warmup** (ramp 1e-5 → 1e-3 over
   the first few hundred iterations) instead of a flat 1e-4. This avoids the blow-up *and* the
   under-training, giving a clean read on whether jitter + neighbor targets help at v2's learning rate.
2. **Densely label one held-out scene** (Tacoma-Centralia_01 — mark *every* echo, or exhaustively every
   truck) and re-score both models. **This is the highest-value single action available** and it
   plausibly **flips the verdict**, since jitter-mv's recall gain is currently being punished by label
   gaps rather than by error.
3. **Build the size/geometry filter** (post-inference, no retraining): measure streak length, keypoint
   spacing, collinearity, and correct blue→red→green ordering; keep truck-sized well-formed echoes and
   drop cars and field texture. This converts jitter-mv's recall into truck precision, and lets us
   *lower* the confidence threshold to recover faint trucks we currently discard.
4. **Best-validation checkpointing / early stopping** — save the lowest-val-loss epoch, not the last.
5. **Calibrate the operating threshold per corridor** (~0.55 was best on TC_01; note the console
   currently renders at 0.5 while all reported F1 was computed at 0.3 — that mismatch should be fixed).
6. **Ablate to isolate the winner** — jitter-only vs neighbor-targets-only vs both, all at matched LR —
   so the next model keeps only what actually helps. The `--jitter 0` control is already built.
7. **Add distinct arid / urban corridors** — the real ceiling. New backgrounds and hard negatives are
   the density-independent fix for the precision problem. More western-forested data is saturated.

**Recommended next build:** the jitter + multi-vehicle recipe, retrained with **LR warmup**, saving the
**best-validation** checkpoint, then scored against a **densely-labeled Tacoma-Centralia_01** with the
**geometry filter** applied. That single run tests every lever this round exposed and is the most
plausible path past v2's calibrated 0.590.

---

## 9. Anticipated questions

**"Did the model get worse?"**
On the raw F1 metric, slightly — 0.490 vs 0.542 on the shared scene. But it found *more real trucks*
(recall 0.600 vs 0.562), and its extra detections are largely real unlabeled vehicles. We can't
distinguish "worse" from "better than our labels can measure" until one scene is densely labeled.

**"Why not just revert the jitter?"**
Because the mechanism is right and historically load-bearing: centered-only chips produced our worst
result ever (0/3 full-scene), and adding jitter fixed it (3/3). The best model on record used it. What
we have not yet done is test it *without* the learning-rate confound.

**"Why is centered recall 0.997 but full-scene F1 only ~0.5?"**
They measure different things and this is the core story of the project. Centered recall asks "here is
a chip with a truck in the middle — do you see it?" (yes, essentially always). Full-scene asks "here is
a raw satellite scene — find every truck yourself." The gap is translation tolerance and precision,
which is exactly what this round targeted.

**"Is ~0.5 F1 good?"**
Vehicles are 1–3 pixels at 3 m resolution. Van Etten 2024 reports 0.49 on PlanetScope trucks; Adamiak
reports mAP ~0.53. We match published results. Our best clean cross-corridor run hit 0.706. And the
project deliverable is **aggregate counts that are stably proportional to reality** — statistically
calibratable per corridor — not perfect per-vehicle detection.

---

_Sources: `models/registry.json` (`kprcnn-adamiak-v2`, `kprcnn-jitter-mv`), `docs/MODEL_COMPARISON.md`,
`docs/PROJECT_STATE.md`, `docs/NEXT_MODEL_PLAN.md`, `CHANGELOG.md` (2026-07-26 entries), and
`data/active/coco/annotations.json` (per-scene counts recomputed 2026-07-27)._


dedscribe mode

model outputs to next model inputs 
keypoint
time 
date
coordinate data
other characteristics
keypoint id

regressors, outcome,