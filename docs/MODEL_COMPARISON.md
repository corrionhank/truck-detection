# Model comparison — `adamiak-v2` vs `jitter-mv`

Your two strongest models head-to-head: what changed, how they were built, how they score, and the levers to
build a better one. Written 2026-07-26. Numbers are from `models/registry.json` + the TC_01 threshold sweep.

---

## TL;DR

- **`adamiak-v2` is still the best *validated* model.** On the one apples-to-apples scene (`Tacoma-Centralia_01`,
  both held it out, thresh 0.3), v2 scores **F1 0.542** vs jitter-mv's **0.49**.
- **But the comparison is confounded and the metric is unreliable.** jitter-mv trained at a lower LR (a
  convergence confound), and its "lower precision" is largely **unlabeled real echoes** (you confirmed this by
  eye), not model error. So jitter-mv is **not** cleanly worse — it detects *more real echoes*, and gets punished
  by partial labels.
- **What jitter-mv proved:** the new mechanisms (translation jitter + multi-vehicle targets) shift the model
  **recall-rich** — it finds more trucks *and* more cars/faint echoes. The open question is converting that recall
  into truck-precision, which is a **post-inference size/geometry filter** job, not a training job.

---

## 1. Side-by-side spec

| | **`adamiak-v2`** | **`jitter-mv`** |
|---|---|---|
| Architecture | Keypoint R-CNN, ResNet-50 + FPN | *same* |
| Anchors | 4/8/16/32/48 px, ratios 0.25–1.25 | *same* |
| Input resize | min 192 / max 320 | *same* |
| Backbone | COCO-pretrained, finetuned | *same* |
| **Training data** | **228 veh / 9 scenes** (388-corpus era) | **466 veh / 14 scenes** (629-corpus) |
| **Chips** | single-vehicle, **64 px** | **96 px** padded → 64 px crop, **multi-vehicle** |
| **Translation jitter** | **none** (echo always centered) | **±16 px** real-pixel jittered crop each epoch |
| **Multi-vehicle targets** | no (neighbor echo = background) | **yes** (629 center + 404 neighbor annotations) |
| **Learning rate** | **1e-3** | **1e-4** (dropped to kill an early loss spike) |
| Epochs / batch / aug | 12 / 4 / rotate+flip+brightness+perspective | *same* (+ jitter) |
| Device | CPU | CPU |
| Held-out split | 3 scenes, name-based | 4 scenes, **footprint-guarded + temporal/spatial labeled** |

---

## 2. Methodology & approach — what actually changed

Both are the same Adamiak-spec detector with identical architecture and anchors. Four things differ:

1. **Translation jitter (the headline change).** v2's chips are always vehicle-centered, so it over-learns
   "echo at dead center" and loses detections the sliding window presents off-center. jitter-mv exports **96 px**
   chips and random-crops a **64 px** window each epoch (offset derived from the keypoints so it never cuts them),
   giving real-pixel translation augmentation. *Effect: higher recall.*
2. **Multi-vehicle targets.** In v2, a neighbor vehicle inside a chip is unlabeled → trained as background,
   teaching the model to suppress real echoes in dense traffic. jitter-mv labels every vehicle in the window
   (404 neighbor annotations added). *Effect: less false suppression → also higher recall / more firing.*
3. **More & broader data.** v2 trained on 9 scenes (228 veh); jitter-mv on 14 scenes (466 veh), including **all 4
   Yakima**, both auburn, and all 3 blaine-bellingham — more eastern/agricultural variety.
4. **Lower LR (1e-4 vs 1e-3).** Not a design choice — the fresh detection heads spiked at 1e-3 (loss 9→528 in the
   smoke test; confirmed 1e-3 oscillates, 1e-4 is smooth), so LR was dropped. **This is a confound:** a lower LR
   can undertrain and produce a less discriminative model (more false positives), so some of jitter-mv's precision
   drop may be LR, not jitter/multi-vehicle.

---

## 3. Data & cleaning (shared pipeline)

Identical and verified clean for both: bands **6/4/2**, per-scene **2–98 % percentile stretch**, `/255`, join
labels↔imagery by the **`scene` text field** (never spatial). The only pipeline difference is the chip export
(64 px single-vehicle for v2; 96 px multi-vehicle for jitter-mv). Labeling policy for both: **only clear,
well-formed echoes** — blurry, malformed, over-bright, and super-small echoes are deliberately left out. That
policy is *encoded into the confidence score* (the model learns "clear echo"), which is why raising the confidence
threshold acts as a rough quality/size filter.

**Known data caveat (critical for reading §4):** scenes are **partially labeled** — precision is a *lower bound*.
You visually confirmed most of jitter-mv's "false positives" are **real small-vehicle or unlabeled echoes**, not
errors. So F1 (which precision feeds) **understates the recall-rich model more**.

---

## 4. Accuracy

### Apples-to-apples: `Tacoma-Centralia_01` (both held out, temporal, thresh 0.3)

| | recall | precision | **F1** | centered | detected (for 80) |
|---|---|---|---|---|---|
| **`adamiak-v2`** | 0.562 | 0.523 | **0.542** | 1.00 | 86 |
| **`jitter-mv`** | **0.600** | 0.414 | **0.490** | 0.988 | 116 |

jitter-mv finds **more** (recall 0.60 vs 0.56, 116 detections vs 86) but at lower labeled-precision. Net F1 dips.

### Threshold sweep on TC_01 (best F1)

| model | best F1 | at threshold |
|---|---|---|
| `adamiak-v2` | **0.590** | 0.55 |
| `jitter-mv` | 0.527 | 0.55 |

Calibration helps both (v2: 0.542→0.590), but v2 stays ahead at every operating point. **However**, this sweep is
scored against partial labels, so it inherits the lower-bound-precision caveat — jitter-mv's extra detections
include real unlabeled echoes counted as FPs.

### Each model's full held-out record (registry, thresh 0.3)

`adamiak-v2`: TC_01 **0.542** · auburn_01 **0.667** · Yakima_01 **0.191** → mean F1 0.467, centered 0.973.
`jitter-mv`: Centralia_01 **0.564** · TC_01 **0.490** · EllensburgPreferredTest_01 0.027 (2 veh — noise) ·
Stanwood_10 0.167 (6 veh — noise) → mean F1 **0.312 (meaningless — dragged by the 2- and 6-vehicle scenes; use
per-scene)**. Centered recall **0.997**.

> Note: the held-out sets differ, so **only TC_01 is directly comparable**. v2's auburn (0.667) and Yakima (0.191)
> scenes were *trained on* by jitter-mv, so there's no head-to-head there. Both Centralia hold-outs for jitter-mv
> are **temporal** (88–96 % footprint overlap with trained TC_02) — same-ground-later-date, not generalization.

---

## 5. Similarities

Same architecture, anchors, backbone, resize, band selection, stretch, join rule, labeling policy, optimizer
(Adam + ReduceLROnPlateau + grad-clip 1.5), CPU training, 12 epochs. Both hit **~0.97–1.0 centered-chip recall**
(they *recognize* a handed echo almost perfectly) — the entire performance gap is in the **full-scene sliding
window**, i.e. translation tolerance and precision, not echo recognition.

---

## 6. Why jitter-mv didn't win the F1 (and why that's not the whole story)

- **Mechanically it did what it should:** jitter + multi-vehicle made it **detect more** — more real trucks
  (recall 0.60 vs 0.56) *and* more cars/faint/unlabeled echoes → more "FPs" against partial labels.
- **The precision drop is partly artifact** (unlabeled real echoes) and **partly confound** (lr 1e-4 may have
  undertrained it, making it fire more loosely).
- **Threshold can't fix it cleanly** because the extra detections are *confident* (they're real echoes) — only a
  **size/geometry gate** separates target trucks from cars/clutter.
- **Mild overfit at the end:** val loss bottomed at epoch 11 (4.715) and ticked up by epoch 12 (4.955), but we
  kept the epoch-12 weights (no best-val checkpoint). So the *registered* jitter-mv is slightly past its best.

---

## 7. How to build a better one (the actionable part)

Ordered by expected value:

1. **De-confound the LR.** Retrain jitter-mv with **LR warmup** (ramp 1e-5 → 1e-3 over the first ~few hundred
   iters) instead of a flat 1e-4. This avoids the early spike *and* the undertraining, giving a clean read on
   whether jitter/multi-vehicle actually help at v2's LR.
2. **Get honest precision — densely label one held-out scene** (e.g. TC_01: mark *every* echo or *exhaustively*
   every truck). Re-score both. This likely **flips the verdict**, because jitter-mv's recall gain is currently
   punished by label gaps. Highest-value single action.
3. **Add the size/geometry filter (non-DL postprocessing).** Measure echo streak length / keypoint spacing /
   bbox; keep truck-sized, well-formed (collinear, correct B→R→G order) echoes; drop cars and field-texture.
   Converts jitter-mv's recall into truck-precision *and* lifts v2. Lets you *lower* the confidence threshold to
   recover faint trucks the slider currently throws away.
4. **Best-val checkpointing / early stopping.** Save the lowest-val-loss epoch, not the last — jitter-mv's ep12
   was slightly overfit vs ep11.
5. **Calibrate the operating threshold** (~0.55 was best on TC_01 for both; the console defaults to 0.5, metrics
   used 0.3). Set the deployment threshold to the truck-optimal point per corridor.
6. **Ablate to isolate the winner** — jitter-only vs multi-vehicle-only vs both, all at matched LR — so the next
   model keeps only what helps.
7. **More distinct arid/urban corridors** (the coverage ceiling) — new backgrounds + hard-negative clutter for
   the precision problem. Skip more western-forested (saturated).

**Recommended next build:** retrain the jitter+multi-vehicle recipe with **LR warmup** (fixes the confound),
save the **best-val** checkpoint, then apply the **size/geometry filter** and re-score against a **densely-labeled
TC_01**. That combination tests every lever this comparison exposed and is the most likely path past v2's 0.59.

---

_Source: `models/registry.json` (both entries), the TC_01 threshold sweep, and the training logs. Companion:
[PROJECT_STATE.md](PROJECT_STATE.md), [NEXT_MODEL_PLAN.md](NEXT_MODEL_PLAN.md)._
