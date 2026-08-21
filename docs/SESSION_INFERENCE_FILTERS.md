# The inference filter chain: the keypoint gate, and three filters that discard trucks

Written 2026-08-20. Self-contained handoff: it assumes no context beyond the repo. Covers what the
keypoint gate is, what a session of chip-size and window-size experiments found, and three predictions
that the experiments falsified.

---

## 1. Context

PlanetScope satellite truck detection with a Keypoint R-CNN. A moving vehicle leaves a blue → red →
green streak because the sensor records its colour bands milliseconds apart; a parked one does not.
Two trained models are in play: `kprcnn-adamiak-v2` (9 scenes, 228 chips) and `kprcnn-warmup-v1`
(20 scenes, 624 chips). Evaluation is 21 labelled scenes holding 789 hand-labelled vehicles.

**The deliverable is aggregate truck counts, not per-vehicle detection.** That matters for reading
everything below: a configuration that finds more trucks while reporting ten times too many is worse
for this project than one that finds fewer and counts correctly. The count ratio column is the one to
watch, not F1.

---

## 2. What the keypoint gate is

For every detection the model returns four things. The pipeline used two of them and discarded the one
that turned out to matter most.

| model output | what it is | used before this session |
|---|---|---|
| `scores` | box classifier confidence, softmax, 0–1 | yes |
| `keypoints` | x, y of the three band positions | yes |
| `keypoints_scores` | **per-keypoint confidence** | **discarded** |
| `boxes` | the detection rectangle | no |

`keypoints_scores` is the raw value of the heatmap at each keypoint's chosen peak. It is an unbounded
logit, roughly −12 to +12 in this data, uncalibrated, and only its ordering is meaningful. It measures
how sharply peaked the heatmap was, so a crisp band localisation scores high and a vague one scores low
or negative.

### Why it carries information the box score does not

Two different heads answer two different questions:

- **Box head:** *does this region look like a moving echo?* A region-level appearance judgement.
- **Keypoint head:** *can I locate three distinct band positions inside it?* A structural judgement
  about the echo's internal geometry.

A patch of bright field texture, lane paint or a rooftop can be region-level echo-like enough to pass
the box head while producing flat keypoint heatmaps, because there is no crisp blue → red → green
triple to lock onto. Measured separability of matched vs unmatched detections:

| feature | AUC (0.5 = useless, 1.0 = perfect) |
|---|---:|
| **mean keypoint score** | **0.891** |
| box score (what the pipeline used) | 0.771 |

### The effect

The gate is a threshold on the mean of the three keypoint scores. On an offline sweep across all 21
labelled scenes, applied to stored detection records:

| model | configuration | precision | recall | F1 | count ratio |
|---|---|---:|---:|---:|---:|
| adamiak-v2 | deployed (top-1, 32 px dedup, thr 0.5) | 0.507 | 0.455 | 0.480 | 0.90 |
| adamiak-v2 | **gate 8.0, all detections, 8 px dedup, thr 0.05** | 0.693 | 0.687 | **0.690** | **0.99** |

Validated leave-one-scene-out on the operating point itself: pick the setting on 20 scenes, apply to
the 21st, pool. Out-of-sample F1 was identical at 0.690, and all 21 folds independently chose the same
setting.

### The caveat that matters most

**The gate value is per-model and does not transfer.** `adamiak-v2`'s optimum is 8.0, `warmup-v1`'s is
3.5. These are raw uncalibrated logits with no meaning outside the model that produced them. Any new
model must be re-tuned from scratch, and the value must be stored per model in the registry, never
fixed as a constant.

---

## 3. Findings

### 3.1 Training time is flat across chip size

Measured on real training iterations, batch 4:

| chip | input after resize | sec/iter | vs 64px | FLOP theory predicted |
|---|---|---:|---:|---:|
| 64px | 192px | 2.75 | 1.00× | 1.00× |
| 48px | 144px | 2.48 | 0.90× | 0.56× |
| 32px | 96px | 3.01 | **1.09×** | 0.25× |
| 24px | 72px | 2.74 | 0.99× | 0.14× |

32 px is marginally *slower* than 64 px. Roughly 7 hours for 16 epochs at any size.

Breakdown of a single iteration explains it:

| chip | forward | backward | optimiser step |
|---|---:|---:|---:|
| 64px | 0.74s | 2.23s | 0.45s |
| 48px | 0.68s | 1.97s | 0.27s |
| 32px | 0.52s | 1.93s | 0.33s |

Only the forward pass scales properly. Backward is 65–70% of the iteration and barely moves, because
gradient computation over 59M parameters has a large fixed floor at small spatial extents, and Adam's
update over those parameters costs the same regardless of input size. Shrinking the image does not
shrink the parameter count, and the parameters are the expensive part.

### 3.2 Inference window size is decoupled from training chip size

Every window is resized to `min_size` before the backbone. Holding the upscale ratio (input = chip × 3)
keeps a truck at the same size in model space at any window size, so the anchor set stays valid. Only
the amount of surrounding context changes.

Confirmed empirically with `warmup-v1`, which was trained at 64 px:

| window | upscale | input | precision | recall | F1 |
|---|---|---|---:|---:|---:|
| 64px | 3× | 192px | 0.068 | 0.273 | 0.109 |
| 48px | 3× | 144px | 0.081 | 0.455 | 0.137 |
| 32px | 3× | 96px | 0.128 | 0.545 | 0.207 |
| 24px | 3× | 72px | 0.110 | **0.727** | 0.190 |
| 32px | 6× | 192px | 0.118 | 0.364 | 0.178 |
| 64px | 2.25× | 144px | **0.194** | 0.636 | **0.298** |

Recall improves monotonically as the window shrinks under scale preservation. Anchor spacing on the
ground stays 24–32 m at every chip size, which is why this works.

### 3.3 There are three filters discarding trucks, not two

In pipeline order:

1. **torchvision's internal ROI NMS at IoU 0.5**, applied *inside* each window, before the pipeline
   sees anything. Our boxes are the three keypoints plus 3 px padding, so two nearby trucks produce
   heavily overlapping boxes and the lower-scoring one is deleted.
2. **top-1 per window** — only the highest-scoring detection in each window is kept.
3. **cross-window duplicate suppression at 96 m**.

Filter 1 is upstream and dominant, and it was not accounted for in any earlier analysis. At threshold
0.5, **only 16.1% of windows contain more than one detection at all**:

| threshold | window-best detections | additional detections | windows with more than one |
|---|---:|---:|---:|
| 0.3 | 3,879 | 1,255 | 24.7% |
| 0.5 | 2,515 | 479 | 16.1% |
| 0.7 | 1,456 | 173 | 10.7% |

Structural blocking by the other two filters, measured against the 773 labelled vehicles:

| filter | setting | trucks blocked |
|---|---|---:|
| top-1 | 64px window, 40px stride | 21.2% |
| top-1 | 32px window, 20px stride | 7.8% |
| top-1 | 24px window, 15px stride | 4.0% |
| dedup | 96 m (current) | 38.0% |
| dedup | 48 m | 14.9% |
| dedup | 24 m | 3.6% |

Dedup runs after windows are merged, so **window size has no effect on it whatsoever**.

### 3.4 Removing top-1 does almost nothing

| config | window | detections | TP | recall | count ratio |
|---|---|---:|---:|---:|---:|
| deployed today | 64px | 44 | 3 | 0.273 | 4.00× |
| keep-all only | 64px | 45 | **3** | **0.273** | 4.09× |
| keep-all + 8 px dedup | 64px | 59 | 3 | 0.273 | 5.36× |
| small window only | 24px | 73 | 8 | 0.727 | 6.64× |
| small window + both | 24px | 112 | 10 | 0.909 | 10.18× |

Removing top-1 recovered one detection and zero true positives, because filter 1 had already deleted
the second truck before the pipeline could keep it.

### 3.5 Small windows work by sidestepping the internal NMS

They do not remove it. A smaller window means two trucks usually land in *different* windows, so each
gets its own detection and the internal NMS never fires on the pair.

### 3.6 Without the gate, finding more trucks makes counting worse

Count ratios in the table above run 4× to 10×. The 24 px configuration found 91% of the labelled
trucks and reported ten times too many. For a counting deliverable that is a regression, not progress.

### 3.7 The gate is what makes a small window usable

Run live through the fixed inference path on the same scene, threshold 0.5, all detections kept,
8 px dedup:

| config | window | detections | TP | FP | FN | precision | recall | F1 | count ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deployed today | 64px | 44 | 3 | 41 | 8 | 0.068 | 0.273 | 0.109 | 4.00× |
| small window, no gate | 24px | 112 | 10 | 102 | 1 | 0.089 | **0.909** | 0.163 | 10.18× |
| 64px + gate 3.5 | 64px | 10 | 2 | 8 | 9 | 0.200 | 0.182 | 0.190 | **0.91×** |
| **24px + gate 3.5** | 24px | 16 | 6 | 10 | 5 | **0.375** | 0.545 | **0.444** | 1.45× |

The two levers do different jobs and compose. The small window finds the trucks: recall goes 0.273 to
0.909. The gate removes what should not have been reported: detections fall from 112 to 16 and the
count ratio from 10.18× to 1.45×. Neither alone is usable. Together, F1 goes from 0.109 to **0.444**,
four times the deployed configuration, and the count lands within 50% of truth instead of ten times it.

The 64 px row is worth noting separately: with the gate it reaches a count ratio of 0.91×, near parity,
but finds only 18% of the trucks. It counts well by accident, with misses and false positives roughly
cancelling. The 24 px row finds three times as many real trucks with better precision, which is the
more honest configuration even though its ratio is further from 1.

---

## 4. Predictions that were falsified

Recorded because the pattern is worth avoiding, not for completeness.

**Claimed smaller chips would nearly halve training time (0.56×).** Actual 0.90×, and 32 px came out at
1.09×, slower than 64 px. FLOP arithmetic was applied to a workload that is not FLOP-bound.

**Predicted rescaling window content to a common input size would break anchor matching.** It did not.
The best F1 in the transfer test was one of the configurations predicted to fail. The prediction used
nearest-anchor-centre matching; torchvision assigns anchors by IoU with 0.7 / 0.3 thresholds across five
aspect ratios, which is far more tolerant than a centre-distance calculation implies.

**Predicted that removing top-1 would recover about 21% of blocked trucks, and that top-1 plus dedup
fixes would give a "96% countable ceiling."** It recovered zero. The analysis measured what the
*pipeline* permits rather than what the *model emits*, and missed the internal NMS entirely.

The common thread in all three: reasoning from structure and asserting a number without running the
experiment. Each was caught by measurement, two of them only after being challenged.

---

## 5. Caveats on every number above

- Most figures in §3.2 and §3.4 come from **one scene, `Bellingham_01_20260425`, with 11 labelled
  vehicles** and TP counts between 3 and 10. Well inside the range where noise could produce the
  observed differences. The monotonic recall trend in §3.2 is more trustworthy than the rest because
  there is a structural mechanism behind it.
- That scene is in `warmup-v1`'s training set, so absolute values are inflated. Comparisons between
  configurations on the same scene and model are unaffected.
- **Precision is a lower bound project-wide.** Only clear, well-formed echoes were labelled, so a
  detection landing on a real but unlabelled truck counts as a false positive. On this scene only 2 of
  41 false positives had any labelled vehicle in frame at all.
- The F1 0.690 result in §2 comes from an offline sweep over stored detection records. It has not yet
  been reproduced through the live inference path.

---

## 6. What is now implemented

Previously hardcoded or available only offline, now parameters on `detect()`, the `/api/detect` body,
and the CLI:

| parameter | default | what it does |
|---|---|---|
| `kp_gate` | off | drop detections whose mean keypoint score is at or below this |
| `nms` | 0.5 | ROI NMS IoU inside each window; raise toward 0.9 to let nearby trucks coexist |
| `top1` | True | keep only the window's best detection |
| `dedup_px` | 32 (96 m) | cross-window suppression radius |
| `chip` | 64 | sliding window size; input resize follows at 3× |

Defaults reproduce the previous behaviour exactly, so existing numbers stay comparable.

---

## 7. Open questions

- ~~Does a small window plus the gate beat a large window plus the gate?~~ **Answered in §3.7: they
  compose. The window governs recall, the gate governs precision and count.** Still needs confirming
  across all 21 scenes.
- Does loosening internal NMS from 0.5 toward 0.9 recover the two-trucks-in-one-window case that
  filter 1 currently removes?
- All of the above needs the full 21-scene sweep rather than one sparse scene.
- The gate interacts with the labelling policy: it selects for crisp, well-formed echoes, which is
  exactly what the annotators marked. Some of its gain may be better agreement with the labels rather
  than better real-world accuracy. Densely labelling one scene would settle it and is the highest-value
  non-modelling task available.

---

_Related: [KEYPOINT_GATE.md](KEYPOINT_GATE.md) for the gate mechanism in depth,
[MULTI_VEHICLE_TARGETS.md](MULTI_VEHICLE_TARGETS.md) for the training-side counterpart, and
[NEXT_APPROACH.md](NEXT_APPROACH.md) for the priority order (note its head-to-head conclusion is stale:
it ranked warmup-v1 above adamiak-v2 from a grid that capped the gate at 4, while adamiak-v2's optimum
is 8.0)._
