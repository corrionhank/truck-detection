# The keypoint-score gate — mechanism, evidence, implementation, caveats

A postprocessing filter for the moving-echo detector. Written as a self-contained handoff: it assumes no
context from the conversation that produced it. Measured 2026-08-03 on two complete inference runs
(`kprcnn-warmup-v1`, `kprcnn-adamiak-v2`) over 24 scenes / 20,950 detections.

---

## 1. One-sentence version

The Keypoint R-CNN emits a *per-keypoint* confidence for each of the three keypoints it predicts; the inference
code discards those values, and they separate real detections from false ones **better than the box score the
code actually uses** — so filtering on them raises F1 substantially with no retraining.

---

## 2. Where the numbers come from (the mechanism)

The model returns, for every detection, a dict with four relevant keys:

| key | shape | what it is | used today? |
|---|---|---|---|
| `scores` | `(N,)` | **box** classifier confidence: "is this region a moving_echo vs background" | ✅ yes |
| `keypoints` | `(N, 3, 3)` | x, y, visibility per keypoint (blue, red, green) | ✅ yes (x,y only) |
| `keypoints_scores` | `(N, 3)` | **per-keypoint** confidence | ❌ **discarded** |
| `boxes` | `(N, 4)` | the detection box | ❌ discarded |

### What `keypoints_scores` actually is

From `torchvision/models/detection/roi_heads.py`, `heatmaps_to_keypoints()`:

```python
pos = roi_map.reshape(num_keypoints, -1).argmax(dim=1)      # peak location per keypoint heatmap
end_scores[i, :] = roi_map[arange(num_keypoints), y_int, x_int]   # the heatmap VALUE at that peak
```

The keypoint head produces one **heatmap per keypoint**. `keypoints` is the *argmax location* of each heatmap;
`keypoints_scores` is the **raw logit value at that peak**. It is:

- **unbounded** — not a probability, no softmax, no sigmoid. Observed range in our data ≈ **−4 to +8**.
- **uncalibrated** — only the *ordering* is meaningful. "3.4" does not mean 3.4 of anything.
- **a peakedness measure** — a sharp, confident heatmap gives a high value; a flat, diffuse heatmap gives a low
  or negative one.

### Why it carries information the box score does not

The two scores answer **different questions**, produced by **different heads**:

- **Box score** (box classification head): *"does this region look like the class 'moving echo'?"* — a
  region-level, appearance-level judgement.
- **Keypoint scores** (keypoint head): *"can I localise three distinct band positions inside this region?"* — a
  structural judgement about the echo's internal geometry.

A patch of bright field texture, lane paint, or a rooftop can be **region-level echo-like** enough to fool the
box head, while producing **flat keypoint heatmaps** because there is no crisp blue → red → green triple to lock
onto. That is precisely the failure mode our worst corridors exhibit, and it is exactly what the keypoint scores
detect and the box score misses.

Physically: a real moving echo is three spatially separated, colour-separated blobs. The keypoint head is the
only part of the network being asked to find that structure, so its confidence is the closest thing the model
produces to "is this the actual physical signature?"

---

## 3. The empirical evidence

Measured over all window-top-1 detections on the 21 labelled scenes: **1,195 matched** (within 18 m of a hand
label) vs **7,430 unmatched**.

### Separability — rank-sum AUC per feature (0.5 = useless, 1.0 = perfect)

| feature | AUC | matched median | unmatched median |
|---|---:|---:|---:|
| **`kp_score_mean`** | **0.891** | **3.45** | **−0.45** |
| `score` (the box score in use today) | 0.771 | 0.663 | 0.212 |
| `streak_len_m` | 0.758 | 33.8 | 27.1 |
| `collinearity_px` (lower = better) | 0.675 | 0.254 | 0.528 |
| `box_area_px` | 0.615 | 146 | 129 |
| `spacing_ratio` (lower = better) | 0.606 | 0.699 | 0.754 |
| `box_h_px` / `box_w_px` | 0.588 / 0.541 | — | — |

The distributions barely overlap: matched detections sit around **+3.4**, unmatched around **−0.45**.

### Effect on F1 — `kprcnn-warmup-v1`, pooled over labelled scenes, on top of score ≥ 0.3

| filter | detections kept | precision | recall | **F1** |
|---|---:|---:|---:|---:|
| baseline (score ≥ 0.3) | 2252 | 0.210 | 0.598 | **0.310** |
| + `kp_score_mean > 0` | 1606 | 0.287 | 0.584 | 0.385 |
| + `kp_score_mean > 1` | 1292 | 0.346 | 0.567 | 0.430 |
| **+ `kp_score_mean > 2`** | — | **0.43** | **0.52** | **0.470** |
| + collinearity < 1 px (geometry only) | 1863 | 0.243 | 0.574 | 0.342 |
| + streak > 25 m (geometry only) | 1692 | 0.264 | 0.567 | 0.360 |
| + all three combined | 1143 | 0.364 | 0.527 | 0.431 |

**Precision roughly doubles while recall drops only ~0.08.** Geometry gates help far less, and add nothing once
the keypoint gate is applied — the keypoint score already encodes what they measure.

### The effect is model-dependent — this is important

| model | F1 without gate | F1 with `kp>2` | gain |
|---|---:|---:|---:|
| `kprcnn-warmup-v1` | 0.310 | 0.476 | **+0.166** |
| `kprcnn-adamiak-v2` | 0.417 | 0.428 | **+0.012** |

The gate is **not** a universal improvement. It removes a specific failure mode — *confident boxes with weak
keypoints* — that `warmup-v1` has in abundance and `adamiak-v2` largely does not. Any new model must be
re-measured; do not assume the gain transfers.

---

## 4. Implementation

Current code, `src/detect_scene.py` (~line 131):

```python
outs = model(imgs)
for (x0, y0), out in zip(chunk, outs):
    if not len(out["scores"]):
        continue
    s = float(out["scores"][0])
    if s < thresh:
        continue
    kp = out["keypoints"][0].numpy()[:, :2] + np.array([x0, y0])
    dets.append((s, kp))
```

With the gate:

```python
outs = model(imgs)
for (x0, y0), out in zip(chunk, outs):
    if not len(out["scores"]):
        continue
    s = float(out["scores"][0])
    if s < thresh:
        continue
    ks = out["keypoints_scores"][0].numpy()      # (3,) — blue, red, green
    if ks.mean() <= kp_gate:                     # kp_gate ≈ 2.0, MUST be tuned per model
        continue
    kp = out["keypoints"][0].numpy()[:, :2] + np.array([x0, y0])
    dets.append((s, kp))
```

Notes for whoever implements it:

- Expose `kp_gate` as a CLI/API parameter (default `None` = disabled) so existing numbers stay reproducible.
- Record it in the registry entry alongside the confidence threshold — it is part of the operating point.
- `min()` across the three keypoints instead of `mean()` is worth testing: it demands *all three* bands be
  confident, which is closer to the physical claim. Untested.
- The gate composes with the confidence threshold; it does not replace it. Best measured combination so far was
  `score ≥ 0.3` **and** `kp_score_mean > 2`.

---

## 5. Caveats — read before trusting this

1. **The threshold is empirical, not principled.** `> 2` was chosen by scanning this dataset. The scores are
   unbounded and uncalibrated, so the value has no meaning outside this model. Re-tune per model.
2. **It was tuned and evaluated on the same data.** There is no held-out validation of the *gate threshold*
   itself, so some of the gain is optimistic. Tuning it on one scene set and testing on another is the honest
   next step.
3. **Precision here is a lower bound.** Scenes are labelled only for clear, well-formed truck echoes; blurry,
   malformed and very small echoes are deliberately skipped, and manual inspection confirmed many "false
   positives" are real unlabelled or small vehicles. **So the gate may partly be removing real detections that
   simply are not labelled.**
   - Interpreted charitably, this is *desirable*: the labelling policy is "clear, well-formed echo," the keypoint
     score measures exactly that, so the gate aligns model output with the labelling policy — and for a
     **truck**-counting deliverable, biasing toward large clean echoes is the right bias.
   - Interpreted strictly: it improves *agreement with labels*, which is not identical to improving *accuracy*.
     A densely-labelled scene would settle this.
4. **It is postprocessing, not a model improvement.** It changes what you keep, not what the model can see. It
   cannot recover a truck the model never detected.

---

## 6. Related finding worth carrying along

With top-1-per-window and cross-window dedup **disabled**, and threshold 0, **80/80 labelled trucks on
`Tacoma-Centralia_01` have a detection within 18 m — a 100 % recall ceiling** (from 931 raw detections). The RPN
and window grid find every truck. Everything lost between that ceiling and the deployed ~0.45 recall is discarded
by postprocessing: top-1-per-window, the 96 m dedup radius, and the score threshold. The keypoint gate is one
lever in that same postprocessing layer, and the layer as a whole is where the remaining headroom is.

---

_Source: `outputs/inference_collect/{kprcnn-warmup-v1,kprcnn-adamiak-v2}/detections.csv`. Analysis dashboard:
the inference-analysis artifact. Related: [PROJECT_STATE.md](PROJECT_STATE.md), [MODEL_COMPARISON.md](MODEL_COMPARISON.md)._
