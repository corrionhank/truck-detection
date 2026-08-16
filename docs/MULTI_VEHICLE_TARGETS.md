# Multi-vehicle training targets: what works, what is broken, and how to fix it

Written 2026-08-15. Self-contained: it assumes no context beyond the repo.

**Short version.** The pipeline already trains on chips containing more than one truck, and 44% of
training chips carry a second vehicle. But roughly a quarter of those second vehicles are silently
discarded from the target every epoch and therefore taught to the model as background. The cause is
the same one-line mistake in two places: a vehicle is only kept if **all three** of its keypoints
land inside the frame, so any truck straddling the edge disappears. torchvision already supports the
correct behaviour; the pipeline just is not using it.

---

## 1. Why more than one vehicle per chip matters

Training chips are 64 pixels, which is 192 metres of ground at 3 m per pixel. On a busy freeway that
is not a small area, so a chip cut around one truck very often contains others.

If those others are not labelled, they are not merely ignored. Detection training treats anything not
covered by a target box as background, so an unlabelled truck sitting in the chip becomes a worked
example of what a truck is not. The model is taught to suppress exactly the thing it is meant to find.

This is a bigger problem here than in ordinary object detection because the deliverable is a **count**.
A detector that reliably finds one truck per group and ignores its neighbours can look respectable on
per-image metrics while systematically under-counting the dense corridors that carry the freight.

## 2. What the pipeline already does

`src/export_coco.py` cuts one chip per labelled vehicle, then adds an annotation for every *other*
labelled vehicle that also falls inside that chip. Each annotation is tagged `center: true/false` so
the trainer knows which vehicle the chip was cut around.

`src/train_detector.py` turns those into a proper multi-instance target. `to_target()` builds N boxes,
N class labels and N keypoint triples, and torchvision's Keypoint R-CNN handles multiple instances per
image natively. Nothing about the architecture limits this.

Current scale of it, straight from `data/active/coco/annotations.json`:

| | count |
|---|---|
| chips | 789 |
| annotations | 1,286 |
| of which neighbours | **497** |
| chips carrying at least one neighbour | 346 (**43.9%**) |

Distribution: 443 chips hold one vehicle, 219 hold two, 103 hold three, 24 hold four.

**This differs between the two trained models.** `kprcnn-warmup-v1` was trained with multi-vehicle
targets. `kprcnn-adamiak-v2` was created on 2026-07-20 and the feature landed on 2026-07-26 in commit
`750b1f7`, so adamiak-v2 was trained with one annotation per chip and every neighbouring truck taught
as background. That is worth remembering when comparing the two: it is one of several differences
between them, and it is not recorded in the registry's `aug` string, which reads identically for both.

## 3. The bug

A vehicle is kept only if every one of its three keypoints is inside the frame. The rule appears twice,
at two different stages, and both use `all()`.

At export time, deciding whether a neighbour becomes an annotation at all
(`src/export_coco.py`, in `main()`):

```python
if w != vid and all(x0 <= c < x0 + export and y0 <= ro < y0 + export
                    for c, ro in veh_px[w]):
    members.append(w)
```

At training time, deciding which annotations survive the random jitter crop from the 96 px export down
to the 64 px model chip (`src/train_detector.py:231`, in `ChipDS.__getitem__`):

```python
keep = [v for v in kps2 if (v >= 0).all() and (v < CHIP).all()]   # vehicles surviving the crop
```

A truck with two keypoints in frame and one just past the edge fails both tests. It is not marked as
uncertain or excluded from the loss. It is simply absent from the target, which in detection training
means the model is told that patch is background.

## 4. How much it costs

Measured over the current export by classifying every neighbour annotation against the 64 px crop:

| crop offset | neighbours fully in frame | **partly in frame, dropped** | fully outside |
|---|---:|---:|---:|
| centre (jitter off) | 334 | **114 — 22.9%** | 49 |
| 8 px shift | 298 | **131 — 26.4%** | 68 |
| 16 px shift (max jitter) | 260 | **90 — 18.1%** | 147 |

So between 18% and 26% of the multi-vehicle annotations are cancelled out on any given epoch, and
because the jitter offset is redrawn every epoch, a given neighbour flips between "labelled truck" and
"labelled background" across epochs. The model receives contradictory supervision for the same patch of
ground.

## 5. Why it matters more than the number suggests

At inference the detector slides a 64 px window on a 40 px stride, so windows overlap by 24 px and every
truck appears in roughly four of them. It is centred in at most one. In the others it sits off-centre,
and near window edges it is partially cut.

Partially-visible trucks are therefore not a rare edge case at inference; they are the normal case. And
they are precisely the case the training data has been teaching the model to reject. Training and
deployment disagree about what a valid detection looks like.

## 6. The fix

torchvision already implements the correct behaviour. Its keypoint loss computes, in
`torchvision/models/detection/roi_heads.py`:

```python
valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
vis = keypoints[..., 2] > 0
valid = (valid_loc & vis).long()
```

and the loss is a cross-entropy over `valid` entries only. An off-frame keypoint is excluded from the
keypoint loss automatically, while the **box still counts as a positive instance**. That is exactly the
partially-visible object case, and it is what the visibility flag exists for.

Three changes:

1. **`export_coco.py`** — include a neighbour when *any* of its keypoints falls in the export window,
   not all three. Change `all(...)` to `any(...)` in the member selection.
2. **`train_detector.py`, `ChipDS.__getitem__`** — keep a vehicle when *any* keypoint survives the crop.
   Change the `keep` comprehension accordingly.
3. **`train_detector.py`, `to_target()`** — clip each box to the chip bounds and set visibility 0 for
   keypoints outside it, leaving their coordinates alone.

Point 3 has a trap worth calling out. `to_target()` currently does:

```python
kps[:, :, 0] = np.clip(kps[:, :, 0], 1, CHIP - 2)
kps[:, :, 1] = np.clip(kps[:, :, 1], 1, CHIP - 2)
```

which is harmless today, because every kept vehicle is fully in frame and the clip is a no-op. The
moment partially-visible vehicles are kept, that clamp would fabricate a keypoint position pinned to the
chip edge and train the model to predict it there. The clamp must be removed along with the visibility
change, or the fix will do more harm than the bug.

Everything is currently written with visibility hardcoded to 2 (`labelled and visible`) in both
`export_coco._annotation` and `to_target`, so the flag is available and simply unused.

## 7. Two caveats before acting

**It requires retraining.** Unlike the inference-side work (duplicate-suppression radius, top-1 removal,
the keypoint gate), this changes what the model learns, so it only takes effect on the next training run
and cannot be evaluated against existing weights.

**It will not show up in the metrics on its own.** `src/detect_scene.py` keeps only the highest-scoring
detection per window. A model that correctly finds two trucks in one window still reports one, so the
benefit is invisible until top-1-per-window is removed. These two changes need to land together, or the
training fix will look like it did nothing. See [NEXT_APPROACH.md](NEXT_APPROACH.md) for the inference
side.

## 8. A related lever, deliberately not conflated

Widening the export margin (currently 16 px, giving a 96 px export around a 64 px chip) would pull more
neighbours into frame and raise the neighbour count. That is a separate change with its own trade-offs,
mainly more jitter range and more disk. It is worth testing, but not at the same time as this fix, or
neither result will be interpretable.

Shrinking the model chip is the opposite lever and would make this worse: at a 32 px export the
neighbour count falls from roughly 490 to 86, discarding most of the multi-vehicle signal.

## 9. Reproducing the measurement

```python
import json, numpy as np, collections
a = json.load(open('data/active/coco/annotations.json'))
CHIP, MARGIN = a['info']['chip_px'], a['info']['margin_px']
per_img = collections.defaultdict(list)
for an in a['annotations']:
    k = np.array(an['keypoints'], float).reshape(3, 3)[:, :2]
    per_img[an['image_id']].append((k, an.get('center', True)))

def classify(k, ox, oy):
    ins = [(ox <= x < ox + CHIP and oy <= y < oy + CHIP) for x, y in k]
    return 'full' if all(ins) else ('partial' if any(ins) else 'out')

c = collections.Counter()
for pts in per_img.values():
    for k, is_centre in pts:
        if not is_centre:
            c[classify(k, MARGIN, MARGIN)] += 1
print(c)   # centre crop: full 334, partial 114, out 49
```

---

_Related: [NEXT_APPROACH.md](NEXT_APPROACH.md) for the inference-side fixes this must ship with, and
[MODELING.md](MODELING.md) for the reference architecture._
