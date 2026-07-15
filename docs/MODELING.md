# Modeling — the detector, the reference method, and what we learned

The model reference for the moving-echo detector: the method we replicate (Adamiak 2025), the annotation →
COCO pipeline it needs, and the lessons from our own runs. For *how to improve* the current model see
[REFINEMENT.md](REFINEMENT.md). The old model code + results are archived (deliberate rebuild) — per-model
methodology cards in [`../archive/models/cards/`](../archive/models/) and the
[registry](../archive/models/registry.json); see [`../archive/README.md`](../archive/README.md).

---

## The task

Per-truck detection in single-capture SuperDove SR imagery over freight corridors. At minimum a point/box +
count; ideally **3 keypoints (blue, red, green)** per truck to recover heading + speed from the inter-band
displacement. The echo detects any mover, but trucks leave the largest streak (2–6 px vs a car's ~1 px), so
they are both the freight-relevant and the most-detectable target.

> **Signal-visibility is settled.** An earlier gate — "does the echo survive in the SR product at 3 m?" — is
> resolved: hand-labeled B→R→G streaks across Bellingham / Centralia / Stanwood SR scenes are clear, and a
> Keypoint R-CNN trained on them detects real echoes. The earlier Bellingham morphology null was a brightness
> detector finding static clutter (lane paint), not the absence of movers. SR's atmospheric correction is
> radiometric, not geometric, so it does not co-register the bands away.

---

## Reference method — Adamiak et al. 2025 (Keypoint R-CNN)

Source: Adamiak et al. 2025, *Int. J. Applied Earth Obs. Geoinformation* 142:104707. The method we replicate.

**Architecture**
- **Keypoint R-CNN** (built on Mask R-CNN), backbone **ResNet-50 + FPN**.
- **2 classes** (moving echo + background), **3 keypoints** per vehicle in temporal order **blue → red → green**.
- They chose Keypoint R-CNN over alternatives (e.g. CenterNet) purely for reproducibility (it ships in
  torchvision) — not a performance claim.
- Trained **from scratch** on **3,236 echoes** (726 images, 80/10/10 split → 502/77/147).

**Anchors — the knob that matters for tiny objects**
- Custom `AnchorGenerator`; **anchor sizes swept: 4, 8, 16, 32, 48 px**; aspect ratios 0.25–1.25.
- Small anchors are what make 1–2 px objects detectable at medium resolution. They ran **~120 runs** mostly
  sweeping anchors, scored by RMSE on a fixed val subset, **locked the winner, then ran all CV folds.**
  → *Sweep anchors first, lock, then train for real.*

**Optimizer / schedule** — Adam; ReduceLROnPlateau on val loss; LR 1e-3 → 1e-5; grad-clip 1.5; inputs 512×512;
augmentation = rotations, H/V flips, brightness, perspective; ~2.5 h/model on a 3090Ti/1080/A4000.

**Keypoint correction (Algorithm 1)** — applied twice, to clean labels before training and refine predictions
after: per band find local intensity peaks (`skimage.h_maxima`, h = 0.02 over a ~11 m neighborhood), build a
k-d tree, and snap each keypoint to the nearest peak within ~7.4 m, else keep the original.

**Evaluation** — primary **mAP** over OKS 0.5–0.95 with a customized per-keypoint OKS constant (reflects the
pixel distance among the 3 connected keypoints, to handle fast vehicles' larger gaps); TP requires keypoint
score > 0.7. Secondary: RMSE on trajectory length (a speed proxy). Adamiak's best mAP was **0.59** — and he
**excluded trucks**; we target them.

**torchvision skeleton**
```python
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator

model = keypointrcnn_resnet50_fpn(weights=None, num_classes=2, num_keypoints=3)
anchor_sizes  = ((4,), (8,), (16,), (32,), (48,))          # small = key for tiny vehicles
aspect_ratios = ((0.25, 0.5, 0.75, 1.0, 1.25),) * len(anchor_sizes)
model.rpn.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
# then: Adam, ReduceLROnPlateau (1e-3→1e-5), grad-clip 1.5, 512×512 inputs.
```

**Our deliberate deviation:** Adamiak trained from scratch on 3,236 echoes; with far fewer labels we
**start from COCO-pretrained weights and finetune** (a pretrained backbone learns from small data far better),
then swap the box head → 1 class and the keypoint head → 3 keypoints. Replication order: (1) get it training
on pretrained weights, (2) sweep anchors, (3) lock the best, train for real.

---

## Annotation → COCO (what training consumes)

Full schema in [DATA.md](DATA.md). Model-relevant essentials:

- **Annotate on the GeoTIFF, never the PNG** — PNGs have no CRS, so keypoints would land in arbitrary image
  space. Load the SR `.tif`, style bands R6/G4/B2 with a 2–98 % stretch for the same true-colour view *with*
  georeferencing.
- **Point layer (GeoPackage):** fields `vehicle_id` (groups a vehicle's 3 points) and `sequence` (1/2/3 =
  blue/red/green capture order). One class; digitize the **streak**, not a dot.
- **Join labels to imagery on the `scene` text field, never a spatial/extent join** — overlapping scenes
  would leak labels across dates. Map world coords → pixel coords via the raster's affine when chipping.

---

## What we learned (our runs)

- **CPU only.** torchvision detection models diverge to NaN on Apple MPS within ~2 iterations (RoIAlign /
  anchor math). CPU is the only correct path today; real speed needs CUDA. See [HARDWARE.md](HARDWARE.md).
- **Center-overfit → jitter fix.** The first model was trained on pixel-centered crops, so it learned "echo at
  dead centre" and a coarse sliding window never centred a truck tightly enough (0/3 on a held-out scene).
  Training on **jittered crops** (vehicle offset ±20 px + flips) fixed it (3/3, scores 0.9+ vs 0.4). This is
  now the standard augmentation.
- **Two metrics, never conflated.** *Centered-chip recall* (one 64 px chip per labeled vehicle — "do you
  recognise it?") is easy: the baseline scores ~0.97. *Full-scene* recall/precision/F1 (find echoes across a
  raw scene) is the deployable task and much harder: the active model scores ~0.40 / 0.68 / 0.50. Always say
  which one. Cross-scene recall is measured with **leave-one-scene-out** CV — never a random split, which
  leaks same-scene cues and inflates the number.
- **Where we stand vs the papers.** Our full-scene truck F1 (~0.50) reproduces Van Etten 2024's PlanetScope
  truck F1 (0.49) at ~half the labels, on CPU. Dense-traffic misses are both papers' central failure mode
  (Van Etten chose segmentation partly to handle dense packing). Trucks are *easier* than cars in PlanetScope.
- **Precision is a lower bound.** Scenes are only partially labeled, so many "false positives" are real but
  unlabeled trucks. Dense labeling is required to score precision honestly.

## Alternatives considered

- **Morphology / white top-hat** — tried; a misleading null. It finds *bright* blobs (lane paint), not
  colour-separated displacement. Measured static clutter and correctly found it static.
- **YOLO small-object detection** — possible for plain counting, but it doesn't natively recover the B/R/G
  keypoints needed for velocity. Only if the goal collapses to counts.
- **Segmentation (Van Etten 2024)** — ResNet+UNet mask + per-contour ellipse fit; handles dense packing better
  than a box detector + NMS. A candidate architecture change, see [REFINEMENT.md](REFINEMENT.md).
- **Road segmentation for corridor masking** — demoted; the clips are already corridor-masked (nodata outside
  the OSM buffer), so `src/make_road_mask.py` is an optional helper, not on the critical path.
