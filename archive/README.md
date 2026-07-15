# archive/ — retired training code (cold storage)

The old **training** scripts, moved here **intact** during a deliberate rebuild: the plan is to rebuild the
training approach from scratch, at a slower, understood pace. **Nothing was deleted — everything here is
recoverable.**

**What stayed active** (not archived): the data pipeline (`src/export_coco.py` + utilities), **inference**
(`src/detect_scene.py`), the **model registry** (`src/model_registry.py` + `models/` + `weights/`), and the
**full web console** (backend + all five tabs). The existing trained models still load and run from the
console — they're the reference baseline the rebuild aims to beat.

## What the archived training setup used (the reference to beat)

- **Input:** one **64×64** true-colour chip per vehicle (bands 6/4/2 = R/G/B, 2–98 % per-scene stretch to
  8-bit), model resized via `min_size=192 / max_size=320`.
- **Model:** Keypoint R-CNN (ResNet-50 + FPN), 1 object class + 3 keypoints (blue/red/green), **finetuned from
  COCO-pretrained weights** (not from scratch — we had hundreds of labels, not Adamiak's 3,236).
- **Split:** **leave-one-scene-out** cross-validation (never a random vehicle split — it leaks same-scene cues
  and inflates the number).
- **Anchor ranges tried:** `small = (8, 16, 32, 64, 128)` and `default = (32, 64, 128, 256, 512)` px. Neither is
  Adamiak's swept winner region (**4–48 px**) — the evidence-based anchor sweep was never run.
- **Realistic result to beat:** full-scene held-out **F1 ≈ 0.50** (recall 0.40 / precision 0.68), matching Van
  Etten 2024's PlanetScope truck F1 (0.49). Centered-chip recall was easy (~0.97); *finding* echoes across a raw
  scene is the hard, deployable number — beat the **0.50 full-scene F1**.
- **Training:** CPU only (torchvision detection ops diverge to NaN on Apple MPS), ~12 epochs, Adam/SGD, batch 4.

Full per-model detail (config, metrics, findings) lives in the **active** `models/registry.json`; methodology
write-ups are in `models/cards/`.

## What's in here (training scripts only)

| Path | What |
|---|---|
| `src/train_model.py` | train + evaluate + register a model (the old entry point) |
| `src/crossval_keypoint.py` | leave-one-scene-out CV; `build_model` / `train_fold` / `eval_scene`; `ANCHOR_SETS` |
| `src/viz_heldout.py` | held-out training + visualization (produced the active model) |
| `src/train_keypoint_rcnn.py`, `_jitter.py` | the original v0 / jitter-augmented v1 training scripts |
| `src/infer_keypoints.py` | chip-level eval + montage (a training-time dev tool, not the deployment path) |

## Open items left for the rebuild (deliberately NOT built)

Keypoint correction · the anchor sweep · a production model on all scenes · threshold calibration · the
geometry/physics filter · velocity. See [`../docs/REFINEMENT.md`](../docs/REFINEMENT.md) for the playbook and
[`../docs/MODELING.md`](../docs/MODELING.md) for the Adamiak replication reference.

## Restoring a training script

```bash
mv archive/src/crossval_keypoint.py src/     # move it back into the active tree
```

Note: active inference (`src/detect_scene.py`) builds its model graph via `src/model_registry.py`, so it does
**not** depend on anything in here. Everything ran on CPU; see [`../docs/HARDWARE.md`](../docs/HARDWARE.md).
