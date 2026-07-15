# archive/ — retired modeling pipeline (cold storage)

The old modeling / training / inference code, the trained models, and the model-capable console, moved here
**intact** during a deliberate rebuild. The data pipeline (annotation export, COCO conversion, chipping) and
the data docs stay in the active repo; only the modeling path was retired, to be rebuilt from scratch at a
slower, understood pace. **Nothing was deleted — everything here is recoverable.**

## What the archived setup used (the reference to beat)

- **Input:** one **64×64** true-colour chip per vehicle (bands 6/4/2 = R/G/B, 2–98 % per-scene stretch to
  8-bit), model resized via `min_size=192 / max_size=320`.
- **Model:** Keypoint R-CNN (ResNet-50 + FPN), 1 object class + 3 keypoints (blue/red/green), **finetuned from
  COCO-pretrained weights** (`weights="DEFAULT"`), box head → 1 class, keypoint head → 3 keypoints. (Not
  trained from scratch — we had hundreds of labels, not Adamiak's 3,236.)
- **Split:** **leave-one-scene-out** cross-validation (never a random vehicle split — that leaks same-scene
  cues and inflates the number).
- **Anchor ranges tried:** `small = (8, 16, 32, 64, 128)` px and `default = (32, 64, 128, 256, 512)` px.
  Neither is Adamiak's swept winner region (**4–48 px**) — the evidence-based anchor sweep was never run.
- **Realistic result to beat:** full-scene held-out **F1 ≈ 0.50** (recall 0.40 / precision 0.68), which matches
  Van Etten 2024's PlanetScope truck F1 (0.49). Centered-chip recall was easy (~0.97); *finding* echoes across a
  raw scene is the hard, deployable number — beat the **0.50 full-scene F1**, not the centered number.
- **Training:** CPU only (torchvision detection ops diverge to NaN on Apple MPS), ~12 epochs, Adam/SGD, batch 4.

**Best model:** `kprcnn-centralia-heldout` — 213 vehicles (3 Centralia scenes), small anchors, rich aug, 12
epochs, held out `Tacoma-Centralia_01_20260429` → the F1 0.50 above. Full per-model detail (config, metrics,
free-text findings) is in `models/registry.json`; methodology write-ups are in `models/cards/`.

## What's in here

| Path | What |
|---|---|
| `src/train_model.py` | train + evaluate + register a model (the old entry point) |
| `src/crossval_keypoint.py` | leave-one-scene-out CV; `build_model` / `train_fold` / `eval_scene`; `ANCHOR_SETS` |
| `src/detect_scene.py` | sliding-window full-scene inference + NMS |
| `src/infer_keypoints.py` | chip-level inference / eval + montage |
| `src/model_registry.py` | load/build any registered model with its anchor set |
| `src/viz_heldout.py` | held-out visualization (produced the active model) |
| `src/train_keypoint_rcnn.py`, `_jitter.py` | the original v0 / jitter-augmented v1 training scripts |
| `models/registry.json`, `models/cards/` | the experiment log + per-model methodology cards |
| `weights/*.pt` | the trained weights (gitignored — on disk, recoverable, not in git) |
| `backend/server.py` | the model-capable backend (registry + `/api/detect` + `/outputs`) |
| `frontend/App.tsx` | the model-capable console (Results / Models / Inference tabs) |

## Open items left for the rebuild (deliberately NOT built)

Keypoint correction · the anchor sweep · a production model on all scenes · threshold calibration · the
geometry/physics filter · velocity. See [`../docs/REFINEMENT.md`](../docs/REFINEMENT.md) for the playbook and
[`../docs/MODELING.md`](../docs/MODELING.md) for the Adamiak replication reference.

## Restoring (if you want a piece back)

Move the file(s) back into the active tree and reinstate the endpoints:

```bash
# a training/inference script
mv archive/src/crossval_keypoint.py src/
# the whole model registry + weights
mv archive/models models && mv archive/weights weights
# the model-capable backend / console (overwrites the data-only ones)
cp archive/backend/server.py backend/server.py
cp archive/frontend/App.tsx  frontend/src/App.tsx
```

Everything ran on CPU; see [`../docs/HARDWARE.md`](../docs/HARDWARE.md) for the smoke test.
