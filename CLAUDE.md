# CLAUDE.md — operating manual for this repo

Rules and context for anyone (human or AI agent) working in **truck-detection**. Read this first.
It is deliberately short; the deep docs live in [`docs/`](docs/) and are indexed in the [README](README.md).

> **Current state: the *training* approach is being rebuilt from scratch.** The training scripts were moved
> intact to [`archive/src/`](archive/) (see [`archive/README.md`](archive/README.md)) — deliberately, to rebuild
> at an understood pace. **Everything else is active:** the data pipeline, **inference** (`detect_scene`), the
> **model registry** + trained weights, and the **full web console**. The existing models still run from the
> console; they're the reference baseline the rebuild aims to beat. Don't resurrect the archived *training* code
> into the active path unless asked — consult it for reference.

---

## What this repo is

The **ML half** of a WSU Transportation Research Group feasibility study: detect **trucks** in single-capture
**PlanetScope SuperDove (~3 m)** satellite imagery over Washington freight corridors.

The signal is the **moving echo** — because the sensor captures its colour bands a fraction of a second apart, a
moving vehicle leaves a colour-separated **blue → red → green** streak. The model is a **Keypoint R-CNN**
predicting 3 keypoints (blue/red/green) for 1 class, following Adamiak et al. 2025.

---

## Repo map — where things live

| Path | What |
|---|---|
| `src/` | **active** pipeline: `import_data.py` (ingest dropped data / exchange bundles), `export_bundle.py` (reference exchange-bundle exporter), `export_coco.py` (labels + GeoTIFFs → COCO + 64×64 chips), `train_detector.py` (train + register a model), `detect_scene.py` (full-scene inference), `model_registry.py` (build/load any registered model), `inspect_scene.py`, `make_road_mask.py` |
| `backend/server.py` | Flask API (port **8787**): `/api/dataset`, `/api/scenes`, `/api/models*`, `/api/detect` |
| `frontend/` | React + Vite console (port **5173**): Dataset / Results / Models / Inference / Spec |
| `models/` | the experiment log: `registry.json` (every trained model + metrics + notes) + methodology `cards/` |
| `weights/` | trained weights (`*.pt`) — **gitignored** (226 MB each), referenced by the registry |
| `data/inbox/` | **drop zone** for new imagery + annotation sets (gitignored) → `import_data.py` ingests it |
| `data/active/` | hot set: `Annotations-RGB.gpkg` (labels, **the only data file in git**) + `imagery/` + `coco/` |
| `data/cold/` | archived unlabeled scenes + old QGIS project — gitignored |
| **`archive/src/`** | **retired training scripts** (`train_model`, `crossval_keypoint`, `train_keypoint_rcnn*`, `viz_heldout`, `infer_keypoints`) — rebuilding |
| `docs/` | context, data schema, model reference, refinement playbook — see README index |

---

## How to run

```bash
# import new data: drop .tif + .gpkg into data/inbox/, then
python3 src/import_data.py --dry-run         # validate the join; then drop --dry-run to ingest + rebuild chips
python3 src/export_coco.py                   # (or regenerate 64×64 chips directly from labels + imagery)
python3 src/train_detector.py --id <id> --name "<name>" --held <scene>   # train + register a model (CPU)
python3 src/detect_scene.py <scene> --thresh 0.3   # full-scene inference with the default weights
python3 backend/server.py                    # data + model API on :8787
cd frontend && npm install && npm run dev     # console on :5173
```

Rebuilding training? The old entry points + the reference setup are in `archive/src/` — see
[`archive/README.md`](archive/README.md) and [`docs/MODELING.md`](docs/MODELING.md).

---

## Hard rules — these will bite you if ignored

**Data + inference (active — always apply):**
1. **Join labels to imagery on the `scene` text field, NOT a spatial/extent join.** Some scenes overlap (e.g.
   Stanwood_01 / Stanwood_08); an extent join leaks one scene's labels onto another. `export_coco.py` already
   does this correctly — preserve it.
2. **Only `data/active/Annotations-RGB.gpkg` is tracked in git.** Imagery is Planet EDU-licensed; weights are
   huge. Never `git add` imagery, `data/*/coco/`, `outputs/`, `weights/`, or the zip.
3. **Anchors must match at load time.** `small` = (8,16,32,64,128), `default` = (32,64,128,256,512). Each registry
   entry stores its anchor set; a small-anchor checkpoint won't load into a default-anchor graph. `model_registry`
   handles this — pass the registry `arch`, don't hardcode.

**For the training rebuild (carry these over from what we learned):**
4. **Training is CPU-only.** torchvision detection models diverge to NaN within ~2 iterations on Apple MPS. Real
   speed needs a CUDA box — see [`docs/HARDWARE.md`](docs/HARDWARE.md).
5. **Evaluate with leave-one-scene-out, never a random split** — a random vehicle split leaks same-scene cues and
   inflates the number.
6. **Don't conflate the two recall metrics** — *centered-chip recall* (one 64 px chip per vehicle; easy, ~0.97)
   vs *full-scene* recall/precision/F1 (find echoes across a raw scene; the real metric, old best F1 ≈ 0.50).
7. **The right anchor size is an open question** — settle it with an evidence-based sweep, don't guess. Adamiak's
   swept region was 4–48 px.

---

## Open items for the rebuild (deliberately not built)

Keypoint correction · the anchor sweep · a production model on all scenes · threshold calibration · the
geometry/physics filter · velocity (per-scene ephemeris Δt — see [`docs/CONTEXT.md`](docs/CONTEXT.md)).
Playbook: [`docs/REFINEMENT.md`](docs/REFINEMENT.md). Reference method: [`docs/MODELING.md`](docs/MODELING.md).

## Conventions

- **When you change behavior, update [`CHANGELOG.md`](CHANGELOG.md)** (newest first) and, if a model changed, its
  card in `models/cards/`.
- Match the surrounding code's style; keep docs terse and high-signal (this project avoids clutter).
- Our full-scene truck F1 (~0.50) reproduced Van Etten 2024's PlanetScope truck F1 (0.49) — the bar the rebuild
  should match, then beat.

## Doc index

Start: this file → [README.md](README.md) → [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (how the system fits
together, and what's archived). Then the [README docs table](README.md#docs).
