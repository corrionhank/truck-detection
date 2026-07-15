# CLAUDE.md — operating manual for this repo

Rules and context for anyone (human or AI agent) working in **truck-detection**. Read this first.
It is deliberately short; the deep docs live in [`docs/`](docs/) and are indexed in the [README](README.md).

> **Current state: the modeling path is being rebuilt from scratch.** The old training / inference / model code
> was moved intact to [`archive/`](archive/) (see [`archive/README.md`](archive/README.md)) — deliberately, so
> it can be rebuilt at an understood pace. **The data pipeline upstream of modeling is active and correct.**
> Do not resurrect the archived modeling code into the active path unless asked; consult it for reference.

---

## What this repo is

The **ML half** of a WSU Transportation Research Group feasibility study: detect **trucks** in single-capture
**PlanetScope SuperDove (~3 m)** satellite imagery over Washington freight corridors.

The signal is the **moving echo** — because the sensor captures its colour bands a fraction of a second apart, a
moving vehicle leaves a colour-separated **blue → red → green** streak. The reference method (to rebuild) is a
**Keypoint R-CNN** predicting 3 keypoints (blue/red/green) for 1 class, following Adamiak et al. 2025.

---

## Repo map — where things live

| Path | What |
|---|---|
| `src/` | **data pipeline** (active): `export_coco.py` (labels + GeoTIFFs → COCO + 64×64 chips), `inspect_scene.py`, `make_road_mask.py` |
| `backend/server.py` | Flask API — **data-only** now (`/api/dataset`, `/api/scenes`), port **8787** |
| `frontend/` | React + Vite console — **data-only** (Dataset + Spec tabs), port **5173** |
| `data/active/` | hot set: `Annotations-RGB.gpkg` (labels, **the only data file in git**) + `imagery/` + `coco/` |
| `data/cold/` | archived unlabeled scenes + old QGIS project — gitignored |
| `docs/` | context, data schema, model reference, refinement playbook — see README index |
| **`archive/`** | **retired modeling code** (training, inference, model registry, weights, model-capable console) — recoverable reference, out of the active path |

---

## How to run (active)

```bash
# Regenerate training chips from labels + imagery
python3 src/export_coco.py

# Data console (two terminals)
python3 backend/server.py                 # data API on :8787
cd frontend && npm install && npm run dev  # UI on :5173
```

Rebuilding the model? The old entry points and the reference setup are in `archive/` — see
[`archive/README.md`](archive/README.md) and [`docs/MODELING.md`](docs/MODELING.md).

---

## Hard rules — these will bite you if ignored

**Data pipeline (active — always apply):**
1. **Join labels to imagery on the `scene` text field, NOT a spatial/extent join.** Some scenes overlap (e.g.
   Stanwood_01 / Stanwood_08); an extent join leaks one scene's labels onto another. `export_coco.py` already
   does this correctly — preserve it.
2. **Only `data/active/Annotations-RGB.gpkg` is tracked in git.** Imagery is Planet EDU-licensed; weights are
   huge. Never `git add` imagery, `data/*/coco/`, `outputs/`, the zip, or anything under `archive/weights/`.

**For the modeling rebuild (carry these over from what we learned):**
3. **Training is CPU-only.** torchvision detection models diverge to NaN within ~2 iterations on Apple MPS. Real
   speed needs a CUDA box — see [`docs/HARDWARE.md`](docs/HARDWARE.md).
4. **Evaluate with leave-one-scene-out, never a random split** — a random vehicle split leaks same-scene cues and
   inflates the number.
5. **Don't conflate the two recall metrics** — *centered-chip recall* (one 64 px chip per vehicle; easy, ~0.97)
   vs *full-scene* recall/precision/F1 (find echoes across a raw scene; the real metric, old best F1 ≈ 0.50).
6. **Anchors must match at load time**, and the right size is an **open question** — settle it with an
   evidence-based sweep on a held-out scene, don't guess. `small` = (8,16,32,64,128), `default` =
   (32,64,128,256,512); Adamiak's swept region was 4–48 px.

---

## Open items for the rebuild (deliberately not built)

Keypoint correction · the anchor sweep · a production model on all scenes · threshold calibration · the
geometry/physics filter · velocity (per-scene ephemeris Δt — see [`docs/CONTEXT.md`](docs/CONTEXT.md)).
Playbook: [`docs/REFINEMENT.md`](docs/REFINEMENT.md). Reference method: [`docs/MODELING.md`](docs/MODELING.md).

## Conventions

- **When you change behavior, update [`CHANGELOG.md`](CHANGELOG.md)** (newest first).
- Match the surrounding code's style; keep docs terse and high-signal (this project avoids clutter).
- Our (old) full-scene truck F1 (~0.50) reproduced Van Etten 2024's PlanetScope truck F1 (0.49) — the bar the
  rebuild should aim to match, then beat.

## Doc index

Start: this file → [README.md](README.md) → [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (how the system fits
together, and what's archived). Then the [README docs table](README.md#docs).
