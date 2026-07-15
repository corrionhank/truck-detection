# truck-detection

ML half of a **WSU Transportation Research Group** feasibility study: detect trucks in single-capture
**PlanetScope SuperDove (~3 m)** satellite imagery over Washington freight corridors, via the
**moving-echo** signal — a colour-separated **blue → red → green** streak a moving vehicle leaves because
the sensor captures its bands a fraction of a second apart. The reference method is a Keypoint R-CNN
(Adamiak et al. 2025).

> **The *training* approach is being rebuilt from scratch.** The training scripts live in
> [`archive/src/`](archive/) (recoverable reference). Everything else — data export, inference, the model
> registry + weights, and the full console — is active, and the existing models still run.

## Layout

- **`src/`** — the pipeline (active):
  - `export_coco.py` — labels + GeoTIFFs → COCO keypoints + 64×64 chips
  - `detect_scene.py` — sliding-window full-scene inference → detections in UTM
  - `model_registry.py` — build/load any registered model · `inspect_scene.py`, `make_road_mask.py` — utilities
- **`backend/`** — Flask API (`/api/dataset`, `/api/scenes`, `/api/models*`, `/api/detect`) for the console
- **`frontend/`** — React/Vite console (Dataset / Results / Models / Inference / Spec tabs)
- **`models/`** — the experiment log: `registry.json` + per-model methodology `cards/`
- **`archive/src/`** — retired **training** scripts (rebuilding) — see [archive/README.md](archive/README.md)
- **`docs/`** — see below
- **`data/active/`** — the current working set: `Annotations-RGB.gpkg` (**339 vehicles / 1017 keypoints**, the only data file tracked in git) + `imagery/` (the 8 labeled scenes) + `coco/` (generated chips)
- **`data/cold/`** — archive (gitignored): unlabeled scenes, old previews, old QGIS project
- **`weights/`, `outputs/`** — trained models + derived artifacts (gitignored; weights indexed by the registry)

## Docs

**Start here:** [CLAUDE.md](CLAUDE.md) (operating manual + hard rules) → [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (how the system fits together).

| File | What |
|---|---|
| [CLAUDE.md](CLAUDE.md) | Operating manual: repo map, how to run, the hard rules and gotchas |
| [CHANGELOG.md](CHANGELOG.md) | Running log of changes + findings (newest first), plus known issues |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | End-to-end system: the data-flow diagram + each component |
| [docs/CONTEXT.md](docs/CONTEXT.md) | Project seed: goal, the moving-echo signal, data on hand, open questions |
| [docs/DATA.md](docs/DATA.md) | Annotation schema, the join, what's exported/consumed, file map, current corpus |
| [docs/MODELING.md](docs/MODELING.md) | The detector: Adamiak replication reference + what we learned |
| [docs/REFINEMENT.md](docs/REFINEMENT.md) | Prioritized playbook for improving the detector (grounded in the papers) |
| [docs/HARDWARE.md](docs/HARDWARE.md) | Can this machine train the model? Smoke test + the CPU-only finding |
| [models/README.md](models/README.md) | The model registry / experiment lab; per-model cards in `models/cards/` |
| [archive/README.md](archive/README.md) | The retired **training** scripts: what they used, the F1 to beat, how to restore |

## Quickstart

```bash
python3 src/export_coco.py                         # labels + imagery -> data/active/coco/ (64×64 chips)
python3 src/detect_scene.py Tacoma-Centralia_02_20260602 --thresh 0.3   # full-scene inference
```

Web console: `python3 backend/server.py` (port 8787) + `cd frontend && npm run dev` (port 5173).

Rebuilding training? The old scripts + the reference setup (64×64 chips, leave-one-scene-out, F1 ≈ 0.50 to beat,
anchor ranges) are documented in [archive/README.md](archive/README.md) and [docs/MODELING.md](docs/MODELING.md).
