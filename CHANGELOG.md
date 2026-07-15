# Changelog

Running log of what changed and what we learned. Newest first. For the current data snapshot see
[docs/DATA.md](docs/DATA.md) §6; for the model reference see [docs/MODELING.md](docs/MODELING.md).

---

## 2026-07-15 — Narrow the archive to *training only* (console + inference restored)

The prior teardown over-reached: it archived the whole modeling stack, including the console pages and the
inference/registry path. Corrected to archive **only the training scripts**, keeping the rest usable.

- **Restored to active:** `src/detect_scene.py` (inference), `src/model_registry.py` + `models/` + `weights/`
  (the registry + trained models), the full backend (`/api/models*`, `/api/detect`, `/outputs`), and the
  **five-tab console** (Dataset / Results / Models / Inference / Spec). The existing models run again.
- **Kept archived (training experiments only):** `train_model.py`, `crossval_keypoint.py`,
  `train_keypoint_rcnn*.py`, `viz_heldout.py`, `infer_keypoints.py` — in `archive/src/`.
- **Decoupled inference from training:** `detect_scene.py` now builds its model graph via the self-contained
  `model_registry` instead of importing from `train_keypoint_rcnn`, so inference no longer depends on any
  archived script. Repointed the registry's `train.script` paths to `archive/src/`.
- Verified: console type-checks + builds, backend imports with all endpoints and preloads the active model,
  `detect_scene` imports with no training-code dependency.

---

## 2026-07-15 — Modeling teardown: rebuilding from scratch

Deliberate reset of the modeling path (the old pipeline worked but was built ahead of understanding). **Nothing
deleted** — retired to [`archive/`](archive/), fully recoverable ([`archive/README.md`](archive/README.md)).

- **Moved to `archive/`:** the training + inference + model code (`crossval_keypoint.py`, `train_model.py`,
  `detect_scene.py`, `infer_keypoints.py`, `model_registry.py`, `viz_heldout.py`, `train_keypoint_rcnn*.py`),
  the trained-model registry + cards (`models/`), the weights (`weights/`), and the model-capable console
  (copies of `backend/server.py` + `frontend/App.tsx`).
- **Left active (upstream of modeling, still correct):** annotation export / COCO conversion / chipping
  (`src/export_coco.py` + utilities), the data docs, the annotations, and the imagery.
- **Console → data-only:** backend now serves just `/api/dataset` + `/api/scenes`; the React console shows only
  Dataset + Spec. The model/registry/detect endpoints and Results/Models/Inference tabs were archived.
- **Reference to beat, recorded** in `archive/README.md`: 64×64 chips, leave-one-scene-out split, finetuned from
  COCO-pretrained, anchor ranges (small 8–128 / default 32–512), **full-scene F1 ≈ 0.50** (matches Van Etten).
- **Settled decision folded in:** inter-band timing is **per-scene ephemeris** (Adamiak Δt =
  (v_sat / (w_bands·d_GSD))⁻¹), not a fixed 800 ms — updated in `docs/CONTEXT.md`.
- **Left as open items for the rebuild** (not built): keypoint correction, the anchor sweep, a production model
  on all scenes, threshold calibration, the geometry filter, velocity. See `docs/REFINEMENT.md`.

---

## 2026-07-15 — Baseline model, per-model cards, docs consolidation

**Baseline model + training entry point**
- `src/train_model.py` — one command to train + evaluate + save + **register** a model (writes `weights/`,
  `models/registry.json`, and a card stub). The single entry point for the experiment lab.
- Trained `base-default` — the **vanilla baseline** (default anchors, no augmentation, no data engineering) on
  the same 3 Centralia scenes / held-out scene as the active model, as the zero-effort reference to measure
  refinements against. Centered-chip recall **0.968**, kp err 0.7 px. (Centered-chip ≫ full-scene; see below.)

**Per-model methodology cards** — every model now has `models/cards/<id>.md` (method, implementation, results,
findings), viewable via a "▸ Methodology" expander in the Models tab (`GET /api/models/<id>/card`).

**`docs/REFINEMENT.md`** — a prioritized, paper-grounded playbook for improving the detector (threshold
calibration, geometry filter, dedup radius, train-on-339, anchor sweep, segmentation, …).

**Documentation overhaul** — added [`CLAUDE.md`](CLAUDE.md) (operating manual + hard rules) and
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (system data-flow). Consolidated the docs, all refreshed:
merged `DATA_LANDSCAPE`→`DATA` §6, `APPROACHES` + `MODEL` lessons → `MODELING`, `MINIMAL_TRAINING_REPO` →
`HARDWARE`; folded `TODO`'s open sensor/velocity questions into `CONTEXT`; dropped `MODEL.md`,
`TRAINING_DATA_REQUEST.md`, and `LABELING_TOOL_CONTRACT.md` (the labeling-contract docs weren't needed).
Refreshed stale counts (15→8
scenes, 16→339 vehicles) across docs + the console Spec tab. Pinned `.claude/settings.local.json` in
`.gitignore` so a fresh clone honours it.

---

## 2026-07-14 — Web console, model registry, and the papers

**Model registry / experiment lab**
- `models/registry.json` — committed log of every model (config, metrics, free-text findings, status).
  Weights stay in `weights/` (gitignored); the log persists so trial-and-error history survives a clone.
- `src/model_registry.py` — builds/loads any model with its **own architecture** (small vs default anchors —
  a small-anchor model won't load into a default-anchor graph). Cached loader + read/update helpers.
- Backend: `GET /api/models`, set-active, update notes/status; `/api/detect` now takes a `model_id`.
- Seeded with the 3 real models and honest notes (including the "mistake found" writeups).

**Web console upgrade**
- **Models tab** — browse active/archived models, set active, archive, edit notes, run inference.
- **Results tab** — real CV + full-scene numbers, replacing the earlier *fabricated* training runs.
- **Dataset tab** — live 339-vehicle data from the backend (was mislabeled "50 vehicles" = echoes).
- **Inference tab** — model selector + richer stats: added **false alarms** and **count error** tiles
  (per Van Etten's counts-over-detection framing).

**Read the reference papers (Adamiak 2025, Van Etten 2024)** — grounded comparison:
- Our full-scene held-out **truck F1 ~0.50 matches Van Etten's PlanetScope truck F1 (0.49)** — reproducing
  the state of the art at ~half the labels, on CPU.
- Adamiak's best mAP is **0.59** (modest, not stellar) and he **excluded trucks** (we target them).
- Our **dense-traffic misses = both papers' central failure mode**; Van Etten chose segmentation partly to
  handle dense packing. Trucks are *easier* than cars in PlanetScope (both papers).
- Unlocked: the SuperDove **green–blue band delta ≈ 800 ms** (Van Etten) → velocity is now attemptable.

---

## 2026-07-13 — Scale-up, cross-validation, data reorg

**Labels scaled ~21× to 339 vehicles / 1017 keypoints across 8 scenes** (from 16). Coverage is ~91% the
Centralia / south-I-5 corridor — volume-rich but concentrated.

**First honest generalization numbers (leave-one-scene-out CV):**
- 15-vehicle model: cross-scene recall swung 0→100% depending on which scenes were held out (data-starved).
- 4-scene Centralia CV (~300 vehicles): **91% overall held-out recall, ~1 px keypoint error** — the data
  scale-up flipped it from "validates the pipeline" to "detects trucks on unseen scenes."
- Full-scene deployment (held-out Tacoma-Centralia_01): **40% recall / 68% precision** — recognizing a
  centered echo (91%) is much easier than *finding* echoes across a raw scene (40%).

**Data reorganized** into `data/active/` (hot: labels + 8 labeled scenes + generated chips) and
`data/cold/` (archive: 10 unlabeled scenes). Only `data/active/Annotations-RGB.gpkg` is tracked in git.
Deleted the superseded 16-vehicle labels; updated all code paths + `.gitignore`.

**New docs:** `DATA.md`, `DATA_LANDSCAPE.md`, `TRAINING_DATA_REQUEST.md`, `LABELING_TOOL_CONTRACT.md`,
`MINIMAL_TRAINING_REPO.md` (fresh-repo bootstrap + hardware smoke test).

**C++ inspector** (`tools/geoinspect/`, gitignored) built with GDAL/GEOS/PROJ — caught a real correctness
trap: a spatial (extent) join leaks Stanwood_08's labels onto Stanwood_01 (overlapping scenes); the pipeline
avoids it by matching on the `scene` text field.

---

## 2026-07-10 — Initial pipeline

- `export_coco.py` — join GeoPackage annotations + GeoTIFFs → COCO keypoints + 64×64 true-colour chips.
- `train_keypoint_rcnn.py` (+ `_jitter`) — fine-tune torchvision Keypoint R-CNN (3 keypoints B/R/G, 1 class).
- `detect_scene.py` — sliding-window full-scene inference → detections in UTM.
- Flask backend + React/Vite web console; pushed the repo to GitHub.
- **Signal confirmed:** the moving echo (colour-separated B→R→G streak) **is visible in the SR product** and
  a Keypoint R-CNN learns it. The earlier "SR may kill the echo" concern came from a brightness detector
  finding static clutter, not from the echo's absence.

---

## Known issues / open items

- **MPS (Apple GPU) diverges** — torchvision detection models explode to NaN within ~2 iterations on MPS;
  training is **CPU-only** (slow). Real speed needs a CUDA box. (See `docs/HARDWARE.md`.)
- **Impossible-geometry detections not yet filtered** — the model sometimes emits physically-impossible
  echoes (zig-zag keypoints, blue/green swapped, red not in the middle). A post-hoc geometry filter
  (collinearity + B→R→G order + spacing ratio, envelope learned from the 339 real labels) is designed but
  **not built**.
- **No production model on all 339 yet** — the current active model is a 3-scene / 12-epoch held-out
  *visualization* throwaway with low confidences (~0.5), so it undersells at threshold 0.5.
- **Threshold sensitivity** — the same model swings from ~2% recall (thr 0.5) to ~40% (thr 0.3); needs
  calibration or a counts-based eval.
- **Diversity gap** — ~91% of labels are one corridor (Centralia); cross-*corridor* transfer is unproven.
- **Velocity** — detection + keypoints work, but pixel-displacement → speed is not yet computed
  (band Δt ≈ 800 ms now known).
