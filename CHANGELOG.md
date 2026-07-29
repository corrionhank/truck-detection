# Changelog

Running log of what changed and what we learned. Newest first. For the current data snapshot see
[docs/DATA.md](docs/DATA.md) §6; for the model reference see [docs/MODELING.md](docs/MODELING.md).

---

## 2026-07-29 — Per-scene CRS: data → 789/21, first arid corridors (Tri-Cities, Spokane)

- **Dropped the project-wide EPSG:32610 requirement from the training path.** The join only ever needed a
  scene's points to agree with **its own** raster; "all of it must be 32610" was a generalization that WA's
  geography (UTM 10/11 seam at 120°W) was always going to break. `export_coco.py` now reprojects points into
  **each raster's native CRS** before the inverse affine — rasters are still never reprojected (resampling
  smears the 1–3 px echo), points are, which is exact. Imagery from anywhere is now ingestible.
- **Replaced the importer's CRS-equality gate with a functional per-scene check** (`align_to_imagery`): do the
  points land inside their raster? That also catches a **mis-stamped CRS** — correct coordinates, wrong CRS
  label — which an equality test reads as fine. Mis-stamps are re-stamped losslessly and reported.
- **Imported `Annotations-New-7-29` → 789 vehicles / 21 scenes** (was 629/19). Both new scenes arrived
  **mis-stamped** (EPSG:32610 label, EPSG:32611 values) and would previously have been silently excluded:
  **`Tri-Cities-New_01` (105 veh)** — the first arid corridor big enough to be a *measurable* spatial hold-out —
  and **`20260622_184442_88_2560_edit` (55 veh)**, a Spokane capture shipped under a raw Planet id. Largest
  corridor share now **35.6 %** (was 45 % at 629, 91 % at 339).
- **Fixed `verify_labels.py`**, which still assumed 64 px chips: it resized the 96 px chip to 512 (5.33×) while
  scaling keypoints by 8, drawing every marker at 1.5× its true position. Read as a data error, was a drawing
  bug. Scale is now derived from the chip; the `kp > 64` out-of-frame test was stale for the same reason.
- Verified: 629/629 pre-existing chips **byte-identical** across both the code change and the re-import, exactly
  160 added, 0 removed; new-scene keypoints centered at (48.0, 48.0) with collinearity 0.0–0.4 px and 0/24
  auto-flags, echoes confirmed by overlay.
- **Pre-training fixes in `train_detector.py`** (the two confounds from the `jitter-mv` post-mortem):
  **`--warmup`** (default 300 iters, linear lr/100 → lr, covering the smoke iters) so the fresh detection heads
  can be trained at the full 1e-3 instead of the flat 1e-4 that undertrained `jitter-mv`; and **best-val
  checkpointing** — the lowest-val-loss epoch is kept and restored at the end (`jitter-mv` shipped epoch 12 at
  val 4.955 when epoch 11 was 4.715), persisted through resume. Registry now records `warmup_iters`,
  `best_val_epoch`, `best_val_loss`, `jitter_px`.

## 2026-07-26 — Trained `jitter-mv` (translation jitter + multi-vehicle chips) + model comparison

- **New refinements built:** padded-chip **translation jitter** (`export_coco.py --margin`; 96 px chips
  random-cropped to 64 px each epoch, legal offset derived from the keypoints so it never cuts them) and
  **multi-vehicle chip targets** (a neighbour echo in the window is now a labelled positive, not
  trained-as-background — 404 neighbour annotations). New tools: `src/neighbor_diagnostic.py`,
  `src/verify_labels.py` (label-on-echo overlay).
- **Spatial-overlap guard** in `train_detector.py` (warns + labels held scenes *temporal* vs *spatial*; the
  name-based split can't see it) + `--exclude` (drop scenes from training without holding them out) + `--jitter`.
- **Trained `kprcnn-jitter-mv`** — 466 veh / 14 scenes, lr 1e-4 (after the fresh heads spiked at 1e-3), 4 held-out
  scenes. Result: **recall-rich, precision-poorer** — TC_01 F1 0.49 vs `adamiak-v2` 0.54 (thr 0.3). But precision
  is a lower bound (partial labels — most "false positives" are real unlabeled / small-vehicle echoes), and the LR
  drop is a confound, so `adamiak-v2` stays the best *validated* model while jitter-mv's recall gain is likely
  real. Full write-up: [docs/MODEL_COMPARISON.md](docs/MODEL_COMPARISON.md).
- **Confidence ≈ quality filter:** labels only mark clear/well-formed echoes, so the score ranks by echo quality;
  raising the threshold roughly separates trucks from cars/noise. The clean version is a non-DL echo-size /
  geometry gate — the next lever to build.

## 2026-07-26 — Data → 629/19, spatial-overlap finding, docs consolidated

- **Imported the `trg-echo-exchange` bundle → 629 vehicles / 19 scenes** (was 538/16; +Yakima-Toppenish_04,
  blaine-bellingham_03, polygon_01; existing 16 replaced with upstream-corrected labels). gpkg committed.
- **Ran a leakage-free eval matrix** (`eval_matrix.py`): failure splits by corridor and the axis is background
  **clutter**, not terrain. Active `adamiak-all` is a **broken run** (0.06 on its own trained scenes — trained on
  all scenes with no held-out, so the LR scheduler never annealed).
- **Finding: every multi-scene corridor is the same footprint re-captured on different dates** (measured on valid
  footprint, not bounds). Name-based leave-one-scene-out never produced spatial hold-outs → distinguish *temporal*
  (later captures of a known corridor) from *spatial* (new corridor) generalization; a spatial-overlap guard is
  needed.
- **Docs consolidated:** the session's working docs merged into [docs/PROJECT_STATE.md](docs/PROJECT_STATE.md);
  the active plan is [docs/NEXT_MODEL_PLAN.md](docs/NEXT_MODEL_PLAN.md); superseded drafts moved to `docs/archive/`.
- New tools: `src/verify_labels.py` (label-on-echo overlay check), `eval_matrix.py` (model×scene matrix).

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
