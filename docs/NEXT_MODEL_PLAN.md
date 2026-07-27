# Pre-Training Fixes, Final

Supersedes all prior drafts. Import is done, `polygon_01` is resolved, build order is settled.

## Data state

629 vehicles / 19 scenes on the working tree. Verify with:

```bash
python3 -c "import json; d=json.load(open('data/active/coco/annotations.json')); \
print(len(d['images']),'chips /', len(set(i['scene'] for i in d['images'])),'scenes')"
```

`polygon_01` is a hand-drawn AOI over Kent/Auburn, ~19 vehicles, **zero footprint overlap with anything else**
(verified: 0.0% vs auburn, Centralia, and all others). Not scratch data. It gets its own row in the split table.

## Step 0, pre-work

Two of these stop being possible once the export rewrite lands.

1. **Hash the chip directory.** Baseline for the `--margin 0` check in Fix 2.
   `find data/active/coco/images -name '*.png' | sort | xargs sha256sum > /tmp/chips-64-baseline.sha256`
2. **Check label registration by eye.** The handshake proves counts, and counts were never the risk. Annotations
   were corrected upstream, which is exactly when a Y-flip lands, and it passes every existing gate. Open 15
   random chips across three-plus corridors including a new scene. Chips are vehicle-centered, so the echo should
   sit at the middle of each. If not, stop. — **DONE** via `src/verify_labels.py` (36 chips / 6 corridors incl.
   new scenes, 0 flags, echoes centered; re-run with `python3 src/verify_labels.py`).
3. **Commit the gpkg.** It is the only reproducible record of 629/19. Everything else is gitignored, and a clean
   checkout reverts to 538/16.

## Overlap data

Valid-data footprint IoU, not bounding boxes:

| pair | IoU | days apart |
|---|---|---|
| Yakima_01 vs _03 / _04 | 99.5% / 100% | 42 / 48 |
| Tacoma-Centralia_01 vs _02 | 97.5% | 34 |
| Centralia_01 vs _02 | 21% | 0 |
| auburn_01 vs _03 | 59% | 50 |
| polygon_01 vs all | 0% | n/a |

Name-based leave-one-scene-out never produced spatial hold-outs. But the overlapping scores sit at or below the
clean ones (v2's auburn 0.67 under `centralia-heldout`'s 0.71, overlapping Yakima still 0.19), so this is
mislabeling rather than inflation. Two generalizations, and each split measures one of them: **temporal** means
later captures of a corridor it has seen, which is the WSDOT per-corridor-calibration condition and not a leak;
**spatial** means transfer to an unlabeled corridor, which needs the whole corridor held out.

## Fix 1, neighbor diagnostic

Runs first. Its result decides the Fix 2 scope, and both touch target construction.

Export writes one vehicle per chip, so a second vehicle in the window is trained as background. Count per scene
in **two bands**, since jitter changes exposure and not just the count: **within 16 px**, in every crop, and
**16 to 48 px**, in some epochs. Old fixed reach was 32. Expect Centralia densest. Report, do not fix. High
counts mean multi-vehicle chips ship inside the Fix 2 rewrite so `export_coco.py` is touched once.

Also dump acquisition date per scene. **Confirmed: all 19 dates fall between April 9 and June 20 (72-day span,
all Apr–Jun)** — `Ellensburg-Yakima_01` (Apr 9) is the early edge, `Yakima-Toppenish_04` (Jun 20) the late one.
One spring window, so clutter exposure is a single-season crop-stage/sun-angle range. That is a stated limitation
for the write-up.

**RESULT (2026-07-26, `src/neighbor_diagnostic.py`):** real and Centralia-concentrated. Overall **16% of chips
carry a neighbor ≤16 px (every crop), 39% at 16–48 px (some epochs)**; on the dense Centralia scenes ≤16 px is
**18–31%** and 16–48 px **52–61%** (TC_01 31/61, C_02 29/61, TC_02 24/52). Sparse corridors (Yakima/Ellensburg)
~0–8%. **→ multi-vehicle chip targets ship inside the Fix 2 export rewrite.**

## Fix 2, padded-chip jitter

**Export.** `--margin` on `export_coco.py`, default 16, exporting `chip + 2*margin` = 96 px with keypoints and
bbox in export space, `chip_px` and `margin_px` recorded. **`--margin 0` must be byte-identical** to the Step 0
baseline.

**Train.** Random 64 px crop, resampled every epoch, offset subtracted from keypoints. Derive the legal offset
from where the keypoints sit, clamped to `[margin - jitter, margin + jitter]`, so it can never cut a keypoint and
never needs to resample. Edge-clamped vehicles narrow automatically. `jitter=0` is the exact center crop, the
ablation control. Seed the RNG per worker. Run `augment()` on the 96 **before** the crop so rotation pulls real
pixels into the corners.

**Blast radius.** `chip_px` in COCO `info` is write-only, safe to redefine. The real dependency is hardcoded
`CHIP, HALF = 64, 32` in `train_detector.py`, driving `augment()` and `to_target()`, which must work in 96-space
with the crop mapping to 64 afterward. Two centered-recall paths need a 96 to 64 center-crop:
`eval_matrix.centered_recall` and `train_detector.eval_centered`. `server.py` is clean.

## Fix 3, spatial guard and split

Build the guard first, then validate the split with it. The existing guard checks **names only**; add a
**footprint-overlap** check that WARNS + labels (not hard-reject) when a held-out scene overlaps a trained scene —
`TC_01` overlaps `TC_02` at 97.5%, so its number is **same-ground-later-date (temporal), not generalization.**

**`--held`: `Centralia_01`, `Tacoma-Centralia_01`, `EllensburgPreferredTest_01`, `Stanwood_10`. Exclude
`polygon_01` from `--train`.** 144 held, **466 trained, Centralia 31%**.

| corridor (in training) | veh | share |
|---|---|---|
| Centralia (TC_02 + C_02) | 145 | 31% |
| Yakima (all 4) | 126 | 27% |
| blaine-bellingham (+Bellingham_01) | 108 | 23% |
| auburn (both, back in training) | 66 | 14% |
| Ellensburg (_01 + Ellensburg-Yakima_01) | 21 | 5% |
| polygon_01 | excluded | — |

Held-out reads: `TC_01` / `Centralia_01` = **temporal** (overlap trained `TC_02`/`C_02`) → the primary
measurement, **`TC_01` vs v2's 0.54**. `Stanwood_10` (6 veh, only Stanwood scene) = spatial but **too small to be
reliable**; `EllensburgPreferredTest_01` (2 veh) = noise. So the spatial signal is weak this run — the temporal
`TC_01` read is the point. Keep all Yakima + auburn in training; measure them in later leave-one-corridor-out runs.

**Record split type per held scene in the registry entry** (temporal vs spatial), not just the scene list, so
`eval_matrix` badges reflect what each number means.

Fallback if 427 feels thin: auburn-only plus a weighted sampler at alpha 0.5, Centralia 36% on 563. The old
objection that a sampler muddies the Yakima read is moot. The 427 plan is preferred, being code-free.

## Notes, not tasks

Active pointer is on `adamiak-all`, 0.06 on its own trained scenes, so console output is currently the worst
model in the registry. Console renders at 0.5 while every reported F1 was computed at 0.3. Set it to 0.3 before
the Yakima false-positive eyeball, and run that eyeball on `adamiak-v2`.

## Build order

0. Pre-work: verify, hash, 15-chip eyeball (done), commit gpkg.
1. Fix 1 diagnostic, two bands plus dates. Report only.
2. Fix 2, single `export_coco.py` rewrite (plus multi-vehicle if warranted), then train-side crop and both
   centered-recall center-crops.
3. Fix 3, guard then `--held`, with split type recorded.
4. Train: pretrained backbone, anchors 4 to 48, jitter on, **no `--set-active`**.
5. Score in `eval_matrix`. Read auburn against 0.71 and `TC_01` against 0.54. Keep the old outputs as baseline.

Post-inference work stays out. Top-1-per-window, threshold sweep, geometry filters and road masking all move the
operating point, so any of them before scoring breaks the comparison.
