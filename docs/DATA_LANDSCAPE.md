# Data landscape — truck-detection

_Snapshot for a research-lead update. As of 2026-07-13._

## One-line summary

Training labels scaled **~21× to 339 vehicles / 1,017 keypoints across 8 scenes**, format-validated and
training-ready; the moving-echo signal is confirmed detectable in the Surface-Reflectance imagery. Coverage
is volume-rich but still **~90% concentrated in the south-I-5 / Centralia corridor**, so the next labeling
should prioritize new corridors. Retraining the detector on the full set is the immediate next step; imagery
growth is paused until the Planet quota resets.

## Labels — 339 vehicles / 1,017 keypoints, 8 scenes

Up from 16 vehicles a few days ago. Each vehicle = 3 keypoints (blue → red → green, the moving-echo streak).
Labeled in the web Annotation Studio, exported to the training repo, and **format-validated**: exports
cleanly to COCO keypoints (339/339, 0 dropped), every vehicle a complete B/R/G triple, EPSG:32610.

| Scene | Corridor | Vehicles |
|---|---|---:|
| Tacoma-Centralia_02_20260602 | south I-5 | 101 |
| Tacoma-Centralia_01_20260429 | south I-5 | 94 |
| Centralia_01_20260511 | south I-5 | 58 |
| Centralia_02_20260511 | south I-5 | 54 |
| Ellensburg_01_20260504 | I-90 | 13 |
| Bellingham_01_20260425 | north I-5 | 11 |
| Stanwood_10_20260511 | north I-5 | 6 |
| EllensburgPreferredTest_01_20260530 | I-90 (test scene) | 2 |
| **Total** | | **339** |

## Geographic coverage (the caveat to flag)

- **~91% (307/339) is one region** — greater Centralia / Tacoma south-I-5 (4 scenes of essentially the same
  corridor).
- The remaining ~9% spans **3 more areas**: Ellensburg (I-90), Bellingham (north I-5), Stanwood (north I-5).
- Strong **volume**, still concentrated **diversity**. For cross-corridor generalization, the next labels are
  worth more spent on new locations than on more Centralia.

## Imagery on hand

- **8 labeled scenes** — 8-band uint16 Surface Reflectance, ~3 m/px, EPSG:32610, native grid; bundled with
  the labels (`data/active/imagery/`).
- **10 unlabeled scenes archived** and available for future labeling (`data/cold/imagery/`): Seattle (I-5)
  plus 9 more Stanwood dates.
- Corpus is **frozen** — Planet EDU quota is over budget, so no new scenes until it resets.

## What's been done with the data

- **Signal question answered.** The moving echo (color-separated B→R→G streak) **is visible in the SR
  product**; a trained Keypoint R-CNN detects real echoes. The earlier "signal may not survive SR" concern
  is resolved (that null came from a brightness detector finding static clutter, not from the absence of
  movers).
- **Detector.** Initial Keypoint R-CNN trained on the *earlier 15-vehicle* set — pipeline validated end to
  end. **Not yet retrained on the 339**; that is the immediate next step and should generalize substantially
  better.
- **Honest generalization check.** Leave-one-scene-out cross-validation on the small set showed cross-scene
  recall swinging 0→100% depending on which scenes were in training — i.e. data-limited, which the jump to
  339 directly addresses.

## Immediate next steps

1. **Retrain** the Keypoint R-CNN on the full 339 and re-run leave-one-scene-out CV for an updated,
   honest cross-scene number.
2. **Diversify labeling** toward under-represented corridors (Seattle, Bellingham, Stanwood, more I-90) over
   more Centralia.
3. **When the Planet quota resets**, prioritize new distinct freight corridors to broaden coverage.

_See [DATA.md](DATA.md) for the schema/file layout and [MODEL.md](MODEL.md) for detector results._
