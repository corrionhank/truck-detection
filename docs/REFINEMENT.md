# Refining the model — a prioritized playbook

The open items for the **modeling rebuild**, ordered by return-on-effort. Grounded in this project's findings
([CHANGELOG.md](../CHANGELOG.md)) and the two reference papers (Adamiak 2025, Van Etten 2024). The old pipeline
these build on is archived — see [`../archive/README.md`](../archive/README.md) and
[MODELING.md](MODELING.md).

**Rule for all of it:** measure every change on the **same held-out scene** using leave-one-scene-out (never a
random split — it inflates the number), against the archived reference numbers to beat: **centered-chip recall
~0.97** and **full-scene F1 ≈ 0.50**. The old training/registration entry point (`train_model.py`, now in
`archive/src/`) is the template for the rebuilt one:

```
python3 <train entry> --id my-variant --name "…" \
  --train <scenes> --held Tacoma-Centralia_01_20260429 --anchors <…> --aug <…> --epochs <…>
```

---

## Tier 1 — cheap, high-value, no retraining

1. **Calibrate the threshold.** The active model's confidences cluster ~0.5, so thr 0.5 collapses recall to
   ~2% while thr 0.3 gives ~40%. Sweep 0.2–0.6 and pick the operating point by F1 *or* by count-error
   stability — not a fixed 0.5.
2. **Physics / geometry filter** (designed, not built). A real echo is **collinear**, ordered **blue→red→green
   with red in the middle**, with a near-constant `|B−R|/|R−G|` spacing ratio and a plausible length. Learn the
   valid envelope from the 339 real labels, then reject detections outside it. Removes the impossible
   detections (zig-zags, swapped ends) → precision up, no retraining. This is what Van Etten's ellipse-fit and
   Adamiak's keypoint-correction do implicitly.
3. **Fix the dedup radius.** `detect_scene.py` suppresses detections within `CHIP/2 = 96 m`. On a busy freeway
   real trucks are closer than that, so neighbours get merged and counted as missed — the dense-traffic
   failure. Drop to ~15–20 m.
4. **Keypoint correction (Adamiak).** Snap each predicted keypoint to the nearest local intensity peak
   (±~7.4 m). Sharpens localization post-hoc; also usable to clean the *labels* before training.

## Tier 2 — retraining, moderate effort

5. **Train the production model on all 339.** The active model is a 3-scene / 12-epoch throwaway. Using all 8
   scenes is the single biggest expected win. (Do this first among Tier 2.)
6. **Hard-negative mining.** Feed the road-brightness false positives back as explicit negatives and retrain —
   directly attacks the dominant FP mode.
7. **Evidence-based anchor choice.** Don't guess: train `default` vs `small` vs Adamiak's `4–48 px` on the same
   held-out scene and compare. (Adamiak swept ~120 runs and chose 4–48 px for the same 4–8 px signal.)
8. **Longer schedule.** 12 epochs is short. Adamiak trained ~2.5 h with Adam + ReduceLROnPlateau. Add a proper
   LR schedule and more epochs once on a GPU.
9. **Geometry-consistency loss.** Penalize non-collinear / mis-ordered *predicted* keypoints during training,
   so the model learns the constraint rather than only being filtered after.

## Tier 3 — data & architecture (bigger bets)

10. **Diversify labels.** ~91% of data is the Centralia corridor. New corridors (Bellingham, Stanwood, Seattle,
    more I-90) are worth far more than more Centralia — needed to prove *cross-corridor* transfer.
11. **Label densely.** Exhaustive labeling makes precision and count-error actually measurable (right now
    "false alarms" are a lower bound because scenes are partly labeled).
12. **Parametric keypoint head.** Predict `(center, angle, length)` and derive B/R/G from the fixed spacing
    ratio → collinearity and ordering are guaranteed *by construction* (a zig-zag becomes unrepresentable).
13. **Segmentation instead of detection** (Van Etten). ResNet+UNet on a mask, fit an ellipse per contour.
    Handles dense packing better than a box detector + NMS — the exact reason he chose it.
14. **Use more bands.** Only 3 of 8 bands (6/4/2) reach the model. Red-edge / NIR / yellow might add echo
    signal (Adamiak found "no relationship between keypoint difficulty and spectral channel" — room to test).
15. **Counts-based objective (Van Etten).** For supplementing WSDOT volume counters, a *stable, calibratable*
    count fraction matters more than per-truck recall. Optimize and report that.

---

## The metric discipline

- **Leave-one-scene-out**, always. A random vehicle split leaks same-scene cues and lies.
- Track **count-error stability across scenes** — the deployable signal, per Van Etten (counts within ~15%
  even when per-object F1 is low).
- Give every model a methodology card recording method + findings (the archived pipeline kept these in a
  registry + `cards/`, now under `archive/models/` — a good pattern to reinstate in the rebuild).

## Suggested order

Tier 1.2 (geometry filter) + 1.1 (threshold) + 1.3 (dedup) first — cheap precision/recall wins, no training.
Then Tier 2.5 (train on 339) as the next model. Then anchor sweep (2.7) to settle that debate with numbers.
