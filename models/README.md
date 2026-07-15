# Model registry — the experiment lab

A place to **store, compare, and look back on** every model you train — active or archived — with its config,
results, and findings, so trial-and-error is tracked instead of lost.

- **`registry.json`** — the log (committed to git). One entry per model: how it was trained, its metrics, and
  free-text **notes** (findings, mistakes, next steps). `active` names the model the Inference tab runs.
- **`../weights/*.pt`** — the actual weights (gitignored; large). Referenced by `weights` in each entry.
- **`../src/model_registry.py`** — builds/loads any entry with the **correct architecture** (each model stores
  its own anchor set; a small-anchor model won't load into a default-anchor graph).

Because only `registry.json` is committed, a fresh clone keeps the full history of what you tried and learned,
even though the weights themselves aren't in git — retrain from the recorded `train.script` + config to
regenerate any model.

## Using it (Models tab)

The web console's **Models** tab lists every model. You can:
- **Set active** — the Inference tab then runs that model.
- **Archive / Unarchive** — keep old attempts without cluttering the active choice.
- **Edit notes** — click the notes box; record findings, mistakes, ideas.
- **Run inference** — jump straight to running the active model on a scene.

Or edit `registry.json` directly.

## Adding a model

1. Train and save weights to `../weights/<name>.pt` (any training script; see `train.script` for examples).
2. Add an entry to `registry.json`:

```json
{
  "id": "short-unique-id",
  "name": "Human-readable name",
  "weights": "<name>.pt",
  "status": "active",
  "created": "YYYY-MM-DD",
  "arch": { "backbone": "resnet50-fpn", "anchors": "small|default", "classes": 2, "keypoints": 3, "min_size": 192, "max_size": 320 },
  "train": { "vehicles": 0, "scenes": [], "epochs": 0, "aug": "…", "device": "cpu", "script": "src/…py" },
  "metrics": { "heldout_recall": 0.0, "heldout_precision": 0.0 },
  "notes": "What you tried, what worked, what didn't, what's next."
}
```

**`arch.anchors` must match how it was trained** (`small` = 8–128 px, `default` = 32–512 px) — otherwise the
weights load into the wrong graph. `metrics` values that are numbers show as tiles in the UI; strings are kept
but not tiled.

## Field reference

| Field | Meaning |
|---|---|
| `id` | stable unique key (used in URLs / the `active` pointer) |
| `status` | `active` or `archived` (display/filtering only; `active` pointer is separate) |
| `arch.anchors` | anchor set — **must match training** |
| `train.script` | the script that produced it (your "model code" link) |
| `metrics` | results; numeric ones render as tiles |
| `notes` | free-text findings — the point of the lab |
