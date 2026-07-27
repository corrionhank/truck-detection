# data/inbox-scenes/ — drop imagery here to run inference on it

This is the **inference-only** drop zone. Put new GeoTIFFs here (or a `.zip` of them), run the importer, and the
scenes become available in the console's **Inference** tab — for running trained models on new ground. It does
**not** touch your annotations or training set.

## What to drop

- **GeoTIFFs** — the same **8-band SuperDove Surface-Reflectance** product the models expect (they read bands
  6/4/2 → R/G/B). Loose files, a folder, or a `.zip` / inference bundle all work.
- **No annotations needed** — this path is purely for inference. (For labeled training data, use `data/inbox/`
  and `import_data.py` instead.)

## Then run

```bash
python3 src/import_scenes.py --dry-run   # validate + report, change nothing
python3 src/import_scenes.py             # move imagery into active/imagery -> Inference tab
```

## Notes

- **Any CRS is fine here** (not just EPSG:32610). Inference doesn't join labels to pixels, so a scene in any UTM
  zone works — detections come back in the tif's own coordinates. (The *training* importer, `import_data.py`,
  does require 32610 because it maps annotation points to pixels.)
- Imported scenes read **"UNSEEN"** for every trained model (they were never trained on), so metrics are absent
  and evaluation is qualitative — eyeball the detection montage.
- Existing models and training data are untouched; this only adds imagery to run against.

_Everything in this folder is gitignored except this README._
