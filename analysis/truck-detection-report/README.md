# Truck detection — model comparison (R project)

Self-contained. Open `truck-detection-report.Rproj` in RStudio and knit either report.
Each is a single file — data loading, helpers, figures and tables all inline.

```r
rmarkdown::render("methodology.Rmd")         # how the detector works -> methodology.html
rmarkdown::render("report.Rmd")              # model comparison       -> report.html
rmarkdown::render("threshold_analysis.Rmd")  # operating point        -> threshold_analysis.html
```

**`methodology.Rmd`** — the method itself: the physical signal, imagery and annotation inputs, preprocessing,
model architecture, training, inference, the complete output schema, and how metrics are defined.
No results — this is the "how it works" document.

**`report.Rmd`** — the A-vs-B comparison: per-config summary statistics, score distributions,
per-scene winner, counting behaviour, failure modes.

**`threshold_analysis.Rmd`** — where each model should actually run: a 0.30–0.975 sweep, F1 optima,
count-parity points, and why raising the threshold does not fix over-counting.

## Layout

```
methodology.Rmd               how the detector works (inputs -> model -> outputs)
report.Rmd                    model comparison
threshold_analysis.Rmd        operating-point analysis
data/  results.csv            252 rows — 21 scenes x 2 models x 6 thresholds
       threshold_sweep.csv     56 rows — 28 thresholds x 2 models (0.30-0.975)
       aggregates.csv           9 rows — model x threshold x provenance group
       scenes.csv              24 rows — scene metadata
       detections_*.csv        per-detection output, one file per model (~13 MB total)
       manifest_*.json         inference parameters per model
report.html                   knitted output
```

## Notes

- `MODEL_A` = `kprcnn-adamiak-v2`, `MODEL_B` = `kprcnn-warmup-v1`; recoded to readable names on load.
- `pooled()` = **micro**: sum TP/FP/FN across scenes then score (weights by vehicle count).
  Mean of per-scene F1 = **macro**, weights each scene equally. They differ materially — see §4.
- `spread()` gives SD / IQR / range / CV of per-scene F1, for consistency questions.
- `load_detections()` column-subsets on demand; those CSVs are ~13 MB.
- Precision is a lower bound throughout — scenes are only partially labelled.

Full column definitions and evaluation caveats: `docs/EVIDENCE_PACK.md` in the parent repo.
