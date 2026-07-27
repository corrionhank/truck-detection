#!/usr/bin/env python3
"""
eval_matrix.py — long-running model x scene evaluation matrix (READ-ONLY; never trains).

Runs each selected model over every loaded scene through the NORMAL inference path
(detect_scene.detect: bands 6/4/2, per-scene 2-98 stretch, /255, 64px chips, resize 192,
each model's own stored anchors via model_registry, current top-1-per-window behavior).

  labeled scene   -> full scoring (precision / recall / F1 / centered-recall vs hand labels)
  unlabeled scene -> count-only (no ground truth)

Crash-survivable:
  - appends each (model,scene) row to the CSV the moment it finishes (flush+fsync)
  - on restart, skips pairs already in the CSV (resume)
  - per-evaluation try/except: logs the traceback and continues
  - times the first eval and logs an extrapolated total

Outputs (outputs/eval_matrix/):
  eval_matrix.csv   one row per model-scene pair
  eval_matrix.log   timestamps, per-eval progress, tracebacks
  summary.md        the markdown summary
  eval_matrix.png   formatted results figure

Run:      python3 eval_matrix.py
Reports:  python3 eval_matrix.py --report-only     # regenerate summary.md + png from the CSV
"""
import os
import sys
import csv
import json
import time
import gc
import datetime
import traceback
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import rasterio
import torch
from PIL import Image

import model_registry as mr
import detect_scene as ds
from export_coco import RED, GREEN, BLUE  # 6,4,2

# ----------------------------------------------------------------- config ----
OUT = REPO / "outputs" / "eval_matrix"
OUT.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT / "eval_matrix.csv"
LOG_PATH = OUT / "eval_matrix.log"
SUMMARY_PATH = OUT / "summary.md"
PNG_PATH = OUT / "eval_matrix.png"
GEO = REPO / "data" / "active" / "imagery"
COCO = REPO / "data" / "active" / "coco"
GSD_M = 3.0            # SuperDove ground sample distance
STRIDE = 40           # detect_scene default (top-1-per-window kept as-is)
MATCH_PX = 6          # detect_scene ground-truth match radius

# The 4 models + one-line rationale (logged before the run).
PICKS = [
    ("kprcnn-adamiak-all",
     "DEPLOYMENT model (active): best-known config, 4-48 anchors + aug, trained on all 16 scenes."),
    ("kprcnn-adamiak-v2",
     "Best generalization read: 3 properly held-out scenes incl. 2 novel corridors (auburn-snoqualmie, Yakima)."),
    ("kprcnn-centralia-heldout",
     "Methodology variation: 'small' 8-128 anchors + archived pipeline, held-out Tacoma-Centralia_01; the F1~0.50 benchmark."),
]
EXCLUDED = [
    ("base-default", "skipped per user request — 3 models is sufficient (would have been the no-aug baseline)"),
    ("kprcnn-adamiak-v1", "near-duplicate of v2 (same anchors/aug, fewer scenes, 1 held-out)"),
    ("kprcnn-3", "archived 'Training & Inference Test', F1 0.10 -> near-broken"),
    ("kprcnn-echo-v0", "15-vehicle overfit demo, 0.0 full-scene recall -> throwaway"),
    ("kprcnn-echo-jitter", "15-vehicle early experiment -> throwaway"),
]
MODEL_IDS = [m for m, _ in PICKS]

FIELDS = ["model_id", "scene", "corridor", "split_status", "threshold", "row_type",
          "true_count", "pred_count", "count_ratio", "precision", "recall", "f1",
          "centered_recall", "valid_km2", "det_per_km2", "eval_seconds", "timestamp"]

SHORT = {
    "kprcnn-adamiak-all": "adamiak-all (deploy)",
    "kprcnn-adamiak-v2": "adamiak-v2",
    "kprcnn-centralia-heldout": "centralia (small-anch)",
    "base-default": "base (default,no-aug)",
}


def log(msg):
    line = f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def corridor_of(scene):
    s = scene.lower()
    if "centralia" in s:
        return "Centralia/south-I-5"
    if "auburn-snoqualmie" in s:
        return "auburn-snoqualmie"
    if "bellingham" in s:                 # Bellingham_01 + blaine-bellingham_*
        return "blaine-bellingham"
    if "ellensburg" in s:                 # Ellensburg-Yakima_* leads with Ellensburg
        return "Ellensburg"
    if "yakima" in s:
        return "Yakima"
    return "other"


def split_status(entry, scene):
    """trained / held-out / unseen from the registry's stored lists; unknown if not recoverable."""
    train = entry.get("train", {}).get("scenes")
    if not train:
        return "unknown"
    if scene in train:
        return "trained"
    m = entry.get("metrics", {})
    held = m.get("heldout_scenes")
    if held is None:
        hs = m.get("heldout_scene")       # older single-scene form
        held = [hs] if hs else []
    if scene in held:
        return "held-out"
    return "unseen"


# ---- centered-chip recall (labeled only): same inference path as training eval ----
_coco = None


def coco_chips():
    global _coco
    if _coco is None:
        _coco = {}
        d = json.loads((COCO / "annotations.json").read_text())
        # keep the CENTER vehicle per image (multi-vehicle exports carry neighbours too)
        anns = {}
        for a in d["annotations"]:
            if a.get("center", True) or a["image_id"] not in anns:
                anns[a["image_id"]] = a
        for im in d["images"]:
            a = anns.get(im["id"])
            if not a:
                continue
            chip = np.asarray(Image.open(COCO / "images" / im["file_name"]).convert("RGB"))
            _coco.setdefault(im["scene"], []).append(chip)
    return _coco


@torch.no_grad()
def centered_recall(model, scene, thresh):
    items = coco_chips().get(scene, [])
    if not items:
        return None
    det = 0
    for chip in items:
        off = (chip.shape[0] - 64) // 2                       # center-crop padded exports 96->64
        crop = chip[off:off + 64, off:off + 64]
        img = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).float() / 255.0
        out = model([img])[0]
        if len(out["scores"]) and float(out["scores"][0]) > thresh:
            det += 1
    return det / len(items)


_valid_km2_cache = {}


def valid_km2(scene):
    if scene not in _valid_km2_cache:
        with rasterio.open(GEO / f"{scene}.tif") as src:
            r = src.read(RED).astype(np.int64)
            g = src.read(GREEN).astype(np.int64)
            b = src.read(BLUE).astype(np.int64)
        valid = (r + g + b) > 0
        _valid_km2_cache[scene] = float(valid.sum()) * (GSD_M * GSD_M) / 1e6
    return _valid_km2_cache[scene]


def evaluate(model, entry, model_id, scene, thresh):
    """Full inference + scoring for one (model, scene). Returns a CSV row dict."""
    res = ds.detect(model, scene, stride=STRIDE, thresh=thresh)   # normal inference path
    pred = res["count"]
    area = valid_km2(scene)
    row = {
        "model_id": model_id, "scene": scene, "corridor": corridor_of(scene),
        "split_status": split_status(entry, scene), "threshold": thresh,
        "pred_count": pred, "valid_km2": round(area, 3),
        "det_per_km2": round(pred / area, 2) if area else "N/A",
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    g = res.get("gt")
    if g:  # labeled -> full scoring
        true = g["labelled"]
        recall = g["recall"] / max(true, 1)
        precision = g["near_label"] / max(pred, 1)
        f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
        row.update({
            "row_type": "scored", "true_count": true,
            "count_ratio": round(pred / true, 3) if true else "N/A",
            "precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "centered_recall": round(centered_recall(model, scene, thresh), 3),
        })
    else:  # unlabeled -> count-only
        row.update({
            "row_type": "count-only", "true_count": "N/A", "count_ratio": "N/A",
            "precision": "N/A", "recall": "N/A", "f1": "N/A", "centered_recall": "N/A",
        })
    return row


def append_row(row):
    new = (not CSV_PATH.exists()) or CSV_PATH.stat().st_size == 0
    with open(CSV_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def load_rows():
    if not CSV_PATH.exists():
        return []
    with open(CSV_PATH) as f:
        return list(csv.DictReader(f))


# ----------------------------------------------------------------- run -------
def run_matrix():
    log("=" * 70)
    log("eval_matrix START")
    log("Models selected (4):")
    for mid, why in PICKS:
        log(f"  + {mid}: {why}")
    log("Notable exclusions:")
    for mid, why in EXCLUDED:
        log(f"  - {mid}: {why}")
    log("NOTE: top-1-per-window is IN EFFECT (detect_scene keeps only the top detection per "
        "window) -> recall is capped; numbers match the current deployment behavior.")

    reg = mr.load()
    scenes = sorted(p.stem for p in GEO.glob("*.tif"))
    labeled = sorted(coco_chips().keys())
    log(f"{len(scenes)} loaded scenes ({len(labeled)} labeled, {len(scenes) - len(labeled)} unlabeled) "
        f"x {len(MODEL_IDS)} models = {len(scenes) * len(MODEL_IDS)} pairs.")

    done = {(r["model_id"], r["scene"]) for r in load_rows()}
    if done:
        log(f"resume: {len(done)} pairs already in CSV -> skipping them.")

    total_remaining = sum((mid, s) not in done for mid in MODEL_IDS for s in scenes)
    log(f"remaining this run: {total_remaining}")
    processed, first_timed = 0, False

    for mid in MODEL_IDS:
        entry = mr.get(reg, mid)
        if entry is None:
            log(f"MODEL NOT FOUND in registry: {mid} -> skipping all its rows.")
            continue
        wpath = REPO / "weights" / entry["weights"]
        if not wpath.exists():
            log(f"WEIGHTS MISSING for {mid}: {wpath} -> skipping all its rows.")
            continue
        if not entry.get("train", {}).get("scenes"):
            log(f"FLAG: {mid} training list NOT recoverable -> split_status=unknown for its rows.")

        thresh = float(entry.get("metrics", {}).get("eval_thresh", 0.3))
        pending = [s for s in scenes if (mid, s) not in done]
        if not pending:
            log(f"{mid}: all scenes already done.")
            continue

        log(f"--- loading {mid} (anchors={entry['arch'].get('anchor_sizes') or entry['arch'].get('anchors')}, "
            f"thresh={thresh}) ---")
        try:
            model = mr.load_weights(entry)
        except Exception as e:
            log(f"LOAD FAILED for {mid}: {e}\n{traceback.format_exc()}")
            continue

        for scene in pending:
            t0 = time.time()
            try:
                row = evaluate(model, entry, mid, scene, thresh)
                dt = time.time() - t0
                row["eval_seconds"] = round(dt, 1)
                append_row(row)
                processed += 1
                extra = (f"f1={row['f1']} P={row['precision']} R={row['recall']}"
                         if row["row_type"] == "scored" else "count-only")
                log(f"[{processed}/{total_remaining}] {mid} / {scene} "
                    f"[{row['split_status']}] pred={row['pred_count']} {extra} ({dt:.1f}s)")
                if not first_timed:
                    first_timed = True
                    est = dt * total_remaining
                    expect_h = total_remaining * 1.5 / 60.0  # ~1.5 min/scene expectation
                    log(f"TIMING: first eval {dt:.1f}s. Rough total for {total_remaining} evals "
                        f"~ {est/60:.0f} min ({est/3600:.1f} h). "
                        f"[~1-2 min/scene expectation would be ~{expect_h:.1f} h; "
                        f"{'in line' if est/3600 <= expect_h * 1.5 else 'SLOWER than expected'}]")
            except Exception as e:
                log(f"ERROR {mid}/{scene}: {e}\n{traceback.format_exc()}")
                continue

        # free ~226 MB before the next model
        mr._model_cache.pop(mid, None)
        del model
        gc.collect()
        log(f"--- finished {mid}, freed weights ---")

    log(f"eval_matrix loop END: processed {processed} this run.")


# -------------------------------------------------------------- reporting ----
def _fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_summary(rows):
    by_model = {mid: [r for r in rows if r["model_id"] == mid] for mid in MODEL_IDS}
    lines = []
    lines.append("# Model evaluation matrix — where the models stand\n")
    lines.append("> Scores are **agreement with the hand labels** (\"clear truck-ish echo\"), not absolute "
                 "accuracy — fine for model comparison, not for a headline accuracy claim.\n")
    lines.append(f"_Generated {datetime.datetime.now().isoformat(timespec='seconds')} · "
                 f"{len([r for r in rows if r['row_type']=='scored'])} scored + "
                 f"{len([r for r in rows if r['row_type']=='count-only'])} count-only rows · "
                 f"top-1-per-window in effect (recall capped) · stride {STRIDE} · match {MATCH_PX}px._\n")

    # 1) held-out generalization ranking (+ trained gap)
    lines.append("## 1 · Generalization ranking — mean F1 on held-out scenes (best→worst)\n")
    lines.append("The honest cross-scene number is the **held-out** column; **trained** is shown alongside so the "
                 "gap (memorization vs. generalization) is visible. *unseen* = never-trained labeled scenes not "
                 "formally designated held-out (bonus generalization signal).\n")
    ranked = []
    for mid in MODEL_IDS:
        rs = by_model[mid]
        held = [_fnum(r["f1"]) for r in rs if r["split_status"] == "held-out" and r["row_type"] == "scored"]
        trained = [_fnum(r["f1"]) for r in rs if r["split_status"] == "trained" and r["row_type"] == "scored"]
        unseen = [_fnum(r["f1"]) for r in rs if r["split_status"] == "unseen" and r["row_type"] == "scored"]
        held = [x for x in held if x is not None]
        trained = [x for x in trained if x is not None]
        unseen = [x for x in unseen if x is not None]
        ranked.append({
            "mid": mid,
            "held_mean": (sum(held) / len(held)) if held else None, "held_n": len(held),
            "trained_mean": (sum(trained) / len(trained)) if trained else None, "trained_n": len(trained),
            "unseen_mean": (sum(unseen) / len(unseen)) if unseen else None, "unseen_n": len(unseen),
        })
    ranked.sort(key=lambda d: (d["held_mean"] is not None, d["held_mean"] or -1), reverse=True)
    lines.append("| rank | model | held-out F1 (n) | trained F1 (n) | gap | unseen F1 (n) |")
    lines.append("|---|---|---|---|---|---|")
    for i, d in enumerate(ranked, 1):
        hm = f"{d['held_mean']:.3f} ({d['held_n']})" if d["held_mean"] is not None else "— (no held-out)"
        tm = f"{d['trained_mean']:.3f} ({d['trained_n']})" if d["trained_mean"] is not None else "—"
        gap = (f"{d['trained_mean']-d['held_mean']:+.3f}"
               if (d["held_mean"] is not None and d["trained_mean"] is not None) else "—")
        um = f"{d['unseen_mean']:.3f} ({d['unseen_n']})" if d["unseen_mean"] is not None else "—"
        lines.append(f"| {i} | `{SHORT.get(d['mid'], d['mid'])}` | {hm} | {tm} | {gap} | {um} |")
    lines.append("")

    # 2) low fit on own training data -> pipeline smell
    lines.append("## 2 · Pipeline check — models scoring poorly on their OWN trained scenes\n")
    LOWFIT = 0.50
    flagged = [d for d in ranked if d["trained_mean"] is not None and d["trained_mean"] < LOWFIT]
    if flagged:
        for d in flagged:
            lines.append(f"- ⚠️ `{SHORT.get(d['mid'], d['mid'])}` mean **trained** F1 = "
                         f"**{d['trained_mean']:.3f}** (< {LOWFIT}). Low fit on data it trained on points at a "
                         f"pipeline/config issue (or, for the no-aug baseline, expected weakness) more than model quality.")
    else:
        lines.append(f"- None below {LOWFIT:.2f}. Every model fits its own training scenes reasonably, so the "
                     f"inference path is not obviously broken for any of them.")
    lines.append("")

    # 3) scenes hard across all 4 models
    lines.append("## 3 · Scenes hard across all 4 models\n")
    scored_scenes = sorted({r["scene"] for r in rows if r["row_type"] == "scored"})
    hard = []
    for s in scored_scenes:
        f1s = [_fnum(r["f1"]) for r in rows if r["scene"] == s and r["row_type"] == "scored"]
        f1s = [x for x in f1s if x is not None]
        precs = [_fnum(r["precision"]) for r in rows if r["scene"] == s and r["row_type"] == "scored"]
        precs = [x for x in precs if x is not None]
        if f1s:
            hard.append((s, sum(f1s) / len(f1s), (sum(precs) / len(precs)) if precs else None, len(f1s)))
    hard.sort(key=lambda t: t[1])
    lines.append("| scene | corridor | mean F1 (all models) | mean precision |")
    lines.append("|---|---|---:|---:|")
    for s, mf1, mp, _n in hard[:5]:
        mp_s = f"{mp:.3f}" if mp is not None else "—"
        lines.append(f"| `{s}` | {corridor_of(s)} | {mf1:.3f} | {mp_s} |")
    lines.append("")
    yak = [t for t in hard if corridor_of(t[0]) == "Yakima"]
    if yak and yak[0][1] == min(t[1] for t in hard):
        s, mf1, mp, _ = yak[0]
        lines.append(f"→ **Yakima is the hardest corridor** (`{s}` mean F1 {mf1:.3f}"
                     + (f", mean precision {mp:.3f}" if mp is not None else "")
                     + ") — the expected precision collapse (many detections, few near a real label).\n")
    elif yak:
        s, mf1, mp, _ = yak[0]
        lines.append(f"→ Yakima note: `{s}` mean F1 {mf1:.3f}"
                     + (f", mean precision {mp:.3f}" if mp is not None else "") + ".\n")

    # 4) unlabeled scenes — relative stability (count-only)
    lines.append("## 4 · Unlabeled scenes — model-to-model agreement (count-only)\n")
    lines.append("Count-only can't separate true from false positives, so the value here is **agreement between "
                 "models on the same scene**, not the absolute number. A model detecting wildly more/fewer than "
                 "its peers on a scene is the flag.\n")
    unl = sorted({r["scene"] for r in rows if r["row_type"] == "count-only"})
    if unl:
        header = "| scene | " + " | ".join(SHORT.get(m, m) for m in MODEL_IDS) + " | median | outliers |"
        lines.append(header)
        lines.append("|" + "---|" * (len(MODEL_IDS) + 3))
        for s in unl:
            counts = {}
            for mid in MODEL_IDS:
                rr = [r for r in rows if r["scene"] == s and r["model_id"] == mid and r["row_type"] == "count-only"]
                counts[mid] = int(rr[0]["pred_count"]) if rr else None
            vals = [c for c in counts.values() if c is not None]
            med = sorted(vals)[len(vals) // 2] if vals else None
            outliers = []
            if med is not None and med > 0:
                for mid, c in counts.items():
                    if c is not None and (c > 2 * med or c < 0.5 * med):
                        outliers.append(SHORT.get(mid, mid))
            cells = " | ".join(str(counts[m]) if counts[m] is not None else "—" for m in MODEL_IDS)
            lines.append(f"| `{s}` | {cells} | {med if med is not None else '—'} | "
                         f"{', '.join(outliers) if outliers else '—'} |")
        lines.append("")
    else:
        lines.append("_(no count-only rows yet)_\n")

    lines.append("---\n_See `eval_matrix.csv` for the full per-pair data and `eval_matrix.png` for the figure._")
    return "\n".join(lines)


def render_png(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    def short_scene(s):
        parts = s.rsplit("_", 1)
        return parts[0] if len(parts) == 2 and parts[1].isdigit() else s

    scored = [r for r in rows if r["row_type"] == "scored"]
    count_only = [r for r in rows if r["row_type"] == "count-only"]
    labeled_scenes = sorted({r["scene"] for r in scored},
                            key=lambda s: (corridor_of(s), s))
    unl_scenes = sorted({r["scene"] for r in count_only})
    models = [m for m in MODEL_IDS if any(r["model_id"] == m for r in rows)]

    lookup = {(r["model_id"], r["scene"]): r for r in rows}
    SPLIT_MARK = {"trained": "T", "held-out": "H", "unseen": "U", "unknown": "?"}

    fig = plt.figure(figsize=(max(13, 0.8 * len(labeled_scenes) + 4), 11.5), dpi=130)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(3, 1, height_ratios=[3.1, 1.5, 1.3], hspace=0.55)

    # ---- Panel A: F1 heatmap (models x labeled scenes) ----
    axA = fig.add_subplot(gs[0])
    M = np.full((len(models), len(labeled_scenes)), np.nan)
    for i, mid in enumerate(models):
        for j, s in enumerate(labeled_scenes):
            r = lookup.get((mid, s))
            if r and r["row_type"] == "scored":
                v = _fnum(r["f1"])
                if v is not None:
                    M[i, j] = v
    im = axA.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    axA.set_xticks(range(len(labeled_scenes)))
    axA.set_xticklabels([short_scene(s) for s in labeled_scenes], rotation=45, ha="right", fontsize=8)
    axA.set_yticks(range(len(models)))
    axA.set_yticklabels([SHORT.get(m, m) for m in models], fontsize=9)
    for i, mid in enumerate(models):
        for j, s in enumerate(labeled_scenes):
            r = lookup.get((mid, s))
            if not r or r["row_type"] != "scored":
                continue
            v = _fnum(r["f1"])
            split = r["split_status"]
            txt = f"{v:.2f}\n{SPLIT_MARK.get(split,'?')}" if v is not None else ""
            axA.text(j, i, txt, ha="center", va="center", fontsize=7,
                     color="black", fontweight="bold" if split == "held-out" else "normal")
            if split == "held-out":
                axA.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                        edgecolor="#1558d6", lw=2.4))
    axA.set_title("Full-scene F1  —  models × labeled scenes\n"
                  "cell letter = split for THAT model (T trained · H held-out [blue box] · U unseen)",
                  fontsize=11, fontweight="bold")
    cbar = fig.colorbar(im, ax=axA, fraction=0.025, pad=0.01)
    cbar.set_label("F1", fontsize=9)

    # ---- Panel B: trained vs held-out mean F1 per model ----
    axB = fig.add_subplot(gs[1])
    tr_means, ho_means, un_means = [], [], []
    for mid in models:
        rs = [r for r in scored if r["model_id"] == mid]
        def mean_for(split):
            xs = [_fnum(r["f1"]) for r in rs if r["split_status"] == split]
            xs = [x for x in xs if x is not None]
            return sum(xs) / len(xs) if xs else np.nan
        tr_means.append(mean_for("trained"))
        ho_means.append(mean_for("held-out"))
        un_means.append(mean_for("unseen"))
    x = np.arange(len(models))
    w = 0.26
    axB.bar(x - w, tr_means, w, label="trained", color="#7fb069")
    axB.bar(x, ho_means, w, label="held-out", color="#1558d6")
    axB.bar(x + w, un_means, w, label="unseen", color="#b0b0b0")
    for xi, (t, h, u) in enumerate(zip(tr_means, ho_means, un_means)):
        for off, val in ((-w, t), (0, h), (w, u)):
            if not np.isnan(val):
                axB.text(xi + off, val + 0.01, f"{val:.2f}", ha="center", va="bottom", fontsize=7)
    axB.set_xticks(x)
    axB.set_xticklabels([SHORT.get(m, m) for m in models], fontsize=9)
    axB.set_ylabel("mean F1", fontsize=9)
    axB.set_ylim(0, 1.0)
    axB.axhline(0.50, ls="--", lw=1, color="#888", label="Van Etten 2024 ~0.50")
    axB.legend(fontsize=8, ncol=4, loc="upper right")
    axB.set_title("Mean F1 by split — trained vs held-out (the generalization gap) vs unseen",
                  fontsize=11, fontweight="bold")

    # ---- Panel C: unlabeled detection counts (relative stability) ----
    axC = fig.add_subplot(gs[2])
    if unl_scenes:
        xC = np.arange(len(unl_scenes))
        wC = 0.8 / max(len(models), 1)
        for k, mid in enumerate(models):
            vals = []
            for s in unl_scenes:
                r = lookup.get((mid, s))
                vals.append(int(r["pred_count"]) if r else np.nan)
            axC.bar(xC + k * wC - 0.4 + wC / 2, vals, wC, label=SHORT.get(mid, mid))
        axC.set_xticks(xC)
        axC.set_xticklabels([short_scene(s) for s in unl_scenes], rotation=20, ha="right", fontsize=8)
        axC.set_ylabel("detections", fontsize=9)
        axC.legend(fontsize=7, ncol=len(models))
        axC.set_title("Unlabeled scenes — detection count per model "
                      "(relative agreement, NOT accuracy: count-only can't tell true from false positives)",
                      fontsize=10, fontweight="bold")
    else:
        axC.axis("off")
        axC.text(0.5, 0.5, "no unlabeled scenes yet", ha="center", va="center")

    fig.suptitle("truck-detection · evaluation matrix   "
                 "(agreement with hand labels, not absolute accuracy · top-1-per-window → recall capped)",
                 fontsize=12.5, fontweight="bold", y=0.995)
    fig.savefig(PNG_PATH, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_reports():
    rows = load_rows()
    if not rows:
        log("no rows in CSV yet -> skipping reports.")
        return
    SUMMARY_PATH.write_text(build_summary(rows))
    log(f"wrote {SUMMARY_PATH.relative_to(REPO)} ({len(rows)} rows).")
    try:
        render_png(rows)
        log(f"wrote {PNG_PATH.relative_to(REPO)}.")
    except Exception as e:
        log(f"PNG render failed: {e}\n{traceback.format_exc()}")


def main():
    if "--report-only" in sys.argv:
        make_reports()
        return
    try:
        run_matrix()
    finally:
        # always regenerate reports from whatever is in the CSV (partials included)
        make_reports()
    rows = load_rows()
    total = len(MODEL_IDS) * len(list(GEO.glob("*.tif")))
    log(f"eval_matrix DONE. CSV has {len(rows)}/{total} pairs.")


if __name__ == "__main__":
    main()
