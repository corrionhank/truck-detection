#!/usr/bin/env python3
"""
Collect full inference statistics for one model across every scene on disk.

Records EVERYTHING the model emits — including the fields the deployment path discards
(`keypoints_scores`, `boxes`, and every detection after the per-window top-1) — plus the
derived echo geometry (streak length, B-R/R-G spacing, collinearity, bearing) that a
non-DL size/geometry filter would gate on.

Each scene is scanned ONCE; the requested thresholds are applied as filters over that single
pass, so N thresholds cost one inference pass, not N (identical results, 1/N the compute).

Outputs (outputs/inference_collect/<model_id>/):
  detections.csv   one row per raw detection (all windows, all detections)
  stats.csv        one row per (scene, threshold): counts, P/R/F1 (asymmetric AND greedy
                   bipartite), centered-chip recall, keypoint error, count error, score spread
  scenes.csv       per-scene metadata (corridor, CRS, size, date, area, windows)
  manifest.json    every parameter the numbers were produced with
  run.log          progress

CPU note: torch defaults to 8 threads. Keep --threads low (2) while a training run is active.

Run:
  python3 src/collect_inference.py --model kprcnn-warmup-v1 --threads 2
  python3 src/collect_inference.py --thresholds 0.3,0.5,0.85
"""
import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

GSD, CHIP = 3.0, 64


def corridor_of(s):
    l = s.lower()
    if "centralia" in l:            return "Centralia/I-5-south"
    if "tri-cities" in l:           return "Tri-Cities (arid)"
    if "spokane" in l:              return "Spokane (arid)"
    if "ellensburg" in l:           return "Ellensburg (arid)"
    if "yakima" in l:               return "Yakima (arid)"
    if "auburn" in l:               return "auburn-snoqualmie/I-90"
    if "bellingham" in l:           return "blaine-bellingham/I-5-north"
    if "stanwood" in l:             return "Stanwood"
    if "polygon" in l:              return "polygon (Kent/Auburn)"
    return "other"


def scene_date(s):
    for tok in s.replace("-", "_").split("_"):
        if len(tok) == 8 and tok.isdigit():
            return f"{tok[:4]}-{tok[4:6]}-{tok[6:]}"
    return ""


def geometry(kp):
    """Echo geometry from the 3 keypoints (blue, red, green) in pixel space."""
    import numpy as np
    b, r, g = kp[0], kp[1], kp[2]
    br = float(np.linalg.norm(r - b)); rg = float(np.linalg.norm(g - r))
    bg = float(np.linalg.norm(g - b))
    v = g - b
    L = max(bg, 1e-9)
    collin = float(abs(v[0] * (r - b)[1] - v[1] * (r - b)[0]) / L)   # perp dist of red from B-G
    return {
        "streak_len_px": round(bg, 3), "streak_len_m": round(bg * GSD, 2),
        "seg_blue_red_px": round(br, 3), "seg_red_green_px": round(rg, 3),
        "spacing_ratio": round(br / rg, 3) if rg > 1e-9 else "",
        "collinearity_px": round(collin, 3),
        "bearing_deg": round(math.degrees(math.atan2(-(v[1]), v[0])) % 360.0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="registry model id (default: newest registry entry)")
    ap.add_argument("--thresholds", default="0.3,0.5,0.85")
    ap.add_argument("--scenes", default=None, help="comma-separated subset (default: every scene on disk)")
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--match-px", dest="match_px", type=float, default=6.0, help="GT match radius (6 px = 18 m)")
    ap.add_argument("--dedup-px", dest="dedup_px", type=float, default=32.0)
    ap.add_argument("--min-valid", dest="min_valid", type=float, default=0.15)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--top1-only", dest="top1_only", action="store_true",
                    help="record only the per-window top detection (default: record all)")
    ap.add_argument("--no-centered", dest="no_centered", action="store_true",
                    help="skip the centered-chip pass (saves time)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(a.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(a.threads))
    import numpy as np, rasterio, torch, geopandas as gpd
    from PIL import Image
    from rasterio.warp import transform as warp_points
    torch.set_num_threads(a.threads)
    import model_registry as mr, detect_scene as ds

    GEO, COCO = REPO / "data/active/imagery", REPO / "data/active/coco"
    reg = mr.load()
    model_id = a.model or reg["models"][-1]["id"]
    entry, model = mr.resolve(reg, model_id); model.eval()

    out_dir = Path(a.out) if a.out else REPO / "outputs" / "inference_collect" / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    det_p, stat_p, scn_p, log_p = (out_dir / n for n in
                                   ("detections.csv", "stats.csv", "scenes.csv", "run.log"))

    def log(m):
        line = f"[{time.strftime('%H:%M:%S')}] {m}"
        print(line, flush=True); open(log_p, "a").write(line + "\n")

    trained = set(entry.get("train", {}).get("scenes") or [])
    md = entry.get("metrics", {})
    held = set(md.get("heldout_scenes") or ([md["heldout_scene"]] if md.get("heldout_scene") else []))
    stype = md.get("heldout_split_type", {})

    def split_of(s):
        if not trained: return "unknown"
        if s in trained: return "trained"
        if s in held:    return f"held-out:{stype.get(s,'unspecified')}"
        return "unseen"

    # ---- labels: all 3 keypoints per vehicle, CRS-aware (raster native; points move) ----
    gdf = gpd.read_file(REPO / "data/active/Annotations-RGB.gpkg", layer="Annotations")
    gdf["scene"] = gdf["scene"].astype(str).str.strip()
    gpkg_crs = gdf.crs

    def gt_vehicles(scene, src):
        g = gdf[gdf["scene"] == scene]
        if not len(g): return []
        xs, ys = list(g.geometry.x), list(g.geometry.y)
        if gpkg_crs is not None and src.crs is not None and gpkg_crs != src.crs:
            xs, ys = warp_points(gpkg_crs, src.crs, xs, ys)
        inv = ~src.transform
        by = {}
        for (vid, seq), x, y in zip(zip(g["vehicle_id"].astype(int), g["sequence"].astype(int)), xs, ys):
            by.setdefault(vid, {})[seq] = np.array(inv * (x, y))
        return [np.stack([v[1], v[2], v[3]]) for v in by.values() if {1, 2, 3} <= set(v)]

    DET_FIELDS = ["model_id", "scene", "corridor", "split", "trained_on", "labelled",
                  "window_x", "window_y", "is_window_top1", "score",
                  "box_x0", "box_y0", "box_x1", "box_y1", "box_w_px", "box_h_px", "box_area_px",
                  "kp_blue_x", "kp_blue_y", "kp_red_x", "kp_red_y", "kp_green_x", "kp_green_y",
                  "kp_score_blue", "kp_score_red", "kp_score_green", "kp_score_mean",
                  "streak_len_px", "streak_len_m", "seg_blue_red_px", "seg_red_green_px",
                  "spacing_ratio", "collinearity_px", "bearing_deg",
                  "blue_utm_x", "blue_utm_y", "red_utm_x", "red_utm_y", "green_utm_x", "green_utm_y",
                  "dist_to_nearest_label_m", "matched_label_idx", "matched_bool", "kp_err_px"]
    STAT_FIELDS = ["model_id", "scene", "corridor", "split", "trained_on", "labelled", "threshold",
                   "true_count", "raw_detections", "kept_after_dedup", "count_ratio", "count_error_pct",
                   "tp", "precision", "recall", "f1",
                   "tp_greedy", "precision_greedy", "recall_greedy", "f1_greedy",
                   "labels_multi_matched", "dets_multi_matching",
                   "centered_recall", "centered_kp_err_px", "matched_kp_err_px",
                   "score_p10", "score_p25", "score_median", "score_p75", "score_p90", "score_mean",
                   "kp_score_mean", "mean_streak_len_m", "median_collinearity_px",
                   "windows", "valid_km2", "det_per_km2", "eval_seconds"]
    SCN_FIELDS = ["scene", "corridor", "split", "date", "crs", "width_px", "height_px",
                  "valid_km2", "windows", "labelled_vehicles"]

    @torch.no_grad()
    def centered_scores(scene):
        """Top score + kp error per labelled vehicle, on the centered chip (96 -> 64 crop)."""
        j = COCO / "annotations.json"
        if a.no_centered or not j.exists(): return [], []
        d = json.loads(j.read_text())
        anns = {}
        for an in d["annotations"]:
            if an.get("center", True) or an["image_id"] not in anns: anns[an["image_id"]] = an
        out_s, out_e = [], []
        for im in d["images"]:
            if im["scene"] != scene: continue
            an = anns.get(im["id"])
            if not an: continue
            chip = np.asarray(Image.open(COCO / "images" / im["file_name"]).convert("RGB"))
            off = (chip.shape[0] - CHIP) // 2
            crop = chip[off:off + CHIP, off:off + CHIP]
            gk = np.array(an["keypoints"], np.float32).reshape(3, 3)[:, :2] - off
            o = model([torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).float() / 255.0])[0]
            if len(o["scores"]):
                out_s.append(float(o["scores"][0]))
                out_e.append(float(np.linalg.norm(o["keypoints"][0].numpy()[:, :2] - gk, axis=1).mean()))
            else:
                out_s.append(0.0)
        return out_s, out_e

    @torch.no_grad()
    def scan(scene):
        with rasterio.open(GEO / f"{scene}.tif") as src:
            rgb = ds.build_rgb(src); tr, crs = src.transform, str(src.crs)
            H, W = rgb.shape[:2]; gts = gt_vehicles(scene, src)
        valid = rgb.sum(2) > 0
        area = float(valid.sum()) * GSD * GSD / 1e6
        origins = [(x, y) for y in range(0, H - CHIP + 1, a.stride) for x in range(0, W - CHIP + 1, a.stride)
                   if valid[y:y + CHIP, x:x + CHIP].mean() >= a.min_valid]
        reds = [g[1] for g in gts]
        recs = []
        for i in range(0, len(origins), a.batch):
            ch = origins[i:i + a.batch]
            imgs = [torch.from_numpy(rgb[y:y + CHIP, x:x + CHIP].copy()).permute(2, 0, 1).float().div(255)
                    for x, y in ch]
            for (x0, y0), o in zip(ch, model(imgs)):
                n = len(o["scores"])
                if not n: continue
                kps, boxes = o["keypoints"].numpy(), o["boxes"].numpy()
                kss = o["keypoints_scores"].numpy() if "keypoints_scores" in o else None
                for j in (range(1) if a.top1_only else range(n)):
                    kp = kps[j][:, :2] + np.array([x0, y0])
                    utm = [tr * (float(p[0]), float(p[1])) for p in kp]
                    d_i, d_min = (None, None)
                    if reds:
                        ds_ = [float(np.linalg.norm(g - kp[1])) for g in reds]
                        d_i = int(np.argmin(ds_)); d_min = ds_[d_i]
                    matched = int(d_min is not None and d_min <= a.match_px)
                    kperr = (round(float(np.linalg.norm(kp - gts[d_i], axis=1).mean()), 3)
                             if matched else "")
                    bx = boxes[j]
                    row = {"model_id": model_id, "scene": scene, "corridor": corridor_of(scene),
                           "split": split_of(scene), "trained_on": int(scene in trained),
                           "labelled": int(bool(gts)),
                           "window_x": x0, "window_y": y0, "is_window_top1": int(j == 0),
                           "score": round(float(o["scores"][j]), 5),
                           "box_x0": round(float(bx[0] + x0), 2), "box_y0": round(float(bx[1] + y0), 2),
                           "box_x1": round(float(bx[2] + x0), 2), "box_y1": round(float(bx[3] + y0), 2),
                           "box_w_px": round(float(bx[2] - bx[0]), 2), "box_h_px": round(float(bx[3] - bx[1]), 2),
                           "box_area_px": round(float((bx[2] - bx[0]) * (bx[3] - bx[1])), 2),
                           "kp_blue_x": round(float(kp[0][0]), 2), "kp_blue_y": round(float(kp[0][1]), 2),
                           "kp_red_x": round(float(kp[1][0]), 2), "kp_red_y": round(float(kp[1][1]), 2),
                           "kp_green_x": round(float(kp[2][0]), 2), "kp_green_y": round(float(kp[2][1]), 2),
                           "kp_score_blue": (round(float(kss[j][0]), 4) if kss is not None else ""),
                           "kp_score_red": (round(float(kss[j][1]), 4) if kss is not None else ""),
                           "kp_score_green": (round(float(kss[j][2]), 4) if kss is not None else ""),
                           "kp_score_mean": (round(float(kss[j].mean()), 4) if kss is not None else ""),
                           "blue_utm_x": round(utm[0][0], 2), "blue_utm_y": round(utm[0][1], 2),
                           "red_utm_x": round(utm[1][0], 2), "red_utm_y": round(utm[1][1], 2),
                           "green_utm_x": round(utm[2][0], 2), "green_utm_y": round(utm[2][1], 2),
                           "dist_to_nearest_label_m": (round(d_min * GSD, 2) if d_min is not None else ""),
                           "matched_label_idx": (d_i if matched else ""), "matched_bool": matched,
                           "kp_err_px": kperr}
                    row.update(geometry(kp))
                    recs.append(row)
        return recs, gts, len(origins), area, crs, W, H

    def dedup(rs):
        kept = []
        for r in sorted(rs, key=lambda r: -r["score"]):
            red = np.array([r["kp_red_x"], r["kp_red_y"]])
            if all(np.linalg.norm(red - np.array([k["kp_red_x"], k["kp_red_y"]])) > a.dedup_px for k in kept):
                kept.append(r)
        return kept

    thresholds = [float(t) for t in a.thresholds.split(",") if t.strip()]
    scenes = ([s.strip() for s in a.scenes.split(",")] if a.scenes
              else sorted(p.stem for p in GEO.glob("*.tif")))

    json.dump({"model_id": model_id, "weights": entry["weights"], "arch": entry["arch"],
               "thresholds": thresholds, "stride": a.stride, "chip_px": CHIP,
               "match_px": a.match_px, "match_m": a.match_px * GSD, "dedup_px": a.dedup_px,
               "dedup_m": a.dedup_px * GSD, "min_valid": a.min_valid, "gsd_m": GSD,
               "top1_only": a.top1_only, "roi_score_thresh": model.roi_heads.score_thresh,
               "roi_nms_thresh": model.roi_heads.nms_thresh,
               "roi_detections_per_img": model.roi_heads.detections_per_img,
               "n_scenes": len(scenes), "generated": time.strftime("%Y-%m-%dT%H:%M:%S")},
              open(out_dir / "manifest.json", "w"), indent=2)

    done = set()
    if stat_p.exists():
        done = {r["scene"] for r in csv.DictReader(open(stat_p))}
        log(f"resume: {len(done)} scene(s) already done — skipping")
    for p, flds in ((det_p, DET_FIELDS), (stat_p, STAT_FIELDS), (scn_p, SCN_FIELDS)):
        if not p.exists():
            csv.DictWriter(open(p, "w", newline=""), fieldnames=flds).writeheader()

    log(f"model={model_id} scenes={len(scenes)} thresholds={thresholds} threads={a.threads} "
        f"top1_only={a.top1_only} match={a.match_px}px({a.match_px*GSD:.0f}m)")

    for si, scene in enumerate(scenes, 1):
        if scene in done: continue
        t0 = time.time()
        try:
            recs, gts, nwin, area, crs, W, H = scan(scene)
            cscores, cerrs = centered_scores(scene)
        except Exception as e:
            log(f"ERROR {scene}: {e}\n{traceback.format_exc()}"); continue
        dt = time.time() - t0

        with open(det_p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=DET_FIELDS); w.writerows(recs)
            f.flush(); os.fsync(f.fileno())
        with open(scn_p, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=SCN_FIELDS).writerow(
                {"scene": scene, "corridor": corridor_of(scene), "split": split_of(scene),
                 "date": scene_date(scene), "crs": crs, "width_px": W, "height_px": H,
                 "valid_km2": round(area, 3), "windows": nwin, "labelled_vehicles": len(gts)})

        top1 = [r for r in recs if r["is_window_top1"]]
        reds_gt = [g[1] for g in gts]
        rows = []
        for t in thresholds:
            sel = [r for r in top1 if r["score"] >= t]
            kept = dedup(sel); n = len(kept)
            sc = [r["score"] for r in kept]
            ks = [r["kp_score_mean"] for r in kept if r["kp_score_mean"] != ""]
            row = {"model_id": model_id, "scene": scene, "corridor": corridor_of(scene),
                   "split": split_of(scene), "trained_on": int(scene in trained),
                   "labelled": int(bool(gts)), "threshold": t,
                   "true_count": (len(gts) if gts else ""), "raw_detections": len(sel),
                   "kept_after_dedup": n, "windows": nwin, "valid_km2": round(area, 3),
                   "det_per_km2": (round(n / area, 2) if area else ""), "eval_seconds": round(dt, 1),
                   "score_p10": (round(float(np.percentile(sc, 10)), 4) if sc else ""),
                   "score_p25": (round(float(np.percentile(sc, 25)), 4) if sc else ""),
                   "score_median": (round(float(np.median(sc)), 4) if sc else ""),
                   "score_p75": (round(float(np.percentile(sc, 75)), 4) if sc else ""),
                   "score_p90": (round(float(np.percentile(sc, 90)), 4) if sc else ""),
                   "score_mean": (round(float(np.mean(sc)), 4) if sc else ""),
                   "kp_score_mean": (round(float(np.mean(ks)), 4) if ks else ""),
                   "mean_streak_len_m": (round(float(np.mean([r["streak_len_m"] for r in kept])), 2) if kept else ""),
                   "median_collinearity_px": (round(float(np.median([r["collinearity_px"] for r in kept])), 3) if kept else ""),
                   "centered_recall": (round(sum(1 for s in cscores if s > t) / len(cscores), 4) if cscores else ""),
                   "centered_kp_err_px": (round(float(np.median(cerrs)), 3) if cerrs else "")}
            if gts:
                kr = [np.array([r["kp_red_x"], r["kp_red_y"]]) for r in kept]
                matched = sum(1 for g in reds_gt if any(np.linalg.norm(g - r) <= a.match_px for r in kr))
                tp = sum(1 for r in kr if any(np.linalg.norm(g - r) <= a.match_px for g in reds_gt))
                R, P = matched / len(gts), tp / max(n, 1)
                lab_multi = sum(1 for g in reds_gt if sum(1 for r in kr if np.linalg.norm(g - r) <= a.match_px) > 1)
                det_multi = sum(1 for r in kr if sum(1 for g in reds_gt if np.linalg.norm(g - r) <= a.match_px) > 1)
                used, tpg = set(), 0
                for r in sorted(kept, key=lambda r: -r["score"]):
                    rp = np.array([r["kp_red_x"], r["kp_red_y"]]); best, bi = None, None
                    for i2, g in enumerate(reds_gt):
                        if i2 in used: continue
                        dd = np.linalg.norm(g - rp)
                        if dd <= a.match_px and (best is None or dd < best): best, bi = dd, i2
                    if bi is not None: used.add(bi); tpg += 1
                Rg, Pg = tpg / len(gts), tpg / max(n, 1)
                errs = [r["kp_err_px"] for r in kept if r["kp_err_px"] != ""]
                row.update({"tp": tp, "precision": round(P, 4), "recall": round(R, 4),
                            "f1": round(2 * P * R / (P + R), 4) if (P + R) else 0.0,
                            "tp_greedy": tpg, "precision_greedy": round(Pg, 4), "recall_greedy": round(Rg, 4),
                            "f1_greedy": round(2 * Pg * Rg / (Pg + Rg), 4) if (Pg + Rg) else 0.0,
                            "labels_multi_matched": lab_multi, "dets_multi_matching": det_multi,
                            "count_ratio": round(n / len(gts), 3),
                            "count_error_pct": round(100 * abs(n - len(gts)) / len(gts), 1),
                            "matched_kp_err_px": (round(float(np.mean(errs)), 3) if errs else "")})
            else:
                row.update({k: "" for k in ("tp", "precision", "recall", "f1", "tp_greedy",
                            "precision_greedy", "recall_greedy", "f1_greedy", "labels_multi_matched",
                            "dets_multi_matching", "count_ratio", "count_error_pct", "matched_kp_err_px")})
            rows.append(row)
        with open(stat_p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=STAT_FIELDS); w.writerows(rows)
            f.flush(); os.fsync(f.fileno())

        log(f"[{si}/{len(scenes)}] {scene} [{split_of(scene)}] {len(recs)} dets / {nwin} win / {dt:.0f}s — "
            + " | ".join(f"thr{r['threshold']}: n={r['kept_after_dedup']}"
                         + (f" F1={r['f1']}" if r["f1"] != "" else "") for r in rows))

    log(f"DONE -> {det_p}  {stat_p}  {scn_p}")


if __name__ == "__main__":
    main()
