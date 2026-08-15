#!/usr/bin/env python3
"""
Sliding-window inference over a whole SuperDove scene.

The chip-level eval (archive/src/infer_keypoints.py) tests the model on the exact 64x64 crops
it trained on. This tests the *deployment* path: take a raw GeoTIFF, slide the
detector across the road corridor, keep confident moving-echo detections, dedupe
across overlapping windows, and map each back to a map coordinate.

With the current overfit v0 weights this is a smoke test of the full-scene path
(+ a recall check on labelled scenes), not a production detector.

Run:  python3 src/detect_scene.py Bellingham_01_20260425 [--stride 40] [--thresh 0.3]
Writes outputs/<scene>_detections.png
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import rasterio
import torch
from PIL import Image, ImageDraw

# Model graph comes from the (self-contained) registry builder, so inference stays
# independent of the archived training scripts. See archive/README.md.
from model_registry import REPO, build_model as _build_model
from export_coco import RED, GREEN, BLUE, stretch_params, apply_stretch

WEIGHTS = REPO / "weights" / "keypoint_rcnn_echo.pt"  # default for the standalone CLI
GEOTIFF_DIR = REPO / "data" / "active" / "imagery"
CHIP = 64
KP_COLORS = [(80, 140, 255), (255, 70, 70), (70, 220, 90)]  # blue, red, green


def build_rgb(src):
    r, g, b = src.read(RED), src.read(GREEN), src.read(BLUE)
    rgb = np.dstack([apply_stretch(r, *stretch_params(r)),
                     apply_stretch(g, *stretch_params(g)),
                     apply_stretch(b, *stretch_params(b))])
    return rgb


def load_gt_vehicles(scene, transform, crs):
    """[(3,2)] pixel coords per labelled vehicle, ordered blue, red, green.

    The GeoPackage is a single CRS (EPSG:32610) while scenes keep their native UTM zone,
    so points from zone-11 scenes MUST be reprojected before the inverse affine — the same
    rule export_coco.py and collect_inference.py follow. Skipping it does not fail loudly:
    the points simply land outside the raster and every label silently disappears."""
    import geopandas as gpd
    from rasterio.warp import transform as warp_points
    gpkg = REPO / "data" / "active" / "Annotations-RGB.gpkg"
    gdf = gpd.read_file(gpkg, layer="Annotations")
    gdf["scene"] = gdf["scene"].astype(str).str.strip()
    g = gdf[gdf["scene"] == scene]
    if not len(g):
        return []
    xs, ys = list(g.geometry.x), list(g.geometry.y)
    if gdf.crs is not None and crs is not None and gdf.crs != crs:
        xs, ys = warp_points(gdf.crs, crs, xs, ys)
    inv = ~transform
    by = {}
    for (vid, seq), x, y in zip(zip(g["vehicle_id"].astype(int), g["sequence"].astype(int)), xs, ys):
        by.setdefault(vid, {})[seq] = np.array(inv * (x, y))
    return [np.stack([v[1], v[2], v[3]]) for v in by.values() if {1, 2, 3} <= set(v)]


def load_gt_reds(scene, transform, crs):
    """Pixel coords of the red (sequence 2) keypoint of each labelled vehicle."""
    return [list(v[1]) for v in load_gt_vehicles(scene, transform, crs)]


def make_det_montage(scene, rgb, kept, gt_reds):
    """Zoomed crop of every detection with its predicted keypoints, for eyeballing."""
    import math
    if not kept:
        return
    SCALE, GAP, cols = 5, 6, 6
    rows = math.ceil(len(kept) / cols)
    cw = CHIP * SCALE
    tw, th = cw + GAP, cw + GAP + 16
    m = Image.new("RGB", (cols * tw + GAP, rows * th + GAP), (18, 18, 22))
    for k, (s, kp) in enumerate(kept):
        cx, cy = int(round(kp[1][0])), int(round(kp[1][1]))
        x0 = max(0, min(cx - CHIP // 2, rgb.shape[1] - CHIP))
        y0 = max(0, min(cy - CHIP // 2, rgb.shape[0] - CHIP))
        crop = Image.fromarray(rgb[y0:y0 + CHIP, x0:x0 + CHIP]).resize((cw, cw), Image.NEAREST)
        d = ImageDraw.Draw(crop)
        pts = [((kx - x0) * SCALE, (ky - y0) * SCALE) for kx, ky in kp]
        for j in range(2):
            d.line([pts[j], pts[j + 1]], fill=(240, 240, 240), width=1)
        for j, (px, py) in enumerate(pts):
            d.ellipse([px - 4, py - 4, px + 4, py + 4], outline=KP_COLORS[j], width=2)
        near = any(np.linalg.norm(np.array(g) - kp[1]) <= 6 for g in gt_reds) if gt_reds else None
        tag = "GT" if near else ("FP?" if near is False else "")
        r0, c0 = k // cols, k % cols
        ox, oy = GAP + c0 * tw, GAP + r0 * th
        m.paste(crop, (ox, oy))
        ImageDraw.Draw(m).text((ox + 2, oy + cw + 2), f"{s:.2f} {tag}", fill=(205, 205, 210))
    out = REPO / "outputs" / f"{scene}_det_montage.png"
    m.save(out)
    print(f"det montage: {out.relative_to(REPO)}")


# ---------------------------------------------------------- comparison chips ---
# ONE CHIP PER OUTCOME, SHOWING WHAT THE MODEL PREDICTED *AND* WHAT IT WAS TAUGHT
#
# The montage above draws predictions only, so a wrong keypoint looks the same as a right
# one. These records carry both sets in chip-local pixels, sorted into the three outcomes,
# so the console can draw them over each other and the difference becomes visible.
#
# Outcomes use the canonical definition (see the eval section): greedy by descending
# confidence, each detection credited to its nearest still-unclaimed label within MATCH_PX,
# the label then consumed. TP is counted over labels, so a second detection on an
# already-claimed label is an FP, and a label nothing reached is an FN.
MATCH_PX = 6.0          # 18 m at 3 m/px


def _chip_png(rgb, x0, y0):
    """Raw CHIP-sized crop as a base64 PNG. Sent at native 64 px; the browser upscales it
    with nearest-neighbour, which keeps it crisp and keeps the payload small."""
    import base64, io
    buf = io.BytesIO()
    Image.fromarray(rgb[y0:y0 + CHIP, x0:x0 + CHIP]).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def greedy_assign(kept, gt_reds):
    """The canonical matching: detections in descending confidence, each credited to its
    nearest still-unclaimed label within MATCH_PX, the label then consumed.

    Returns (verdict, claimed_by): detection index -> label index or None, and label index
    -> detection index. TP is len(claimed_by), counted over LABELS, so a second detection on
    an already-claimed label is a false positive rather than a second true positive."""
    claimed_by, verdict = {}, {}
    for di, (s_, kp) in sorted(enumerate(kept), key=lambda t: -t[1][0]):
        best, bi = None, None
        for gi, g in enumerate(gt_reds):
            if gi in claimed_by:
                continue
            d = float(np.linalg.norm(np.asarray(g, float) - kp[1]))
            if d <= MATCH_PX and (best is None or d < best):
                best, bi = d, gi
        if bi is not None:
            claimed_by[bi] = di
        verdict[di] = bi
    return verdict, claimed_by


def build_chip_records(rgb, kept, gt_vehicles, limit=400):
    """{'tp': [...], 'fp': [...], 'fn': [...]} — one record per outcome.

    Each record carries the chip image plus every predicted and every labelled vehicle
    whose keypoints fall inside that chip, so neighbours show up as context rather than
    vanishing. Coordinates are chip-local (0..CHIP)."""
    H, W = rgb.shape[:2]
    dets = sorted(enumerate(kept), key=lambda t: -t[1][0])       # descending confidence
    gts = [np.asarray(v, float) for v in gt_vehicles]
    gt_reds = [v[1] for v in gts]

    verdict, claimed_by = greedy_assign(kept, gt_reds)

    def window(cx, cy):
        return (max(0, min(int(round(cx)) - CHIP // 2, W - CHIP)),
                max(0, min(int(round(cy)) - CHIP // 2, H - CHIP)))

    def inside(pts, x0, y0):
        return all(x0 <= px < x0 + CHIP and y0 <= py < y0 + CHIP for px, py in pts)

    def pack(kind, cx, cy, subject_pred, subject_gt, extra):
        x0, y0 = window(cx, cy)
        rec = {"kind": kind, "chip": _chip_png(rgb, x0, y0), "origin": [x0, y0],
               "pred": ([[round(float(a - x0), 2), round(float(b - y0), 2)] for a, b in subject_pred]
                        if subject_pred is not None else None),
               "gt": ([[round(float(a - x0), 2), round(float(b - y0), 2)] for a, b in subject_gt]
                      if subject_gt is not None else None)}
        # neighbours that also land in frame, so a busy chip reads correctly
        rec["other_pred"] = [[[round(float(a - x0), 2), round(float(b - y0), 2)] for a, b in kp]
                             for _, (s, kp) in dets
                             if (subject_pred is None or not np.array_equal(kp, subject_pred))
                             and inside(kp, x0, y0)]
        rec["other_gt"] = [[[round(float(a - x0), 2), round(float(b - y0), 2)] for a, b in g]
                           for g in gts
                           if (subject_gt is None or not np.array_equal(g, subject_gt))
                           and inside(g, x0, y0)]
        rec.update(extra)
        return rec

    tp, fp = [], []
    for di, (s, kp) in dets:
        gi = verdict[di]
        if gi is not None:
            g = gts[gi]
            err = float(np.linalg.norm(kp - g, axis=1).mean())
            tp.append(pack("tp", kp[1][0], kp[1][1], kp, g,
                           {"score": round(s, 4), "vehicle": gi,
                            "err_px": round(err, 2), "err_m": round(err * 3.0, 1),
                            "red_err_px": round(float(np.linalg.norm(kp[1] - g[1])), 2)}))
        else:
            d = (min(float(np.linalg.norm(g - kp[1])) for g in gt_reds) * 3.0) if gt_reds else None
            fp.append(pack("fp", kp[1][0], kp[1][1], kp, None,
                           {"score": round(s, 4),
                            "dist_to_label_m": (round(d, 1) if d is not None else None)}))

    fn = []
    for gi, g in enumerate(gts):
        if gi in claimed_by:
            continue
        fn.append(pack("fn", g[1][0], g[1][1], None, g, {"score": None, "vehicle": gi}))

    return {"tp": tp[:limit], "fp": fp[:limit], "fn": fn[:limit],
            "totals": {"tp": len(tp), "fp": len(fp), "fn": len(fn)},
            "truncated": {"tp": len(tp) > limit, "fp": len(fp) > limit, "fn": len(fn) > limit},
            "chip_px": CHIP, "match_px": MATCH_PX, "gsd_m": 3.0}


def load_model(weights=None, anchors="default"):
    """Load a Keypoint R-CNN once from a raw weights path (standalone CLI use).
    `anchors` must match how the weights were trained. The backend does NOT use this —
    it builds + loads via model_registry (which stores each model's anchor set)."""
    wpath = Path(weights) if weights else WEIGHTS
    if not wpath.is_absolute():
        wpath = REPO / wpath
    model = _build_model({"anchors": anchors})
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model.eval()
    return model


@torch.no_grad()
def detect(model, scene, stride=40, thresh=0.3, min_valid=0.15, batch=12, chips=False):
    """Run sliding-window detection on one scene. Returns a structured dict and
    writes the montage/preview/JSON to outputs/. Pure Python types so a web
    backend can JSON-serialise the result directly."""
    tif = GEOTIFF_DIR / f"{scene}.tif"
    if not tif.exists():
        raise FileNotFoundError(f"no GeoTIFF for {scene!r}")

    with rasterio.open(tif) as src:
        rgb = build_rgb(src)
        transform, scene_crs = src.transform, src.crs
        H, W = rgb.shape[:2]
    valid = rgb.sum(2) > 0

    # window origins on a stride grid, kept only where enough road is present
    origins = []
    for y0 in range(0, H - CHIP + 1, stride):
        for x0 in range(0, W - CHIP + 1, stride):
            if valid[y0:y0 + CHIP, x0:x0 + CHIP].mean() >= min_valid:
                origins.append((x0, y0))
    print(f"scene {scene}: {W}x{H}px, {len(origins)} road windows (stride {stride})")

    # run windows through the model in batches
    dets = []  # (score, [ (kx,ky) x3 ] in full-scene px)
    for i in range(0, len(origins), batch):
        chunk = origins[i:i + batch]
        imgs = [torch.from_numpy(rgb[y:y + CHIP, x:x + CHIP].copy())
                .permute(2, 0, 1).float().div(255) for x, y in chunk]
        outs = model(imgs)
        for (x0, y0), out in zip(chunk, outs):
            if not len(out["scores"]):
                continue
            s = float(out["scores"][0])
            if s < thresh:
                continue
            kp = out["keypoints"][0].numpy()[:, :2] + np.array([x0, y0])
            dets.append((s, kp))

    # dedupe overlapping windows: greedy by score, suppress reds within CHIP/2 px
    dets.sort(key=lambda d: -d[0])
    kept = []
    for s, kp in dets:
        red = kp[1]
        if all(np.linalg.norm(red - k[1][1]) > CHIP / 2 for k in kept):
            kept.append((s, kp))
    print(f"detections > {thresh}: {len(dets)} raw -> {len(kept)} after dedupe")

    # save detections as JSON (red keypoint = sequence 2 anchor -> UTM)
    import json
    recs = []
    for s, kp in kept:
        x, y = transform * (float(kp[1][0]), float(kp[1][1]))
        recs.append({"score": round(s, 3),
                     "keypoints_px": [[round(float(a), 2), round(float(b), 2)] for a, b in kp],
                     "red_utm": [round(x, 2), round(y, 2)]})
    (REPO / "outputs" / f"{scene}_detections.json").write_text(json.dumps(recs, indent=2))

    # ---- recall / precision vs ground truth (if this scene is labelled) ----
    gt_vehicles = load_gt_vehicles(scene, transform, scene_crs)
    gt_reds = [list(v[1]) for v in gt_vehicles]
    gt_stats = None
    if gt_reds:
        # One TP count for both metrics. The previous version used a label-centric numerator
        # for recall and a detection-centric one for precision, so two detections on one label
        # inflated recall relative to precision; they now share the consuming count.
        _, claimed = greedy_assign(kept, gt_reds)
        tp = len(claimed)
        gt_stats = {"labelled": len(gt_reds), "recall": tp,
                    "near_label": tp, "elsewhere": len(kept) - tp}
        print(f"vs ground truth: {tp}/{len(gt_reds)} labelled trucks matched (<={MATCH_PX:.0f}px, "
              f"one detection per label); {len(kept)-tp}/{len(kept)} detections off-label")

    # ---- per-outcome comparison chips (predicted vs labelled keypoints) ----
    chips = build_chip_records(rgb, kept, gt_vehicles) if (chips and gt_vehicles) else None

    # ---- zoomed montage of each detection's crop, keypoints drawn ----
    make_det_montage(scene, rgb, kept, gt_reds)

    # ---- preview: crop to the valid bbox, upscale, draw detections ----
    ys, xs = np.where(valid)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    crop = rgb[y1:y2, x1:x2]
    scale = max(1, min(4, 2200 // max(crop.shape[:2])))
    prev = Image.fromarray(crop).resize(
        (crop.shape[1] * scale, crop.shape[0] * scale), Image.NEAREST)
    d = ImageDraw.Draw(prev)
    for s, kp in kept:
        pts = [((kx - x1) * scale, (ky - y1) * scale) for kx, ky in kp]
        for j in range(2):
            d.line([pts[j], pts[j + 1]], fill=(255, 255, 255), width=1)
        for j, (px, py) in enumerate(pts):
            d.ellipse([px - 4, py - 4, px + 4, py + 4], outline=KP_COLORS[j], width=2)

    out = REPO / "outputs" / f"{scene}_detections.png"
    prev.save(out)
    print(f"preview: {out.relative_to(REPO)}  ({prev.size[0]}x{prev.size[1]})")

    return {
        "scene": scene,
        "stride": stride,
        "thresh": thresh,
        "count": len(kept),
        "detections": recs,
        "gt": gt_stats,
        "montage": f"{scene}_det_montage.png",
        "preview": f"{scene}_detections.png",
        "chips": chips,
    }


def main(scene, stride, thresh, min_valid, batch, weights, anchors):
    model = load_model(weights, anchors)
    print(f"weights: {(Path(weights) if weights else WEIGHTS)}  (anchors={anchors})")
    result = detect(model, scene, stride=stride, thresh=thresh,
                    min_valid=min_valid, batch=batch)
    print(f"-> {result['count']} detections; montage outputs/{result['montage']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--min_valid", type=float, default=0.15)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--weights", default=None, help="override weights path")
    ap.add_argument("--anchors", choices=["small", "default"], default="default",
                    help="anchor set the weights were trained with")
    main(**vars(ap.parse_args()))
