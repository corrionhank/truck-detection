#!/usr/bin/env python3
"""
Sliding-window inference over a whole SuperDove scene.

The chip-level eval (infer_keypoints.py) tests the model on the exact 64x64 crops
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

from train_keypoint_rcnn import build_model, WEIGHTS, REPO
from export_coco import RED, GREEN, BLUE, stretch_params, apply_stretch

GEOTIFF_DIR = REPO / "imagery" / "all_geotiffs"
CHIP = 64
KP_COLORS = [(80, 140, 255), (255, 70, 70), (70, 220, 90)]  # blue, red, green


def build_rgb(src):
    r, g, b = src.read(RED), src.read(GREEN), src.read(BLUE)
    rgb = np.dstack([apply_stretch(r, *stretch_params(r)),
                     apply_stretch(g, *stretch_params(g)),
                     apply_stretch(b, *stretch_params(b))])
    return rgb


def load_gt_reds(scene, transform):
    """Pixel coords of the red (sequence 2) keypoint of each labelled vehicle."""
    import geopandas as gpd
    gpkg = REPO / "Annotations-RGB.gpkg"
    gdf = gpd.read_file(gpkg, layer="Annotations")
    gdf["scene"] = gdf["scene"].astype(str).str.strip()
    g = gdf[(gdf["scene"] == scene) & (gdf["sequence"] == 2)]
    inv = ~transform
    return [list(inv * (row.geometry.x, row.geometry.y)) for _, row in g.iterrows()]


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


def load_model(weights=None):
    """Load a Keypoint R-CNN once; reuse across detect() calls (e.g. a server)."""
    wpath = Path(weights) if weights else WEIGHTS
    if not wpath.is_absolute():
        wpath = REPO / wpath
    model = build_model()
    model.load_state_dict(torch.load(wpath, map_location="cpu"))
    model.eval()
    return model


@torch.no_grad()
def detect(model, scene, stride=40, thresh=0.3, min_valid=0.15, batch=12):
    """Run sliding-window detection on one scene. Returns a structured dict and
    writes the montage/preview/JSON to outputs/. Pure Python types so a web
    backend can JSON-serialise the result directly."""
    tif = GEOTIFF_DIR / f"{scene}.tif"
    if not tif.exists():
        raise FileNotFoundError(f"no GeoTIFF for {scene!r}")

    with rasterio.open(tif) as src:
        rgb = build_rgb(src)
        transform = src.transform
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
    gt_reds = load_gt_reds(scene, transform)
    gt_stats = None
    if gt_reds:
        det_reds = [kp[1] for _, kp in kept]
        matched_gt = sum(1 for g in gt_reds
                         if any(np.linalg.norm(np.array(g) - dr) <= 6 for dr in det_reds))
        tp = sum(1 for dr in det_reds
                 if any(np.linalg.norm(np.array(g) - dr) <= 6 for g in gt_reds))
        gt_stats = {"labelled": len(gt_reds), "recall": matched_gt,
                    "near_label": tp, "elsewhere": len(kept) - tp}
        print(f"vs ground truth: recall {matched_gt}/{len(gt_reds)} labelled trucks matched (<=6px); "
              f"{tp}/{len(kept)} detections near a label, {len(kept)-tp} elsewhere")

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
    }


def main(scene, stride, thresh, min_valid, batch, weights):
    model = load_model(weights)
    print(f"weights: {(Path(weights) if weights else WEIGHTS)}")
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
    main(**vars(ap.parse_args()))
