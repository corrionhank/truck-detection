#!/usr/bin/env python3
"""
Train holding out one scene, then VISUALIZE the model's detections on it:
true positives, false positives (misidentifications), and misses.

Honest by construction — the held-out scene is never trained on. Renders:
  outputs/<held>_tpfpfn.png       full scene, road-cropped: TP green, FP red, miss yellow
  outputs/<held>_falsepos.png     zoomed montage of every false positive
  outputs/<held>_missed.png       zoomed montage of every missed labelled vehicle

Run:  python3 src/viz_heldout.py --held Tacoma-Centralia_01_20260429
"""
import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import rasterio
import torch
from PIL import Image, ImageDraw

import crossval_keypoint as cv
import detect_scene as ds
from export_coco import REPO

CHIP = 64
GREEN, RED, YELLOW = (70, 220, 90), (240, 70, 70), (240, 200, 60)


def classify(det_reds, gt_reds, tol=6.0):
    """Match detection red-keypoints to GT red-keypoints. Returns (tp, fp, missed) index lists."""
    gt = [np.array(g, dtype=float) for g in gt_reds]
    matched, tp, fp = set(), [], []
    for i, dr in enumerate(det_reds):
        near = [j for j, g in enumerate(gt) if np.linalg.norm(np.array(dr) - g) <= tol]
        if near:
            tp.append(i); matched.update(near)
        else:
            fp.append(i)
    missed = [j for j in range(len(gt)) if j not in matched]
    return tp, fp, missed


def crop_montage(rgb, centers, kpts_list, colors, title_fmt, path, cols=6):
    """Zoomed 64px crop around each center, keypoints drawn; saved as a labelled montage."""
    if not centers:
        Image.new("RGB", (300, 40), (18, 18, 22)).save(path)
        return
    S, GAP = 5, 6
    cw = CHIP * S
    rows = math.ceil(len(centers) / cols)
    tw, th = cw + GAP, cw + GAP + 16
    m = Image.new("RGB", (cols * tw + GAP, rows * th + GAP), (18, 18, 22))
    H, W = rgb.shape[:2]
    for k, (cx, cy) in enumerate(centers):
        x0 = max(0, min(int(cx) - CHIP // 2, W - CHIP))
        y0 = max(0, min(int(cy) - CHIP // 2, H - CHIP))
        crop = Image.fromarray(rgb[y0:y0 + CHIP, x0:x0 + CHIP]).resize((cw, cw), Image.NEAREST)
        d = ImageDraw.Draw(crop)
        if kpts_list is not None:
            for (kx, ky), col in zip(kpts_list[k], ((80, 140, 255), (255, 70, 70), (70, 220, 90))):
                px, py = (kx - x0) * S, (ky - y0) * S
                d.ellipse([px - 4, py - 4, px + 4, py + 4], outline=col, width=2)
        else:
            d.rectangle([2, 2, cw - 2, cw - 2], outline=colors, width=3)
        r0, c0 = k // cols, k % cols
        ox, oy = GAP + c0 * tw, GAP + r0 * th
        m.paste(crop, (ox, oy))
        ImageDraw.Draw(m).text((ox + 2, oy + cw + 2), title_fmt.format(k=k + 1), fill=(205, 205, 210))
    m.save(path)


def main(held, train_scenes, epochs, stride, thresh):   # NB: training needs grad; detect() is no_grad internally
    data = cv.load_scene_vehicles()
    if train_scenes:
        train = train_scenes
    else:
        train = [s for s in data if s != held]  # everything else that is labelled
    print(f"train on {train}\nhold out + visualize: {held}", flush=True)

    wpath = REPO / "weights" / f"kprcnn_heldout_{held}.pt"
    if wpath.exists():                                    # reuse a prior train (skip the ~20 min)
        model = cv.build_model(cv.ANCHORS["small"])       # small anchors must match how it was trained
        model.load_state_dict(torch.load(wpath, map_location="cpu"))
        print(f"reused {wpath.relative_to(REPO)} (skip training)", flush=True)
    else:
        model = cv.train_fold(data, train, epochs, 0.005, 4, "small", "rich", 0, repeat=1)
        torch.save(model.state_dict(), wpath)
        print(f"saved {wpath.relative_to(REPO)}", flush=True)
    model.eval()                                          # <-- inference mode (the bug fix)

    # sliding-window detection on the held-out scene (detect() also writes its own montage/json)
    result = ds.detect(model, held, stride=stride, thresh=thresh)
    dets = result["detections"]
    det_reds = [d["keypoints_px"][1] for d in dets]           # red keypoint = anchor
    det_kpts = [d["keypoints_px"] for d in dets]

    # scene rgb + GT red keypoints in pixel space
    with rasterio.open(ds.GEOTIFF_DIR / f"{held}.tif") as src:
        rgb = ds.build_rgb(src)
        gt_reds = ds.load_gt_reds(held, src.transform)

    tp, fp, missed = classify(det_reds, gt_reds)
    n_gt = len(gt_reds)
    recall = len(tp) / max(n_gt, 1)
    precision = len(tp) / max(len(dets), 1)
    print(f"\nHELD-OUT {held}:  GT={n_gt}  detections={len(dets)}", flush=True)
    print(f"  TP {len(tp)}  |  FP {len(fp)}  |  MISSED {len(missed)}", flush=True)
    print(f"  recall {recall:.0%}   precision {precision:.0%}", flush=True)

    # ---- full-scene overlay, road-cropped ----
    valid = rgb.sum(2) > 0
    ys, xs = np.where(valid)
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = rgb[y1:y2, x1:x2]
    scale = max(1, min(4, 2400 // max(crop.shape[:2])))
    prev = Image.fromarray(crop).resize((crop.shape[1] * scale, crop.shape[0] * scale), Image.NEAREST)
    d = ImageDraw.Draw(prev)
    def pt(x, y): return ((x - x1) * scale, (y - y1) * scale)
    for i in tp:                       # true positives = green circle
        px, py = pt(*det_reds[i]); d.ellipse([px-6, py-6, px+6, py+6], outline=GREEN, width=2)
    for i in fp:                       # false positives = red cross
        px, py = pt(*det_reds[i])
        d.line([px-6, py, px+6, py], fill=RED, width=2); d.line([px, py-6, px, py+6], fill=RED, width=2)
    for j in missed:                   # missed GT = yellow square
        px, py = pt(*gt_reds[j]); d.rectangle([px-6, py-6, px+6, py+6], outline=YELLOW, width=2)
    out = REPO / "outputs" / f"{held}_tpfpfn.png"
    prev.save(out)
    print(f"  overlay  -> {out.relative_to(REPO)}  (green=TP, red=FP, yellow=miss)", flush=True)

    # ---- montages of the errors ----
    crop_montage(rgb, [tuple(det_reds[i]) for i in fp], [det_kpts[i] for i in fp], RED,
                 "FP {k}", REPO / "outputs" / f"{held}_falsepos.png")
    crop_montage(rgb, [tuple(gt_reds[j]) for j in missed], None, YELLOW,
                 "miss {k}", REPO / "outputs" / f"{held}_missed.png")
    print(f"  false-positive montage -> outputs/{held}_falsepos.png", flush=True)
    print(f"  missed-vehicle montage -> outputs/{held}_missed.png", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--held", default="Tacoma-Centralia_01_20260429")
    ap.add_argument("--train", default=None, help="comma-separated train scenes (default: other Centralia)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--stride", type=int, default=24)
    ap.add_argument("--thresh", type=float, default=0.5)
    a = ap.parse_args()
    default_train = ["Centralia_01_20260511", "Centralia_02_20260511", "Tacoma-Centralia_02_20260602"]
    train = a.train.split(",") if a.train else default_train
    main(a.held, train, a.epochs, a.stride, a.thresh)
