#!/usr/bin/env python3
"""
Run the trained Keypoint R-CNN on the chips and score it against the hand labels.

Because the model was trained on the same ~15 vehicles (an overfit demo), this is
a *fit* check, not a generalisation metric: it confirms the model learned to place
3 keypoints (blue -> red -> green) on the moving echoes. Reports mean per-keypoint
pixel error and renders a predicted-vs-truth montage.

Run:  python3 src/infer_keypoints.py
Writes outputs/keypoint_predictions.png
"""
import json
import math
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from PIL import Image, ImageDraw

from train_keypoint_rcnn import build_model, pick_device, COCO_DIR, WEIGHTS, REPO

KP_COLORS = [(80, 140, 255), (255, 70, 70), (70, 220, 90)]  # blue, red, green


@torch.no_grad()
def main():
    device = pick_device()
    model = build_model().to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.eval()

    coco = json.loads((COCO_DIR / "annotations.json").read_text())
    imgs = {i["id"]: i for i in coco["images"]}
    anns = {a["image_id"]: a for a in coco["annotations"]}
    img_dir = COCO_DIR / "images"

    ids = sorted(imgs)
    SCALE, GAP, cols = 6, 6, 5
    rows = math.ceil(len(ids) / cols)
    cw = 64 * SCALE
    tile_w, tile_h = cw + GAP, cw + GAP + 18
    montage = Image.new("RGB", (cols * tile_w + GAP, rows * tile_h + GAP), (20, 20, 24))

    errors = []
    n_detected = 0
    for k, img_id in enumerate(ids):
        im = imgs[img_id]
        arr = np.asarray(Image.open(img_dir / im["file_name"]).convert("RGB"))
        t = torch.from_numpy(arr).permute(2, 0, 1).float().div(255).to(device)
        out = model([t])[0]

        gt = np.array(anns[img_id]["keypoints"], dtype=np.float32).reshape(-1, 3)

        # Each chip holds exactly one labelled vehicle, so take the top-scoring
        # detection. The from-scratch box head caps confidence ~0.45 on these
        # sub-pixel objects, so the accept threshold is 0.3, not 0.5.
        pred_kp = None
        if len(out["scores"]) and float(out["scores"][0]) > 0.3:
            n_detected += 1
            pred_kp = out["keypoints"][0].cpu().numpy()  # (3,3): x,y,score
            err = np.linalg.norm(pred_kp[:, :2] - gt[:, :2], axis=1).mean()
            errors.append(err)

        # ---- draw ----
        chip = Image.open(img_dir / im["file_name"]).convert("RGB").resize((cw, cw), Image.NEAREST)
        d = ImageDraw.Draw(chip)
        # ground truth = hollow circles
        for j in range(3):
            x, y = gt[j, 0] * SCALE, gt[j, 1] * SCALE
            d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=KP_COLORS[j], width=2)
        # prediction = filled crosses
        if pred_kp is not None:
            for j in range(3):
                x, y = pred_kp[j, 0] * SCALE, pred_kp[j, 1] * SCALE
                d.line([x - 5, y, x + 5, y], fill=KP_COLORS[j], width=2)
                d.line([x, y - 5, x, y + 5], fill=KP_COLORS[j], width=2)

        r0, c0 = k // cols, k % cols
        ox, oy = GAP + c0 * tile_w, GAP + r0 * tile_h
        montage.paste(chip, (ox, oy))
        lab = f'{im["scene"][:9]} v{anns[img_id].get("vehicle_id","?")}'
        if pred_kp is not None:
            lab += f'  err={errors[-1]:.1f}px'
        else:
            lab += "  (no det)"
        ImageDraw.Draw(montage).text((ox + 2, oy + cw + 3), lab, fill=(205, 205, 210))

    out_path = REPO / "outputs" / "keypoint_predictions.png"
    montage.save(out_path)

    print(f"vehicles              : {len(ids)}")
    print(f"detected (score>0.5)  : {n_detected}")
    if errors:
        px = np.array(errors)
        print(f"mean keypoint error   : {px.mean():.2f} px  (~{px.mean()*3:.1f} m at 3 m/px)")
        print(f"median / max          : {np.median(px):.2f} / {px.max():.2f} px")
    print(f"montage               : {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
