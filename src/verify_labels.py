#!/usr/bin/env python3
"""
Label-overlay verifier — eyeball that annotations sit ON the moving echoes.

The importer proves the label COUNT is right; it cannot prove the keypoints land in the
right PLACE. A Y-flip or affine error at the annotation source passes every count check and
then trains the model on garbage locations. This renders the exact 64x64 chips the model
trains on (data/active/coco/) with each vehicle's blue/red/green keypoints drawn on top, so
a mis-registered set is obvious: the drawn points won't sit on the coloured streak.

Selection defaults to a spread weighted toward the newest / most-recently-corrected scenes
(the ones most likely to carry a registration error), plus a few from established scenes as a
control.

Run:  python3 src/verify_labels.py                       # default weighted sample
      python3 src/verify_labels.py --scenes <s1,s2> --n 8 # specific scenes
Writes outputs/label_verify.png
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
COCO = REPO / "data" / "active" / "coco"
OUT = REPO / "outputs" / "label_verify.png"

# blue, red, green — the capture order (sequence 1/2/3)
KP_COLORS = [(70, 140, 255), (255, 60, 60), (60, 220, 90)]

# scenes to prioritise: the 3 new/corrected-in scenes first, then a few established controls
NEW_SCENES = ["Yakima-Toppenish_04_20260620", "blaine-bellingham_03_20260504", "polygon_01_20260503"]
CONTROL_SCENES = ["Centralia_01_20260511", "Tacoma-Centralia_02_20260602", "auburn-snoqualmie_03_20260614"]


def load_by_scene():
    """{scene: [(chip_path, kp[3,2]), ...]} from the COCO export."""
    d = json.loads((COCO / "annotations.json").read_text())
    anns = {a["image_id"]: a for a in d["annotations"]}
    by = {}
    for im in d["images"]:
        a = anns.get(im["id"])
        if not a:
            continue
        kp = np.array(a["keypoints"], dtype=np.float32).reshape(3, 3)[:, :2]
        by.setdefault(im["scene"], []).append((COCO / "images" / im["file_name"], kp, a.get("vehicle_id", "?")))
    return by


def pick(by, scenes, n_per):
    """Evenly-spaced sample of up to n_per vehicles from each scene (deterministic)."""
    picks = []
    for s in scenes:
        items = by.get(s, [])
        if not items:
            continue
        idx = np.linspace(0, len(items) - 1, min(n_per, len(items))).round().astype(int)
        for i in sorted(set(idx.tolist())):
            picks.append((s, *items[i]))
    return picks


def collinearity_px(kp):
    """Perpendicular distance of the red (middle) point from the blue->green line, in px.
    A real echo is ~collinear, so a large value flags a suspicious / mis-ordered label."""
    b, r, g = kp
    bg = g - b
    L = np.linalg.norm(bg)
    if L < 1e-6:
        return 0.0
    rb = r - b
    return float(abs(bg[0] * rb[1] - bg[1] * rb[0]) / L)   # 2D cross magnitude


def main(a):
    by = load_by_scene()
    scenes = [s.strip() for s in a.scenes.split(",")] if a.scenes else (NEW_SCENES + CONTROL_SCENES)
    picks = pick(by, scenes, a.n)
    if not picks:
        raise SystemExit(f"no chips found for {scenes}. available: {sorted(by)}")

    # Chip size comes from the export (64 legacy, 96 with --margin 16). Derive the keypoint
    # scale from it — hardcoding it desynchronises the markers from the image and draws every
    # point offset from its true position, which reads as a data error but is a drawing bug.
    src_px = Image.open(picks[0][1]).size[0]
    GAP, cols, cw = 8, 6, 512
    SCALE = cw / src_px
    tw, th = cw + GAP, cw + GAP + 30
    rows = math.ceil(len(picks) / cols)
    montage = Image.new("RGB", (cols * tw + GAP, rows * th + GAP), (16, 16, 20))
    md = ImageDraw.Draw(montage)

    flags = 0
    for k, (scene, chip_path, kp, vid) in enumerate(picks):
        chip = Image.open(chip_path).convert("RGB").resize((cw, cw), Image.NEAREST)
        d = ImageDraw.Draw(chip)
        pts = [(float(x) * SCALE, float(y) * SCALE) for x, y in kp]
        for j in range(2):                                   # B-R and R-G segments
            d.line([pts[j], pts[j + 1]], fill=(235, 235, 235), width=2)
        for j, (px, py) in enumerate(pts):
            d.ellipse([px - 7, py - 7, px + 7, py + 7], outline=KP_COLORS[j], width=3)

        col = collinearity_px(kp)
        oob = bool((kp < 0).any() or (kp > src_px).any())    # keypoint outside the chip
        suspicious = col > 4.0 or oob
        flags += suspicious
        r0, c0 = k // cols, k % cols
        ox, oy = GAP + c0 * tw, GAP + r0 * th
        montage.paste(chip, (ox, oy))
        tag = f"{scene.split('_')[0][:16]} v{vid}  collin {col:.1f}px" + (" !!" if suspicious else "")
        md.text((ox + 3, oy + cw + 4), tag, fill=(255, 120, 120) if suspicious else (200, 200, 205))

    OUT.parent.mkdir(exist_ok=True)
    montage.save(OUT)
    print(f"rendered {len(picks)} chips from {len(scenes)} scene(s) -> {OUT.relative_to(REPO)}")
    print(f"auto-flags (non-collinear >4px or keypoint out-of-frame): {flags}/{len(picks)} "
          f"— these are hints; the real check is visual: do the B/R/G dots sit on the coloured streak?")
    per = {}
    for s, *_ in picks:
        per[s] = per.get(s, 0) + 1
    print("sampled:", ", ".join(f"{s.split('_')[0]}={n}" for s, n in per.items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default=None, help="comma-separated scenes (default: new + control sample)")
    ap.add_argument("--n", type=int, default=6, help="max vehicles sampled per scene")
    main(ap.parse_args())
