#!/usr/bin/env python3
"""
Leave-one-scene-out cross-validation for the moving-echo Keypoint R-CNN.

Everything else in this repo trains and evaluates on the same ~15 vehicles, so the
numbers only show the pipeline *fits*. This asks the real question: does a model
trained on some scenes detect trucks in a scene it has NEVER seen? With 3 labelled
scenes that is a 3-fold leave-one-scene-out CV — the first honest cross-scene
generalization signal we have.

On top of the honest split it adds the two "push the model further" levers:
  * richer augmentation — rotation + scale + photometric, on top of the translation
    jitter + flips from train_keypoint_rcnn_jitter.py. Image and keypoints are warped
    by the SAME cv2 affine matrix, so they can't drift out of sync.
  * small RPN anchors — the default anchors (32..512 px) are huge for a 4-8 px streak;
    --anchors small uses (8,16,32,64,128).

Run:  python3 src/crossval_keypoint.py --epochs 20 --anchors small --aug rich
Expectation: with 15 vehicles this may generalize poorly — that result is the point.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import rasterio
import torch
from torchvision.models.detection import (
    keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

from export_coco import load_vehicles, RED, GREEN, BLUE, stretch_params, apply_stretch, REPO

CHIP, HALF = 64, 32
GEOTIFF_DIR = REPO / "data" / "active" / "imagery"
NUM_KP, NUM_CLASSES = 3, 2
ANCHORS = {"small": (8, 16, 32, 64, 128), "default": (32, 64, 128, 256, 512)}


def build_model(anchor_sizes):
    # 1 size x 3 ratios per level = 3 anchors/location — same count as the pretrained
    # RPN head, so only the anchor SCALES change; pretrained weights still load.
    ag = AnchorGenerator(tuple((s,) for s in anchor_sizes),
                         ((0.5, 1.0, 2.0),) * len(anchor_sizes))
    model = keypointrcnn_resnet50_fpn(
        weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT,
        min_size=192, max_size=320, rpn_anchor_generator=ag)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, NUM_CLASSES)
    in_kp = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_kp, NUM_KP)
    return model


def load_scene_vehicles():
    """{scene: (rgb_uint8_HxWx3, [kp_px(3,2), ...])} for every labelled scene."""
    by_scene, _ = load_vehicles()
    out = {}
    for scene, vehicles in by_scene.items():
        tif = GEOTIFF_DIR / f"{scene}.tif"
        if not tif.exists():
            continue
        with rasterio.open(tif) as src:
            r, g, b = src.read(RED), src.read(GREEN), src.read(BLUE)
            rgb = np.dstack([apply_stretch(r, *stretch_params(r)),
                             apply_stretch(g, *stretch_params(g)),
                             apply_stretch(b, *stretch_params(b))])
            inv = ~src.transform
            kps = [np.array([list(inv * (x, y)) for _, x, y in pts], dtype=np.float32)
                   for pts in vehicles.values()]
        out[scene] = (rgb, kps)
    return out


class SceneDS(torch.utils.data.Dataset):
    """Augmented 64px chips cropped from the given scenes' rasters."""

    def __init__(self, data, scenes, aug, repeat=6, seed=0):
        self.rng = np.random.default_rng(seed)
        self.aug = aug
        self.repeat = repeat
        self.items = []  # (rgb, kp_px)
        for s in scenes:
            rgb, kps = data[s]
            for kp in kps:
                self.items.append((rgb, kp))
        self.n = len(self.items)

    def __len__(self):
        return self.n * self.repeat

    def __getitem__(self, idx):
        rgb, kp = self.items[idx % self.n]
        H, W = rgb.shape[:2]
        cx, cy = kp.mean(0)
        jit = 20 if self.aug != "none" else 0
        dx, dy = self.rng.uniform(-jit, jit, 2) if jit else (0.0, 0.0)
        x0 = max(0, min(int(round(cx + dx)) - HALF, W - CHIP))
        y0 = max(0, min(int(round(cy + dy)) - HALF, H - CHIP))
        crop = rgb[y0:y0 + CHIP, x0:x0 + CHIP].copy()
        k = kp - np.array([x0, y0], dtype=np.float32)

        if self.aug != "none":
            if self.rng.random() < 0.5:
                crop = crop[:, ::-1]; k[:, 0] = CHIP - 1 - k[:, 0]
            if self.rng.random() < 0.5:
                crop = crop[::-1, :]; k[:, 1] = CHIP - 1 - k[:, 1]
        crop = np.ascontiguousarray(crop)

        if self.aug == "rich":
            # rotation + scale via ONE affine matrix applied to image AND keypoints
            ang = self.rng.uniform(-25, 25)
            sc = self.rng.uniform(0.85, 1.20)
            M = cv2.getRotationMatrix2D((HALF, HALF), ang, sc)
            crop = cv2.warpAffine(crop, M, (CHIP, CHIP), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)
            ones = np.ones((3, 1), dtype=np.float32)
            k = (np.hstack([k, ones]) @ M.T).astype(np.float32)
            # photometric: brightness * contrast
            bright = self.rng.uniform(0.8, 1.2)
            crop = np.clip(crop.astype(np.float32) * bright, 0, 255).astype(np.uint8)

        # keep keypoints inside the frame (bail on the rare out-of-bounds warp)
        k[:, 0] = np.clip(k[:, 0], 1, CHIP - 2)
        k[:, 1] = np.clip(k[:, 1], 1, CHIP - 2)

        img = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        pad = 3.0
        bx0, by0 = max(0.0, k[:, 0].min() - pad), max(0.0, k[:, 1].min() - pad)
        bx1, by1 = min(float(CHIP), k[:, 0].max() + pad), min(float(CHIP), k[:, 1].max() + pad)
        bw, bh = max(bx1 - bx0, 4.0), max(by1 - by0, 4.0)
        kpts = np.concatenate([k, np.full((3, 1), 2.0, np.float32)], axis=1)
        target = {
            "boxes": torch.tensor([[bx0, by0, bx0 + bw, by0 + bh]], dtype=torch.float32),
            "labels": torch.ones(1, dtype=torch.int64),
            "keypoints": torch.tensor(kpts[None], dtype=torch.float32),
            "image_id": torch.tensor([idx]),
            "area": torch.tensor([bw * bh], dtype=torch.float32),
            "iscrowd": torch.zeros(1, dtype=torch.int64),
        }
        return img, target


@torch.no_grad()
def eval_scene(model, data, scene, thresh=0.3):
    """Held-out eval: centered chip per vehicle, does the model detect + localize it?"""
    model.eval()
    rgb, kps = data[scene]
    H, W = rgb.shape[:2]
    n_det, errs = 0, []
    for kp in kps:
        cx, cy = kp.mean(0)
        x0 = max(0, min(int(round(cx)) - HALF, W - CHIP))
        y0 = max(0, min(int(round(cy)) - HALF, H - CHIP))
        crop = rgb[y0:y0 + CHIP, x0:x0 + CHIP].copy()
        t = torch.from_numpy(crop).permute(2, 0, 1).float().div(255)
        out = model([t])[0]
        if len(out["scores"]) and float(out["scores"][0]) > thresh:
            n_det += 1
            pred = out["keypoints"][0].numpy()[:, :2]
            gt = kp - np.array([x0, y0], dtype=np.float32)
            errs.append(float(np.linalg.norm(pred - gt, axis=1).mean()))
    recall = n_det / max(len(kps), 1)
    med_err = float(np.median(errs)) if errs else float("nan")
    return len(kps), n_det, recall, med_err


def train_fold(data, train_scenes, epochs, lr, batch, anchors, aug, seed, repeat=6):
    torch.manual_seed(seed)
    ds = SceneDS(data, train_scenes, aug=aug, repeat=repeat, seed=seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True,
                                         collate_fn=lambda b: tuple(zip(*b)))
    model = build_model(ANCHORS[anchors])
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    warm = torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, total_iters=len(loader))
    step = 0
    for epoch in range(1, epochs + 1):
        running = 0.0
        for imgs, targets in loader:
            loss = sum(model(list(imgs), list(targets)).values())
            if not torch.isfinite(loss):
                opt.zero_grad(); step += 1
                if step <= len(loader): warm.step()
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step(); step += 1
            if step <= len(loader): warm.step()
            running += float(loss.detach())
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"    epoch {epoch:2d}/{epochs}  loss={running/len(loader):.3f}", flush=True)
    return model


def main(epochs, lr, batch, anchors, aug, seed, scene_filter=None, repeat=6):
    torch.manual_seed(seed)
    print(f"device: cpu  | anchors: {anchors} {ANCHORS[anchors]}  | aug: {aug}  | epochs: {epochs} | repeat: {repeat}")
    data = load_scene_vehicles()
    if scene_filter:
        missing = [s for s in scene_filter if s not in data]
        if missing:
            print(f"  ! requested scenes not labeled/found: {missing}")
        data = {s: data[s] for s in scene_filter if s in data}
    scenes = sorted(data)
    print(f"scenes: {[f'{s} ({len(data[s][1])}v)' for s in scenes]}\n")

    rows = []
    for held in scenes:
        train_scenes = [s for s in scenes if s != held]
        print(f"FOLD  train={train_scenes}  test={held}")
        model = train_fold(data, train_scenes, epochs, lr, batch, anchors, aug, seed, repeat)
        n, det, recall, err = eval_scene(model, data, held)
        print(f"  -> held-out {held}: detected {det}/{n} vehicles "
              f"(recall {recall:.0%}), median kp err {err:.1f} px\n", flush=True)
        rows.append((held, n, det, recall, err))

    print("=" * 62)
    print(f"LEAVE-ONE-SCENE-OUT CV  (anchors={anchors}, aug={aug})")
    tot_n = sum(r[1] for r in rows); tot_det = sum(r[2] for r in rows)
    for held, n, det, recall, err in rows:
        print(f"  {held:26s} recall {det}/{n} ({recall:.0%})  med_err {err:.1f}px")
    print(f"  {'OVERALL':26s} recall {tot_det}/{tot_n} ({tot_det/max(tot_n,1):.0%})")
    print("=" * 62)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--anchors", choices=list(ANCHORS), default="small")
    ap.add_argument("--aug", choices=["none", "basic", "rich"], default="rich")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scenes", default=None, help="comma-separated subset of scenes to use")
    ap.add_argument("--repeat", type=int, default=6, help="augmented samples per vehicle per epoch")
    a = ap.parse_args()
    main(a.epochs, a.lr, a.batch, a.anchors, a.aug, a.seed,
         scene_filter=a.scenes.split(",") if a.scenes else None, repeat=a.repeat)
