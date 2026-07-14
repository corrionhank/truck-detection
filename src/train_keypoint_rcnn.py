#!/usr/bin/env python3
"""
Fine-tune a Keypoint R-CNN on the SuperDove moving-echo COCO dataset.

This replicates the Adamiak et al. 2025 method at toy scale: Keypoint R-CNN
(ResNet-50 + FPN), one object class ("moving_echo"), 3 keypoints per vehicle in
capture order blue -> red -> green.

DATA REALITY: only ~15 labelled vehicles across 3 scenes exist right now. That is
far too few to train a model that generalises (Adamiak had 3,236). This script is
the end-to-end pipeline validation called for in CONTEXT.md sec 9 step 5 -- it will
*overfit*. We start from COCO-pretrained weights (transfer learning) so the tiny
dataset has a fighting chance of placing keypoints sensibly.

Run:  python3 src/train_keypoint_rcnn.py --epochs 60
Saves weights to weights/keypoint_rcnn_echo.pt
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # MPS ops fall back to CPU

import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import (
    keypointrcnn_resnet50_fpn,
    KeypointRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor

REPO = Path(__file__).resolve().parent.parent
COCO_DIR = REPO / "data" / "active" / "coco"
WEIGHTS = REPO / "weights" / "keypoint_rcnn_echo.pt"
NUM_KP = 3        # blue, red, green
NUM_CLASSES = 2   # background + moving_echo


class EchoCocoDataset(torch.utils.data.Dataset):
    """Minimal COCO-keypoints loader -> torchvision detection targets."""

    def __init__(self, coco_json):
        coco = json.loads(Path(coco_json).read_text())
        self.img_dir = Path(coco_json).parent / "images"
        self.images = {im["id"]: im for im in coco["images"]}
        # one annotation per image in this dataset, but group defensively
        self.by_image = {}
        for a in coco["annotations"]:
            self.by_image.setdefault(a["image_id"], []).append(a)
        self.ids = sorted(self.by_image)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        im = self.images[img_id]
        arr = np.asarray(Image.open(self.img_dir / im["file_name"]).convert("RGB"))
        img = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

        boxes, labels, kpts, areas = [], [], [], []
        for a in self.by_image[img_id]:
            x, y, w, h = a["bbox"]
            w = max(w, 4.0); h = max(h, 4.0)  # guard against degenerate boxes
            boxes.append([x, y, x + w, y + h])
            labels.append(1)
            k = np.array(a["keypoints"], dtype=np.float32).reshape(-1, 3)
            kpts.append(k)
            areas.append(w * h)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "keypoints": torch.as_tensor(np.stack(kpts), dtype=torch.float32),
            "image_id": torch.tensor([img_id]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }
        return img, target


def build_model():
    """COCO-pretrained Keypoint R-CNN, head re-sized to 3 keypoints / 1 class."""
    model = keypointrcnn_resnet50_fpn(
        weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT,
        min_size=192, max_size=320,  # chips are 64px; no need for the 800px default
    )
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, NUM_CLASSES)
    in_kp = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_kp, NUM_KP)
    return model


def pick_device(prefer="cpu"):
    # NOTE: MPS produces NaN losses with torchvision's detection models
    # (RoIAlign / anchor math). CPU is the reliable default for this tiny job.
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(epochs, lr, batch, seed, device):
    torch.manual_seed(seed)
    device = pick_device(device)
    print(f"device: {device}")

    ds = EchoCocoDataset(COCO_DIR / "annotations.json")
    print(f"training vehicles: {len(ds)}")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=True,
        collate_fn=lambda b: tuple(zip(*b)),
    )

    model = build_model().to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 3), gamma=0.3)

    # Linear LR warmup over the first epoch so the fresh heads don't blow up.
    warmup_iters = len(loader)
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.01, total_iters=warmup_iters)
    step = 0

    for epoch in range(1, epochs + 1):
        running, last = 0.0, {}
        for imgs, targets in loader:
            imgs = [i.to(device) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                print(f"  ! non-finite loss at epoch {epoch}, skipping step")
                opt.zero_grad(); step += 1
                if step <= warmup_iters:
                    warmup.step()
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            opt.step()
            step += 1
            if step <= warmup_iters:
                warmup.step()
            running += float(loss.detach())
            last = {k: float(v.detach()) for k, v in loss_dict.items()}
        sched.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            parts = " ".join(f"{k.split('_')[-1]}={last.get(k, 0):.3f}" for k in last)
            print(f"epoch {epoch:3d}/{epochs}  loss={running/len(loader):.4f}   [{parts}]")

    WEIGHTS.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS)
    print(f"saved weights -> {WEIGHTS.relative_to(REPO)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    main(**vars(ap.parse_args()))
