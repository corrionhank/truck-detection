#!/usr/bin/env python3
"""
Retrain the Keypoint R-CNN WITH translation + flip augmentation.

The v0 model (train_keypoint_rcnn.py) trained on 15 pixel-centered crops, so it
only fires when a vehicle sits dead-center in the window -> a coarse sliding
window over a scene finds nothing. This version crops *jittered* 64px windows
straight from the GeoTIFFs (vehicle offset by a random +/-JIT px, random flips),
teaching the model to detect a vehicle anywhere in the window. Then a coarse
sliding window should work.

Same architecture / transfer-learning setup as v0; only the data pipeline differs.

Run:  python3 src/train_keypoint_rcnn_jitter.py --epochs 30
Saves weights/keypoint_rcnn_echo_jitter.pt
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import rasterio
import torch

from train_keypoint_rcnn import build_model, pick_device, REPO
from export_coco import load_vehicles, RED, GREEN, BLUE, stretch_params, apply_stretch

CHIP = 64
HALF = CHIP // 2
GEOTIFF_DIR = REPO / "data" / "active" / "imagery"
WEIGHTS = REPO / "weights" / "keypoint_rcnn_echo_jitter.pt"


class JitteredSceneDataset(torch.utils.data.Dataset):
    """Crop random-offset windows from the scene rasters around each vehicle."""

    def __init__(self, jit=20, repeat=8, seed=0):
        self.jit = jit
        self.rng = np.random.default_rng(seed)
        by_scene, _ = load_vehicles()

        self.rgb = {}          # scene -> HxWx3 uint8
        self.vehicles = []     # (scene, kpts_px[3,2] in blue,red,green order)
        for scene, vehicles in by_scene.items():
            tif = GEOTIFF_DIR / f"{scene}.tif"
            if not tif.exists():
                continue
            with rasterio.open(tif) as src:
                r, g, b = src.read(RED), src.read(GREEN), src.read(BLUE)
                self.rgb[scene] = np.dstack([apply_stretch(r, *stretch_params(r)),
                                             apply_stretch(g, *stretch_params(g)),
                                             apply_stretch(b, *stretch_params(b))])
                inv = ~src.transform
                for vid, pts in vehicles.items():
                    kp = np.array([list(inv * (x, y)) for _, x, y in pts], dtype=np.float32)
                    self.vehicles.append((scene, kp))
        self.n = len(self.vehicles)
        self.repeat = repeat

    def __len__(self):
        return self.n * self.repeat

    def __getitem__(self, idx):
        scene, kp = self.vehicles[idx % self.n]
        rgb = self.rgb[scene]
        H, W = rgb.shape[:2]
        cx, cy = kp.mean(0)

        # random window offset; clamp so the window stays inside the scene
        dx, dy = self.rng.uniform(-self.jit, self.jit, 2)
        x0 = int(round(cx + dx)) - HALF
        y0 = int(round(cy + dy)) - HALF
        x0 = max(0, min(x0, W - CHIP))
        y0 = max(0, min(y0, H - CHIP))
        crop = rgb[y0:y0 + CHIP, x0:x0 + CHIP].copy()
        k = kp - np.array([x0, y0], dtype=np.float32)

        # random flips (geometric only; keypoint identities are band-based, not L/R)
        if self.rng.random() < 0.5:
            crop = crop[:, ::-1]; k[:, 0] = CHIP - 1 - k[:, 0]
        if self.rng.random() < 0.5:
            crop = crop[::-1, :]; k[:, 1] = CHIP - 1 - k[:, 1]
        crop = np.ascontiguousarray(crop)

        pad = 3.0
        bx0 = max(0.0, k[:, 0].min() - pad); by0 = max(0.0, k[:, 1].min() - pad)
        bx1 = min(float(CHIP), k[:, 0].max() + pad); by1 = min(float(CHIP), k[:, 1].max() + pad)
        bw = max(bx1 - bx0, 4.0); bh = max(by1 - by0, 4.0)

        img = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        kpts = np.concatenate([k, np.full((3, 1), 2.0, dtype=np.float32)], axis=1)
        target = {
            "boxes": torch.tensor([[bx0, by0, bx0 + bw, by0 + bh]], dtype=torch.float32),
            "labels": torch.ones(1, dtype=torch.int64),
            "keypoints": torch.tensor(kpts[None], dtype=torch.float32),
            "image_id": torch.tensor([idx]),
            "area": torch.tensor([bw * bh], dtype=torch.float32),
            "iscrowd": torch.zeros(1, dtype=torch.int64),
        }
        return img, target


def main(epochs, lr, batch, seed, jit, repeat, device):
    torch.manual_seed(seed)
    device = pick_device(device)
    print(f"device: {device}")

    ds = JitteredSceneDataset(jit=jit, repeat=repeat, seed=seed)
    print(f"vehicles: {ds.n}  |  jitter: +/-{jit}px  |  samples/epoch: {len(ds)}")
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch, shuffle=True, collate_fn=lambda b: tuple(zip(*b)))

    model = build_model().to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 3), gamma=0.3)
    warmup_iters = len(loader)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=warmup_iters)
    step = 0

    for epoch in range(1, epochs + 1):
        running, last = 0.0, {}
        for imgs, targets in loader:
            imgs = [i.to(device) for i in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            if not torch.isfinite(loss):
                opt.zero_grad(); step += 1
                if step <= warmup_iters: warmup.step()
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step(); step += 1
            if step <= warmup_iters: warmup.step()
            running += float(loss.detach())
            last = {k: float(v.detach()) for k, v in loss_dict.items()}
        sched.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            parts = " ".join(f"{k.split('_')[-1]}={last.get(k,0):.3f}" for k in last)
            print(f"epoch {epoch:3d}/{epochs}  loss={running/len(loader):.4f}   [{parts}]")

    WEIGHTS.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS)
    print(f"saved weights -> {WEIGHTS.relative_to(REPO)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jit", type=int, default=20)
    ap.add_argument("--repeat", type=int, default=8)
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    main(**vars(ap.parse_args()))
