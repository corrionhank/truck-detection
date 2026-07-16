#!/usr/bin/env python3
"""
Fresh moving-echo detector — Keypoint R-CNN, following Adamiak et al. 2025.

Built against the intact data pipeline: reads the 64x64 COCO chips from
data/active/coco/ (produced by export_coco.py), builds the model via model_registry
(so the console / detect_scene can rebuild + load it), trains, evaluates on a held-out
scene (leave-one-scene-out), then registers the model + writes a methodology card.

Architecture (Adamiak):
  - Keypoint R-CNN, torchvision keypointrcnn_resnet50_fpn (ResNet-50 + FPN).
  - 2 classes (moving echo + background), 3 keypoints per vehicle (blue -> red -> green).

Deviations from Adamiak (our setup differs — recorded in the card):
  - Finetune from COCO-pretrained backbone (weights="DEFAULT"), not trained from scratch
    (our label count << their 3,236).
  - 64x64 chips, not their 512x512 images.
  - Leave-one-scene-out split, not random 80/10/10.

Training config (from the paper, adopted directly):
  - Adam; ReduceLROnPlateau on validation loss; LR 1e-3 -> 1e-5; grad clip 1.5.
  - Augmentation: random rotation, H/V flips, brightness, perspective.
  - Loss: the composite loss torchvision returns in training mode (not assembled by hand).

Anchors are the one high-leverage knob (Adamiak swept them). This first build uses small
anchors in that spirit (sizes 4/8/16/32/48, ratios 0.25/0.5/0.75/1.0/1.25). Our chips are
64x64 (not 512), so the exact sizes may not transfer — pass --anchor-sizes / --aspect-ratios
to sweep later. NOT swept here on purpose.

Run (CPU; MPS diverges — see docs/HARDWARE.md):
  python3 src/train_detector.py --id kprcnn-adamiak-v1 --name "Keypoint R-CNN - Adamiak v1"
"""
import argparse
import datetime
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.models.detection import keypointrcnn_resnet50_fpn

import model_registry as mr
import detect_scene as ds

REPO = Path(__file__).resolve().parent.parent
COCO = REPO / "data" / "active" / "coco"
CHIP, HALF = 64, 32


# ---------------------------------------------------------------- data ------
def load_coco():
    """{scene: [(chip uint8 HxWx3, kps float32[3,2] in chip px), ...]} from the export."""
    d = json.loads((COCO / "annotations.json").read_text())
    anns = {a["image_id"]: a for a in d["annotations"]}
    by_scene = {}
    for im in d["images"]:
        a = anns.get(im["id"])
        if not a:
            continue
        chip = np.asarray(Image.open(COCO / "images" / im["file_name"]).convert("RGB"))
        kp = np.array(a["keypoints"], dtype=np.float32).reshape(3, 3)[:, :2]
        by_scene.setdefault(im["scene"], []).append((chip, kp))
    return by_scene


# ------------------------------------------------- augmentation (Adamiak) ---
def augment(chip, kp, rng):
    """Adamiak's augmentation: rotation, H/V flips, brightness, perspective — applied to
    the image AND the keypoints. No translation jitter (not in Adamiak's list)."""
    c, k = chip.copy(), kp.copy()
    if rng.random() < 0.5:                                   # horizontal flip
        c = c[:, ::-1]; k[:, 0] = CHIP - 1 - k[:, 0]
    if rng.random() < 0.5:                                   # vertical flip
        c = c[::-1, :]; k[:, 1] = CHIP - 1 - k[:, 1]
    c = np.ascontiguousarray(c)

    ang = rng.uniform(-180, 180)                             # rotation about center
    M = cv2.getRotationMatrix2D((HALF, HALF), ang, 1.0)
    c = cv2.warpAffine(c, M, (CHIP, CHIP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    k = (np.hstack([k, np.ones((3, 1), np.float32)]) @ M.T).astype(np.float32)

    jit = rng.uniform(-5, 5, (4, 2)).astype(np.float32)      # mild perspective warp
    src = np.array([[0, 0], [CHIP, 0], [CHIP, CHIP], [0, CHIP]], np.float32)
    P = cv2.getPerspectiveTransform(src, src + jit)
    c = cv2.warpPerspective(c, P, (CHIP, CHIP), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    kh = np.hstack([k, np.ones((3, 1), np.float32)]) @ P.T
    k = (kh[:, :2] / kh[:, 2:3]).astype(np.float32)

    c = np.clip(c.astype(np.float32) * rng.uniform(0.8, 1.2), 0, 255).astype(np.uint8)  # brightness
    return np.ascontiguousarray(c), k


def to_target(k):
    """torchvision Keypoint R-CNN target: 1 box + 1 label + 3 keypoints (visible)."""
    k = k.copy()
    k[:, 0] = np.clip(k[:, 0], 1, CHIP - 2)
    k[:, 1] = np.clip(k[:, 1], 1, CHIP - 2)
    pad = 3.0
    x0, y0 = max(0.0, k[:, 0].min() - pad), max(0.0, k[:, 1].min() - pad)
    x1, y1 = min(float(CHIP), k[:, 0].max() + pad), min(float(CHIP), k[:, 1].max() + pad)
    w, h = max(x1 - x0, 4.0), max(y1 - y0, 4.0)
    kpts = np.concatenate([k, np.full((3, 1), 2.0, np.float32)], axis=1)  # v=2 visible
    return {
        "boxes": torch.tensor([[x0, y0, x0 + w, y0 + h]], dtype=torch.float32),
        "labels": torch.ones(1, dtype=torch.int64),          # class 1 = moving_echo
        "keypoints": torch.tensor(kpts[None], dtype=torch.float32),
    }


class ChipDS(torch.utils.data.Dataset):
    def __init__(self, items, repeat, seed, aug):
        self.items, self.repeat, self.aug = items, repeat, aug
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items) * self.repeat

    def __getitem__(self, i):
        chip, kp = self.items[i % len(self.items)]
        if self.aug:
            chip, kp = augment(chip, kp, self.rng)
        img = torch.from_numpy(np.ascontiguousarray(chip)).permute(2, 0, 1).float() / 255.0
        return img, to_target(kp)


def collate(b):
    return tuple(zip(*b))


# ----------------------------------------------------------------- model ----
def build_finetune(arch):
    """Custom-anchor graph from the registry builder, with the COCO-pretrained ResNet-50 +
    FPN backbone injected (the transferable part). The detection heads (RPN / box / keypoint)
    are trained from scratch — they must be, since our anchors / classes / keypoints differ
    from the COCO-person model. This is our 'finetune from weights=DEFAULT' deviation."""
    model = mr.build_model(arch)
    pre = keypointrcnn_resnet50_fpn(weights="DEFAULT")
    model.backbone.load_state_dict(pre.backbone.state_dict())
    del pre
    return model


# ----------------------------------------------------------------- eval -----
@torch.no_grad()
def eval_centered(model, items, thresh):
    """Centered-chip recall on the held-out scene: one 64px chip per vehicle."""
    model.eval()
    det, errs = 0, []
    for chip, kp in items:
        img = torch.from_numpy(np.ascontiguousarray(chip)).permute(2, 0, 1).float() / 255.0
        out = model([img])[0]
        if len(out["scores"]) and float(out["scores"][0]) > thresh:
            det += 1
            pred = out["keypoints"][0].numpy()[:, :2]
            errs.append(float(np.linalg.norm(pred - kp, axis=1).mean()))
    return det / max(len(items), 1), (float(np.median(errs)) if errs else float("nan"))


def eval_full_scene(model, scene, thresh):
    """Full-scene recall / precision / F1 via the deployment path (detect_scene)."""
    model.eval()
    res = ds.detect(model, scene, stride=40, thresh=thresh)
    g = res.get("gt")
    if not g:
        return None
    recall = g["recall"] / max(g["labelled"], 1)
    precision = g["near_label"] / max(res["count"], 1)
    f1 = (2 * recall * precision / (recall + precision)) if (recall + precision) > 0 else 0.0
    return {"labelled": g["labelled"], "detected": res["count"], "tp": g["near_label"],
            "recall": recall, "precision": precision, "f1": f1}


# ---------------------------------------------------------------- train -----
def train(train_items, val_items, arch, epochs, batch, lr, repeat, seed, ckpt_path):
    torch.manual_seed(seed)
    model = mr.build_model(arch)
    dl = DataLoader(ChipDS(train_items, repeat, seed, aug=True), batch_size=batch,
                    shuffle=True, collate_fn=collate, num_workers=0)
    vdl = DataLoader(ChipDS(val_items, 1, seed + 1, aug=False), batch_size=batch,
                     shuffle=False, collate_fn=collate, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)                       # Adam
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(                     # ReduceLROnPlateau
        opt, mode="min", factor=0.3, patience=2, min_lr=1e-5)

    start_ep = 1
    if ckpt_path.exists():
        # resume: the checkpoint already holds trained weights (skip the pretrained download)
        ck = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_ep = ck["epoch"] + 1
        print(f"resumed from checkpoint @ epoch {ck['epoch']} -> continuing at {start_ep}", flush=True)
    else:
        # fresh start: inject the COCO-pretrained backbone, then a 2-iter smoke (fail fast)
        pre = keypointrcnn_resnet50_fpn(weights="DEFAULT")
        model.backbone.load_state_dict(pre.backbone.state_dict()); del pre
        model.train()
        for j, (imgs, tgts) in enumerate(dl):
            loss = sum(model(list(imgs), list(tgts)).values())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5); opt.step()
            print(f"smoke iter {j+1}: loss={float(loss.detach()):.3f} finite={torch.isfinite(loss).item()}", flush=True)
            if j >= 1:
                break

    for ep in range(start_ep, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for imgs, tgts in dl:
            loss = sum(model(list(imgs), list(tgts)).values())             # composite loss
            if not torch.isfinite(loss):
                opt.zero_grad(); continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)        # grad clip 1.5
            opt.step(); tot += float(loss.detach()); n += 1
        vtot, vn = 0.0, 0                                                   # validation loss
        with torch.no_grad():
            for imgs, tgts in vdl:
                vl = sum(model(list(imgs), list(tgts)).values())
                if torch.isfinite(vl):
                    vtot += float(vl); vn += 1
        vloss = vtot / max(vn, 1)
        sched.step(vloss)                                                  # step on val loss
        print(f"epoch {ep:2d}/{epochs}  train_loss={tot/max(n,1):.3f}  "
              f"val_loss={vloss:.3f}  lr={opt.param_groups[0]['lr']:.2e}", flush=True)
        # checkpoint every epoch so a kill costs one epoch, not the whole run (resumable)
        torch.save({"epoch": ep, "model": model.state_dict(),
                    "opt": opt.state_dict(), "sched": sched.state_dict()}, ckpt_path)
    model.eval()
    return model


# ----------------------------------------------------------- register + card
def card_md(entry, n_train, train_scenes, centered, kp_err, full):
    a, t, m = entry["arch"], entry["train"], entry["metrics"]
    fs = (f"- Full-scene held-out (recall / precision / F1): **{full['recall']:.2f} / "
          f"{full['precision']:.2f} / {full['f1']:.2f}**  ·  {full['tp']}/{full['detected']} on-label, "
          f"{full['labelled']} labelled" if full else "- Full-scene held-out: n/a")
    return f"""# {entry['name']}

`{entry['id']}` · created {entry['created']} · weights `{entry['weights']}`

## What this is
A **fresh** Keypoint R-CNN moving-echo detector following **Adamiak et al. 2025**, rebuilt from scratch on the
intact data pipeline (not derived from the archived training scripts). It replaces the archived experiments as
the current, understood baseline for the detector rebuild.

## Relative to the base data and the earlier models
- **Base data:** the same 339-vehicle / 8-scene corpus, exported to **64×64 COCO chips** by `export_coco.py`
  (see [DATA.md](../../docs/DATA.md) §6). Unchanged and upstream of this model.
- **`base-default` (baseline):** the zero-effort reference (default anchors, no augmentation). This model adds
  Adamiak's architecture choices + augmentation + small anchors on top of that starting point.
- **Benchmark to beat:** the archived `kprcnn-centralia-heldout` reached **full-scene F1 ≈ 0.50** (matching Van
  Etten 2024's PlanetScope truck F1 0.49) on the same held-out scene. This first pass is measured against that.

## Methodology (Adamiak architecture)
- **Model:** Keypoint R-CNN, `keypointrcnn_resnet50_fpn` (ResNet-50 + FPN).
- **Classes:** {a.get('classes', 2)} (moving echo + background). **Keypoints:** {a.get('keypoints', 3)} per
  vehicle, order blue → red → green.
- **Anchors (the high-leverage knob):** sizes `{a.get('anchor_sizes')}`, ratios `{a.get('aspect_ratios')}` —
  small anchors in Adamiak's spirit (their swept best was 4–48 px on 512² images). **Not swept here.** Sweep
  later via `--anchor-sizes` / `--aspect-ratios` (the registry stores them, so `model_registry` rebuilds the
  right graph). Input resized `min_size={a.get('min_size')}` / `max_size={a.get('max_size')}`.
- **Training:** Adam · ReduceLROnPlateau on validation loss · LR 1e-3 → 1e-5 · grad-clip 1.5 · composite
  torchvision loss · {t.get('epochs')} epochs · batch {t.get('batch')} · {t.get('device')}.
- **Augmentation:** {t.get('aug')}.
- **Data:** trained on {n_train} vehicles across {len(train_scenes)} scenes; held out **{m.get('heldout_scene')}**.

## Deviations from Adamiak (our setup differs)
- **Finetuned from the COCO-pretrained backbone** (`weights="DEFAULT"`), not trained from scratch — our label
  count is far below their 3,236. Detection heads (RPN / box / keypoint) are trained fresh (custom
  anchors/classes/keypoints).
- **64×64 chips**, not their 512×512 images (kept our chip size; anchors may need a sweep because of it).
- **Leave-one-scene-out** split, not random 80/10/10 (a random split leaks same-scene cues and inflates).

## Results (held-out {m.get('heldout_scene')})
- Centered-chip recall: **{centered:.3f}**  ·  keypoint error: {kp_err:.1f} px  (easy metric — recognising a
  centered echo).
{fs}
- The full-scene numbers are the deployable metric and are **threshold-dependent** (evaluated at
  thr {m.get('eval_thresh')}); threshold calibration is a separate, not-yet-built item.

## Not built yet (deliberately, for later)
Keypoint correction · the anchor sweep · threshold calibration · the geometry/physics filter · velocity.
See [REFINEMENT.md](../../docs/REFINEMENT.md).
"""


def main(a):
    by_scene = load_coco()
    if a.held not in by_scene:
        raise SystemExit(f"held-out scene {a.held!r} not found. scenes: {sorted(by_scene)}")
    train_scenes = [s for s in sorted(by_scene) if s != a.held]
    all_train = [it for s in train_scenes for it in by_scene[s]]
    held_items = by_scene[a.held]

    # carve a small random val subset for the LR scheduler ONLY (the held-out scene stays
    # pure for the test metric — the leave-one-scene-out split is train vs. that scene).
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(all_train))
    nval = max(8, int(0.12 * len(all_train)))
    val_items = [all_train[i] for i in perm[:nval]]
    train_items = [all_train[i] for i in perm[nval:]]

    arch = {
        "backbone": "resnet50-fpn", "classes": 2, "keypoints": 3,
        "anchor_sizes": [int(x) for x in a.anchor_sizes.split(",")],
        "aspect_ratios": [float(x) for x in a.aspect_ratios.split(",")],
        "min_size": a.min_size, "max_size": a.max_size,
    }
    print(f"train {len(train_items)} chips ({len(train_scenes)} scenes) + {len(val_items)} val  |  "
          f"held-out {a.held} ({len(held_items)})", flush=True)
    print(f"anchors sizes={arch['anchor_sizes']} ratios={arch['aspect_ratios']}  "
          f"epochs={a.epochs} batch={a.batch} repeat={a.repeat} lr={a.lr}", flush=True)

    ckpt_path = REPO / "weights" / f"{a.id}.ckpt.pt"
    (REPO / "weights").mkdir(exist_ok=True)
    model = train(train_items, val_items, arch, a.epochs, a.batch, a.lr, a.repeat, a.seed, ckpt_path)

    centered, kp_err = eval_centered(model, held_items, a.thresh)
    full = eval_full_scene(model, a.held, a.thresh)
    print(f"\ncentered-chip recall {centered:.3f} (kp err {kp_err:.1f}px)", flush=True)
    if full:
        print(f"full-scene recall {full['recall']:.2f} precision {full['precision']:.2f} "
              f"F1 {full['f1']:.2f}  ({full['tp']}/{full['detected']} on-label / {full['labelled']} labelled)",
              flush=True)

    # save weights + register + card
    (REPO / "weights").mkdir(exist_ok=True)
    wpath = REPO / "weights" / f"{a.id}.pt"
    torch.save(model.state_dict(), wpath)

    metrics = {"heldout_scene": a.held, "eval_thresh": a.thresh,
               "heldout_recall_centered": round(centered, 3), "heldout_kp_err_px": round(kp_err, 2)}
    if full:
        metrics.update({"heldout_recall": round(full["recall"], 3),
                        "heldout_precision": round(full["precision"], 3),
                        "heldout_f1": round(full["f1"], 3)})
    entry = {
        "id": a.id, "name": a.name, "weights": f"{a.id}.pt",
        "status": "active" if a.set_active else "archived",
        "created": a.date, "card": f"cards/{a.id}.md",
        "arch": {"backbone": "resnet50-fpn", "anchors": "custom",
                 "anchor_sizes": arch["anchor_sizes"], "aspect_ratios": arch["aspect_ratios"],
                 "classes": 2, "keypoints": 3, "min_size": a.min_size, "max_size": a.max_size},
        "train": {"vehicles": len(train_items), "scenes": train_scenes, "epochs": a.epochs,
                  "batch": a.batch, "lr": a.lr, "aug": "rotate+flip+brightness+perspective (Adamiak)",
                  "finetune": "COCO-pretrained backbone (weights=DEFAULT)",
                  "device": "cpu", "script": "src/train_detector.py"},
        "metrics": metrics,
        "notes": a.notes or ("Fresh Adamiak-spec rebuild (finetuned backbone, 64px chips, small anchors, "
                             "leave-one-scene-out). First pass — see the card for methodology and how it "
                             "relates to base-default and the F1~0.50 benchmark."),
    }
    reg = mr.load()
    reg["models"] = [m for m in reg["models"] if m["id"] != a.id] + [entry]
    if a.set_active:
        reg["active"] = a.id
    mr.save(reg)

    (REPO / "models" / "cards").mkdir(parents=True, exist_ok=True)
    (REPO / "models" / entry["card"]).write_text(
        card_md(entry, len(train_items), train_scenes, centered, kp_err, full))

    ckpt_path.unlink(missing_ok=True)   # training done — drop the resume checkpoint
    print(f"\nregistered '{a.id}' ({entry['status']}) -> weights/{a.id}.pt, models/{entry['card']}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--held", default="Tacoma-Centralia_01_20260429", help="held-out scene (leave-one-scene-out)")
    p.add_argument("--anchor-sizes", dest="anchor_sizes", default="4,8,16,32,48")
    p.add_argument("--aspect-ratios", dest="aspect_ratios", default="0.25,0.5,0.75,1.0,1.25")
    p.add_argument("--min-size", dest="min_size", type=int, default=192)
    p.add_argument("--max-size", dest="max_size", type=int, default=320)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--repeat", type=int, default=3, help="augmented samples per vehicle per epoch")
    p.add_argument("--thresh", type=float, default=0.3, help="eval confidence threshold")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set-active", dest="set_active", action="store_true")
    p.add_argument("--notes", default=None)
    p.add_argument("--date", default=datetime.date.today().isoformat())
    main(p.parse_args())
