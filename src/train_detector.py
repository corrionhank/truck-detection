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
import math
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
# WHAT THE MODEL ACTUALLY SEES
#
# We never hand the network a whole satellite scene. A scene is ~4000x4000 px of mostly empty
# ground, and the thing we want is 3-5 px across — the network would spend all its capacity on
# field and rooftop. Instead export_coco.py cuts one small square ("chip") centred on each
# labelled vehicle, and we train on those. At inference time detect_scene.py slides a window of
# the same size across the full scene, so the model only ever sees one chip-sized view either way.
#
# Two details worth teaching:
#   - Chips are grouped BY SCENE, not thrown in one pile, because we evaluate leave-one-scene-out.
#     Holding out random vehicles would let the model memorise a scene's lighting and road colour
#     from its other vehicles and score well without having learned anything transferable.
#   - A chip carries the centre vehicle FIRST, then any neighbour whose echo also fell inside the
#     window. On a busy freeway trucks are close together, so a neighbour lands in frame often. If
#     we labelled only the centre one, every neighbour would be taught to the model as background —
#     we would be actively training it to ignore real trucks.
def load_coco():
    """{scene: [(chip uint8 SxSx3, kps float32[N,3,2] center-vehicle-first, in export px), ...]}.

    S is the exported chip size (CHIP + 2*margin); the model crops CHIP from it at train time.
    Each chip carries the center vehicle plus any neighbours whose echo fell inside the window
    (multi-vehicle targets); with legacy single-vehicle chips N == 1 and S == CHIP."""
    d = json.loads((COCO / "annotations.json").read_text())
    per_img = {}
    for a in d["annotations"]:
        per_img.setdefault(a["image_id"], []).append(a)
    by_scene = {}
    for im in d["images"]:
        anns = per_img.get(im["id"])
        if not anns:
            continue
        anns = sorted(anns, key=lambda a: (not a.get("center", True), a["id"]))  # center vehicle first
        chip = np.asarray(Image.open(COCO / "images" / im["file_name"]).convert("RGB"))
        kps = np.stack([np.array(a["keypoints"], np.float32).reshape(3, 3)[:, :2] for a in anns])
        by_scene.setdefault(im["scene"], []).append((chip, kps))
    return by_scene


# ------------------------------------------------- augmentation (Adamiak) ---
# MAKING A FEW HUNDRED LABELS LOOK LIKE MANY THOUSANDS
#
# A network this size wants far more examples than we have. Augmentation gets there by showing the
# same vehicle differently every epoch: flipped, rotated, slightly brightened, slightly warped. The
# model sees a fresh image each time and cannot memorise any single one.
#
# Rotation by any angle is legitimate here in a way it is not for most vision tasks. A photo of a
# car has an up; a satellite echo does not — roads run in every direction, so a streak at 200 deg is
# a perfectly real thing to see. That makes full 360 deg rotation free extra data rather than a lie.
#
# The ordering matters and is the easiest thing to get wrong. We rotate the LARGER exported image
# (96 px) and crop the 64 px training chip afterwards. Rotate a 64 px chip directly and the corners
# have no source pixels, so they get filled with mirrored padding — and the model happily learns to
# recognise that mirrored texture, which exists nowhere in a real scene. Rotating the padded export
# means the corners get filled with genuine neighbouring ground instead.
#
# Translation is deliberately NOT done here. Shifting an image leaves the same empty-corner problem,
# so it is handled in ChipDS by moving the crop window instead — real pixels, nothing invented.
def augment(img, pts, rng, size):
    """Adamiak's augmentation: rotation, H/V flips, brightness, perspective — applied to the
    SxS image AND all (M,2) keypoints in image space. Runs on the full export (e.g. 96px) BEFORE
    the train-time crop, so rotation pulls real pixels into the corners instead of reflected pad.
    Translation is handled separately by the jittered crop in ChipDS (not here)."""
    c, k = img.copy(), pts.copy()
    if rng.random() < 0.5:                                   # horizontal flip
        c = c[:, ::-1]; k[:, 0] = size - 1 - k[:, 0]
    if rng.random() < 0.5:                                   # vertical flip
        c = c[::-1, :]; k[:, 1] = size - 1 - k[:, 1]
    c = np.ascontiguousarray(c)
    half = size / 2.0

    ang = rng.uniform(-180, 180)                             # rotation about center
    M = cv2.getRotationMatrix2D((half, half), ang, 1.0)
    c = cv2.warpAffine(c, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    k = (np.hstack([k, np.ones((len(k), 1), np.float32)]) @ M.T).astype(np.float32)

    jit = rng.uniform(-5, 5, (4, 2)).astype(np.float32)      # mild perspective warp
    src = np.array([[0, 0], [size, 0], [size, size], [0, size]], np.float32)
    P = cv2.getPerspectiveTransform(src, src + jit)
    c = cv2.warpPerspective(c, P, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    kh = np.hstack([k, np.ones((len(k), 1), np.float32)]) @ P.T
    k = (kh[:, :2] / kh[:, 2:3]).astype(np.float32)

    c = np.clip(c.astype(np.float32) * rng.uniform(0.8, 1.2), 0, 255).astype(np.uint8)  # brightness
    return np.ascontiguousarray(c), k


# THE ANSWER KEY WE HAND THE MODEL
#
# For each training chip torchvision expects a "target": for every vehicle in it, a box, a class
# label, and the keypoints. We care about the keypoints, so why a box at all? Because an R-CNN works
# in two stages — first it proposes regions that might contain something, then it looks inside a
# chosen region to place keypoints. It needs a region to be graded on. So we derive the box from the
# keypoints themselves: the smallest rectangle containing all three, plus 3 px of padding.
#
# labels = 1 everywhere because we have exactly one class, "moving echo"; class 0 is background and
# is never listed explicitly — anything not covered by a box is background by omission. The trailing
# "2" appended to each keypoint is torchvision's visibility flag, meaning "labelled and visible",
# which is always true here since an annotator only marks blobs they can actually see.
def to_target(kps):
    """torchvision Keypoint R-CNN target for N vehicles in a chip: N boxes + N labels + N
    keypoint-triples, all clipped into CHIP space. kps is float32[N,3,2]."""
    kps = kps.copy()
    kps[:, :, 0] = np.clip(kps[:, :, 0], 1, CHIP - 2)
    kps[:, :, 1] = np.clip(kps[:, :, 1], 1, CHIP - 2)
    boxes = []
    for k in kps:
        pad = 3.0
        x0, y0 = max(0.0, k[:, 0].min() - pad), max(0.0, k[:, 1].min() - pad)
        x1, y1 = min(float(CHIP), k[:, 0].max() + pad), min(float(CHIP), k[:, 1].max() + pad)
        w, h = max(x1 - x0, 4.0), max(y1 - y0, 4.0)
        boxes.append([x0, y0, x0 + w, y0 + h])
    kpts = np.concatenate([kps, np.full((len(kps), 3, 1), 2.0, np.float32)], axis=2)  # v=2 visible
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.ones(len(kps), dtype=torch.int64),   # class 1 = moving_echo
        "keypoints": torch.tensor(kpts, dtype=torch.float32),
    }


# WHY THE CROP MOVES AROUND (translation jitter)
#
# export_coco.py centres each vehicle in its exported image. Train on those directly and every truck
# the model has ever seen sat in the exact middle of the frame — so it partly learns "the truck is
# in the middle" as if that were a property of trucks. At inference the sliding window lands wherever
# it lands and the truck is usually off-centre, so that shortcut quietly costs recall.
#
# The fix: export a 96 px image, then each epoch cut the 64 px training chip at a RANDOM offset. The
# vehicle now appears left, right, high, low — but always drawn from real pixels, because the margin
# we are sliding into is genuine surrounding ground. Nothing is invented or mirrored.
#
# _offset() picks that random origin under one hard constraint: the crop must still contain the
# centre vehicle's keypoints. It computes the legal range of origins and draws inside it, so the
# jitter can never cut a vehicle in half and produce a wrong answer key. Validation uses the exact
# centre crop instead (jitter off), because a validation number that moves for random reasons cannot
# be compared between epochs.
class ChipDS(torch.utils.data.Dataset):
    """Serves CHIP-sized (64px) training tensors cropped from the SxS export chips. When S > CHIP
    (padded export) the crop origin is jittered every epoch — real-pixel translation augmentation —
    with the legal range derived from the center vehicle's keypoints so it can never cut them and
    never needs to resample. jitter=0 forces the exact center crop (the ablation control). Legacy
    64px chips (S == CHIP) degrade to the old center crop automatically."""
    def __init__(self, items, repeat, seed, aug, jitter=None):
        self.items, self.repeat, self.aug, self.jitter = items, repeat, aug, jitter
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items) * self.repeat

    def _offset(self, coords, S, margin, jitter):
        """Random 1-D crop origin that keeps every coord in [origin, origin+CHIP), clamped to the
        jitter window [margin-jitter, margin+jitter] and the valid range [0, S-CHIP]."""
        lo = max(int(np.floor(coords.max())) - CHIP + 1, margin - jitter, 0)
        hi = min(int(np.floor(coords.min())), margin + jitter, S - CHIP)
        if hi < lo:                                          # vehicle wider than the window: center it
            return max(0, min(int(round(float(coords.mean())) - CHIP // 2), S - CHIP))
        return int(self.rng.integers(lo, hi + 1))

    def __getitem__(self, i):
        chip, kps = self.items[i % len(self.items)]          # chip SxSx3, kps (N,3,2) center-first
        S = chip.shape[0]
        margin = (S - CHIP) // 2
        pts = kps.reshape(-1, 2).astype(np.float32).copy()
        if self.aug:
            chip, pts = augment(chip, pts, self.rng, S)
        kps2 = pts.reshape(len(kps), 3, 2)
        if self.aug and margin > 0:                          # jittered crop (real-pixel translation aug)
            jitter = margin if self.jitter is None else self.jitter
            ox = self._offset(kps2[0, :, 0], S, margin, jitter)
            oy = self._offset(kps2[0, :, 1], S, margin, jitter)
        else:
            ox = oy = margin                                 # deterministic center crop (val / legacy)
        crop = chip[oy:oy + CHIP, ox:ox + CHIP]
        kps2 = kps2 - np.array([ox, oy], np.float32)
        keep = [v for v in kps2 if (v >= 0).all() and (v < CHIP).all()]   # vehicles surviving the crop
        if not keep:                                         # safety: center always kept
            keep = [np.clip(kps2[0], 0.5, CHIP - 1.5)]
        img = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).float() / 255.0
        return img, to_target(np.stack(keep).astype(np.float32))


def collate(b):
    return tuple(zip(*b))


# ----------------------------------------------------------------- model ----
# BORROWING MOST OF THE NETWORK INSTEAD OF LEARNING IT (transfer learning)
#
# A ResNet-50 has ~25 M parameters in its feature extractor alone. Learning that from a few hundred
# labelled trucks is hopeless. But the early layers of any vision network learn generic things —
# edges, corners, blobs, texture — that are the same whether the input is a photograph of a dog or a
# satellite tile. So we take the backbone from a model already trained on COCO (millions of ordinary
# photographs) and keep it, which is the single reason this works at our label count.
#
# The heads on top are built fresh, and must be: COCO's model predicts 80 classes and 17 human body
# keypoints, ours predicts 1 class and 3 band positions, so the final layers are the wrong shape.
# They are also the layers that encode what the task IS, which is exactly the part we want learned
# from our data rather than inherited.
#
# The trade this makes: the network starts out good at seeing and knowing nothing about trucks,
# instead of starting out ignorant of both.
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
# TWO RECALL NUMBERS THAT MEAN VERY DIFFERENT THINGS — DO NOT CONFLATE THEM
#
# eval_centered() asks the easy question: given a chip with a truck already centred in it, does the
# model notice? It scores ~0.97, and it is nearly meaningless on its own — we handed it the answer's
# location. It is still worth measuring, because if it ever drops the model is broken outright.
#
# eval_full_scene() asks the real question: turned loose on a raw scene with no hints, sliding across
# thousands of windows of mostly empty ground, how many trucks does it find and how much of what it
# reports is real? That is the number the project lives or dies by, and it is far lower (~0.50 F1).
#
# The gap between the two is the entire difficulty of the task: recognising a truck you have been
# pointed at is easy; finding it unaided among tens of thousands of look-alike patches is not.
@torch.no_grad()
def eval_centered(model, items, thresh):
    """Centered-chip recall on the held-out scene: the center CHIP crop per vehicle (96->64)."""
    model.eval()
    det, errs = 0, []
    for chip, kps in items:
        off = (chip.shape[0] - CHIP) // 2
        crop = chip[off:off + CHIP, off:off + CHIP]
        cen = kps[0] - off                                   # center vehicle keypoints in crop space
        img = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1).float() / 255.0
        out = model([img])[0]
        if len(out["scores"]) and float(out["scores"][0]) > thresh:
            det += 1
            pred = out["keypoints"][0].numpy()[:, :2]
            errs.append(float(np.linalg.norm(pred - cen, axis=1).mean()))
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
# THE TRAINING LOOP, AND THE FOUR GUARDS AROUND IT
#
# The loop itself is ordinary: show the model a batch, it returns how wrong it was (the loss), the
# gradient says which direction each weight should move, the optimiser (Adam) takes a step. Repeat
# for a set number of passes over the data (epochs). Everything below is the machinery that keeps
# that loop from going wrong in ways we have actually been bitten by:
#
#   1. DEVICE CHOICE. Apple's GPU (MPS) runs this model but silently produces garbage — the loss
#      drifts to NaN with no error raised. So we probe it with a few real steps and fall back to CPU
#      unless the numbers stay sane. Slow and correct beats fast and wrong.
#   2. LEARNING-RATE WARMUP. The borrowed backbone is good; the fresh heads are random. At full
#      learning rate from step one, the random heads take a huge step and blow up the whole model —
#      we measured loss going 9 -> 528. Warmup starts at 1/100th of the rate and ramps up, letting
#      the heads settle before they are allowed to move fast.
#   3. GRADIENT CLIPPING. A single odd batch can produce an enormous gradient. Clipping caps its
#      length, so one bad example cannot undo an epoch of progress.
#   4. BEST-VAL CHECKPOINTING. More training is not monotonically better — at some point the model
#      starts fitting quirks of the training scenes and gets worse on everything else. So after every
#      epoch we score a held-back slice, keep whichever epoch scored best, and ship that one. An
#      earlier run shipped its final epoch when an earlier epoch was measurably better.
#
# ReduceLROnPlateau handles the other end: when validation loss stops improving it cuts the learning
# rate, which is how the model goes from broad strokes early to fine adjustments late.
def _to(imgs, tgts, device):
    """Move a collated batch (tuple of images, tuple of target dicts) onto the device."""
    return ([im.to(device) for im in imgs],
            [{k: v.to(device) for k, v in t.items()} for t in tgts])


def _mps_probe_ok():
    """torchvision detection models RUN on Apple MPS but silently diverge (loss -> NaN) on many
    PyTorch versions — no exception is raised. Probe with a few real train steps and accept MPS
    only if the loss stays finite and bounded; otherwise the caller falls back to CPU."""
    dev = torch.device("mps")
    m = keypointrcnn_resnet50_fpn(weights=None, weights_backbone=None,
                                  num_classes=2, num_keypoints=3, min_size=192, max_size=320).to(dev)
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=1e-3, momentum=0.9)
    for _ in range(4):
        imgs = [torch.rand(3, 192, 192, device=dev)]
        tgt = [{"boxes": torch.tensor([[40., 40., 150., 150.]], device=dev),
                "labels": torch.ones(1, dtype=torch.int64, device=dev),
                "keypoints": torch.tensor([[[60., 60., 1.], [95., 95., 1.], [130., 130., 1.]]], device=dev)}]
        loss = sum(m(imgs, tgt).values())
        opt.zero_grad(); loss.backward(); opt.step()
        v = float(loss.detach())
        if not math.isfinite(v) or abs(v) > 1e4:
            return False
    return True


def pick_device(prefer="auto"):
    """Prefer a GPU, fall back to CPU. Order: CUDA (real speedup) -> MPS (only if it passes the
    divergence probe) -> CPU. `prefer` in {auto, gpu, cuda, mps, cpu} forces a choice."""
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer in ("auto", "gpu", "cuda") and torch.cuda.is_available():
        print("  device: CUDA available -> using GPU", flush=True)
        return torch.device("cuda")
    if prefer in ("auto", "gpu", "mps") and torch.backends.mps.is_available():
        try:
            if _mps_probe_ok():
                print("  device: MPS probe passed -> using Apple GPU", flush=True)
                return torch.device("mps")
            print("  device: MPS available but DIVERGES on this model -> falling back to CPU", flush=True)
        except Exception as e:
            print(f"  device: MPS probe errored ({e}) -> CPU", flush=True)
    return torch.device("cpu")


def warmup_lr(opt, step, total, base):
    """Linear LR ramp base/100 -> base over the first `total` optimiser steps, then hands off to
    ReduceLROnPlateau. The fresh detection heads spike at full LR from iteration 1 (jitter-mv's
    smoke test went loss 9 -> 528 at 1e-3), which is why that run dropped to a flat 1e-4 — and a
    flat low LR undertrains, confounding its precision drop. Warmup avoids both."""
    if total <= 0 or step >= total:
        return
    for g in opt.param_groups:
        g["lr"] = base * (0.01 + 0.99 * (step + 1) / total)


def train(train_items, val_items, arch, epochs, batch, lr, repeat, seed, ckpt_path, aug_on=True,
          device=None, jitter=None, warmup=0):
    torch.manual_seed(seed)
    device = device or pick_device()
    print(f"  training on: {device.type}", flush=True)
    model = mr.build_model(arch)
    dl = DataLoader(ChipDS(train_items, repeat, seed, aug=aug_on, jitter=jitter), batch_size=batch,
                    shuffle=True, collate_fn=collate, num_workers=0)
    vdl = DataLoader(ChipDS(val_items, 1, seed + 1, aug=False), batch_size=batch,
                     shuffle=False, collate_fn=collate, num_workers=0)

    start_ep, step = 1, 0
    best = {"vloss": float("inf"), "epoch": None, "state": None}
    if ckpt_path.exists():
        # resume: the checkpoint already holds trained weights (skip the pretrained download)
        ck = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ck["model"]); model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        opt.load_state_dict(ck["opt"])
        for st in opt.state.values():                                      # optimizer state -> device
            for k, v in st.items():
                if torch.is_tensor(v):
                    st[k] = v.to(device)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.3, patience=2, min_lr=1e-5)
        sched.load_state_dict(ck["sched"])
        start_ep = ck["epoch"] + 1
        best = ck.get("best", best)
        step = warmup                                                      # warmup already served
        print(f"resumed from checkpoint @ epoch {ck['epoch']} -> continuing at {start_ep}"
              + (f" (best val {best['vloss']:.3f} @ epoch {best['epoch']})" if best["epoch"] else ""),
              flush=True)
    else:
        # fresh start: inject the COCO-pretrained backbone, then a 2-iter smoke (fail fast)
        pre = keypointrcnn_resnet50_fpn(weights="DEFAULT")
        model.backbone.load_state_dict(pre.backbone.state_dict()); del pre
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)                  # Adam
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(                # ReduceLROnPlateau
            opt, mode="min", factor=0.3, patience=2, min_lr=1e-5)
        if warmup:
            print(f"  LR warmup: {lr/100:.1e} -> {lr:.1e} over {warmup} iters", flush=True)
        model.train()
        for j, (imgs, tgts) in enumerate(dl):
            warmup_lr(opt, step, warmup, lr); step += 1                    # smoke runs warmed too
            imgs, tgts = _to(imgs, tgts, device)
            loss = sum(model(imgs, tgts).values())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5); opt.step()
            print(f"smoke iter {j+1}: loss={float(loss.detach()):.3f} finite={torch.isfinite(loss).item()}", flush=True)
            if j >= 1:
                break

    for ep in range(start_ep, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for imgs, tgts in dl:
            warmup_lr(opt, step, warmup, lr); step += 1
            imgs, tgts = _to(imgs, tgts, device)
            loss = sum(model(imgs, tgts).values())                         # composite loss
            if not torch.isfinite(loss):
                opt.zero_grad(); continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)        # grad clip 1.5
            opt.step(); tot += float(loss.detach()); n += 1
        vtot, vn = 0.0, 0                                                   # validation loss
        with torch.no_grad():
            for imgs, tgts in vdl:
                imgs, tgts = _to(imgs, tgts, device)
                vl = sum(model(imgs, tgts).values())
                if torch.isfinite(vl):
                    vtot += float(vl); vn += 1
        vloss = vtot / max(vn, 1)
        sched.step(vloss)                                                  # step on val loss
        cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        improved = vloss < best["vloss"]
        if improved:                                                       # keep the BEST epoch, not the last
            best = {"vloss": vloss, "epoch": ep, "state": {k: v.clone() for k, v in cpu_state.items()}}
        print(f"epoch {ep:2d}/{epochs}  train_loss={tot/max(n,1):.3f}  "
              f"val_loss={vloss:.3f}  lr={opt.param_groups[0]['lr']:.2e}"
              + ("  *best*" if improved else ""), flush=True)
        # checkpoint every epoch (model on CPU for portability) so a kill costs one epoch
        torch.save({"epoch": ep, "model": cpu_state, "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "best": best}, ckpt_path)
    # Ship the lowest-val-loss epoch. jitter-mv shipped epoch 12 (val 4.955) when epoch 11 was
    # better (4.715) — there was no best-val checkpoint, so the registered weights were past peak.
    if best["state"] is not None:
        model.load_state_dict(best["state"])
        print(f"\nbest-val weights: epoch {best['epoch']}/{epochs} (val {best['vloss']:.3f})"
              + ("" if best["epoch"] == epochs else "  <- final epoch was worse; restored"), flush=True)
    model.eval()
    model.to("cpu")                                                        # eval + save on CPU (detect_scene builds CPU tensors)
    return model, best


# ----------------------------------------------------------- register + card
# WHY EVERY RUN WRITES ITSELF DOWN
#
# Finishing a run produces three things: the weights, an entry in models/registry.json, and a
# human-readable card. The registry entry is not bookkeeping for its own sake — it is required to
# use the model again. The anchor sizes are part of the network's SHAPE, so weights trained with
# one anchor set cannot be loaded into a graph built with another. Storing the architecture beside
# the weights is what lets the console rebuild the right graph months later.
#
# It also makes the experiment log honest. Each entry records the exact scene list, epochs, learning
# rate, augmentation and the resulting metrics, so a later comparison between two models is a
# comparison of recorded configurations rather than of recollection.
def card_md(entry, n_train, train_scenes, held, per_scene, mean_cen, mean_f1):
    a, t, m = entry["arch"], entry["train"], entry["metrics"]
    if held:
        rows = []
        for hs in held:
            r = per_scene.get(hs, {})
            rp = f"{r['recall']:.2f} / {r['precision']:.2f}" if "recall" in r else "— / —"
            f1 = f"{r['f1']:.2f}" if "f1" in r else "—"
            rows.append(f"| `{hs}` | {r.get('vehicles', '?')} | {r.get('centered_recall', '—')} | {rp} | **{f1}** |")
        results_block = (
            "Every number below is measured on scenes the model **never saw in training** (data-separated —\n"
            "no leakage). *Centered* recall is the easy \"recognise a centered echo\" metric; *full* R/P/F1 is the\n"
            f"deployable sliding-window metric (threshold {m.get('eval_thresh')}).\n\n"
            "| held-out (untrained) scene | veh | centered | full R / P | full F1 |\n"
            "|---|---:|---:|---:|---:|\n" + "\n".join(rows) +
            f"\n\n**Mean across held-out scenes: centered {mean_cen} · full-scene F1 {mean_f1}.**")
    else:
        results_block = (
            "**Trained on all scenes — no held-out test set.** This is a **deployment model**: there are no\n"
            "labels held back to score against. Evaluate it qualitatively by running inference on **new imagery**\n"
            "in the console's Inference tab (it shows detections + the montage, with no metrics to validate\n"
            "against). For a measured generalization number, train a sibling model that holds a few scenes out.")
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
- **Data:** trained on {n_train} vehicles across {len(train_scenes)} scenes; {('held out ' + str(len(held)) + ' untrained scene(s) for testing (below)') if held else 'no held-out set (all scenes trained)'}.

## Deviations from Adamiak (our setup differs)
- **Finetuned from the COCO-pretrained backbone** (`weights="DEFAULT"`), not trained from scratch — our label
  count is far below their 3,236. Detection heads (RPN / box / keypoint) are trained fresh (custom
  anchors/classes/keypoints).
- **64×64 chips**, not their 512×512 images (kept our chip size; anchors may need a sweep because of it).
- **Leave-one-scene-out** split, not random 80/10/10 (a random split leaks same-scene cues and inflates).

## Results

{results_block}

## Not built yet (deliberately, for later)
Keypoint correction · the anchor sweep · threshold calibration · the geometry/physics filter · velocity.
See [REFINEMENT.md](../../docs/REFINEMENT.md).
"""


# CHECKING THAT "HELD OUT" REALLY MEANS HELD OUT
#
# The obvious leak is training and testing on the same scene NAME, and main() already refuses that.
# This catches the non-obvious one. Most corridors here are the same patch of ground photographed on
# different dates, so two differently-named scenes can cover nearly identical terrain. Test on one
# having trained on the other and the model gets to recognise roads, buildings and field boundaries
# it has already memorised — the score looks like generalisation but isn't.
#
# So we compare actual ground footprints, reprojected onto a common grid, and label each held-out
# scene: "spatial" (genuinely new place — a real transfer test) or "temporal" (same ground, later
# date — a much weaker claim). It warns rather than blocks, because a temporal test is still worth
# running; it just must not be reported as proof the model works somewhere new.
def footprint_split_check(train_scenes, held_scenes, decim=8):
    """Beyond the name-based leakage guard: reproject valid footprints to catch SPATIAL overlap a
    name split can't see (every corridor here is one footprint re-captured on different dates).
    Warns per held scene and returns {scene: 'temporal'|'spatial'|'unknown'} — temporal = overlaps
    a trained scene (same-ground-later-date, NOT generalization); spatial = a genuinely new place."""
    import rasterio
    from rasterio.warp import reproject, Resampling
    from export_coco import RED, GREEN, BLUE
    geo = REPO / "data" / "active" / "imagery"

    def vmask(p):
        with rasterio.open(p) as s:
            h, w = max(1, s.height // decim), max(1, s.width // decim)
            r = s.read(RED, out_shape=(h, w)).astype(np.int64)
            g = s.read(GREEN, out_shape=(h, w)); b = s.read(BLUE, out_shape=(h, w))
            return (r + g + b) > 0, s.transform * s.transform.scale(s.width / w, s.height / h), s.crs

    tmasks = {t: vmask(geo / f"{t}.tif") for t in train_scenes if (geo / f"{t}.tif").exists()}
    labels = {}
    print("spatial-overlap guard (held-out footprint vs trained scenes):", flush=True)
    for h in held_scenes:
        hp = geo / f"{h}.tif"
        if not hp.exists():
            labels[h] = "unknown"; print(f"  {h}: imagery missing -> UNKNOWN", flush=True); continue
        hm, ht, hc = vmask(hp)
        best, who = 0.0, None
        for t, (tm, tt, tc) in tmasks.items():
            dst = np.zeros(hm.shape, "uint8")
            reproject(tm.astype("uint8"), dst, src_transform=tt, src_crs=tc,
                      dst_transform=ht, dst_crs=hc, resampling=Resampling.nearest)
            frac = float((hm & (dst > 0)).sum()) / max(int(hm.sum()), 1)
            if frac > best:
                best, who = frac, t
        labels[h] = "temporal" if best > 0.2 else "spatial"
        if labels[h] == "temporal":
            print(f"  ! {h}: TEMPORAL — {best*100:.0f}% footprint overlap with trained {who} "
                  f"(same-ground-later-date, NOT spatial generalization)", flush=True)
        else:
            print(f"  {h}: spatial (clean; max overlap {best*100:.0f}%)", flush=True)
    return labels


# SPLITTING THE DATA — THE STEP MOST LIKELY TO PRODUCE A FLATTERING LIE
#
# Three separate roles, and mixing any two of them inflates the result:
#   - TRAIN scenes: the model learns from these.
#   - HELD-OUT scenes: never trained on, used once at the end for the honest score.
#   - VAL: a small random slice carved out of the TRAINING chips. Its only job is to tell the
#     learning-rate scheduler when progress has stalled and to pick the best epoch. It is drawn from
#     training scenes on purpose, so the held-out scenes stay completely untouched until the end.
#
# The hard rule is leave-one-SCENE-out, never a random split of vehicles. Split randomly and the
# same scene appears on both sides: the model sees that road's exact surface, lighting and vehicle
# mix during training and is then tested on it. The score comes out much higher and means nothing,
# because deployment always means a scene the model has never seen.
def main(a):
    by_scene = load_coco()
    held = [s.strip() for s in a.held.split(",") if s.strip()]
    bad = [s for s in held if s not in by_scene]
    if bad:
        raise SystemExit(f"held-out scene(s) not found: {bad}. available: {sorted(by_scene)}")
    if a.train:
        train_scenes = [s.strip() for s in a.train.split(",") if s.strip()]
        bad_t = [s for s in train_scenes if s not in by_scene]
        if bad_t:
            raise SystemExit(f"train scene(s) not found: {bad_t}. available: {sorted(by_scene)}")
    else:
        train_scenes = [s for s in sorted(by_scene) if s not in held]   # everything not held out
    excl = [s.strip() for s in (a.exclude or "").split(",") if s.strip()]
    if excl:
        train_scenes = [s for s in train_scenes if s not in excl]       # dropped from training, NOT held out
        print(f"excluded from training (neither trained nor scored): {excl}", flush=True)
    leak = sorted(set(train_scenes) & set(held))
    if leak:
        raise SystemExit(f"LEAKAGE: scene(s) in BOTH train and held-out: {leak}")
    if not train_scenes:
        raise SystemExit("no training scenes selected")
    split_type = footprint_split_check(train_scenes, held)              # spatial guard (names can't see it)
    all_train = [it for s in train_scenes for it in by_scene[s]]

    # carve a small random val subset for the LR scheduler ONLY (the held-out scenes stay
    # pure for the test metric — the split is train scenes vs. held-out scenes).
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(all_train))
    nval = max(8, int(0.12 * len(all_train)))
    val_items = [all_train[i] for i in perm[:nval]]
    if held:
        train_items = [all_train[i] for i in perm[nval:]]   # held-out testing: val disjoint from train
    else:
        train_items = all_train                             # deployment (no held-out): train on ALL chips;
        #                                                     val overlaps train, used only for the LR signal

    arch = {
        "backbone": "resnet50-fpn", "classes": 2, "keypoints": 3,
        "anchor_sizes": [int(x) for x in a.anchor_sizes.split(",")],
        "aspect_ratios": [float(x) for x in a.aspect_ratios.split(",")],
        "min_size": a.min_size, "max_size": a.max_size,
    }
    print(f"train {len(train_items)} chips / {len(train_scenes)} scenes + {len(val_items)} val  |  "
          f"held-out {held} ({sum(len(by_scene[s]) for s in held)} veh)", flush=True)
    print(f"anchors sizes={arch['anchor_sizes']} ratios={arch['aspect_ratios']}  "
          f"epochs={a.epochs} batch={a.batch} repeat={a.repeat} lr={a.lr}", flush=True)

    device = pick_device(a.device)
    ckpt_path = REPO / "weights" / f"{a.id}.ckpt.pt"
    (REPO / "weights").mkdir(exist_ok=True)
    model, best = train(train_items, val_items, arch, a.epochs, a.batch, a.lr, a.repeat, a.seed, ckpt_path,
                        aug_on=(a.aug != "none"), device=device,
                        jitter=(None if a.jitter < 0 else a.jitter), warmup=a.warmup)

    # evaluate each held-out scene independently — the honest, untrained test set
    per_scene, f1s, cens = {}, [], []
    for hs in held:
        c, e = eval_centered(model, by_scene[hs], a.thresh)
        f = eval_full_scene(model, hs, a.thresh)
        row = {"vehicles": len(by_scene[hs]), "centered_recall": round(c, 3), "kp_err_px": round(e, 2)}
        if f:
            row.update({"recall": round(f["recall"], 3), "precision": round(f["precision"], 3),
                        "f1": round(f["f1"], 3), "detected": f["detected"], "labelled": f["labelled"]})
            f1s.append(f["f1"])
        cens.append(c)
        per_scene[hs] = row
        line = f"\n[{hs}]  centered {c:.3f} (kp {e:.1f}px)"
        if f:
            line += f"  |  full R/P/F1 {f['recall']:.2f}/{f['precision']:.2f}/{f['f1']:.2f}"
        print(line, flush=True)
    mean_cen = round(sum(cens) / len(cens), 3) if cens else None
    mean_f1 = round(sum(f1s) / len(f1s), 3) if f1s else None
    if held:
        print(f"\nMEAN over {len(held)} held-out scene(s): centered {mean_cen}  full-scene F1 {mean_f1}", flush=True)
    else:
        print("\ntrained on ALL scenes — no held-out set (deployment model; evaluate on new imagery)", flush=True)

    (REPO / "weights").mkdir(exist_ok=True)
    wpath = REPO / "weights" / f"{a.id}.pt"
    torch.save(model.state_dict(), wpath)

    metrics = {"heldout_scenes": held, "heldout_split_type": split_type,
               "eval_thresh": a.thresh, "per_scene": per_scene}
    if mean_cen is not None:
        metrics["heldout_recall_centered_mean"] = mean_cen
    if mean_f1 is not None:
        metrics["heldout_f1_mean"] = mean_f1
    entry = {
        "id": a.id, "name": a.name, "weights": f"{a.id}.pt",
        "status": "active" if a.set_active else "archived",
        "created": a.date, "card": f"cards/{a.id}.md",
        "arch": {"backbone": "resnet50-fpn", "anchors": "custom",
                 "anchor_sizes": arch["anchor_sizes"], "aspect_ratios": arch["aspect_ratios"],
                 "classes": 2, "keypoints": 3, "min_size": a.min_size, "max_size": a.max_size},
        "train": {"vehicles": len(train_items), "scenes": train_scenes, "epochs": a.epochs,
                  "batch": a.batch, "lr": a.lr, "warmup_iters": a.warmup,
                  "best_val_epoch": best["epoch"],
                  "best_val_loss": round(best["vloss"], 4) if best["epoch"] else None,
                  "jitter_px": (None if a.jitter < 0 else a.jitter),
                  "aug": "rotate+flip+brightness+perspective (Adamiak)" if a.aug != "none" else "none",
                  "finetune": "COCO-pretrained backbone (weights=DEFAULT)",
                  "device": device.type, "script": "src/train_detector.py"},
        "metrics": metrics,
        "notes": a.notes or (
            (f"Adamiak-spec detector (finetuned backbone, 64px chips, small anchors), trained on ALL "
             f"{len(train_scenes)} scenes — no held-out set (deployment model; evaluate on new imagery).")
            if not held else
            (f"Adamiak-spec detector (finetuned backbone, 64px chips, small anchors). Trained on "
             f"{len(train_scenes)} scenes, tested on {len(held)} untrained held-out scene(s): "
             f"{', '.join(held)}. See the card for per-scene results.")),
    }
    reg = mr.load()
    reg["models"] = [m for m in reg["models"] if m["id"] != a.id] + [entry]
    if a.set_active:
        reg["active"] = a.id
    mr.save(reg)

    (REPO / "models" / "cards").mkdir(parents=True, exist_ok=True)
    (REPO / "models" / entry["card"]).write_text(
        card_md(entry, len(train_items), train_scenes, held, per_scene, mean_cen, mean_f1))

    ckpt_path.unlink(missing_ok=True)   # training done — drop the resume checkpoint
    print(f"\nregistered '{a.id}' ({entry['status']}) -> weights/{a.id}.pt, models/{entry['card']}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--held", default="Tacoma-Centralia_01_20260429",
                   help="comma-separated held-out (untrained) TEST scenes")
    p.add_argument("--exclude", default=None,
                   help="comma-separated scenes to drop from training WITHOUT holding them out (neither trained nor scored)")
    p.add_argument("--train", default=None,
                   help="comma-separated TRAIN scenes (default: every scene not held out)")
    p.add_argument("--anchor-sizes", dest="anchor_sizes", default="4,8,16,32,48")
    p.add_argument("--aspect-ratios", dest="aspect_ratios", default="0.25,0.5,0.75,1.0,1.25")
    p.add_argument("--min-size", dest="min_size", type=int, default=192)
    p.add_argument("--max-size", dest="max_size", type=int, default=320)
    p.add_argument("--device", choices=["auto", "gpu", "cuda", "mps", "cpu"], default="auto",
                   help="auto = CUDA if present, else MPS if it passes the divergence probe, else CPU")
    p.add_argument("--aug", choices=["none", "adamiak"], default="adamiak",
                   help="augmentation: 'adamiak' (rotate/flip/brightness/perspective) or 'none'")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=300,
                   help="linear LR warmup iters (lr/100 -> lr) before ReduceLROnPlateau takes over; "
                        "0 disables. Lets the fresh heads settle at full lr instead of spiking")
    p.add_argument("--repeat", type=int, default=3, help="augmented samples per vehicle per epoch")
    p.add_argument("--jitter", type=int, default=-1,
                   help="max crop-translation px (default: derive from chip margin; 0 = center-crop ablation control)")
    p.add_argument("--thresh", type=float, default=0.3, help="eval confidence threshold")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set-active", dest="set_active", action="store_true")
    p.add_argument("--notes", default=None)
    p.add_argument("--date", default=datetime.date.today().isoformat())
    main(p.parse_args())
