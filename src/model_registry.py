#!/usr/bin/env python3
"""
Model registry — the experiment lab's backbone.

Reads/writes models/registry.json (the committed experiment log: config + metrics +
notes per model) and builds/loads any registered model with the CORRECT architecture.
Different models use different anchor sets (a small-anchor model's weights will not
load into a default-anchor graph correctly), so the arch is stored per model and the
builder honors it. Weights themselves live in weights/ (gitignored, large).

Used by backend/server.py; also usable standalone:
    from model_registry import load, get, build_model, load_weights
"""
import json
from pathlib import Path

import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator

REPO = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO / "models" / "registry.json"
WEIGHTS_DIR = REPO / "weights"

ANCHOR_SETS = {"small": (8, 16, 32, 64, 128), "default": (32, 64, 128, 256, 512)}

_model_cache = {}  # id -> loaded torch model (lazy; weights are 236 MB each)


def load():
    """The whole registry dict ({'active': id, 'models': [...]})."""
    return json.loads(REGISTRY_PATH.read_text())


def save(reg):
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2) + "\n")


def get(reg, model_id):
    """The entry for one model id, or None."""
    return next((m for m in reg["models"] if m["id"] == model_id), None)


def build_model(arch):
    """Construct the Keypoint R-CNN graph described by an arch dict (random init;
    weights are loaded separately). weights=None avoids the 226 MB COCO download —
    load_weights() overwrites everything anyway."""
    sizes = ANCHOR_SETS[arch.get("anchors", "default")]
    ag = AnchorGenerator(tuple((s,) for s in sizes), ((0.5, 1.0, 2.0),) * len(sizes))
    model = keypointrcnn_resnet50_fpn(
        weights=None, weights_backbone=None,
        min_size=arch.get("min_size", 192), max_size=arch.get("max_size", 320),
        rpn_anchor_generator=ag)
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, arch.get("classes", 2))
    in_kp = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(in_kp, arch.get("keypoints", 3))
    return model


def load_weights(entry):
    """Build + load the weights for a registry entry, cached by id. Eval mode."""
    mid = entry["id"]
    if mid not in _model_cache:
        wpath = WEIGHTS_DIR / entry["weights"]
        if not wpath.exists():
            raise FileNotFoundError(f"weights missing for '{mid}': {wpath}")
        model = build_model(entry["arch"])
        model.load_state_dict(torch.load(wpath, map_location="cpu"))
        model.eval()
        _model_cache[mid] = model
    return _model_cache[mid]


def resolve(reg, model_id=None):
    """Pick the requested model, else the active one. Returns (entry, model)."""
    entry = get(reg, model_id) if model_id else get(reg, reg.get("active"))
    if entry is None:
        raise KeyError(f"model not found: {model_id or reg.get('active')}")
    return entry, load_weights(entry)


if __name__ == "__main__":  # quick check: list registered models + weight availability
    reg = load()
    print(f"active: {reg['active']}")
    for m in reg["models"]:
        have = (WEIGHTS_DIR / m["weights"]).exists()
        print(f"  [{m['status']:8}] {m['id']:26} {m['weights']:45} {'ok' if have else 'MISSING'}")
