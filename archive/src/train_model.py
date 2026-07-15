#!/usr/bin/env python3
"""
Train a Keypoint R-CNN with a chosen config, evaluate it on a held-out scene, save the
weights, and register it in models/registry.json (+ a markdown card stub under
models/cards/). The single entry point for adding a model to the experiment lab.

Held-out recall here is the *centered-chip* measure (one 64px chip per labeled vehicle:
"do you recognise it?") — fast, and directly comparable across models. The harder
full-scene deployment number is obtained by running the model in the Inference tab.

Example — the vanilla BASELINE (default anchors, no augmentation, no data engineering):
  python3 src/train_model.py --id base-default --name "Base (default anchors, no aug)" \
    --train Centralia_01_20260511,Centralia_02_20260511,Tacoma-Centralia_02_20260602 \
    --held Tacoma-Centralia_01_20260429 --anchors default --aug none --epochs 12
"""
import argparse
import datetime
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

import crossval_keypoint as cv
import model_registry as mr
from export_coco import REPO


def card_stub(entry, n_train):
    a, t, m = entry["arch"], entry["train"], entry["metrics"]
    return f"""# {entry['name']}

`{entry['id']}` · created {entry['created']} · weights `{entry['weights']}`

## What this model is
_One-line purpose — edit me._

## Methodology
- **Architecture:** Keypoint R-CNN (ResNet-50 + FPN), COCO-pretrained, head → 1 class + 3 keypoints (B/R/G).
- **Anchors:** `{a['anchors']}` ({'8-128 px' if a['anchors']=='small' else '32-512 px'}).
- **Augmentation:** `{t['aug']}`.
- **Data:** {n_train} vehicles across {len(t['scenes'])} scene(s): {', '.join(t['scenes'])}.
- **Training:** {t['epochs']} epochs · batch {t.get('batch','?')} · lr {t.get('lr','?')} · {t['device']}.

## Results (held-out {m.get('heldout_scene','?')})
- Centered-chip recall: **{m.get('heldout_recall_centered','?')}**  ·  keypoint error: {m.get('heldout_kp_err_px','?')} px
- Full-scene deployment: _run in the Inference tab and record here._

## Findings / what worked / what didn't
_Edit me — this is the point of the card._

## Next ideas
_Edit me._
"""


def main(a):
    data = cv.load_scene_vehicles()
    train = [s.strip() for s in a.train.split(",")]
    n_train = sum(len(data[s][1]) for s in train if s in data)
    print(f"train {train} ({n_train} veh)  held {a.held}  anchors={a.anchors} aug={a.aug} "
          f"epochs={a.epochs}", flush=True)

    model = cv.train_fold(data, train, a.epochs, a.lr, a.batch, a.anchors, a.aug, a.seed, repeat=a.repeat)

    weights = a.weights or f"{a.id}.pt"
    (REPO / "weights").mkdir(exist_ok=True)
    torch.save(model.state_dict(), REPO / "weights" / weights)

    n, det, recall, err = cv.eval_scene(model, data, a.held)
    print(f"held-out {a.held}: {det}/{n} centered (recall {recall:.0%}), kp err {err:.1f} px", flush=True)

    entry = {
        "id": a.id, "name": a.name, "weights": weights, "status": a.status,
        "created": a.date, "card": f"cards/{a.id}.md",
        "arch": {"backbone": "resnet50-fpn", "anchors": a.anchors, "classes": 2, "keypoints": 3,
                 "min_size": 192, "max_size": 320},
        "train": {"vehicles": n_train, "scenes": train, "epochs": a.epochs, "batch": a.batch,
                  "lr": a.lr, "aug": a.aug, "anchors": a.anchors, "device": "cpu", "script": "src/train_model.py"},
        "metrics": {"heldout_scene": a.held, "heldout_recall_centered": round(recall, 3),
                    "heldout_kp_err_px": round(err, 1)},
        "notes": a.notes or f"Trained via train_model.py — anchors={a.anchors}, aug={a.aug}.",
    }
    reg = mr.load()
    reg["models"] = [m for m in reg["models"] if m["id"] != a.id] + [entry]
    if a.set_active:
        reg["active"] = a.id
    mr.save(reg)

    card = REPO / "models" / "cards" / f"{a.id}.md"
    card.parent.mkdir(parents=True, exist_ok=True)
    if not card.exists():
        card.write_text(card_stub(entry, n_train))
    print(f"registered '{a.id}'  (+ card {card.relative_to(REPO)})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--train", required=True, help="comma-separated train scenes")
    ap.add_argument("--held", required=True, help="held-out scene for eval")
    ap.add_argument("--anchors", choices=["small", "default"], default="default")
    ap.add_argument("--aug", choices=["none", "basic", "rich"], default="none")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--status", choices=["active", "archived"], default="archived")
    ap.add_argument("--set-active", action="store_true")
    ap.add_argument("--notes", default=None)
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    main(ap.parse_args())
