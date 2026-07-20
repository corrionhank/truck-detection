#!/usr/bin/env python3
"""
Local backend for the Truck Detection web console.

Bridges the React UI to the real Python model:
  GET  /api/dataset             -> true annotation counts read from the GeoPackage
  GET  /api/scenes              -> every GeoTIFF scene available to run on
  GET  /api/models              -> the model registry (active + all models, metadata)
  POST /api/models/active {id}  -> set the default model
  POST /api/models/<id> {...}   -> update a model's notes / status
  POST /api/detect {scene,...}  -> runs a model (model_id or the active one) on a scene
  GET  /outputs/<file>          -> serves the generated PNGs

Models are described in models/registry.json and built/loaded on demand by
model_registry (each with its own anchor config). Run: python3 backend/server.py (8787).
The Vite dev server proxies /api and /outputs here (see vite.config.ts).
"""
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import geopandas as gpd
from flask import Flask, jsonify, request, send_from_directory

import detect_scene as ds
import model_registry as mr

GPKG = REPO / "data" / "active" / "Annotations-RGB.gpkg"
GEOTIFF_DIR = REPO / "data" / "active" / "imagery"
OUTPUTS = REPO / "outputs"

# in-flight training jobs (id -> Popen); logs persist to weights/<id>.train.log
_train_jobs = {}
ANCHOR_PRESETS = {
    "small":   ("8,16,32,64,128", "0.5,1.0,2.0"),
    "default": ("32,64,128,256,512", "0.5,1.0,2.0"),
    "adamiak": ("4,8,16,32,48", "0.25,0.5,0.75,1.0,1.25"),
}

app = Flask(__name__)

# Preload the active model so the first detection isn't slow; others load on demand.
try:
    _entry, _ = mr.resolve(mr.load())
    print(f"active model ready: {_entry['id']} ({_entry['weights']})")
except Exception as e:
    print(f"WARNING: could not preload active model: {e}")


def dataset_stats():
    """Real counts straight from the annotation GeoPackage."""
    g = gpd.read_file(GPKG, layer="Annotations")
    g["scene"] = g["scene"].astype(str).str.strip()
    g = g[g["scene"].str.len() > 0]
    per_scene = {}
    for s in sorted(g["scene"].unique()):
        sub = g[g["scene"] == s]
        per_scene[s] = {"vehicles": int(sub["vehicle_id"].nunique()),
                        "echoes": int(len(sub))}
    return {
        "vehicles": int(g["vehicle_id"].nunique()),
        "echoes": int(len(g)),            # keypoints; 3 per vehicle (blue/red/green)
        "scenes_labelled": len(per_scene),
        "per_scene": per_scene,
    }


def all_scenes():
    """Every scene GeoTIFF, tagged with its labelled-vehicle count (0 if none)."""
    labelled = dataset_stats()["per_scene"]
    scenes = []
    for tif in sorted(GEOTIFF_DIR.glob("*.tif")):
        name = tif.stem
        scenes.append({"name": name, "vehicles": labelled.get(name, {}).get("vehicles", 0)})
    return scenes


@app.get("/api/dataset")
def api_dataset():
    return jsonify(dataset_stats())


@app.get("/api/scenes")
def api_scenes():
    return jsonify({"scenes": all_scenes()})


@app.get("/api/models")
def api_models():
    return jsonify(mr.load())


@app.post("/api/models/active")
def api_models_active():
    mid = (request.get_json(force=True, silent=True) or {}).get("id")
    reg = mr.load()
    if mr.get(reg, mid) is None:
        return jsonify({"error": f"unknown model: {mid}"}), 404
    reg["active"] = mid
    mr.save(reg)
    return jsonify({"active": mid})


@app.post("/api/models/<model_id>")
def api_models_update(model_id):
    """Edit a model's notes and/or status (archived | active)."""
    body = request.get_json(force=True, silent=True) or {}
    reg = mr.load()
    entry = mr.get(reg, model_id)
    if entry is None:
        return jsonify({"error": f"unknown model: {model_id}"}), 404
    if "notes" in body:
        entry["notes"] = str(body["notes"])
    if body.get("status") in ("active", "archived"):
        entry["status"] = body["status"]
    mr.save(reg)
    return jsonify(entry)


@app.get("/api/models/<model_id>/card")
def api_model_card(model_id):
    """The model's methodology markdown (models/cards/<id>.md)."""
    entry = mr.get(mr.load(), model_id)
    if entry is None:
        return jsonify({"error": f"unknown model: {model_id}"}), 404
    card = REPO / "models" / entry.get("card", f"cards/{model_id}.md")
    return jsonify({"markdown": card.read_text() if card.exists() else ""})


@app.post("/api/detect")
def api_detect():
    body = request.get_json(force=True, silent=True) or {}
    scene = body.get("scene")
    stride = int(body.get("stride", 40))
    thresh = float(body.get("thresh", 0.5))
    if not scene:
        return jsonify({"error": "missing 'scene'"}), 400
    try:
        entry, model = mr.resolve(mr.load(), body.get("model_id"))
        result = ds.detect(model, scene, stride=stride, thresh=thresh)
    except (FileNotFoundError, KeyError) as e:
        return jsonify({"error": str(e)}), 404
    result["model_id"] = entry["id"]
    result["model_name"] = entry["name"]
    # leakage guard: was this scene in the model's training set? -> metrics on it are inflated
    trained = entry.get("train", {}).get("scenes", []) or []
    mtr = entry.get("metrics", {})
    heldout = mtr.get("heldout_scenes") or ([mtr["heldout_scene"]] if mtr.get("heldout_scene") else [])
    result["eval_split"] = "train" if scene in trained else ("heldout" if scene in heldout else "unseen")
    result["montage_url"] = f"/outputs/{result['montage']}"
    result["preview_url"] = f"/outputs/{result['preview']}"
    return jsonify(result)


@app.post("/api/train")
def api_train():
    """Launch a training run as a background subprocess (src/train_detector.py). Returns
    immediately; poll /api/train/status?id=<id>. The run self-registers the model on success."""
    b = request.get_json(force=True, silent=True) or {}
    mid = (b.get("id") or "").strip()
    name = (b.get("name") or "").strip()
    train_scenes = [s for s in (b.get("train_scenes") or []) if s]
    held_scenes = [s for s in (b.get("held_scenes") or []) if s]
    if not mid or not name:
        return jsonify({"error": "id and name are required"}), 400
    if not train_scenes:
        return jsonify({"error": "select at least one training scene"}), 400
    if not held_scenes:
        return jsonify({"error": "select at least one held-out (test) scene"}), 400
    overlap = sorted(set(train_scenes) & set(held_scenes))
    if overlap:
        return jsonify({"error": f"scene(s) in both train and held-out: {overlap}"}), 400
    if mid in _train_jobs and _train_jobs[mid].poll() is None:
        return jsonify({"error": f"a training job for '{mid}' is already running"}), 409

    if b.get("anchors") == "custom":
        sizes = b.get("anchor_sizes", "4,8,16,32,48")
        ratios = b.get("aspect_ratios", "0.25,0.5,0.75,1.0,1.25")
    else:
        sizes, ratios = ANCHOR_PRESETS.get(b.get("anchors", "adamiak"), ANCHOR_PRESETS["adamiak"])
    cmd = [sys.executable, str(REPO / "src" / "train_detector.py"),
           "--id", mid, "--name", name,
           "--train", ",".join(train_scenes), "--held", ",".join(held_scenes),
           "--anchor-sizes", sizes, "--aspect-ratios", ratios,
           "--aug", b.get("aug", "adamiak"),
           "--epochs", str(int(b.get("epochs", 12))),
           "--batch", str(int(b.get("batch", 4))),
           "--lr", str(float(b.get("lr", 1e-3))),
           "--repeat", str(int(b.get("repeat", 2)))]
    (REPO / "weights").mkdir(exist_ok=True)
    logp = REPO / "weights" / f"{mid}.train.log"
    proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=open(logp, "w"), stderr=subprocess.STDOUT)
    _train_jobs[mid] = proc
    return jsonify({"id": mid, "started": True, "cmd": " ".join(shlex.quote(c) for c in cmd)})


@app.get("/api/train/status")
def api_train_status():
    mid = request.args.get("id", "")
    logp = REPO / "weights" / f"{mid}.train.log"
    proc = _train_jobs.get(mid)
    if proc is None and not logp.exists():
        return jsonify({"id": mid, "state": "none"})
    log = logp.read_text() if logp.exists() else ""
    done = f"registered '{mid}'" in log
    poll = proc.poll() if proc is not None else "untracked"
    if done:
        state = "done"
    elif "Traceback" in log or (isinstance(poll, int) and poll not in (0,)):
        state = "failed"
    elif poll is None:
        state = "running"
    else:
        state = "stopped"                       # log exists but no live process (e.g. backend restarted)
    m = re.findall(r"epoch\s+(\d+)/(\d+)", log)
    prog = {"epoch": int(m[-1][0]), "total": int(m[-1][1])} if m else None
    return jsonify({"id": mid, "state": state, "progress": prog,
                    "log_tail": "\n".join(log.splitlines()[-24:])})


@app.get("/outputs/<path:fname>")
def serve_output(fname):
    return send_from_directory(OUTPUTS, fname)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    app.run(host="127.0.0.1", port=args.port, debug=False)
