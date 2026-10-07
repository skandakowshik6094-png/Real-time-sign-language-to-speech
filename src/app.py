"""
app.py -- Flask web server for SignSense.

Endpoints:
  GET  /                   -- Dashboard HTML
  GET  /video_feed         -- MJPEG camera stream
  GET  /events             -- Server-Sent Events (detection results)
  GET  /status             -- JSON status snapshot
  POST /control/start      -- Start detection  {camera_id: int}
  POST /control/stop       -- Stop detection
  POST /tts/toggle         -- Toggle TTS on/off
  POST /tts/settings       -- {rate: int, volume: float 0-1}
  POST /threshold          -- {threshold: float 0-1}
  POST /buffer/clear       -- Reset server-side cooldown state
"""
from __future__ import annotations

import json
import os
import sys
import time
import threading

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from flask import Flask, Response, render_template, jsonify, request, stream_with_context

from detection_engine import DetectionEngine

app = Flask(__name__, template_folder=os.path.join(_SRC_DIR, "templates"))
engine = DetectionEngine()

# ── Training state (shared between routes and background thread) ────────────
_train_state: dict = {"state": "idle", "progress": 0, "total": 0,
                      "logs": [], "error": None}
_train_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", classes=engine.CLASSES)


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------

@app.route("/video_feed")
def video_feed():
    """MJPEG stream of annotated camera frames."""
    def generate():
        while True:
            with engine._frame_lock:
                frame = engine.latest_frame
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            time.sleep(0.03)          # ~33 fps cap

    return Response(
        stream_with_context(generate()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"},
    )


@app.route("/events")
def events():
    """Server-Sent Events stream for real-time detection results."""
    def generate():
        q = engine.subscribe()
        try:
            # Immediately send current status so the UI can sync on connect
            yield f"data: {json.dumps({'type': 'status', **engine.get_status()})}\n\n"
            while True:
                try:
                    event = q.get(timeout=15.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except Exception:
                    # Heartbeat keeps the connection alive through proxies
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        finally:
            engine.unsubscribe(q)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Detection control
# ---------------------------------------------------------------------------

@app.route("/control/start", methods=["POST"])
def start():
    data = request.get_json(silent=True) or {}
    engine.start(camera_id=int(data.get("camera_id", 0)))
    return jsonify({"ok": True, "running": engine.running})


@app.route("/control/stop", methods=["POST"])
def stop():
    engine.stop()
    return jsonify({"ok": True, "running": engine.running})


@app.route("/status")
def status():
    return jsonify(engine.get_status())


# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

@app.route("/tts/toggle", methods=["POST"])
def tts_toggle():
    engine.tts_enabled = not engine.tts_enabled
    return jsonify({"ok": True, "tts_enabled": engine.tts_enabled})


@app.route("/tts/settings", methods=["POST"])
def tts_settings():
    data = request.get_json(silent=True) or {}
    if "rate" in data:
        engine.tts_rate = int(data["rate"])
    if "volume" in data:
        engine.tts_volume = float(data["volume"])
    return jsonify({"ok": True, "tts_rate": engine.tts_rate, "tts_volume": engine.tts_volume})


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------

@app.route("/threshold", methods=["POST"])
def set_threshold():
    data = request.get_json(silent=True) or {}
    engine.threshold = max(0.05, min(0.99, float(data.get("threshold", engine.threshold))))
    return jsonify({"ok": True, "threshold": engine.threshold})


# ---------------------------------------------------------------------------
# Buffer (server resets TTS cooldown; actual buffer lives in the browser)
# ---------------------------------------------------------------------------

@app.route("/buffer/clear", methods=["POST"])
def clear_buffer():
    engine._current_letter = ""   # silence TTS immediately
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Symbol Manager
# ---------------------------------------------------------------------------

@app.route("/symbols")
def list_symbols():
    """Return all classes with sample counts from the landmark dataset."""
    from landmark_trainer import load_landmark_dataset
    classes = engine.CLASSES
    dataset = load_landmark_dataset()
    counts: dict = {}
    for s in dataset.get("samples", []):
        lbl = s.get("label", "")
        counts[lbl] = counts.get(lbl, 0) + 1
    return jsonify({"classes": [{"name": c, "samples": counts.get(c, 0)} for c in classes]})


@app.route("/symbols/delete", methods=["POST"])
def delete_symbol():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").lower().strip()
    if not name:
        return jsonify({"ok": False, "error": "No name provided"}), 400
    ok = engine.delete_class(name)
    return jsonify({"ok": ok})


@app.route("/collect/start", methods=["POST"])
def collect_start():
    data = request.get_json(silent=True) or {}
    class_name = data.get("class_name", "").strip()
    n_samples = int(data.get("n_samples", 60))
    replace = bool(data.get("replace", False))
    if not class_name:
        return jsonify({"ok": False, "error": "No class_name"}), 400
    if not engine.running:
        return jsonify({"ok": False, "error": "Start detection first"}), 400
    engine.start_collecting(class_name, n_samples, replace)
    return jsonify({"ok": True})


@app.route("/collect/stop", methods=["POST"])
def collect_stop():
    result = engine.stop_collecting()
    return jsonify({"ok": True, **result})


@app.route("/collect/status")
def collect_status():
    return jsonify(engine.get_collection_status())


@app.route("/train/start", methods=["POST"])
def train_start():
    global _train_state
    with _train_lock:
        if _train_state.get("state") == "running":
            return jsonify({"ok": False, "error": "Already training"}), 400
        _train_state = {"state": "running", "progress": 0, "total": 40,
                        "logs": [], "error": None}

    def _do_train():
        global _train_state
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset, random_split
            from landmark_trainer import load_landmark_dataset
            from landmark_model import LandmarkClassifier
            from utils.setup import get_classes

            dataset = load_landmark_dataset()
            samples  = dataset.get("samples", [])
            classes  = get_classes()
            num_classes = len(classes)

            # ── Collect valid samples & track which classes have data ────────
            X_data, y_data = [], []
            classes_with_data: set = set()
            for item in samples:
                lbl   = item.get("label")
                feats = item.get("features", [])
                if lbl not in classes or len(feats) != 63:
                    continue
                c_idx = classes.index(lbl)
                if c_idx < num_classes:
                    X_data.append(feats)
                    y_data.append(c_idx)
                    classes_with_data.add(lbl)

            if not X_data:
                with _train_lock:
                    _train_state["state"] = "error"
                    _train_state["error"] = "No valid training samples found in the dataset"
                return

            data_indices = sorted(classes.index(c) for c in classes_with_data)

            with _train_lock:
                _train_state["logs"].append(
                    f"New data: {len(X_data)} samples for {sorted(classes_with_data)}"
                )

            # ── Load existing checkpoint as starting point ───────────────────
            ckpt_path = "checkpoints/landmark_model.pt"
            model = LandmarkClassifier(num_classes=num_classes)

            if os.path.exists(ckpt_path):
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    if isinstance(ckpt, dict) and "state_dict" in ckpt:
                        old_cls = ckpt.get("classes", [])
                        old_n   = len(old_cls)
                        # Recreate old model and load weights
                        old_m = LandmarkClassifier(num_classes=old_n)
                        old_m.load_state_dict(ckpt["state_dict"])
                        # Transfer backbone and old output neurons
                        with torch.no_grad():
                            for layer in ("fc1", "bn1", "fc2", "bn2"):
                                getattr(model, layer).load_state_dict(
                                    getattr(old_m, layer).state_dict()
                                )
                            n_copy = min(old_n, num_classes)
                            model.fc3.weight.data[:n_copy] = old_m.fc3.weight.data[:n_copy].clone()
                            model.fc3.bias.data[:n_copy]   = old_m.fc3.bias.data[:n_copy].clone()
                    with _train_lock:
                        _train_state["logs"].append(
                            "Loaded existing checkpoint — fine-tuning (old symbols preserved)"
                        )
                except Exception as e:
                    with _train_lock:
                        _train_state["logs"].append(f"Could not load checkpoint ({e}) — training fresh")

            # ── Freeze backbone so old features stay intact ──────────────────
            for name, param in model.named_parameters():
                param.requires_grad = ("fc3" in name)

            # Keep frozen BN layers in eval mode (use stored running stats)
            model.bn1.eval()
            model.bn2.eval()

            # ── Gradient mask: only update fc3 rows for classes WITH data ────
            def _mask_weight(grad):
                m = torch.zeros_like(grad)
                for i in data_indices:
                    m[i] = 1.0
                return grad * m

            def _mask_bias(grad):
                m = torch.zeros_like(grad)
                for i in data_indices:
                    m[i] = 1.0
                return grad * m

            model.fc3.weight.register_hook(_mask_weight)
            model.fc3.bias.register_hook(_mask_bias)

            # ── Training loop ────────────────────────────────────────────────
            EPOCHS = 60   # more epochs since only the output layer is trained
            with _train_lock:
                _train_state["total"] = EPOCHS

            X_t = torch.tensor(X_data, dtype=torch.float32)
            y_t = torch.tensor(y_data, dtype=torch.long)
            ds  = TensorDataset(X_t, y_t)
            val_sz   = max(1, int(len(ds) * 0.2))
            train_ds, val_ds = random_split(ds, [len(ds) - val_sz, val_sz])
            tr_ld = DataLoader(train_ds, batch_size=16, shuffle=True,  drop_last=False)
            va_ld = DataLoader(val_ds,   batch_size=16, drop_last=False)

            trainable = [p for p in model.parameters() if p.requires_grad]
            opt  = optim.Adam(trainable, lr=1e-3, weight_decay=1e-4)
            crit = nn.CrossEntropyLoss()

            for epoch in range(1, EPOCHS + 1):
                model.fc3.train()   # only fc3 needs train mode
                ep_loss = 0.0
                for Xb, yb in tr_ld:
                    opt.zero_grad()
                    loss = crit(model(Xb), yb)
                    loss.backward()
                    opt.step()
                    ep_loss += loss.item()

                with _train_lock:
                    _train_state["progress"] = epoch

                if epoch % 10 == 0 or epoch == EPOCHS:
                    model.eval()
                    correct = total = 0
                    with torch.no_grad():
                        for Xv, yv in va_ld:
                            preds = model(Xv).argmax(-1)
                            correct += (preds == yv).sum().item()
                            total   += len(yv)
                    acc = correct / max(1, total)
                    with _train_lock:
                        _train_state["logs"].append(
                            f"Epoch {epoch}/{EPOCHS} — Loss {ep_loss/len(tr_ld):.4f} | Val {acc*100:.1f}%"
                        )

            # ── Save & hot-reload ────────────────────────────────────────────
            torch.save({"state_dict": model.state_dict(),
                        "classes": classes, "num_classes": num_classes}, ckpt_path)
            engine._load_model()
            engine.model.eval()
            engine.CLASSES = classes

            with _train_lock:
                _train_state["state"] = "done"
                _train_state["logs"].append(
                    f"✅ Saved & reloaded — {num_classes} classes "
                    f"({len(classes_with_data)} updated, "
                    f"{num_classes - len(classes_with_data)} preserved from old model)"
                )

        except Exception as exc:
            import traceback
            with _train_lock:
                _train_state["state"] = "error"
                _train_state["error"] = str(exc)

    threading.Thread(target=_do_train, daemon=True, name="TrainThread").start()
    return jsonify({"ok": True})



@app.route("/train/status")
def train_status_route():
    with _train_lock:
        return jsonify(dict(_train_state))



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("  \033[1;35mSignSense\033[0m  --  web frontend at \033[1;36mhttp://localhost:5000\033[0m")
    print()
    app.run(debug=False, threaded=True, host="0.0.0.0", port=5000)
