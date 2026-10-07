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
        # Normalize to lowercase to match config class names
        lbl = s.get("label", "").lower().strip()
        counts[lbl] = counts.get(lbl, 0) + 1
    return jsonify({"classes": [{"name": c, "samples": counts.get(c.lower().strip(), 0)} for c in classes]})


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


@app.route("/debug/model")
def debug_model():
    """Temporary debug: test model inference live from the engine instance."""
    import torch, numpy as np
    try:
        model = engine.model
        model.eval()

        # 1. Test with zeros
        x0 = torch.zeros(1, 63)
        i0, c0, p0 = model.predict_probs(x0)

        # 2. Test with realistic hand-like features (normalized, wrist=0)
        rng = np.random.default_rng(42)
        hand_like = rng.normal(0, 0.3, 63).astype(np.float32)
        hand_like[:3] = 0.0  # wrist = origin
        x1 = torch.tensor(hand_like).unsqueeze(0)
        i1, c1, p1 = model.predict_probs(x1)

        # 3. Check prob buffer
        buf_len = len(engine._prob_buf)
        latest  = engine.latest_detection
        thresh  = engine.threshold

        return jsonify({
            "num_classes": model.num_classes,
            "model_training_mode": model.training,
            "bn1_tracked": int(model.bn1.num_batches_tracked.item()),
            "zero_input": {"class": engine.CLASSES[i0], "conf_pct": round(c0*100,1)},
            "random_input": {"class": engine.CLASSES[i1], "conf_pct": round(c1*100,1)},
            "prob_buf_len": buf_len,
            "latest_detection": latest,
            "threshold": thresh,
            "top3_probs_random": sorted(
                [{"cls": engine.CLASSES[i], "p": round(p,4)} for i,p in enumerate(p1)],
                key=lambda x: -x["p"])[:3],
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()})


@app.route("/train/retrain_all", methods=["POST"])
def train_retrain_all():
    """Retrain the ENTIRE model from scratch on ALL saved samples.
    Use this when the backbone is corrupted / model gives wrong class for everything.
    Training data (landmark_dataset.json) is NOT deleted — only the checkpoint is rebuilt."""
    global _train_state
    with _train_lock:
        if _train_state.get("state") == "running":
            return jsonify({"ok": False, "error": "Already training"}), 400
        _train_state = {"state": "running", "progress": 0, "total": 150,
                        "logs": ["🔄 Full retrain from scratch — backbone + output layer"], "error": None}

    def _do_retrain_all():
        global _train_state
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset, random_split
            from landmark_trainer import load_landmark_dataset
            from landmark_model import LandmarkClassifier
            from utils.setup import get_classes

            dataset     = load_landmark_dataset()
            samples     = dataset.get("samples", [])
            all_classes = get_classes()
            num_classes = len(all_classes)
            # Normalize to lowercase for case-insensitive matching
            # (dataset stores labels as uppercase, config uses lowercase)
            class_to_idx = {c.lower().strip(): i for i, c in enumerate(all_classes)}

            X_data, y_data = [], []
            class_counts: dict = {}
            for item in samples:
                raw_lbl = item.get("label", "")
                lbl     = raw_lbl.lower().strip()   # normalize
                feats   = item.get("features", [])
                if lbl not in class_to_idx or len(feats) != 63:
                    continue
                X_data.append(feats)
                y_data.append(class_to_idx[lbl])
                class_counts[lbl] = class_counts.get(lbl, 0) + 1

            if not X_data:
                with _train_lock:
                    _train_state["state"] = "error"
                    _train_state["error"] = "No training samples found in dataset."
                return

            with _train_lock:
                _train_state["logs"].append(
                    f"📦 {len(X_data)} samples across {len(class_counts)} classes"
                )

            # Train from scratch — no weight copying
            model = LandmarkClassifier(num_classes=num_classes)

            # 3× augmentation with Gaussian noise
            X_t = torch.tensor(X_data, dtype=torch.float32)
            y_t = torch.tensor(y_data, dtype=torch.long)
            noise = 0.02
            X_aug = torch.cat([
                X_t,
                X_t + torch.randn_like(X_t) * noise,
                X_t + torch.randn_like(X_t) * noise,
            ], dim=0)
            y_aug = y_t.repeat(3)

            ds     = TensorDataset(X_aug, y_aug)
            val_sz = max(1, int(len(ds) * 0.15))
            train_ds, val_ds = random_split(ds, [len(ds) - val_sz, val_sz])
            tr_ld = DataLoader(train_ds, batch_size=32, shuffle=True,  drop_last=False)
            va_ld = DataLoader(val_ds,   batch_size=32, drop_last=False)

            EPOCHS = 150
            with _train_lock:
                _train_state["total"] = EPOCHS

            opt   = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
            crit  = nn.CrossEntropyLoss(label_smoothing=0.05)

            best_acc   = 0.0
            best_state = None
            patience   = 25
            no_improve = 0

            for epoch in range(1, EPOCHS + 1):
                model.train()
                ep_loss = 0.0
                for Xb, yb in tr_ld:
                    opt.zero_grad()
                    loss = crit(model(Xb), yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    opt.step()
                    ep_loss += loss.item()
                sched.step()

                with _train_lock:
                    _train_state["progress"] = epoch

                if epoch % 15 == 0 or epoch == EPOCHS:
                    model.eval()
                    correct = total = 0
                    with torch.no_grad():
                        for Xv, yv in va_ld:
                            preds = model(Xv).argmax(-1)
                            correct += (preds == yv).sum().item()
                            total   += len(yv)
                    acc = correct / max(1, total)
                    lr  = sched.get_last_lr()[0]
                    with _train_lock:
                        _train_state["logs"].append(
                            f"Epoch {epoch}/{EPOCHS} — Loss {ep_loss/max(1,len(tr_ld)):.4f}"
                            f" | Val {acc*100:.1f}% | LR {lr:.5f}"
                        )
                    if acc > best_acc + 0.001:
                        best_acc   = acc
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}
                        no_improve = 0
                    else:
                        no_improve += 15
                    if no_improve >= patience and epoch >= 45:
                        with _train_lock:
                            _train_state["logs"].append(
                                f"⏹ Early stop at epoch {epoch} — best val {best_acc*100:.1f}%"
                            )
                        break

            if best_state:
                model.load_state_dict(best_state)

            model.eval()
            ckpt_path = "checkpoints/landmark_model.pt"
            torch.save({"state_dict": model.state_dict(),
                        "classes": all_classes,
                        "num_classes": num_classes}, ckpt_path)
            engine._load_model()
            engine.model.eval()
            engine.CLASSES = all_classes

            with _train_lock:
                _train_state["state"] = "done"
                _train_state["logs"].append(
                    f"✅ Full retrain done — {num_classes} classes"
                    f"  |  Best val acc: {best_acc*100:.1f}%"
                )

        except Exception as exc:
            import traceback
            with _train_lock:
                _train_state["state"] = "error"
                _train_state["error"] = str(exc)

    threading.Thread(target=_do_retrain_all, daemon=True, name="RetrainAll").start()
    return jsonify({"ok": True})


@app.route("/train/start", methods=["POST"])
def train_start():
    global _train_state
    with _train_lock:
        if _train_state.get("state") == "running":
            return jsonify({"ok": False, "error": "Already training"}), 400
        _train_state = {"state": "running", "progress": 0, "total": 100,
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

            dataset     = load_landmark_dataset()
            samples     = dataset.get("samples", [])
            all_classes = get_classes()
            num_classes = len(all_classes)

            # Identify which classes are already in the checkpoint
            ckpt_path   = "checkpoints/landmark_model.pt"
            old_classes = []
            old_state   = {}

            if os.path.exists(ckpt_path):
                try:
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    if isinstance(ckpt, dict) and "state_dict" in ckpt:
                        old_classes = ckpt.get("classes", [])
                        old_state   = ckpt["state_dict"]
                except Exception as e:
                    with _train_lock:
                        _train_state["logs"].append(
                            f"Warning: Could not read checkpoint ({e}) - training fresh"
                        )

            new_classes = [c for c in all_classes if c not in old_classes]

            if not new_classes:
                with _train_lock:
                    _train_state["state"] = "done"
                    _train_state["logs"].append(
                        "No new symbols to train. Collect samples for a new symbol first."
                    )
                return

            with _train_lock:
                _train_state["logs"].append(
                    f"New: {new_classes}  |  Preserved: {old_classes}"
                )

            # Gather samples only for NEW classes
            new_indices = {c: all_classes.index(c) for c in new_classes}
            X_data, y_data = [], []
            classes_with_data = set()

            for item in samples:
                raw_lbl = item.get("label", "")
                lbl     = raw_lbl.lower().strip()   # normalize case
                feats   = item.get("features", [])
                if lbl not in new_classes or len(feats) != 63:
                    continue
                X_data.append(feats)
                y_data.append(new_indices[lbl])
                classes_with_data.add(lbl)

            if not X_data:
                with _train_lock:
                    _train_state["state"] = "error"
                    _train_state["error"] = (
                        f"No samples found for {new_classes}. Collect data first."
                    )
                return

            missing = [c for c in new_classes if c not in classes_with_data]
            if missing:
                with _train_lock:
                    _train_state["logs"].append(f"No data yet for: {missing} - skipping.")

            trainable_indices = sorted(new_indices[c] for c in classes_with_data)

            with _train_lock:
                _train_state["logs"].append(
                    f"{len(X_data)} samples for {sorted(classes_with_data)}"
                )

            # Build model and carry over all existing weights
            model = LandmarkClassifier(num_classes=num_classes)

            if old_state and old_classes:
                old_n = len(old_classes)
                try:
                    old_m = LandmarkClassifier(num_classes=old_n)
                    old_m.load_state_dict(old_state)
                    with torch.no_grad():
                        # Copy backbone layers verbatim
                        for layer in ("fc1", "bn1", "fc2", "bn2"):
                            getattr(model, layer).load_state_dict(
                                getattr(old_m, layer).state_dict()
                            )
                        # Copy old output neurons exactly (new neurons start random)
                        n_copy = min(old_n, num_classes)
                        model.fc3.weight.data[:n_copy] = old_m.fc3.weight.data[:n_copy].clone()
                        model.fc3.bias.data[:n_copy]   = old_m.fc3.bias.data[:n_copy].clone()
                    with _train_lock:
                        _train_state["logs"].append(
                            f"Copied {n_copy} old symbol weights. Backbone frozen."
                        )
                except Exception as e:
                    with _train_lock:
                        _train_state["logs"].append(f"Weight copy failed ({e}) - fresh start")

            # Freeze backbone, unlock only fc3
            for param in model.parameters():
                param.requires_grad = False
            model.fc3.weight.requires_grad = True
            model.fc3.bias.requires_grad   = True

            # BN stays in eval mode — never update running stats
            model.bn1.eval()
            model.bn2.eval()

            # Rows to protect in fc3 (old class rows that must never change)
            protect_rows = [i for i in range(num_classes) if i not in trainable_indices]

            # Data augmentation: Gaussian noise (3x dataset)
            X_t = torch.tensor(X_data, dtype=torch.float32)
            y_t = torch.tensor(y_data, dtype=torch.long)
            noise = 0.025
            X_aug = torch.cat([X_t,
                                X_t + torch.randn_like(X_t) * noise,
                                X_t + torch.randn_like(X_t) * noise], dim=0)
            y_aug = y_t.repeat(3)

            ds     = TensorDataset(X_aug, y_aug)
            val_sz = max(1, int(len(ds) * 0.15))
            train_ds, val_ds = random_split(ds, [len(ds) - val_sz, val_sz])
            tr_ld = DataLoader(train_ds, batch_size=32, shuffle=True,  drop_last=False)
            va_ld = DataLoader(val_ds,   batch_size=32, drop_last=False)

            # Adam (NOT AdamW) with no weight_decay — AdamW's decay would modify
            # old fc3 rows every step even with zero gradient, causing drift.
            EPOCHS = 100
            with _train_lock:
                _train_state["total"] = EPOCHS

            opt   = optim.Adam([model.fc3.weight, model.fc3.bias], lr=3e-3)
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
            crit  = nn.CrossEntropyLoss(label_smoothing=0.05)

            best_acc   = 0.0
            best_state = None
            patience   = 20
            no_improve = 0

            for epoch in range(1, EPOCHS + 1):
                model.train()
                # Re-lock BN each epoch — model.train() re-enables train mode for BN
                model.bn1.eval()
                model.bn2.eval()
                ep_loss = 0.0
                for Xb, yb in tr_ld:
                    opt.zero_grad()
                    loss = crit(model(Xb), yb)
                    loss.backward()
                    # Zero old rows directly — no hooks, no weight decay leakage
                    with torch.no_grad():
                        if model.fc3.weight.grad is not None:
                            model.fc3.weight.grad[protect_rows] = 0.0
                        if model.fc3.bias.grad is not None:
                            model.fc3.bias.grad[protect_rows] = 0.0
                    nn.utils.clip_grad_norm_(
                        [model.fc3.weight, model.fc3.bias], max_norm=2.0
                    )
                    opt.step()
                    ep_loss += loss.item()
                sched.step()

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
                    lr  = sched.get_last_lr()[0]
                    with _train_lock:
                        _train_state["logs"].append(
                            f"Epoch {epoch}/{EPOCHS} - Loss {ep_loss/max(1,len(tr_ld)):.4f}"
                            f" | Val {acc*100:.1f}% | LR {lr:.5f}"
                        )
                    if acc > best_acc + 0.001:
                        best_acc   = acc
                        best_state = {k: v.clone() for k, v in model.state_dict().items()}
                        no_improve = 0
                    else:
                        no_improve += 10
                    if no_improve >= patience and epoch >= 30:
                        with _train_lock:
                            _train_state["logs"].append(
                                f"Early stop at epoch {epoch} - best val {best_acc*100:.1f}%"
                            )
                        break

            if best_state:
                model.load_state_dict(best_state)

            # Save and hot-reload
            model.eval()
            torch.save({"state_dict": model.state_dict(),
                        "classes": all_classes,
                        "num_classes": num_classes}, ckpt_path)
            engine._load_model()
            engine.model.eval()
            engine.CLASSES = all_classes

            with _train_lock:
                _train_state["state"] = "done"
                _train_state["logs"].append(
                    f"Done - {len(classes_with_data)} new symbol(s): {sorted(classes_with_data)}"
                    f"  |  Best val acc: {best_acc*100:.1f}%"
                    f"  |  {len(old_classes)} old symbol(s) preserved"
                )

        except Exception as exc:
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
