"""
detection_engine.py -- Thread-safe detection engine for SignSense.
"""
from __future__ import annotations

import os
import sys
import cv2
import time
import torch
import queue
import threading
import subprocess

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from utils.setup import get_classes, get_colors
from utils.logger import get_logger
from hand_landmarks import HandLandmarkDetector
from landmark_model import LandmarkClassifier

logger = get_logger("engine")


class DetectionEngine:
    def __init__(self) -> None:
        self.CLASSES: list = get_classes()
        self.COLORS: list = get_colors()

        self.model = LandmarkClassifier(num_classes=len(self.CLASSES))
        self.CKPT_PATH = "checkpoints/landmark_model.pt"
        self._load_model()
        self.model.eval()

        self.running: bool = False
        self.camera_id: int = 0
        self._thread = None

        self.latest_frame: bytes | None = None
        self._frame_lock = threading.Lock()

        self._subscribers: list = []
        self._sub_lock = threading.Lock()

        self.latest_detection: dict = {"letter": None, "confidence": 0.0}

        self.tts_enabled: bool = True
        self.tts_rate: int = 160
        self.tts_volume: float = 1.0
        # Poll-based TTS: detection loop writes here; worker reads every interval
        self._current_letter: str = ""
        self._tts_event = threading.Event()
        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True, name="TTSWorker")
        self._tts_thread.start()

        self.threshold: float = 0.40
        self._hand_detector = None

        # ── Collection state ──────────────────────────────────────────────
        self._collecting: bool = False
        self._collect_class: str = ""
        self._collect_target: int = 0
        self._collect_count: int = 0
        self._collect_buffer: list = []

    def _load_model(self) -> None:
        if not os.path.exists(self.CKPT_PATH):
            logger.warning(f"Checkpoint not found at '{self.CKPT_PATH}'; using random weights.")
            return
        try:
            data = torch.load(self.CKPT_PATH, map_location="cpu", weights_only=False)
            if isinstance(data, dict) and "state_dict" in data:
                ckpt_classes = data.get("classes", self.CLASSES)
                if len(ckpt_classes) != self.model.num_classes:
                    self.model.update_num_classes(len(ckpt_classes))
                    self.CLASSES = ckpt_classes
                self.model.load_state_dict(data["state_dict"])
            else:
                self.model.load_state_dict(data)
            logger.success(f"Loaded model from '{self.CKPT_PATH}'")
        except Exception as exc:
            logger.warning(f"Could not load checkpoint: {exc}")

    def _tts_worker(self) -> None:
        """Speak _current_letter by spawning an isolated Python subprocess.

        Using a subprocess sidesteps COM threading conflicts between pyttsx3
        and the Flask/OpenCV/MediaPipe threads that plague in-process TTS.
        Any in-flight utterance is killed before the new one starts so the
        voice always says the *current* letter.
        """
        INTERVAL = 0.5          # max seconds between utterances
        proc = None

        while True:
            self._tts_event.wait(timeout=INTERVAL)
            self._tts_event.clear()

            if not self.tts_enabled:
                if proc and proc.poll() is None:
                    proc.kill()
                continue

            text = self._current_letter
            if not text:
                continue

            # Kill previous utterance so we always say the latest letter
            if proc and proc.poll() is None:
                proc.kill()

            try:
                rate = self.tts_rate
                vol  = self.tts_volume
                script = (
                    f'import pyttsx3; e=pyttsx3.init(); '
                    f'e.setProperty("rate",{rate}); '
                    f'e.setProperty("volume",{vol}); '
                    f'e.say("{text}"); e.runAndWait()'
                )
                kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
                if sys.platform == "win32":
                    kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                proc = subprocess.Popen([sys.executable, "-c", script], **kwargs)
            except Exception as exc:
                logger.warning(f"TTS subprocess error: {exc}")

    def _make_tts_engine(self):
        """(Legacy stub kept for API compatibility.)"""
        raise NotImplementedError("TTS now runs via subprocess")

    def _speak(self, text: str) -> None:
        """Signal the TTS worker to say *text* on its next cycle."""
        if self.tts_enabled:
            self._current_letter = text
            self._tts_event.set()

    def subscribe(self):
        q: queue.Queue = queue.Queue(maxsize=64)
        with self._sub_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q) -> None:
        with self._sub_lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def _publish(self, event: dict) -> None:
        with self._sub_lock:
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass

    def start(self, camera_id: int = 0) -> None:
        if self.running:
            return
        self.camera_id = camera_id

        self.running = True
        self._thread = threading.Thread(target=self._detection_loop, daemon=True, name="DetectionLoop")
        self._thread.start()
        logger.info(f"Detection engine started on camera {camera_id}")

    def stop(self) -> None:
        self.running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._hand_detector:
            self._hand_detector.close()
            self._hand_detector = None
        with self._frame_lock:
            self.latest_frame = None
        self.latest_detection = {"letter": None, "confidence": 0.0}
        logger.info("Detection engine stopped")

    def get_status(self) -> dict:
        return {
            "running": self.running,
            "tts_enabled": self.tts_enabled,
            "tts_rate": self.tts_rate,
            "tts_volume": self.tts_volume,
            "threshold": self.threshold,
            "latest_detection": self.latest_detection,
            "classes": self.CLASSES,
        }

    def _detection_loop(self) -> None:
        if self._hand_detector is None:
            self._hand_detector = HandLandmarkDetector(show_coordinates=True)

        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            logger.error(f"Could not open camera {self.camera_id}")
            self.running = False
            return

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Frame read failed -- stopping detection")
                    break

                frame, hand_pts_list, feature_vectors = self._hand_detector.process_and_draw(frame)

                detected_something = False
                for pts, feats in zip(hand_pts_list, feature_vectors):
                    if len(feats) != 63:
                        continue

                    # ── Collection mode ──────────────────────────────────
                    if self._collecting:
                        if self._collect_count < self._collect_target:
                            self._collect_buffer.append(feats)
                            self._collect_count += 1
                            self._publish({
                                "type": "collect_progress",
                                "count": self._collect_count,
                                "target": self._collect_target,
                                "class_name": self._collect_class,
                            })
                            pct = int(self._collect_count / self._collect_target * 100)
                            bar_w = int(frame.shape[1] * 0.6)
                            bar_x, bar_y = 20, frame.shape[0] - 40
                            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 14),
                                          (40, 40, 40), -1)
                            cv2.rectangle(frame, (bar_x, bar_y),
                                          (bar_x + int(bar_w * pct / 100), bar_y + 14),
                                          (0, 220, 120), -1)
                            cv2.putText(
                                frame,
                                f"Collecting '{self._collect_class}': {self._collect_count}/{self._collect_target}",
                                (20, frame.shape[0] - 52),
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 128), 2, cv2.LINE_AA,
                            )
                            if self._collect_count >= self._collect_target:
                                self._collecting = False
                                threading.Thread(
                                    target=self._finalize_collection,
                                    daemon=True, name="CollectFinalize",
                                ).start()
                        break  # one hand is sufficient for collection

                    # ── Normal classification ────────────────────────────
                    x_tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
                    class_idx, confidence, _ = self.model.predict_probs(x_tensor)

                    if confidence > self.threshold and class_idx < len(self.CLASSES):
                        detected_something = True
                        cls_name = self.CLASSES[class_idx]
                        color = tuple(self.COLORS[class_idx]) if class_idx < len(self.COLORS) else (0, 255, 0)

                        self._speak(cls_name)

                        detection = {
                            "type": "detection",
                            "letter": cls_name,
                            "confidence": round(confidence * 100, 1),
                        }
                        self.latest_detection = {"letter": cls_name, "confidence": round(confidence * 100, 1)}
                        self._publish(detection)

                        wx, wy = pts[0][:2]
                        label = f"{cls_name} ({round(confidence * 100, 1)}%)"
                        x2 = min(frame.shape[1] - 10, wx + len(label) * 16)
                        cv2.rectangle(frame, (max(10, wx - 100), max(10, wy - 60)), (x2, max(40, wy - 10)), color, -1)
                        cv2.putText(frame, label, (max(15, wx - 95), max(32, wy - 20)),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                if not detected_something and not self._collecting:
                    self._current_letter = ""   # silence TTS when no hand visible
                    if self.latest_detection.get("letter"):
                        self.latest_detection = {"letter": None, "confidence": 0.0}
                        self._publish({"type": "detection", "letter": None, "confidence": 0.0})

                ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
                if ok:
                    with self._frame_lock:
                        self.latest_frame = jpeg.tobytes()

        finally:
            cap.release()
            self.running = False

    # ── Collection API ────────────────────────────────────────────────────

    def start_collecting(self, class_name: str, n_samples: int = 60,
                         replace: bool = False) -> None:
        """Begin capturing landmark features for *class_name*.

        If *replace* is True, existing samples for this class are deleted
        before capture starts.
        """
        class_name = class_name.lower().strip()
        if replace:
            self._delete_class_samples(class_name)
        self._collect_class = class_name
        self._collect_target = max(10, n_samples)
        self._collect_count = 0
        self._collect_buffer = []
        self._collecting = True
        logger.info(f"Collection started for '{class_name}' ({n_samples} samples target)")

    def stop_collecting(self) -> dict:
        """Abort an in-progress collection. Partial data is discarded."""
        self._collecting = False
        stopped_at = self._collect_count
        self._collect_buffer = []
        self._collect_count = 0
        logger.info(f"Collection stopped early at {stopped_at} samples")
        return {"stopped_at": stopped_at, "class_name": self._collect_class}

    def get_collection_status(self) -> dict:
        return {
            "collecting": self._collecting,
            "class_name": self._collect_class,
            "count": self._collect_count,
            "target": self._collect_target,
        }

    def _finalize_collection(self) -> None:
        """Save buffered samples to disk and reload CLASSES."""
        class_name = self._collect_class
        samples = list(self._collect_buffer)
        self._collect_buffer = []
        if not samples or not class_name:
            return
        try:
            import json as _json
            # 1. Ensure class exists in config.json
            config_path = os.path.join(_SRC_DIR, "config.json")
            with open(config_path) as f:
                cfg = _json.load(f)
            if class_name not in cfg["classes"]:
                cfg["classes"].append(class_name)
                import random
                cfg.setdefault("colors", []).append(
                    [random.randint(50, 255) for _ in range(3)]
                )
                with open(config_path, "w") as f:
                    _json.dump(cfg, f, indent=2)
                logger.info(f"Added new class '{class_name}' to config.json")
            class_idx = cfg["classes"].index(class_name)

            # 2. Append to landmark dataset
            from landmark_trainer import load_landmark_dataset, save_landmark_dataset
            dataset = load_landmark_dataset()
            for feats in samples:
                dataset["samples"].append({
                    "label": class_name,
                    "class_id": class_idx,
                    "features": feats,
                })
            save_landmark_dataset(dataset)

            # 3. Reload engine class list
            self.CLASSES = cfg["classes"]
            self.COLORS = cfg.get("colors", self.COLORS)

            self._publish({"type": "collect_done",
                           "class_name": class_name,
                           "count": len(samples)})
            logger.success(f"Saved {len(samples)} samples for '{class_name}'")
        except Exception as exc:
            logger.error(f"_finalize_collection error: {exc}")

    def _delete_class_samples(self, class_name: str) -> int:
        """Remove all landmark dataset samples for *class_name*. Returns count removed."""
        try:
            from landmark_trainer import load_landmark_dataset, save_landmark_dataset
            ds = load_landmark_dataset()
            before = len(ds.get("samples", []))
            ds["samples"] = [
                s for s in ds.get("samples", []) if s.get("label") != class_name
            ]
            removed = before - len(ds["samples"])
            save_landmark_dataset(ds)
            logger.info(f"Removed {removed} samples for '{class_name}'")
            return removed
        except Exception as exc:
            logger.error(f"_delete_class_samples error: {exc}")
            return 0

    def delete_class(self, class_name: str) -> bool:
        """Remove *class_name* from config.json and all its dataset samples."""
        import json as _json
        config_path = os.path.join(_SRC_DIR, "config.json")
        try:
            with open(config_path) as f:
                cfg = _json.load(f)
            if class_name in cfg["classes"]:
                idx = cfg["classes"].index(class_name)
                cfg["classes"].pop(idx)
                if "colors" in cfg and idx < len(cfg["colors"]):
                    cfg["colors"].pop(idx)
                with open(config_path, "w") as f:
                    _json.dump(cfg, f, indent=2)
            self._delete_class_samples(class_name)
            self.CLASSES = cfg["classes"]
            self.COLORS = cfg.get("colors", [])
            logger.info(f"Deleted class '{class_name}'")
            return True
        except Exception as exc:
            logger.error(f"delete_class error: {exc}")
            return False
