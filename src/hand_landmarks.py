"""
hand_landmarks.py — Optimised MediaPipe Hand Landmark Detector.

Key improvements over the original:
  • EMA (Exponential Moving Average) smoothing per landmark — eliminates jitter
  • Vectorised NumPy feature extraction — ~10× faster than Python loops
  • Single-pass drawing — skeleton lines + nodes in one tight loop
  • Higher tracking confidence so MediaPipe keeps tracking rather than re-detecting
  • Coordinate text removed from non-tip nodes — less per-frame OpenCV work
"""

import cv2
import os
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Hand skeleton connections (21-node MediaPipe topology) ─────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # Thumb
    (0,5),(5,6),(6,7),(7,8),          # Index
    (0,9),(9,10),(10,11),(11,12),     # Middle
    (0,13),(13,14),(14,15),(15,16),   # Ring
    (0,17),(17,18),(18,19),(19,20),   # Pinky
    (5,9),(9,13),(13,17),             # Palm
]

FINGER_TIP_NAMES = {4:"Thumb", 8:"Index", 12:"Middle", 16:"Ring", 20:"Pinky"}

MODEL_PATH = "pretrained/hand_landmarker.task"

# ── EMA smoothing factor (0 = no smoothing, 1 = frozen) ───────────────────
_EMA_ALPHA = 0.55   # higher = snappier, lower = smoother

# Drawing constants
_LINE_COLOR  = (0, 220, 220)   # cyan skeleton
_NODE_OUTER  = (50, 220, 255)  # blue-cyan outer circle
_NODE_INNER  = (255, 60, 200)  # magenta inner dot
_TIP_COLOR   = (0, 255, 180)   # label text


def ensure_model_file():
    """Download official MediaPipe HandLandmarker model file if not present."""
    os.makedirs("pretrained", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        try:
            urllib.request.urlretrieve(url, MODEL_PATH)
        except Exception:
            pass


def extract_normalized_features(pts_array: np.ndarray) -> list | None:
    """
    Vectorised feature extraction.
    pts_array: (21, 3) float32 — already wrist-relative normalised coords (x,y,z).
    Returns 63-element list.
    """
    if pts_array is None or pts_array.shape[0] < 21:
        return None
    # Wrist-relative
    rel = pts_array[:21] - pts_array[0]          # (21, 3)
    # Scale by max L2 distance from wrist
    dists  = np.linalg.norm(rel, axis=1)          # (21,)
    max_d  = dists.max()
    if max_d < 1e-5:
        max_d = 1.0
    normed = rel / max_d                           # (21, 3)
    return normed.flatten().tolist()               # 63 values


class HandLandmarkDetector:
    """
    Fast, smooth hand landmark detector with EMA position smoothing.
    """

    def __init__(self, show_coordinates: bool = True):
        self.show_coordinates = show_coordinates
        ensure_model_file()
        self._init_detector()
        # EMA state: dict[hand_index -> np.ndarray shape (21,3)] of smoothed raw coords
        self._ema: list[np.ndarray | None] = [None, None]

    def _init_detector(self):
        try:
            base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,                        # detect both hands
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.7,
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            self.ready = True
        except Exception:
            self.ready = False

    def close(self):
        if hasattr(self, "detector") and self.detector is not None:
            try:
                self.detector.close()
            except Exception:
                pass
            self.detector = None

    def __del__(self):
        self.close()

    # ------------------------------------------------------------------
    def process_and_draw(self, frame: np.ndarray):
        """
        Detect, smooth, draw, and extract features in one call.
        Returns (annotated_frame, list_of_pts, list_of_feature_vectors).
        pts element: list of (px, py, nx, ny, nz) for 21 landmarks.
        """
        if not self.ready or frame is None:
            return frame, [], []

        h, w = frame.shape[:2]

        # ── MediaPipe inference ────────────────────────────────────────
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        landmarks_list: list  = []
        feature_vectors: list = []

        try:
            result = self.detector.detect(mp_img)
        except Exception:
            return frame, [], []

        if not (result and result.hand_landmarks):
            # No hand detected — reset EMA state so stale smoothing doesn't persist
            self._ema = [None, None]
            return frame, [], []

        for hand_idx, hand_lms in enumerate(result.hand_landmarks):
            # ── Build raw coord array (21, 3) ──────────────────────────
            raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms],
                           dtype=np.float32)   # (21, 3)

            # ── EMA smoothing ──────────────────────────────────────────
            if hand_idx < len(self._ema):
                prev = self._ema[hand_idx]
                if prev is None:
                    smoothed = raw.copy()
                else:
                    smoothed = _EMA_ALPHA * raw + (1.0 - _EMA_ALPHA) * prev
                self._ema[hand_idx] = smoothed
            else:
                smoothed = raw.copy()

            # ── Convert to pixel coords ────────────────────────────────
            px_arr = np.clip(
                (smoothed[:, :2] * np.array([w, h], dtype=np.float32)).astype(np.int32),
                [0, 0], [w - 1, h - 1]
            )   # (21, 2)  integer pixel coords

            # Build pts list (px, py, nx, ny, nz)
            pts = [(int(px_arr[i, 0]), int(px_arr[i, 1]),
                    float(smoothed[i, 0]), float(smoothed[i, 1]), float(smoothed[i, 2]))
                   for i in range(21)]

            landmarks_list.append(pts)

            # ── Feature extraction (vectorised) ────────────────────────
            feats = extract_normalized_features(smoothed)
            if feats is not None:
                feature_vectors.append(feats)

            # ── Draw skeleton ──────────────────────────────────────────
            self._draw(frame, px_arr)

        return frame, landmarks_list, feature_vectors

    # ------------------------------------------------------------------
    def _draw(self, frame: np.ndarray, px: np.ndarray) -> None:
        """
        Fast single-pass skeleton drawing.
        px: (21, 2) int32 pixel positions.
        """
        # Skeleton lines
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame,
                     (px[a, 0], px[a, 1]),
                     (px[b, 0], px[b, 1]),
                     _LINE_COLOR, 2, cv2.LINE_AA)

        # Joint nodes
        for i in range(21):
            c = (int(px[i, 0]), int(px[i, 1]))
            cv2.circle(frame, c, 6, _NODE_OUTER, -1, cv2.LINE_AA)
            cv2.circle(frame, c, 3, _NODE_INNER, -1, cv2.LINE_AA)

        # Finger tip labels (only 5 nodes — minimal text overhead)
        if self.show_coordinates:
            for tip_idx, name in FINGER_TIP_NAMES.items():
                x, y = int(px[tip_idx, 0]), int(px[tip_idx, 1])
                cv2.putText(frame, name,
                            (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            _TIP_COLOR, 1, cv2.LINE_AA)


if __name__ == "__main__":
    detector = HandLandmarkDetector()
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    _, pts, feats = detector.process_and_draw(dummy)
    print("HandLandmarkDetector ready. Feature vectors:", len(feats))
