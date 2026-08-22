import cv2
import os
import urllib.request
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 21 Standard Hand Landmark connections
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index Finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle Finger
    (9, 10), (10, 11), (11, 12), (0, 9),
    # Ring Finger
    (13, 14), (14, 15), (15, 16), (0, 13),
    # Pinky Finger
    (17, 18), (18, 19), (19, 20), (0, 17),
    # Palm connections
    (5, 9), (9, 13), (13, 17)
]

FINGER_TIP_NAMES = {
    4: "Thumb",
    8: "Index",
    12: "Middle",
    16: "Ring",
    20: "Pinky"
}

MODEL_PATH = "pretrained/hand_landmarker.task"

def ensure_model_file():
    """Download official MediaPipe HandLandmarker model file if not present."""
    os.makedirs("pretrained", exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        try:
            urllib.request.urlretrieve(url, MODEL_PATH)
        except Exception:
            pass

def extract_normalized_features(hand_nodes):
    """
    Normalizes 21 3D hand landmarks relative to wrist origin and hand scale.
    Returns 63-dimensional feature vector.
    """
    if len(hand_nodes) < 21:
        return None

    # Wrist is node index 0
    wx, wy, wz = hand_nodes[0][2], hand_nodes[0][3], hand_nodes[0][4]

    rel_coords = []
    max_dist = 0.0
    for node in hand_nodes[:21]:
        dx = node[2] - wx
        dy = node[3] - wy
        dz = node[4] - wz
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)
        if dist > max_dist:
            max_dist = dist
        rel_coords.append((dx, dy, dz))

    if max_dist < 1e-5:
        max_dist = 1.0

    features = []
    for dx, dy, dz in rel_coords:
        features.extend([dx / max_dist, dy / max_dist, dz / max_dist])

    return features

class HandLandmarkDetector:
    """
    High-Precision Neural Network Hand Landmark & Finger Joint Detector using MediaPipe 1.0.
    """
    def __init__(self, show_coordinates=True):
        self.show_coordinates = show_coordinates
        ensure_model_file()
        self._init_detector()

    def _init_detector(self):
        try:
            base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            self.ready = True
        except Exception:
            self.ready = False

    def close(self):
        if hasattr(self, 'detector') and self.detector is not None:
            try:
                self.detector.close()
            except Exception:
                pass
            self.detector = None

    def __del__(self):
        self.close()

    def process_and_draw(self, frame):
        """
        Runs neural network hand landmarker, draws exact finger joint nodes, skeleton lines, and coordinates.
        Returns (annotated_frame, list_of_hands_pts, list_of_63d_features).
        """
        if not self.ready or frame is None:
            return frame, [], []

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        landmarks_list = []
        feature_vectors = []

        try:
            result = self.detector.detect(mp_image)
            if result and result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    pts = []
                    for lm in hand_landmarks:
                        px = max(0, min(w - 1, int(lm.x * w)))
                        py = max(0, min(h - 1, int(lm.y * h)))
                        pts.append((px, py, lm.x, lm.y, lm.z))

                    landmarks_list.append(pts)
                    feats = extract_normalized_features(pts)
                    if feats is not None:
                        feature_vectors.append(feats)

                    self.draw_hand_nodes(frame, pts)
        except Exception:
            pass

        return frame, landmarks_list, feature_vectors

    def draw_hand_nodes(self, frame, nodes):
        """
        Draws glowing node circles on finger joints, skeleton lines, and coordinates on finger tips.
        """
        h, w, _ = frame.shape

        # Draw skeleton connection lines
        for p1_idx, p2_idx in HAND_CONNECTIONS:
            if p1_idx < len(nodes) and p2_idx < len(nodes):
                x1, y1 = nodes[p1_idx][:2]
                x2, y2 = nodes[p2_idx][:2]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)

        # Draw 21 finger joint node circles & coordinate labels
        for idx, node in enumerate(nodes):
            px, py, nx, ny = node[:4]
            
            # Draw joint node circle (cyan outer ring, magenta center dot)
            cv2.circle(frame, (px, py), 6, (255, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 3, (255, 0, 255), -1, cv2.LINE_AA)

            # Display finger tip label & coordinates
            if idx in FINGER_TIP_NAMES and self.show_coordinates:
                finger_name = FINGER_TIP_NAMES[idx]
                coord_text = f"{finger_name} ({round(nx, 2)}, {round(ny, 2)})"
                cv2.putText(frame, coord_text, (px + 8, py - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

if __name__ == "__main__":
    detector = HandLandmarkDetector()
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    out, pts, feats = detector.process_and_draw(dummy)
    print("MediaPipe Neural HandLandmarkDetector ready! Features:", len(feats))
