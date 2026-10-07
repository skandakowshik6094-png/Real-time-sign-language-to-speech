import os
import cv2
import time
import torch
import threading
import queue
import pyttsx3
from utils.setup import get_classes, get_colors
from utils.logger import get_logger
from hand_landmarks import HandLandmarkDetector
from landmark_model import LandmarkClassifier

logger = get_logger("realtime")

# ---------------------------------------------------------------------------
# Text-to-speech: runs in a background thread so it never stalls the camera
# ---------------------------------------------------------------------------
_tts_queue: queue.Queue = queue.Queue()

def _tts_worker():
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)   # slightly slower than default for clarity
    engine.setProperty("volume", 1.0)
    while True:
        text = _tts_queue.get()       # blocks until something arrives
        if text is None:              # sentinel – shut down
            break
        engine.say(text)
        engine.runAndWait()

_tts_thread = threading.Thread(target=_tts_worker, daemon=True)
_tts_thread.start()

def speak(text: str):
    """Queue a word / letter for TTS. Non-blocking."""
    _tts_queue.put(text)

CLASSES = get_classes()
COLORS = get_colors()
num_classes = len(CLASSES)

model = LandmarkClassifier(num_classes=num_classes)
CKPT_PATH = "checkpoints/landmark_model.pt"

if os.path.exists(CKPT_PATH):
    try:
        data = torch.load(CKPT_PATH, map_location="cpu")
        if isinstance(data, dict) and "state_dict" in data:
            ckpt_classes = data.get("classes", CLASSES)
            if len(ckpt_classes) != model.num_classes:
                model.update_num_classes(len(ckpt_classes))
                CLASSES = ckpt_classes
            model.load_state_dict(data["state_dict"])
        else:
            model.load_state_dict(data)
        logger.success(f"Loaded Landmark Neural Model from '{CKPT_PATH}'")
    except Exception as e:
        logger.warning(f"Could not load checkpoint '{CKPT_PATH}': {e}")

model.eval()
hand_detector = HandLandmarkDetector(show_coordinates=True)

def run_realtime_detection(camera_id=0):
    logger.realtime("Starting high-precision MediaPipe Landmark Recognition...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        logger.error(f"Could not open camera {camera_id}")
        return

    frame_count = 0
    fps_start_time = time.time()

    # TTS cooldown state – avoid speaking the same letter on every frame
    last_spoken: str = ""
    last_spoken_time: float = 0.0
    TTS_COOLDOWN = 1.5  # seconds before the same letter can be spoken again

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            break

        frame, hand_pts_list, feature_vectors = hand_detector.process_and_draw(frame)

        for hand_idx, (pts, feats) in enumerate(zip(hand_pts_list, feature_vectors)):
            if len(feats) == 63:
                x_tensor = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
                class_idx, confidence, probs = model.predict_probs(x_tensor)

                if confidence > 0.40 and class_idx < len(CLASSES):
                    cls_name = CLASSES[class_idx]
                    color = COLORS[class_idx] if class_idx < len(COLORS) else (0, 255, 0)

                    # Speak the letter if it changed or enough time has elapsed
                    now = time.time()
                    if cls_name != last_spoken or (now - last_spoken_time) >= TTS_COOLDOWN:
                        speak(cls_name)
                        last_spoken = cls_name
                        last_spoken_time = now
                else:
                    cls_name = "Detecting Sign..."
                    color = (200, 200, 200)

                # Wrist location for placing text overlay
                wx, wy = pts[0][:2]
                label_text = f"{cls_name} ({round(confidence * 100, 1)}%)"

                # Draw label background card
                cv2.rectangle(frame, (max(10, wx - 100), max(10, wy - 60)),
                              (min(frame.shape[1]-10, wx + len(label_text)*16), max(40, wy - 10)),
                              color, -1)
                cv2.putText(frame, label_text, (max(15, wx - 95), max(32, wy - 20)),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        frame_count += 1
        if frame_count % 30 == 0:
            elapsed_time = time.time() - fps_start_time
            fps = 30 / max(1e-5, elapsed_time)
            fps_start_time = time.time()

        cv2.imshow("MediaPipe 3D Landmark Sign Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.realtime("Stopping real-time recognition...")
            break

    cap.release()
    hand_detector.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_detection()
