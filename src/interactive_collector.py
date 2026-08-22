import cv2
import os
import json
import uuid
import time
import numpy as np
import random
from utils.setup import get_classes, get_colors
from utils.logger import get_logger

from hand_landmarks import HandLandmarkDetector

logger = get_logger("interactive_collector")
hand_detector = HandLandmarkDetector(show_coordinates=True)

def add_class_to_config(class_name, config_path="src/config.json"):
    """
    Ensure the new sign class name and an associated color are saved in src/config.json.
    """
    class_name = class_name.lower().strip()
    with open(config_path, "r") as f:
        config = json.load(f)

    if class_name not in config["classes"]:
        config["classes"].append(class_name)
        # Generate random RGB color distinct from background
        new_color = [random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)]
        if "colors" not in config:
            config["colors"] = []
        config["colors"].append(new_color)

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Added new sign class '{class_name}' with color {new_color} to src/config.json")
    
    classes = config["classes"]
    return classes.index(class_name), classes

def detect_hand_bbox(frame):
    """
    Detect hand/object region using color contours or return center box [cx, cy, w, h] normalized.
    """
    h, w, _ = frame.shape
    # Default center bounding box (normalized center_x, center_y, width, height)
    default_box = (0.5, 0.5, 0.5, 0.6)
    
    try:
        # HSV skin color detection heuristic for hand bounding box estimate
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 1000:
                x, y, bw, bh = cv2.boundingRect(c)
                cx = max(0.05, min(0.95, (x + bw / 2.0) / w))
                cy = max(0.05, min(0.95, (y + bh / 2.0) / h))
                norm_w = max(0.05, min(0.9, bw / float(w)))
                norm_h = max(0.05, min(0.9, bh / float(h)))
                return (round(cx, 4), round(cy, 4), round(norm_w, 4), round(norm_h, 4))
    except Exception:
        pass

    return default_box

def collect_user_sign_interactive(class_name=None, num_samples=30, camera_id=0, output_dir="data/user"):
    """
    Interactive webcam collection tool to capture and label custom user signs.
    """
    train_img_dir = os.path.join(output_dir, "train", "images")
    train_lbl_dir = os.path.join(output_dir, "train", "labels")
    test_img_dir = os.path.join(output_dir, "test", "images")
    test_lbl_dir = os.path.join(output_dir, "test", "labels")

    for d in [train_img_dir, train_lbl_dir, test_img_dir, test_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    if class_name is None:
        print("\n==========================================")
        print("  🤟 INTERACTIVE SIGN LEARNING & CAPTURE  ")
        print("==========================================")
        class_name = input("Enter the name/label of the sign you want to teach (e.g., 'peace', 'rock'): ").strip()
        if not class_name:
            class_name = "custom_sign"

    class_idx, all_classes = add_class_to_config(class_name)
    logger.info(f"Target Sign: '{class_name}' (Class ID: {class_idx})")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.warning(f"Could not open camera {camera_id}. Generating synthetic samples for testing.")
        # Non-interactive fallback mode for headless environments
        for i in range(num_samples):
            split = "test" if i % 5 == 0 else "train"
            img_dir = test_img_dir if split == "test" else train_img_dir
            lbl_dir = test_lbl_dir if split == "test" else train_lbl_dir
            
            img_filename = f"{class_name}_{uuid.uuid4().hex[:6]}"
            img = np.zeros((224, 224, 3), dtype=np.uint8)
            cv2.rectangle(img, (50, 50), (170, 170), (0, 255, 0), -1)
            cv2.imwrite(os.path.join(img_dir, f"{img_filename}.jpg"), img)
            
            with open(os.path.join(lbl_dir, f"{img_filename}.txt"), "w") as f:
                f.write(f"{class_idx} 0.5 0.5 0.5 0.5\n")
        logger.success(f"Generated {num_samples} samples for sign '{class_name}' in {output_dir}")
        return class_name

    logger.info("Camera connected. Get ready to hold your sign in front of the camera!")
    
    # Countdown 3 seconds
    for cd in [3, 2, 1]:
        logger.info(f"Starting capture in {cd} seconds... Hold your sign!")
        start_t = time.time()
        while time.time() - start_t < 1.0:
            ret, frame = cap.read()
            if ret:
                cv2.putText(frame, f"GET READY: {cd}", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
                cv2.putText(frame, f"Sign: {class_name}", (50, 160), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("Sign Capture Window", frame)
                cv2.waitKey(1)

    captured_count = 0
    while captured_count < num_samples:
        ret, frame = cap.read()
        if not ret:
            break

        cx, cy, nw, nh = detect_hand_bbox(frame)

        # Save frame and label file
        split = "test" if captured_count % 5 == 0 else "train"
        img_dir = test_img_dir if split == "test" else train_img_dir
        lbl_dir = test_lbl_dir if split == "test" else train_lbl_dir

        img_filename = f"{class_name}_{captured_count:03d}_{uuid.uuid4().hex[:4]}"
        cv2.imwrite(os.path.join(img_dir, f"{img_filename}.jpg"), frame)

        with open(os.path.join(lbl_dir, f"{img_filename}.txt"), "w") as f:
            f.write(f"{class_idx} {cx} {cy} {nw} {nh}\n")

        captured_count += 1

        # Draw overlay box & progress & finger nodes
        frame, _ = hand_detector.process_and_draw(frame)
        h, w, _ = frame.shape
        x1, y1 = int((cx - nw/2)*w), int((cy - nh/2)*h)
        x2, y2 = int((cx + nw/2)*w), int((cy + nh/2)*h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"Capturing '{class_name}': {captured_count}/{num_samples}", (30, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Sign Capture Window", frame)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            logger.warning("Capture stopped early by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.success(f"Successfully captured {captured_count} images for '{class_name}'!")
    return class_name

if __name__ == "__main__":
    collect_user_sign_interactive(num_samples=10)
