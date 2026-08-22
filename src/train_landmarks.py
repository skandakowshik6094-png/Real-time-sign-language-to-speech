import os
import sys
import argparse

from landmark_trainer import collect_user_landmark_data, train_landmark_classifier
from realtime import run_realtime_detection
from utils.logger import get_logger

logger = get_logger("train_landmarks")

def main():
    parser = argparse.ArgumentParser(description="MediaPipe 3D Landmark Sign Recognition & Interactive Training")
    parser.add_argument("--sign", type=str, default="", help="Name of the sign to teach (e.g. 'peace', 'thumbs_up')")
    parser.add_argument("--samples", type=int, default=60, help="Number of 3D landmark feature samples to capture")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs for LandmarkClassifier")
    parser.add_argument("--skip_capture", action="store_true", help="Skip capture step and train directly on existing landmark dataset")
    parser.add_argument("--skip_train", action="store_true", help="Skip training step and go straight to live prediction")
    args = parser.parse_args()

    logger.print_banner()

    if not args.skip_train and not args.skip_capture:
        logger.info("\n=== STEP 1: CAPTURE HAND LANDMARK SIGN DATA ===")
        sign_label = collect_user_landmark_data(
            class_name=args.sign if args.sign else None,
            num_samples=args.samples
        )
    elif args.sign:
        sign_label = args.sign
    else:
        sign_label = "custom_sign"

    if not args.skip_train:
        logger.info("\n=== STEP 2: FAST NEURAL LANDMARK TRAINING (2 SECONDS) ===")
        train_landmark_classifier(epochs=args.epochs)

    logger.info("\n=== STEP 3: LIVE MEDIAPIPE LANDMARK RECOGNITION ===")
    run_realtime_detection()

if __name__ == "__main__":
    main()
