import os
import sys
import argparse
from interactive_collector import collect_user_sign_interactive
from interactive_train import run_interactive_fine_tuning
from realtime import run_realtime_detection
from utils.logger import get_logger

logger = get_logger("train_my_sign")

def main():
    parser = argparse.ArgumentParser(description="Interactive Self-Training & Prediction for Custom Sign Language Recognition")
    parser.add_argument("--sign", type=str, default="", help="Name of the sign to capture (e.g. 'peace', 'thumbs_up')")
    parser.add_argument("--samples", type=int, default=30, help="Number of images to capture")
    parser.add_argument("--epochs", type=int, default=15, help="Number of fine-tuning epochs")
    parser.add_argument("--skip_capture", action="store_true", help="Skip capture step and train directly on existing data")
    parser.add_argument("--skip_train", action="store_true", help="Skip training step and go straight to live prediction")
    args = parser.parse_args()

    logger.print_banner()

    if not args.skip_train and not args.skip_capture:
        logger.info("\n=== STEP 1: CAPTURE YOUR SIGN ===")
        sign_label = collect_user_sign_interactive(
            class_name=args.sign if args.sign else None,
            num_samples=args.samples
        )
    elif args.sign:
        sign_label = args.sign
    else:
        sign_label = "custom_sign"

    if not args.skip_train:
        logger.info("\n=== STEP 2: FINE-TUNE MODEL ON YOUR SIGN ===")
        run_interactive_fine_tuning(epochs=args.epochs, freeze_epochs=3)

    logger.info("\n=== STEP 3: LIVE PREDICTION & TESTING ===")
    run_realtime_detection()

if __name__ == "__main__":
    main()
