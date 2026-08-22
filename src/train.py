import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from torch import optim, save

from data import DETRData, AUTSLDETRData, CombinedDETRData, BalancedDatasetSampler
from model import DETR
from loss import DETRLoss, HungarianMatcher
from metrics import evaluate_model
from autsl_processor import prepare_autsl_dataset, generate_label_report
from utils.logger import get_logger
from utils.setup import get_classes
from utils.rich_handlers import TrainingHandler, rich_training_context
from utils.boxes import stacker

def parse_args():
    parser = argparse.ArgumentParser(description="SignDETR Training Script with AUTSL Integration & Fine-Tuning")
    parser.add_argument("--dataset", type=str, choices=["existing", "autsl", "combined"], default="existing",
                        help="Dataset selection: 'existing', 'autsl', or 'combined'")
    parser.add_argument("--freeze_epochs", type=int, default=5,
                        help="Number of initial epochs to freeze backbone for transfer learning")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Total number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to resume checkpoint (defaults to latest in checkpoints/ or pretrained/4426_model.pt)")
    return parser.parse_args()

def get_latest_checkpoint():
    checkpoints_dir = "checkpoints"
    if os.path.exists(checkpoints_dir):
        pts = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pt")]
        if len(pts) > 0:
            # Sort by epoch index
            pts_sorted = sorted(pts, key=lambda x: int(x.split("_")[0]) if x.split("_")[0].isdigit() else -1)
            latest = os.path.join(checkpoints_dir, pts_sorted[-1])
            return latest
    if os.path.exists("pretrained/4426_model.pt"):
        return "pretrained/4426_model.pt"
    return None

def main():
    args = parse_args()
    logger = get_logger("training")
    logger.print_banner()

    os.makedirs("checkpoints", exist_ok=True)
    existing_classes = get_classes()

    # Ensure AUTSL dataset is prepared
    if args.dataset in ["autsl", "combined"]:
        prepare_autsl_dataset(base_path="data/autsl")

    # Run label audit report before training
    autsl_info = [
        {"label_name": "hello"},
        {"label_name": "merhaba"},
        {"label_name": "iloveyou"},
        {"label_name": "thankyou"},
        {"label_name": "goodbye"},
        {"label_name": "water"}
    ]
    generate_label_report(autsl_info, existing_classes)

    # Initialize Datasets and DataLoaders according to --dataset argument
    if args.dataset == "existing":
        train_dataset = DETRData("data/train", train=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=stacker, drop_last=True, shuffle=True)
    elif args.dataset == "autsl":
        train_dataset = AUTSLDETRData("data/autsl/train", train=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=stacker, drop_last=True, shuffle=True)
    elif args.dataset == "combined":
        existing_train = DETRData("data/train", train=True)
        autsl_train = AUTSLDETRData("data/autsl/train", train=True)
        train_dataset = CombinedDETRData(existing_path="data/train", autsl_path="data/autsl/train", train=True)
        sampler = BalancedDatasetSampler(len(existing_train), len(autsl_train))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=stacker, drop_last=True, sampler=sampler)

    # Prepare Evaluation Validation DataLoaders
    existing_val_dataset = DETRData("data/test", train=False)
    existing_val_loader = DataLoader(existing_val_dataset, batch_size=args.batch_size, collate_fn=stacker, drop_last=True)

    autsl_val_loader = None
    if os.path.exists("data/autsl/test"):
        autsl_val_dataset = AUTSLDETRData("data/autsl/test", train=False)
        if len(autsl_val_dataset) > 0:
            autsl_val_loader = DataLoader(autsl_val_dataset, batch_size=args.batch_size, collate_fn=stacker, drop_last=True)

    num_classes = len(existing_classes)
    model = DETR(num_classes=num_classes)

    # Resume checkpoint selection
    checkpoint_to_load = args.checkpoint if args.checkpoint else get_latest_checkpoint()
    if checkpoint_to_load and os.path.exists(checkpoint_to_load):
        logger.info(f"Resuming / Loading weights from checkpoint: {checkpoint_to_load}")
        model.load_pretrained(checkpoint_to_load)
    else:
        logger.info("Initializing model from scratch.")

    model.log_model_info()

    opt = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, len(train_loader) * 30, T_mult=2)

    weights = {"class_weighting": 1, "bbox_weighting": 5, "giou_weighting": 2}
    matcher = HungarianMatcher(weights)
    criterion = DETRLoss(num_classes=num_classes, matcher=matcher, weight_dict=weights, eos_coef=0.1)

    train_batches = len(train_loader)
    epochs = args.epochs

    training_config = {
        "Dataset Mode": args.dataset,
        "Total Epochs": epochs,
        "Freeze Epochs": args.freeze_epochs,
        "Batch Size": args.batch_size,
        "Train Batches": train_batches,
        "Learning Rate": args.lr,
        "Optimizer": "Adam",
        "Checkpoint Loaded": checkpoint_to_load or "None"
    }
    logger.print_table("🏋️ Fine-Tuning & Training Configuration", list(training_config.keys()), [list(training_config.values())])

    with rich_training_context() as training_handler:
        for epoch in range(epochs):
            # Backbone freezing / unfreezing management for transfer learning
            if epoch < args.freeze_epochs:
                logger.info(f"Epoch {epoch+1}/{epochs}: Backbone is FROZEN for fine-tuning stability.")
                for param in model.backbone.parameters():
                    param.requires_grad = False
            else:
                if epoch == args.freeze_epochs:
                    logger.info(f"Epoch {epoch+1}/{epochs}: Unfreezing backbone for full fine-tuning!")
                for param in model.backbone.parameters():
                    param.requires_grad = True

            model.train()
            train_epoch_loss = 0.0

            with training_handler.create_training_progress() as epoch_progress:
                epoch_task = epoch_progress.add_task(
                    f"[bold blue] Epoch {epoch+1}/{epochs}",
                    train_loss=0.0,
                    test_loss=0.0,
                    total=train_batches
                )

                for batch_idx, batch in enumerate(train_loader):
                    X, y = batch
                    try:
                        yhat = model(X)
                        loss_dict = criterion(yhat, y)
                        weight_dict = criterion.weight_dict

                        losses = (loss_dict["labels"]["loss_ce"] * weight_dict["class_weighting"] +
                                  loss_dict["boxes"]["loss_bbox"] * weight_dict["bbox_weighting"] +
                                  loss_dict["boxes"]["loss_giou"] * weight_dict["giou_weighting"])

                        train_epoch_loss += losses.item()

                        opt.zero_grad()
                        losses.backward()
                        opt.step()

                        epoch_progress.update(
                            epoch_task,
                            advance=1,
                            train_loss=round(train_epoch_loss / max(1, batch_idx + 1), 5)
                        )
                    except Exception as e:
                        logger.error(f"Training error at epoch {epoch+1}, batch {batch_idx}: {str(e)}")
                        sys.exit(1)

                scheduler.step()

            # Comprehensive Evaluation on Validation Sets
            orig_metrics = evaluate_model(model, existing_val_loader, criterion, num_classes)
            logger.info(f"Epoch {epoch+1} [Original Validation] - Loss: {orig_metrics['loss']}, Acc: {orig_metrics['accuracy']}, Prec: {orig_metrics['precision']}, Rec: {orig_metrics['recall']}, F1: {orig_metrics['f1_score']}")

            if autsl_val_loader is not None:
                autsl_metrics = evaluate_model(model, autsl_val_loader, criterion, num_classes)
                logger.info(f"Epoch {epoch+1} [AUTSL Validation] - Loss: {autsl_metrics['loss']}, Acc: {autsl_metrics['accuracy']}, Prec: {autsl_metrics['precision']}, Rec: {autsl_metrics['recall']}, F1: {autsl_metrics['f1_score']}")

            # Save checkpoint after EVERY epoch
            checkpoint_path = f"checkpoints/{epoch+1}_model.pt"
            save(model.state_dict(), checkpoint_path)
            training_handler.save_checkpoint_status(checkpoint_path, epoch+1)

    logger.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    main()