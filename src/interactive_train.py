import os
import torch
from torch.utils.data import DataLoader
from torch import optim, save

from data import DETRData, AUTSLDETRData, CombinedDETRData, BalancedDatasetSampler
from model import DETR
from loss import DETRLoss, HungarianMatcher
from metrics import evaluate_model
from utils.logger import get_logger
from utils.setup import get_classes
from utils.boxes import stacker

logger = get_logger("interactive_train")

def run_interactive_fine_tuning(epochs=15, freeze_epochs=3, batch_size=4, lr=1e-5):
    """
    Fine-tunes the SignDETR model on the newly captured user sign dataset.
    """
    os.makedirs("checkpoints", exist_ok=True)
    existing_classes = get_classes()
    num_classes = len(existing_classes)

    logger.info(f"Starting interactive fine-tuning for {num_classes} total classes: {existing_classes}")

    user_train_path = "data/user/train"
    if not os.path.exists(user_train_path) or len(os.listdir(os.path.join(user_train_path, "labels"))) == 0:
        logger.warning(f"No user sign data found at {user_train_path}. Using existing dataset for fine-tuning.")
        train_dataset = DETRData("data/train", train=True)
    else:
        # Merge existing data and newly captured user data
        existing_ds = DETRData("data/train", train=True)
        user_ds = DETRData("data/user/train", train=True)
        train_dataset = CombinedDETRData(existing_path="data/train", autsl_path="data/user/train", train=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=stacker, drop_last=False, shuffle=True)

    val_dataset = DETRData("data/test", train=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=stacker, drop_last=False)

    model = DETR(num_classes=num_classes)

    # Check for latest checkpoint or pretrained model
    latest_ckpt = None
    if os.path.exists("checkpoints"):
        pts = [f for f in os.listdir("checkpoints") if f.endswith(".pt")]
        if pts:
            pts_sorted = sorted(pts, key=lambda x: int(x.split("_")[0]) if x.split("_")[0].isdigit() else -1)
            latest_ckpt = os.path.join("checkpoints", pts_sorted[-1])
    if not latest_ckpt and os.path.exists("pretrained/4426_model.pt"):
        latest_ckpt = "pretrained/4426_model.pt"

    if latest_ckpt:
        logger.info(f"Loading base weights from: {latest_ckpt}")
        model.load_pretrained(latest_ckpt)

    # Ensure output head matches current class count
    model.update_num_classes(num_classes)

    opt = optim.Adam(model.parameters(), lr=lr)
    weights = {"class_weighting": 1, "bbox_weighting": 5, "giou_weighting": 2}
    matcher = HungarianMatcher(weights)
    criterion = DETRLoss(num_classes=num_classes, matcher=matcher, weight_dict=weights, eos_coef=0.1)

    logger.info(f"Fine-tuning model for {epochs} epochs (Backbone frozen for first {freeze_epochs} epochs)...")

    for epoch in range(1, epochs + 1):
        if epoch <= freeze_epochs:
            for param in model.backbone.parameters():
                param.requires_grad = False
        else:
            for param in model.backbone.parameters():
                param.requires_grad = True

        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            X, y = batch
            try:
                yhat = model(X)
                loss_dict = criterion(yhat, y)
                w_dict = criterion.weight_dict
                loss = (loss_dict["labels"]["loss_ce"] * w_dict["class_weighting"] +
                        loss_dict["boxes"]["loss_bbox"] * w_dict["bbox_weighting"] +
                        loss_dict["boxes"]["loss_giou"] * w_dict["giou_weighting"])

                epoch_loss += loss.item()
                opt.zero_grad()
                loss.backward()
                opt.step()
            except Exception as e:
                logger.error(f"Error at epoch {epoch}, batch {batch_idx}: {str(e)}")

        val_metrics = evaluate_model(model, val_loader, criterion, num_classes)
        logger.info(f"Epoch {epoch}/{epochs} - Loss: {round(epoch_loss/max(1, len(train_loader)), 4)} | Val Acc: {val_metrics['accuracy']} | Val F1: {val_metrics['f1_score']}")

    # Save fine-tuned interactive model checkpoint
    user_ckpt_path = "checkpoints/user_model.pt"
    save(model.state_dict(), user_ckpt_path)
    logger.success(f"Fine-tuning complete! Model saved to '{user_ckpt_path}'")
    return user_ckpt_path

if __name__ == "__main__":
    run_interactive_fine_tuning(epochs=3, freeze_epochs=1)
