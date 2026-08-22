import torch
import numpy as np
from utils.logger import get_logger

logger = get_logger("metrics")

def evaluate_model(model, dataloader, criterion, num_classes, device=None):
    """
    Evaluates DETR model on a given dataloader and returns comprehensive evaluation metrics:
    - Loss
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    total_loss = 0.0
    total_batches = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in dataloader:
            if isinstance(X, list) or isinstance(X, tuple):
                X = torch.stack([x if isinstance(x, torch.Tensor) else torch.tensor(x) for x in X]).to(device)
            else:
                X = X.to(device)

            yhat = model(X)
            loss_dict = criterion(yhat, y)
            weight_dict = criterion.weight_dict
            
            loss = (loss_dict['labels']['loss_ce'] * weight_dict['class_weighting'] + 
                    loss_dict['boxes']['loss_bbox'] * weight_dict['bbox_weighting'] + 
                    loss_dict['boxes']['loss_giou'] * weight_dict['giou_weighting'])
            
            total_loss += loss.item()
            total_batches += 1

            # Extract top predicted class logits per sample
            # pred_logits: [batch_size, num_queries, num_classes + 1]
            logits = yhat['pred_logits'] # [b, q, num_classes+1]
            probs = logits.softmax(-1)[:, :, :-1] # exclude no-object background class
            max_probs, max_classes = probs.max(-1) # [b, q]

            for b in range(len(y)):
                tgt_labels = y[b]['labels'].cpu().numpy()
                pred_labels = max_classes[b].cpu().numpy()
                pred_scores = max_probs[b].cpu().numpy()

                # Get top confident predictions matching target box count
                num_tgt = len(tgt_labels)
                top_q_indices = np.argsort(pred_scores)[-num_tgt:] if num_tgt > 0 else np.argsort(pred_scores)[-1:]
                top_preds = pred_labels[top_q_indices]

                for t_lbl in tgt_labels:
                    all_targets.append(int(t_lbl))
                for p_lbl in top_preds:
                    all_preds.append(int(p_lbl))

    avg_loss = total_loss / max(1, total_batches)

    # Compute classification metrics across predictions and ground truths
    if len(all_targets) == 0:
        return {
            "loss": round(avg_loss, 5),
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }

    preds_arr = np.array(all_preds[:len(all_targets)])
    tgts_arr = np.array(all_targets)

    correct = (preds_arr == tgts_arr).sum()
    accuracy = float(correct / len(tgts_arr))

    # Compute macro Precision, Recall, and F1
    precisions, recalls, f1s = [], [], []
    for cls_idx in range(num_classes):
        tp = int(((preds_arr == cls_idx) & (tgts_arr == cls_idx)).sum())
        fp = int(((preds_arr == cls_idx) & (tgts_arr != cls_idx)).sum())
        fn = int(((preds_arr != cls_idx) & (tgts_arr == cls_idx)).sum())

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    return {
        "loss": round(avg_loss, 5),
        "accuracy": round(accuracy, 4),
        "precision": round(macro_precision, 4),
        "recall": round(macro_recall, 4),
        "f1_score": round(macro_f1, 4)
    }
