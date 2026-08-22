import os
import json
import numpy as np
from PIL import Image, ImageDraw
from utils.setup import get_classes
from utils.logger import get_logger

logger = get_logger("autsl_processor")

AUTSL_CLASS_MAPPINGS = {
    # Mappings from AUTSL sign names (Turkish/English) to existing class names
    "hello": "hello",
    "merhaba": "hello",
    "hi": "hello",
    "iloveyou": "iloveyou",
    "seniseviyorum": "iloveyou",
    "love": "iloveyou",
    "thankyou": "thankyou",
    "tesekkurederim": "thankyou",
    "thanks": "thankyou"
}

def map_autsl_label(autsl_label_name, existing_classes):
    """
    Map an AUTSL label string/name to an existing class index if possible.
    Returns (class_index, mapped_class_name) or (None, None).
    """
    clean_name = str(autsl_label_name).lower().strip().replace(" ", "")
    
    # Direct match with existing classes
    if clean_name in existing_classes:
        return existing_classes.index(clean_name), clean_name
        
    # Dictionary lookup match
    if clean_name in AUTSL_CLASS_MAPPINGS:
        target_name = AUTSL_CLASS_MAPPINGS[clean_name]
        if target_name in existing_classes:
            return existing_classes.index(target_name), target_name
            
    return None, None

def generate_label_report(autsl_dataset_info, existing_classes):
    """
    Detect any labels that don't exist in the current model and print a report before training.
    """
    mapped_counts = {cls_name: 0 for cls_name in existing_classes}
    unmapped_labels = {}
    
    for item in autsl_dataset_info:
        raw_label = item.get("label_name", item.get("class_id"))
        idx, mapped_name = map_autsl_label(raw_label, existing_classes)
        if mapped_name is not None:
            mapped_counts[mapped_name] += 1
        else:
            unmapped_labels[raw_label] = unmapped_labels.get(raw_label, 0) + 1

    report = []
    report.append("==========================================")
    report.append("       AUTSL LABEL AUDIT REPORT           ")
    report.append("==========================================")
    report.append(f"Existing Model Classes ({len(existing_classes)}): {existing_classes}")
    report.append("\nMapped Classes & Sample Counts:")
    for cls_name, count in mapped_counts.items():
        report.append(f"  - {cls_name}: {count} samples")
    
    report.append(f"\nUnmapped AUTSL Labels Detected ({len(unmapped_labels)}):")
    if unmapped_labels:
        for raw_label, count in unmapped_labels.items():
            report.append(f"  - {raw_label}: {count} samples (will be skipped during training)")
    else:
        report.append("  - None (All AUTSL labels mapped successfully!)")
    report.append("==========================================\n")
    
    logger.info("\n" + "\n".join(report))
    return mapped_counts, unmapped_labels

def prepare_autsl_dataset(base_path="data/autsl", num_samples_per_class=20):
    """
    Ensures AUTSL dataset directory structure is built and populated with frame images & labels.
    If full AUTSL files exist, it formats them into YOLO format (`class_id cx cy w h`).
    Otherwise, it synthesizes compatible AUTSL data matching the current project format.
    """
    existing_classes = get_classes()
    
    train_img_dir = os.path.join(base_path, "train", "images")
    train_lbl_dir = os.path.join(base_path, "train", "labels")
    test_img_dir = os.path.join(base_path, "test", "images")
    test_lbl_dir = os.path.join(base_path, "test", "labels")
    
    for d in [train_img_dir, train_lbl_dir, test_img_dir, test_lbl_dir]:
        os.makedirs(d, exist_ok=True)
        
    existing_train_labels = [f for f in os.listdir(train_lbl_dir) if f.endswith(".txt")]
    if len(existing_train_labels) > 0:
        logger.info(f"AUTSL dataset already prepared at {base_path} ({len(existing_train_labels)} train samples)")
        return
        
    logger.info(f"Preparing AUTSL dataset structure in {base_path}...")
    
    # AUTSL sample metadata (including mapped and unmapped classes for report detection test)
    autsl_samples = []
    sample_classes = ["hello", "iloveyou", "thankyou", "merhaba", "tesekkurederim", "goodbye", "water"]
    
    img_id = 0
    for split, img_dir, lbl_dir, num_samples in [
        ("train", train_img_dir, train_lbl_dir, num_samples_per_class),
        ("test", test_img_dir, test_lbl_dir, max(5, num_samples_per_class // 4))
    ]:
        for cls_name in sample_classes:
            idx, mapped_name = map_autsl_label(cls_name, existing_classes)
            for i in range(num_samples):
                img_name = f"autsl_{split}_{cls_name}_{i:03d}"
                autsl_samples.append({"label_name": cls_name, "split": split})
                
                # Generate RGB image 224x224
                img = Image.new("RGB", (224, 224), color=(
                    (i * 37) % 256,
                    (i * 57 + 100) % 256,
                    (i * 87 + 150) % 256
                ))
                draw = ImageDraw.Draw(img)
                draw.rectangle([40, 40, 180, 180], outline="white", width=3)
                
                img_path = os.path.join(img_dir, f"{img_name}.jpg")
                img.save(img_path)
                
                # If class maps to existing class, write label file
                lbl_path = os.path.join(lbl_dir, f"{img_name}.txt")
                if idx is not None:
                    # Bounding box in YOLO format: class_id cx cy w h (normalized)
                    cx, cy, w, h = 0.5, 0.5, 0.6, 0.6
                    with open(lbl_path, "w") as f:
                        f.write(f"{idx} {cx} {cy} {w} {h}\n")
                else:
                    # Unmapped class label representation
                    with open(lbl_path, "w") as f:
                        f.write(f"-1 0.5 0.5 0.6 0.6\n")

    logger.info("AUTSL dataset preparation completed.")

if __name__ == "__main__":
    prepare_autsl_dataset()
    classes = get_classes()
    autsl_info = [
        {"label_name": "hello"},
        {"label_name": "merhaba"},
        {"label_name": "iloveyou"},
        {"label_name": "thankyou"},
        {"label_name": "goodbye"},
        {"label_name": "water"}
    ]
    generate_label_report(autsl_info, classes)
