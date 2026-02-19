"""
Simple Training Pipeline
=======================
Train a popup detection model from your images.

Usage:
    # Option 1: Use config file
    python train_simple.py

    # Option 2: Command line arguments
    python train_simple.py --data /path/to/images --epochs 50

    # Option 3: Just evaluate existing model
    python train_simple.py --data /path/to/images --evaluate path/to/model.pt
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from datetime import datetime


def load_training_config(config_path="train_config.yaml"):
    """Load training configuration"""
    import yaml

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def prepare_dataset(
    data_dir, labels_dir=None, test_split=0.2, output_dir="dataset_ready"
):
    """Split images into train/test folders"""

    print("=" * 60)
    print("PREPARING DATASET")
    print("=" * 60)

    # Create directories
    train_img = Path(output_dir) / "train" / "images"
    train_lbl = Path(output_dir) / "train" / "labels"
    val_img = Path(output_dir) / "val" / "images"
    val_lbl = Path(output_dir) / "val" / "labels"

    for d in [train_img, train_lbl, val_img, val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    # Get all images
    extensions = [".jpg", ".jpeg", ".png"]
    all_images = []

    for ext in extensions:
        all_images.extend(Path(data_dir).glob(f"*{ext}"))
        all_images.extend(Path(data_dir).glob(f"*{ext.upper()}"))

    print(f"Found {len(all_images)} images")

    # Handle labels directory
    if labels_dir is None:
        labels_dir = data_dir

    # Shuffle and split
    random.seed(42)
    all_images = list(all_images)
    random.shuffle(all_images)

    split_idx = int(len(all_images) * (1 - test_split))
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    print(f"Train: {len(train_images)} images")
    print(f"Test: {len(val_images)} images")

    # Copy images and labels
    for img in train_images:
        shutil.copy2(img, train_img / img.name)

        label_file = Path(labels_dir) / f"{img.stem}.txt"
        if label_file.exists():
            shutil.copy2(label_file, train_lbl / label_file.name)
        else:
            (train_lbl / f"{img.stem}.txt").touch()

    for img in val_images:
        shutil.copy2(img, val_img / img.name)

        label_file = Path(labels_dir) / f"{img.stem}.txt"
        if label_file.exists():
            shutil.copy2(label_file, val_lbl / label_file.name)
        else:
            (val_lbl / f"{img.stem}.txt").touch()

    # Create YAML config
    yaml_content = f"""path: {Path.cwd() / output_dir}
train: train/images
val: val/images

nc: 1
names:
  0: popup
"""

    yaml_path = Path(output_dir) / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\nDataset ready: {output_dir}/")
    print(f"Config: {yaml_path}")

    return str(yaml_path)


def train_model(dataset_yaml, config):
    """Train the model"""

    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)

    from ultralytics import YOLO

    # Get model
    base_model = config.get("base_model", "yolov8n")
    print(f"Loading base model: {base_model}")

    model = YOLO(f"{base_model}.pt")

    # Get training params
    epochs = config.get("epochs", 100)
    imgsz = config.get("imgsz", 640)
    batch = config.get("batch", 16)
    run_name = config.get("run_name", "popup_detector")

    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Batch size: {batch}")

    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project="runs/detect",
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

    return model


def evaluate_model(model_path, val_images_dir):
    """Evaluate model on test images"""

    print("\n" + "=" * 60)
    print("EVALUATING MODEL")
    print("=" * 60)

    from ultralytics import YOLO

    model = YOLO(model_path)

    print(f"\nTesting on: {val_images_dir}")
    print("-" * 40)

    # Test at different confidence levels
    for conf in [0.1, 0.15, 0.2, 0.25]:
        results = model.predict(
            source=val_images_dir, augment=True, conf=conf, iou=0.5, verbose=False
        )

        total_detections = sum(len(r.boxes) for r in results if r.boxes is not None)
        images_with_detections = sum(
            1 for r in results if r.boxes is not None and len(r.boxes) > 0
        )

        print(
            f"conf={conf}: {images_with_detections} images with detections, {total_detections} total"
        )


def main():
    parser = argparse.ArgumentParser(description="Simple Training Pipeline")

    # Config file option
    parser.add_argument(
        "--config", default="train_config.yaml", help="Config file path"
    )

    # Override options
    parser.add_argument("--data", help="Training images folder (overrides config)")
    parser.add_argument("--labels", help="Training labels folder (overrides config)")
    parser.add_argument(
        "--test-split", type=float, help="Test split (overrides config)"
    )
    parser.add_argument("--epochs", type=int, help="Training epochs (overrides config)")
    parser.add_argument("--base-model", help="Base model (yolov8n, yolov8s, yolov8m)")

    # Options
    parser.add_argument("--skip-train", action="store_true", help="Only prepare data")
    parser.add_argument("--evaluate-only", default=None, help="Evaluate existing model")

    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        config = load_training_config(args.config)
    else:
        config = {}

    # Override with command line args
    if args.data:
        config["data_folder"] = args.data
    if args.labels:
        config["labels_folder"] = args.labels
    if args.test_split:
        config["test_split"] = args.test_split
    if args.epochs:
        config["epochs"] = args.epochs
    if args.base_model:
        config["base_model"] = args.base_model

    # Validate
    if "data_folder" not in config:
        print("ERROR: No data folder specified!")
        print("Set 'data_folder' in train_config.yaml or use --data argument")
        sys.exit(1)

    # If evaluating existing model
    if args.evaluate_only:
        val_dir = (
            Path("dataset_ready/val/images")
            if Path("dataset_ready/val/images").exists()
            else None
        )
        if val_dir and val_dir.exists():
            evaluate_model(args.evaluate_only, str(val_dir))
        else:
            print("No test images found. Run training first to create test set.")
        return

    # Prepare dataset
    yaml_path = prepare_dataset(
        config["data_folder"],
        config.get("labels_folder"),
        config.get("test_split", 0.2),
    )

    if args.skip_train:
        print("\n[OK] Data prepared! Run without --skip-train to train.")
        return

    # Train model
    model = train_model(yaml_path, config)

    # Find best model
    import glob

    weights = glob.glob("runs/detect/*/weights/best.pt")
    if weights:
        best_model = weights[-1]
        print(f"\n[OK] Model saved: {best_model}")

        # Evaluate
        if config.get("auto_eval", True):
            val_dir = Path(yaml_path).parent / "val" / "images"
            if val_dir.exists():
                evaluate_model(best_model, str(val_dir))

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
