"""
Popup Detector - Extract Crops for OCR
======================================
This script detects popups in images and crops them for OCR processing.

Cell 1: Detection + Cropping
Cell 2: S3 Upload (run separately when ready)
"""

import os
import random
import json
import yaml
import glob
from pathlib import Path
from datetime import datetime

# Check for ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    print("Installing ultralytics...")
    import subprocess

    subprocess.run(["pip", "install", "ultralytics"])
    from ultralytics import YOLO


# ============================================================
# CELL 1: LOAD CONFIG & DETECT
# ============================================================


def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_random_images(folder, num_images, extensions=None):
    """Get random images from a folder"""
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png"]

    all_images = []
    for ext in extensions:
        all_images.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        all_images.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))

    if num_images >= len(all_images):
        return all_images

    return random.sample(all_images, num_images)


def detect_and_crop(model, image_path, conf, iou, augment):
    """Run detection on a single image and return crops"""
    results = model.predict(
        source=image_path, conf=conf, iou=iou, augment=augment, verbose=False
    )

    if len(results) == 0 or results[0].boxes is None:
        return []

    result = results[0]
    img = result.orig_img

    crops = []
    boxes = result.boxes

    for i, box in enumerate(boxes):
        # Get coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf_score = float(box.conf[0])

        # Crop the image
        crop = img[int(y1) : int(y2), int(x1) : int(x2)]

        crops.append(
            {
                "crop": crop,
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "confidence": conf_score,
                "image_name": Path(image_path).name,
                "box_index": i,
            }
        )

    return crops


def save_crops(crops, output_folder, base_name):
    """Save cropped images and coordinates"""
    os.makedirs(output_folder, exist_ok=True)

    saved_files = []
    coordinates = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for crop_data in crops:
        crop = crop_data["crop"]

        # Save crop image
        filename = f"{base_name}_box{crop_data['box_index']}_{timestamp}.jpg"
        filepath = os.path.join(output_folder, filename)

        import cv2

        cv2.imwrite(filepath, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

        saved_files.append(filename)

        # Save coordinates
        coordinates.append(
            {
                "filename": filename,
                "source_image": crop_data["image_name"],
                "x1": crop_data["x1"],
                "y1": crop_data["y1"],
                "x2": crop_data["x2"],
                "y2": crop_data["y2"],
                "confidence": round(crop_data["confidence"], 4),
            }
        )

    return saved_files, coordinates


def run_detection(config=None):
    """Main detection and cropping pipeline"""

    # Load config
    if config is None:
        config = load_config()

    print("=" * 60)
    print("POPUP DETECTOR - EXTRACT FOR OCR")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Model: {config.get('model_path', 'models/popup_detector_best.pt')}")
    print(f"  Conf threshold: {config.get('conf_threshold', 0.15)}")
    print(f"  IOU threshold: {config.get('iou_threshold', 0.5)}")
    print(f"  Images per run: {config.get('images_per_run', 50)}")
    print(f"  Output folder: {config.get('output_folder', 'output/crops')}")
    print()

    # Load model
    model_path = config.get("model_path", "models/popup_detector_best.pt")
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    print("Model loaded!\n")

    # Get random images
    input_folder = config.get("input_folder", "test_images")
    num_images = config.get("images_per_run", 50)
    extensions = config.get("image_extensions", [".jpg", ".jpeg", ".png"])

    print(f"Selecting {num_images} random images from {input_folder}...")
    image_files = get_random_images(input_folder, num_images, extensions)

    if not image_files:
        print(f"ERROR: No images found in {input_folder}")
        return

    print(f"Found {len(image_files)} images to process\n")

    # Process images
    all_coordinates = []
    total_crops = 0

    for i, img_path in enumerate(image_files):
        try:
            crops = detect_and_crop(
                model,
                img_path,
                config.get("conf_threshold", 0.15),
                config.get("iou_threshold", 0.5),
                config.get("augment", True),
            )

            if crops:
                output_folder = config.get("output_folder", "output/crops")
                base_name = Path(img_path).stem

                files, coords = save_crops(crops, output_folder, base_name)
                all_coordinates.extend(coords)
                total_crops += len(files)

                print(
                    f"  [{i + 1}/{len(image_files)}] {Path(img_path).name}: {len(crops)} popups detected"
                )
            else:
                print(
                    f"  [{i + 1}/{len(image_files)}] {Path(img_path).name}: No popups"
                )

        except Exception as e:
            print(f"  [{i + 1}/{len(image_files)}] ERROR: {e}")

    # Save coordinates JSON
    if all_coordinates and config.get("save_coordinates", True):
        coords_file = os.path.join(
            config.get("output_folder", "output/crops"), "detection_coordinates.json"
        )
        with open(coords_file, "w") as f:
            json.dump(all_coordinates, f, indent=2)
        print(f"\nCoordinates saved to: {coords_file}")

    print("\n" + "=" * 60)
    print("DETECTION COMPLETE")
    print("=" * 60)
    print(f"Images processed: {len(image_files)}")
    print(f"Total popup crops: {total_crops}")
    print(f"Output folder: {config.get('output_folder', 'output/crops')}")
    print("\nReady for S3 upload (run Cell 2)!")


# Run Cell 1
if __name__ == "__main__":
    run_detection()
