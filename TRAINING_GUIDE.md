# Training Guide

## Quick Start

```bash
# Train with default settings (from train_config.yaml)
python train_simple.py

# Or with command line options
python train_simple.py --data /path/to/images --epochs 50
```

## Configuration File (train_config.yaml)

### Data Settings
```yaml
# Folder containing your training images
data_folder: dataset_final_train/train/images

# Folder containing labels (leave empty if same as images)
labels_folder: dataset_final_train/train/labels
```

### Split Settings
```yaml
# 20% of data will be used for testing
test_split: 0.2
```

### Model Settings
```yaml
# Base model options:
# yolov8n - Fastest, smallest (good for testing)
# yolov8s  - Small, balanced
# yolov8m  - Medium, better accuracy
base_model: yolov8n
```

### Training Settings
```yaml
# How many times to see all data
# 100 = standard training
# 50 = faster training
# 200 = more training (may overfit)
epochs: 100

# Image resolution
# 640 = standard (faster)
# 1280 = higher quality (slower)
imgsz: 640

# Batch size (how many images at once)
# 16 = standard
# 8 = uses less memory
# 32 = faster training, needs more RAM
batch: 16
```

## Examples

### Example 1: Quick test with fewer epochs
```bash
python train_simple.py --epochs 20 --base-model yolov8n
```

### Example 2: Higher quality model
```bash
python train_simple.py --base-model yolov8m --epochs 150 --imgsz 1280
```

### Example 3: Custom data folder
```bash
python train_simple.py --data /path/to/my/images --labels /path/to/my/labels
```

### Example 4: Only prepare data (no training)
```bash
python train_simple.py --skip-train
```

### Example 5: Evaluate existing model
```bash
python train_simple.py --data /path/to/images --evaluate path/to/model.pt
```

## Understanding Parameters

| Parameter | What it does | Recommended |
|-----------|--------------|-------------|
| `epochs` | How many times the model sees all images | 100 for production, 20-50 for testing |
| `base_model` | Starting model | yolov8n for speed, yolov8m for accuracy |
| `imgsz` | Image resolution | 640 standard, 1280 for better quality |
| `batch` | Images processed at once | 16 standard, lower if out of memory |
| `test_split` | % for testing | 0.2 = 20% for testing |

## Output

After training:
- Model saved to: `runs/detect/<run_name>/weights/best.pt`
- Evaluation results printed automatically
