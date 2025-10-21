# YOLO v8 Training Project

A comprehensive project setup for training YOLO v8 models with proper data management, configuration, and utilities.

## Project Structure

```
yolo_v8_training_project/
├── configs/
│   └── yolo_config.yaml          # Training configuration
├── data/
│   ├── dataset.yaml              # Dataset configuration template
│   ├── images/
│   │   ├── train/                # Training images
│   │   ├── val/                  # Validation images
│   │   └── test/                 # Test images
│   └── labels/
│       ├── train/                # Training labels
│       ├── val/                  # Validation labels
│       └── test/                 # Test labels
├── models/                       # Trained models will be saved here
├── scripts/
│   ├── train.py                  # Main training script
│   ├── inference.py              # Inference script
│   └── data_preparation.py       # Data preparation utilities
├── utils/
│   ├── data_utils.py             # Data handling utilities
│   └── training_utils.py         # Training utilities
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. **Clone or download this project structure**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "from ultralytics import YOLO; print('YOLO v8 installed successfully!')"
   ```

## Dataset Preparation

### 1. Organize Your Dataset

Your dataset should follow this structure:
```
data/
├── images/
│   ├── train/          # Training images (.jpg, .png, etc.)
│   ├── val/            # Validation images
│   └── test/           # Test images (optional)
└── labels/
    ├── train/          # Training labels (.txt files in YOLO format)
    ├── val/            # Validation labels
    └── test/           # Test labels (optional)
```

### 2. YOLO Label Format

Each label file should contain one line per object:
```
class_id center_x center_y width height
```

Where all coordinates are normalized (0-1) relative to image dimensions.

### 3. Dataset Configuration

Edit `data/dataset.yaml` to match your dataset:
- Update `nc` (number of classes)
- Update `names` (class names list)
- Adjust paths if needed

### 4. Validate Your Dataset

```bash
python scripts/data_preparation.py --action validate --data-path data/
```

### 5. Get Dataset Statistics

```bash
python scripts/data_preparation.py --action stats --data-path data/
```

## Training

### Basic Training

```bash
python scripts/train.py --data data/dataset.yaml --model yolo8n.pt --epochs 100
```

### Advanced Training with Custom Configuration

```bash
python scripts/train.py \
    --config configs/yolo_config.yaml \
    --data data/dataset.yaml \
    --model yolo8s.pt \
    --epochs 200 \
    --batch 32 \
    --imgsz 640 \
    --device 0
```

### Training Parameters

- `--model`: Model variant (yolo8n.pt, yolo8s.pt, yolo8m.pt, yolo8l.pt, yolo8x.pt)
- `--epochs`: Number of training epochs
- `--batch`: Batch size
- `--imgsz`: Image size for training
- `--device`: Device to use (cpu, 0, 0,1,2,3)
- `--workers`: Number of worker threads
- `--project`: Project directory for saving results
- `--name`: Experiment name

### Resume Training

```bash
python scripts/train.py --resume --project runs/train --name exp
```

## Inference

### Single Image

```bash
python scripts/inference.py \
    --model models/best.pt \
    --source path/to/image.jpg \
    --conf 0.25 \
    --save-txt
```

### Video

```bash
python scripts/inference.py \
    --model models/best.pt \
    --source path/to/video.mp4 \
    --conf 0.25
```

### Directory of Images

```bash
python scripts/inference.py \
    --model models/best.pt \
    --source path/to/images/ \
    --conf 0.25 \
    --save-txt \
    --save-crop
```

### Inference Parameters

- `--model`: Path to trained model
- `--source`: Source (image, video, or directory)
- `--conf`: Confidence threshold
- `--iou`: IoU threshold for NMS
- `--save-txt`: Save results to .txt files
- `--save-crop`: Save cropped detection boxes
- `--show-labels`: Show labels on output
- `--show-conf`: Show confidence scores

## Data Preparation Utilities

### Split Dataset

If you have a single directory with images and labels, split them into train/val/test:

```bash
python scripts/data_preparation.py \
    --action split \
    --data-path path/to/your/dataset \
    --output-path data/ \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```

### Create Dataset YAML

```bash
python scripts/data_preparation.py \
    --action create_yaml \
    --data-path data/ \
    --class-names class1 class2 class3 \
    --yaml-file my_dataset.yaml
```

## Configuration

### Training Configuration

Edit `configs/yolo_config.yaml` to customize:
- Model parameters
- Training hyperparameters
- Augmentation settings
- Loss function weights
- Learning rate schedule

### Key Configuration Options

- **Model**: Choose from yolo8n.pt (nano) to yolo8x.pt (extra large)
- **Augmentation**: Adjust HSV, rotation, scaling, etc.
- **Loss Weights**: Balance box, classification, and DFL losses
- **Learning Rate**: Initial LR, momentum, weight decay
- **Validation**: IoU threshold, confidence threshold

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir runs/train
```

### Weights & Biases

Add to your training script:
```python
import wandb
wandb.init(project="yolo-v8-training")
```

## Model Variants

| Model | Size | Speed | mAP | Parameters |
|-------|------|-------|-----|------------|
| YOLOv8n | 6.2MB | 8.7ms | 37.3 | 3.2M |
| YOLOv8s | 21.5MB | 11.5ms | 44.9 | 11.2M |
| YOLOv8m | 49.7MB | 20.1ms | 50.2 | 25.9M |
| YOLOv8l | 83.7MB | 25.4ms | 52.9 | 43.7M |
| YOLOv8x | 136.7MB | 32.0ms | 53.9 | 68.2M |

## Tips for Better Training

1. **Data Quality**: Ensure high-quality, diverse training data
2. **Data Augmentation**: Use appropriate augmentation for your use case
3. **Learning Rate**: Start with default values, adjust based on training curves
4. **Batch Size**: Use largest batch size that fits in GPU memory
5. **Image Size**: Higher resolution generally improves accuracy but increases training time
6. **Validation**: Monitor validation metrics to avoid overfitting
7. **Early Stopping**: Stop training when validation metrics plateau

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or image size
2. **Poor Performance**: Check data quality and augmentation settings
3. **Training Stalls**: Verify learning rate and optimizer settings
4. **Label Format Errors**: Ensure labels are in correct YOLO format

### Getting Help

- Check the [Ultralytics documentation](https://docs.ultralytics.com/)
- Review training logs in `logs/` directory
- Validate your dataset structure and labels

## Label Studio Integration

This project includes full integration with Label Studio for annotation workflow.

### Label Studio Setup

1. **Install and setup Label Studio:**
   ```bash
   # Install Label Studio
   python scripts/labelstudio_setup.py --install
   
   # Initialize database
   python scripts/labelstudio_setup.py --init-db
   
   # Start server
   python scripts/labelstudio_setup.py --start-server
   ```

2. **Create a project:**
   - Open http://localhost:8080 in your browser
   - Create a new project
   - Import the configuration from `configs/labelstudio_config.xml`

3. **Import images for annotation:**
   ```bash
   python scripts/labelstudio_import.py \
       --project-id <PROJECT_ID> \
       --images-dir path/to/your/images
   ```

4. **Annotate your images in Label Studio web interface**

5. **Export annotations to YOLO format:**
   ```bash
   python scripts/labelstudio_export.py \
       --project-id <PROJECT_ID> \
       --output-dir data/ \
       --split-ratio 0.8,0.1,0.1
   ```

### Label Studio Workflow

1. **Setup**: Install and configure Label Studio
2. **Import**: Upload your images to a Label Studio project
3. **Annotate**: Use the web interface to draw bounding boxes and assign labels
4. **Export**: Convert annotations to YOLO format for training
5. **Train**: Use the exported data to train your YOLO v8 model

### Label Studio Scripts

- `scripts/labelstudio_setup.py`: Setup and manage Label Studio
- `scripts/labelstudio_import.py`: Import images into Label Studio
- `scripts/labelstudio_export.py`: Export annotations to YOLO format
- `utils/labelstudio_utils.py`: Core integration utilities

### Label Studio Configuration

The project includes a pre-configured Label Studio template (`configs/labelstudio_config.xml`) with:
- All 80 COCO classes
- Rectangle labeling interface
- Color-coded labels for easy identification
- Zoom and rotation controls

## License

This project is provided as-is for educational and research purposes.
