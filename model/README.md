# Playground Activity Recognition Model

A skeleton-based action recognition system for playground activity classification, inspired by [MP-GCN](https://github.com/mgiant/MP-GCN) (Multi-Person Graph Convolutional Network).

## Features

- **Multi-Stream Input**: Supports Joint (J), Bone (B), Velocity (V), and Motion (M) streams
- **Panoramic Graph**: Person-object and person-person interaction modeling
- **Skeleton Normalization**: Hip-centered with torso-scaled coordinates
- **Object Integration**: Incorporates static object positions from cameras
- **Regularization**: Label smoothing, mixup, and dropout
- **Multi-Person Support**: Handles up to 6 persons per frame

## Architecture

The model follows the ST-GCN (Spatial-Temporal Graph Convolutional Network) architecture:

```
Input (N, C, T, V, M)
    ↓
Batch Normalization
    ↓
┌─────────────────────────────────────┐
│  ST-GCN Block                       │
│  ├── Spatial Graph Convolution      │
│  ├── Multi-Scale Temporal Conv      │
│  └── Residual Connection            │
└─────────────────────────────────────┘
    ↓ (×5 blocks with stride=2)
Global Average Pooling
    ↓
Dropout
    ↓
Fully Connected → 7 classes
```

## Activity Classes

1. `no_activity` - No significant activity
2. `Social_People` - Social interaction between people
3. `Transit` - Walking/moving through the area
4. `Play_Object_Normal` - Normal play with playground equipment
5. `Play_Object_Risk` - Risky play behavior
6. `Adult_Assisting` - Adult helping/supervising children
7. `sliding` - Sliding activity

## Installation

```bash
cd /home/sformador/equipo1/finalfinal

# Install dependencies
pip install torch numpy pyyaml scikit-learn
```

## Quick Start

### Test the Model

```bash
python model/test_model.py
```

### Train the Model

```bash
# Basic training
python model/train.py

# Lightweight model with custom parameters
python model/train.py --lite --epochs 50 --batch_size 8 --lr 0.001

# Full training with all options
python model/train.py \
    --poses_root runs/pose_batch \
    --annotations annotationsv2.json \
    --objects objects.yaml \
    --epochs 100 \
    --batch_size 16 \
    --lr 0.001 \
    --dropout 0.3 \
    --label_smoothing 0.1 \
    --mixup_alpha 0.2
```

## Data Format

### Pose Data (YOLO Format)
Located in `runs/pose_batch/<camera>/<video>/labels/`:
```
class cx cy w h kp1_x kp1_y kp1_conf ... kp17_x kp17_y kp17_conf
```

### Annotations (JSON)
File: `annotationsv2.json`
```json
{
    "data": {
        "video_id": "camera_videoname",
        "camera": "camera_name"
    },
    "annotations": [{
        "result": [{
            "value": {"choices": ["activity_label"]}
        }]
    }]
}
```

### Objects Config (YAML)
File: `objects.yaml`
```yaml
camera_name:
  resolution: [2560, 1440]
  objects:
    - name: bench
      type: bench
      centroid: [0.7, 0.4]
```

## Model Files

```
model/
├── __init__.py          # Package exports
├── config.py            # Configuration and constants
├── graphs.py            # Basic skeleton graph
├── panoramic_graph.py   # Multi-person panoramic graph
├── blocks.py            # GCN building blocks
├── networks.py          # Model architectures
├── preprocessing.py     # Data preprocessing
├── dataset.py           # PyTorch dataset and loaders
├── trainer.py           # Training pipeline
├── train.py             # Training script
└── test_model.py        # Testing script
```

## Key Concepts

### Skeleton Graph
- 17 COCO keypoints per person
- Spatial edges follow human body topology
- Inter-person edges connect hip joints
- Object nodes connect to hand joints

### Normalization
1. **Centering**: Skeleton centered at hip midpoint
2. **Scaling**: Normalized by shoulder width (torso length)
3. **Temporal**: Uniform sampling to fixed length (32 frames)

### Multi-Stream Features
- **Joint (J)**: Raw joint positions + relative to center
- **Bone (B)**: Bone vectors (child - parent joint)
- **Velocity (V)**: Temporal difference of joints
- **Motion (M)**: Temporal difference of bones

## Training Details

### Regularization
- **Label Smoothing**: 0.1 (softens one-hot labels)
- **Mixup**: α=0.2 (interpolates samples)
- **Dropout**: 0.3 (applied before FC layer)
- **Weight Decay**: 1e-4 (L2 regularization)

### Learning Rate Schedule
- **Warmup**: Linear warmup for 5 epochs
- **Decay**: Cosine annealing after warmup

### Data Augmentation
- Random rotation (±15°)
- Random scale (0.9-1.1×)
- Random horizontal flip (with joint swapping)
- Gaussian noise

## Usage Example

```python
from model import (
    PlaygroundGCNLite,
    create_data_loaders,
    ModelConfig,
    Trainer
)

# Configure
config = ModelConfig()
config.epochs = 50
config.batch_size = 16

# Load data
train_loader, val_loader = create_data_loaders(
    poses_root='runs/pose_batch',
    annotations_path='annotationsv2.json',
    objects_yaml='objects.yaml',
    config=config
)

# Create model
model = PlaygroundGCNLite(
    num_classes=7,
    num_persons=4,
    num_joints=25,  # 17 keypoints + 8 objects
    dropout=0.3
)

# Train
trainer = Trainer(model, train_loader, val_loader, config)
trainer.train()
```

## Performance

Initial training results (3 epochs, 251 train / 63 val samples):
- **Accuracy**: ~46%
- **F1 Score**: ~40%

*Note: Performance improves significantly with more epochs and hyperparameter tuning.*

## References

- [MP-GCN Paper](https://link.springer.com/chapter/10.1007/978-3-031-73202-7_15): "Skeleton-based Group Activity Recognition via Spatial-Temporal Panoramic Graph"
- [ST-GCN](https://github.com/yysijie/st-gcn): Spatial Temporal Graph Convolutional Networks
- [YOLO Pose](https://docs.ultralytics.com/tasks/pose/): Ultralytics YOLO pose estimation

## License

MIT License
