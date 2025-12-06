# Deliverables - Playground Activity Recognition with MP-GCN

## Structure

```
deliverables/
├── graphs/                      # Graph adjacency matrices
│   ├── A0_self.npy             # Self-loop adjacency (identity)
│   ├── A_intra.npy             # Intra-person body + object connections
│   ├── A_inter.npy             # Inter-person connections
│   ├── A_combined.npy          # Combined [A0, A_intra, A_inter]
│   ├── A_intra_raw.npy         # Unnormalized A_intra (for visualization)
│   ├── A_panoramic_full.npy    # Full panoramic graph (reference)
│   ├── connect_joint.npy       # Parent joint indices for bone computation
│   └── graph_metadata.json     # Graph configuration and metadata
│
├── data/
│   ├── videos.csv              # Video index (video_id, camera, label)
│   ├── annotations.json        # Raw annotations
│   └── npy/                    # Per-sample .npy files (if available)
│
├── configs/
│   └── objects.yaml            # Object centroids per camera
│
├── metrics/
│   ├── training_history.json   # Full training history
│   └── training_summary.json   # Best metrics summary
│
├── checkpoints/
│   ├── best_model.pth          # Best model checkpoint (symlink)
│   ├── best_model_config.json  # Model configuration
│   ├── final_model.pth         # Final model checkpoint (symlink)
│   └── final_model_config.json # Model configuration
│
└── README.md                   # This file
│
└── pose_batch.zip # Containing all of the yolo-extracted poses from the videos
```

## Graph Details

### Per-Person Graph (Used by Model)
- **Nodes**: 25 (17 COCO keypoints + 8 objects)
- **Scales**: 3 (self-loops, intra-person, inter-person placeholder)
- **Shape**: (3, 25, 25)

### Adjacency Types
1. **A0 (Self)**: Identity matrix - self-connections
2. **A_intra**: Body skeleton topology + object-hand connections
   - Skeleton: COCO format edges
   - Objects: Connected to wrists (joints 9, 10)
3. **A_inter**: Inter-person connections (pelvis-to-pelvis)

### Streams (J, B, JM, BM)
- **J**: Joint coordinates (C, T, V, M)
- **B**: Bone vectors (joint - parent_joint)
- **JM**: Joint motion (temporal difference)
- **BM**: Bone motion (temporal difference)

## Model Architecture
- **Type**: PlaygroundGCNLite (per-person ST-GCN)
- **Classes**: 5 (Social_People, Transit, Play_Object_Normal, Adult_Assisting, sliding)
- **Input**: (N, C, T, V, M) = (batch, 2, 32, 25, 6)

## How to Load Graphs

```python
import numpy as np

# Load combined adjacency
A = np.load('deliverables/graphs/A_combined.npy')
print(f"Adjacency shape: {A.shape}")  # (3, 25, 25)

# Load individual matrices
A0 = np.load('deliverables/graphs/A0_self.npy')
A_intra = np.load('deliverables/graphs/A_intra.npy')
A_inter = np.load('deliverables/graphs/A_inter.npy')

# Load connect_joint for bone computation
connect_joint = np.load('deliverables/graphs/connect_joint.npy')
```

