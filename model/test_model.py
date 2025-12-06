#!/usr/bin/env python3
"""
Test script to verify the model and data pipeline work correctly.

This script:
1. Tests data loading and preprocessing
2. Creates a small model and runs a forward pass
3. Verifies shapes and outputs are correct
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

print("=" * 60)
print("Testing Playground Activity Recognition Model")
print("=" * 60)

# Test 1: Config and imports
print("\n[1] Testing imports...")
try:
    from model import (
        ACTIVITY_LABELS,
        NUM_KEYPOINTS,
        ModelConfig,
        create_graph,
        create_panoramic_graph,
        PlaygroundGCN,
        PlaygroundGCNLite,
    )
    print("✓ All imports successful")
    print(f"  Activity labels: {list(ACTIVITY_LABELS.keys())}")
    print(f"  Num keypoints: {NUM_KEYPOINTS}")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test 2: Graph construction
print("\n[2] Testing graph construction...")
try:
    # Simple graph
    graph = create_graph(num_persons=1, max_hop=2, labeling='spatial')
    print(f"✓ Simple graph: {graph}")
    print(f"  Adjacency shape: {graph.A.shape}")
    
    # Panoramic graph
    pgraph = create_panoramic_graph(num_persons=4, num_objects=2, max_hop=2)
    print(f"✓ Panoramic graph: {pgraph}")
    print(f"  Nodes: {pgraph.num_nodes} (persons={pgraph.num_persons}, objects={pgraph.num_objects})")
    print(f"  Adjacency shape: {pgraph.A.shape}")
except Exception as e:
    print(f"✗ Graph error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Model construction
print("\n[3] Testing model construction...")
try:
    config = ModelConfig()
    config.max_persons = 4
    config.num_keypoints = 17
    config.base_channels = 32  # Smaller for testing
    
    # Create graph
    graph = create_panoramic_graph(num_persons=4, num_objects=0)
    A = torch.FloatTensor(graph.A)
    
    # Create model
    model = PlaygroundGCNLite(
        num_classes=7,
        num_persons=4,
        num_joints=17,
        num_objects=0,
        A=A,
        dropout=0.1
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {num_params:,} parameters")
except Exception as e:
    print(f"✗ Model construction error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Forward pass
print("\n[4] Testing forward pass...")
try:
    # Create dummy input: (N, C, T, V, M)
    batch_size = 2
    channels = 2
    seq_length = 48
    num_joints = 17
    num_persons = 4
    
    x = torch.randn(batch_size, channels, seq_length, num_joints, num_persons)
    print(f"  Input shape: {x.shape}")
    
    model.eval()
    with torch.no_grad():
        logits, features = model(x)
    
    print(f"✓ Forward pass successful")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Features shape: {features.shape}")
    print(f"  Sample predictions: {logits.softmax(dim=1)[0].numpy().round(3)}")
except Exception as e:
    print(f"✗ Forward pass error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Data preprocessing functions
print("\n[5] Testing data preprocessing...")
try:
    from model.preprocessing import (
        normalize_skeleton,
        temporal_sample,
        compute_bone_features,
        compute_motion_features,
        prepare_multi_stream_input
    )
    from model.config import CONNECT_JOINT
    
    # Create dummy pose data
    T, M, V, C = 50, 4, 17, 2
    poses = np.random.rand(T, M, V, C).astype(np.float32)
    
    # Test normalization
    normalized = normalize_skeleton(poses, None)
    print(f"✓ Normalization: {poses.shape} -> {normalized.shape}")
    
    # Test temporal sampling
    sampled = temporal_sample(normalized, 32)
    print(f"✓ Temporal sampling: {normalized.shape} -> {sampled.shape}")
    
    # Convert to (C, T, V, M)
    data = sampled.transpose(3, 0, 2, 1)
    
    # Test bone computation
    connect_joint = np.array(CONNECT_JOINT)
    bones = compute_bone_features(data, connect_joint)
    print(f"✓ Bone features: {data.shape} -> {bones.shape}")
    
    # Test motion computation
    motion = compute_motion_features(data)
    print(f"✓ Motion features: {data.shape} -> {motion.shape}")
    
    # Test multi-stream input (JBVM)
    multi_stream = prepare_multi_stream_input(data, connect_joint, streams='JBVM')
    print(f"✓ Multi-stream input: {data.shape} -> {multi_stream.shape}")
    
except Exception as e:
    print(f"✗ Preprocessing error: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Check real data availability
print("\n[6] Checking data availability...")
base_dir = Path(__file__).parent.parent
poses_root = base_dir / "runs" / "pose_batch"
annotations_path = base_dir / "annotationsv2.json"
objects_yaml = base_dir / "objects.yaml"

if poses_root.exists():
    cameras = list(poses_root.iterdir())
    print(f"✓ Found poses directory with {len(cameras)} cameras")
    for cam in cameras[:3]:
        videos = list(cam.iterdir())
        print(f"  {cam.name}: {len(videos)} videos")
else:
    print(f"⚠ Poses directory not found: {poses_root}")

if annotations_path.exists():
    import json
    with open(annotations_path) as f:
        annotations = json.load(f)
    print(f"✓ Found annotations with {len(annotations)} entries")
else:
    print(f"⚠ Annotations file not found: {annotations_path}")

if objects_yaml.exists():
    import yaml
    with open(objects_yaml) as f:
        objects = yaml.safe_load(f)
    print(f"✓ Found objects config with {len(objects)} cameras")
else:
    print(f"⚠ Objects YAML not found: {objects_yaml}")

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
print("\nTo train the model, run:")
print("  cd model && python train.py --lite --epochs 50 --batch_size 8")
