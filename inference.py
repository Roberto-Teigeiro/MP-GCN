#!/usr/bin/env python3
"""
Inference script for Playground Activity Recognition

Usage:
    python inference.py <video_path.mp4> [--checkpoint <path>] [--objects <path>]

Example:
    python inference.py /path/to/video.mp4
    python inference.py video.mp4 --checkpoint checkpoints/best_model.pth
"""

import sys
import os
import argparse
from pathlib import Path

# Add model to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import cv2
from tqdm import tqdm

# Label mapping (from training)
LABEL_MAP = {
    0: 'Social_People',
    1: 'Transit',
    2: 'Play_Object_Normal',
    3: 'Adult_Assisting',
    4: 'sliding',
}


def load_model(checkpoint_path, device):
    """Load the trained model."""
    from model.networks import PlaygroundGCN
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    
    # Get num_nodes from checkpoint
    A_shape = ckpt['model_state_dict']['A'].shape
    num_nodes = A_shape[-1]  # Total nodes including objects
    num_objects = num_nodes - 17
    
    # The checkpoint is from PlaygroundGCN (multi-stream model)
    # Note: num_joints parameter actually means total nodes (joints + objects)
    model = PlaygroundGCN(
        num_classes=config['num_classes'],
        num_persons=config['max_persons'],
        num_joints=num_nodes,  # Total nodes = 25 (17 joints + 8 objects)
        num_objects=0,  # Already included in num_joints
        in_channels=config['input_channels'],
        base_channels=config['base_channels'],
        num_streams=config.get('num_streams', 2),  # num streams from checkpoint config
        dropout=config['dropout'],
        use_attention=config.get('use_attention', True)
    )
    
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Store additional config
    config['num_nodes'] = num_nodes
    config['num_objects'] = num_objects
    
    return model, config


def extract_poses_yolo(video_path, max_persons=6):
    """Extract poses from video using YOLO."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    
    # Load YOLO pose model
    yolo_path = Path(__file__).parent / "yolov8m-pose.pt"
    if not yolo_path.exists():
        # Try to download or use default
        print("Downloading YOLOv8 pose model...")
        model = YOLO('yolov8m-pose.pt')
    else:
        model = YOLO(str(yolo_path))
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  Video: {total_frames} frames, {fps:.1f} FPS, {width}x{height}")
    
    poses = []
    
    for _ in tqdm(range(total_frames), desc="  Extracting poses"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO
        results = model(frame, verbose=False)
        
        # Initialize frame poses
        frame_poses = np.zeros((max_persons, 17, 2), dtype=np.float32)
        
        if results[0].keypoints is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy()
            
            for i, kp in enumerate(keypoints[:max_persons]):
                # Normalize to [0, 1]
                frame_poses[i, :, 0] = kp[:, 0] / width
                frame_poses[i, :, 1] = kp[:, 1] / height
        
        poses.append(frame_poses)
    
    cap.release()
    return np.array(poses), fps


def normalize_skeleton(poses):
    """Normalize skeleton by centering at hip and scaling by torso."""
    T, M, V, C = poses.shape
    normalized = poses.copy()
    
    for t in range(T):
        for m in range(M):
            # Get hip center (average of joints 11, 12)
            left_hip = normalized[t, m, 11]
            right_hip = normalized[t, m, 12]
            
            if np.any(left_hip > 0) and np.any(right_hip > 0):
                center = (left_hip + right_hip) / 2
            elif np.any(normalized[t, m, 0] > 0):  # Use nose
                center = normalized[t, m, 0]
            else:
                continue
            
            # Center skeleton
            valid = np.any(normalized[t, m] > 0, axis=1)
            normalized[t, m, valid] -= center
            
            # Scale by shoulder width
            left_shoulder = normalized[t, m, 5]
            right_shoulder = normalized[t, m, 6]
            
            if np.any(left_shoulder != 0) and np.any(right_shoulder != 0):
                scale = np.linalg.norm(right_shoulder - left_shoulder)
                if scale > 0.01:
                    normalized[t, m, valid] /= scale
    
    return normalized


def add_objects(poses, objects, num_target_objects=8):
    """Add object centroids to pose data."""
    T, M, V, C = poses.shape
    V_new = V + num_target_objects
    
    result = np.zeros((T, M, V_new, C), dtype=poses.dtype)
    result[:, :, :V, :] = poses
    
    # Add objects (replicated for each person and frame)
    if objects is not None and len(objects) > 0:
        num_objects = min(len(objects), num_target_objects)
        for t in range(T):
            for m in range(M):
                result[t, m, V:V+num_objects, :] = objects[:num_objects]
    
    return result


def temporal_sample(data, target_length=32):
    """Sample or pad to target temporal length."""
    T = data.shape[0]
    
    if T == target_length:
        return data
    elif T > target_length:
        indices = np.linspace(0, T - 1, target_length, dtype=int)
        return data[indices]
    else:
        padding = np.zeros((target_length - T,) + data.shape[1:], dtype=data.dtype)
        return np.concatenate([data, padding], axis=0)


def prepare_input(poses, config, num_objects=8):
    """Prepare poses for model input (multi-stream format)."""
    from model.config import CONNECT_JOINT
    
    # Normalize
    poses = normalize_skeleton(poses)
    
    # Add objects (zeros if no objects available)
    poses = add_objects(poses, None, num_target_objects=num_objects)
    
    # Temporal sample
    poses = temporal_sample(poses, config['sequence_length'])
    
    # Adjust persons
    T, M, V, C = poses.shape
    target_M = config['max_persons']
    if M > target_M:
        poses = poses[:, :target_M]
    elif M < target_M:
        padding = np.zeros((T, target_M - M, V, C), dtype=poses.dtype)
        poses = np.concatenate([poses, padding], axis=1)
    
    # Convert to (C, T, V, M)
    data = poses.transpose(3, 0, 2, 1)
    
    # Compute bone stream
    connect_joint = np.array(CONNECT_JOINT)
    # Extend for objects
    if V > 17:
        obj_connections = np.full(V - 17, 9)
        connect_joint = np.concatenate([connect_joint, obj_connections])
    
    joints = data.copy()
    bones = np.zeros_like(joints)
    for v in range(V):
        parent = connect_joint[v] if v < len(connect_joint) else -1
        if 0 <= parent < V:
            bones[:, :, v, :] = joints[:, :, v, :] - joints[:, :, parent, :]
    
    # Compute joint motion (temporal first-order difference) -> JM
    joint_motion = np.zeros_like(joints)
    joint_motion[:, :-1, :, :] = joints[:, 1:, :, :] - joints[:, :-1, :, :]

    # Compute bone motion (BM)
    bone_motion = np.zeros_like(bones)
    bone_motion[:, :-1, :, :] = bones[:, 1:, :, :] - bones[:, :-1, :, :]

    # Stack streams: (I, C, T, V, M) where I=4 (J, B, JM, BM)
    multi_stream = np.stack([joints, bones, joint_motion, bone_motion], axis=0)
    
    # Add batch dimension: (1, I, C, T, V, M)
    return multi_stream[np.newaxis, ...]


def get_camera_objects(video_path, objects_yaml):
    """Get object centroids for camera based on video path."""
    if objects_yaml is None or not Path(objects_yaml).exists():
        return None
    
    import yaml
    with open(objects_yaml, 'r') as f:
        all_objects = yaml.safe_load(f)
    
    video_name = Path(video_path).stem.lower()
    
    # Map video name to camera
    camera_patterns = {
        'columpios_cam4': ['columpios_cam4', 'columpioscam4'],
        'columpiosCam1': ['columpioscam1'],
        'columpiosCam2': ['columpioscam2'],
        'columpiosCam3': ['columpioscam3'],
        'hundido_cam4': ['hundido_cam4', 'hundidocam4'],
        'hundido_cam5': ['hundido_cam5', 'hundidocam5'],
        'hundido_cam7': ['hundido_cam7', 'hundidocam7'],
    }
    
    for camera, patterns in camera_patterns.items():
        for pattern in patterns:
            if pattern in video_name:
                if camera in all_objects:
                    objs = all_objects[camera].get('objects', [])
                    if objs:
                        return np.array([obj['centroid'] for obj in objs])
    
    return None


def sliding_window_predict(model, poses, config, device, window_size=48, stride=24):
    """Run prediction with sliding windows."""
    T = poses.shape[0]
    num_objects = config.get('num_objects', 8)
    
    if T < window_size:
        # Single prediction for short videos
        input_data = prepare_input(poses, config, num_objects)
        input_tensor = torch.FloatTensor(input_data).to(device)
        
        with torch.no_grad():
            logits, _ = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
        
        return probs.cpu().numpy()[0]
    
    # Sliding window
    all_probs = []
    for start in range(0, T - window_size + 1, stride):
        window = poses[start:start + window_size]
        input_data = prepare_input(window, config, num_objects)
        input_tensor = torch.FloatTensor(input_data).to(device)
        
        with torch.no_grad():
            logits, _ = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
        
        all_probs.append(probs.cpu().numpy()[0])
    
    # Average predictions
    return np.mean(all_probs, axis=0)


def main():
    parser = argparse.ArgumentParser(description='Playground Activity Recognition Inference')
    parser.add_argument('video', type=str, help='Path to video file (.mp4)')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='Path to model checkpoint')
    parser.add_argument('--objects', type=str, default=None,
                        help='Path to objects.yaml')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    parser.add_argument('--window-size', type=int, default=48,
                        help='Sliding window size (frames)')
    parser.add_argument('--stride', type=int, default=24,
                        help='Sliding window stride')
    
    args = parser.parse_args()
    
    # Find paths
    base_dir = Path(__file__).parent
    
    if args.checkpoint is None:
        args.checkpoint = base_dir / "checkpoints" / "best_model.pth"
    
    if args.objects is None:
        args.objects = base_dir / "objects.yaml"
    
    # Check video exists
    if not Path(args.video).exists():
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("="*60)
    print("Playground Activity Recognition")
    print("="*60)
    print(f"Video: {args.video}")
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # Load model
    print("\nLoading model...")
    model, config = load_model(args.checkpoint, device)
    config['num_objects'] = config.get('num_objects', 8)
    print(f"  Classes: {list(LABEL_MAP.values())}")
    
    # Extract poses
    print("\nExtracting poses from video...")
    poses, fps = extract_poses_yolo(args.video, max_persons=config['max_persons'])
    
    if len(poses) == 0:
        print("ERROR: No poses extracted from video")
        sys.exit(1)
    
    # Get objects for this camera
    objects = get_camera_objects(args.video, args.objects)
    if objects is not None:
        print(f"  Found {len(objects)} objects for this camera")
    
    # Run inference
    print("\nRunning inference...")
    probs = sliding_window_predict(
        model, poses, config, device,
        window_size=args.window_size,
        stride=args.stride
    )
    
    # Get prediction
    pred_idx = np.argmax(probs)
    pred_label = LABEL_MAP[pred_idx]
    confidence = probs[pred_idx] * 100
    
    # Print results
    print("\n" + "="*60)
    print("PREDICTION")
    print("="*60)
    print(f"\n  🎯 Activity: {pred_label}")
    print(f"  📊 Confidence: {confidence:.1f}%")
    print("\n  All probabilities:")
    for idx, prob in enumerate(probs):
        label = LABEL_MAP[idx]
        bar = "█" * int(prob * 30)
        marker = " ◀" if idx == pred_idx else ""
        print(f"    {label:20s} {prob*100:5.1f}% {bar}{marker}")
    
    print("="*60)
    
    return pred_label, confidence


if __name__ == "__main__":
    main()
