"""
Data Preprocessing for Playground Activity Recognition

Handles:
- YOLO pose loading and parsing
- Skeleton normalization (center at hip, scale by torso)
- Temporal sampling and padding
- Multi-person handling
- Object centroid integration
"""
import os
import json
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from .config import (
    NUM_KEYPOINTS, ACTIVITY_LABELS, CONNECT_JOINT, 
    CENTER_JOINT, ModelConfig
)


def parse_yolo_pose_line(line: str) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Parse a single YOLO pose detection line.
    
    YOLO format: class cx cy w h kp1_x kp1_y kp1_conf ... kp17_x kp17_y kp17_conf
    
    Returns:
        class_id: Detection class (0 = person)
        keypoints: (17, 2) array of normalized x,y coordinates
        confidences: (17,) array of keypoint confidences
    """
    parts = line.strip().split()
    if len(parts) < 5 + 17 * 3:
        return None, None, None
    
    class_id = int(parts[0])
    # Skip bbox: cx, cy, w, h
    kp_start = 5
    
    keypoints = np.zeros((NUM_KEYPOINTS, 2))
    confidences = np.zeros(NUM_KEYPOINTS)
    
    for i in range(NUM_KEYPOINTS):
        idx = kp_start + i * 3
        if idx + 2 < len(parts):
            keypoints[i, 0] = float(parts[idx])      # x
            keypoints[i, 1] = float(parts[idx + 1])  # y
            confidences[i] = float(parts[idx + 2])   # confidence
    
    return class_id, keypoints, confidences


def load_video_poses(labels_dir: str, max_frames: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all pose detections from a video's labels directory.
    
    Args:
        labels_dir: Path to directory containing frame-by-frame label files
        max_frames: Optional maximum number of frames to load
        
    Returns:
        poses: List of (num_persons, 17, 2) arrays per frame
        confidences: List of (num_persons, 17) arrays per frame
    """
    label_files = sorted(
        Path(labels_dir).glob("*.txt"),
        key=lambda x: int(x.stem.split('_')[-1])  # Sort by frame number
    )
    
    if max_frames:
        label_files = label_files[:max_frames]
    
    all_poses = []
    all_confs = []
    
    for label_file in label_files:
        frame_poses = []
        frame_confs = []
        
        with open(label_file, 'r') as f:
            for line in f:
                class_id, kps, confs = parse_yolo_pose_line(line)
                if kps is not None and class_id == 0:  # Only persons
                    frame_poses.append(kps)
                    frame_confs.append(confs)
        
        if frame_poses:
            all_poses.append(np.array(frame_poses))
            all_confs.append(np.array(frame_confs))
        else:
            # Empty frame
            all_poses.append(np.zeros((0, NUM_KEYPOINTS, 2)))
            all_confs.append(np.zeros((0, NUM_KEYPOINTS)))
    
    return all_poses, all_confs


def normalize_skeleton(keypoints: np.ndarray, confidences: np.ndarray,
                       center_joint: int = 0, scale_joints: Tuple[int, int] = (5, 6)) -> np.ndarray:
    """
    Normalize skeleton by centering and scaling.
    
    Args:
        keypoints: (T, M, V, C) or (M, V, C) array of keypoints
        confidences: Confidence values for filtering
        center_joint: Joint index to use as center (0=nose, or use hip midpoint)
        scale_joints: Joints to use for scale (shoulders or hips)
        
    Returns:
        Normalized keypoints with same shape
    """
    kps = keypoints.copy()
    
    # Handle different input shapes
    if kps.ndim == 3:
        kps = kps[np.newaxis, ...]  # Add time dimension
        squeeze = True
    else:
        squeeze = False
    
    T, M, V, C = kps.shape
    
    for t in range(T):
        for m in range(M):
            # Get center point (use hip midpoint: average of joints 11 and 12)
            left_hip = kps[t, m, 11, :]
            right_hip = kps[t, m, 12, :]
            
            # Check if hips are valid
            if np.any(left_hip > 0) and np.any(right_hip > 0):
                center = (left_hip + right_hip) / 2
            elif np.any(kps[t, m, center_joint, :] > 0):
                center = kps[t, m, center_joint, :]
            else:
                # Find any valid joint as fallback
                valid_mask = np.any(kps[t, m] > 0, axis=1)
                if np.any(valid_mask):
                    center = kps[t, m, valid_mask].mean(axis=0)
                else:
                    continue  # Skip empty detections
            
            # Center the skeleton
            valid_joints = np.any(kps[t, m] > 0, axis=1)
            kps[t, m, valid_joints] -= center
            
            # Scale by torso length (distance between shoulders)
            left_shoulder = kps[t, m, 5, :]
            right_shoulder = kps[t, m, 6, :]
            
            if np.any(left_shoulder != 0) and np.any(right_shoulder != 0):
                torso_length = np.linalg.norm(right_shoulder - left_shoulder)
                if torso_length > 0.01:  # Avoid division by very small numbers
                    kps[t, m, valid_joints] /= torso_length
    
    if squeeze:
        kps = kps[0]
    
    return kps


def temporal_sample(data: np.ndarray, target_length: int, mode: str = 'uniform') -> np.ndarray:
    """
    Sample or pad temporal sequence to target length.
    
    Args:
        data: (T, ...) array
        target_length: Desired temporal length
        mode: 'uniform' for uniform sampling, 'random' for random crop
        
    Returns:
        Sampled data of shape (target_length, ...)
    """
    T = data.shape[0]
    
    if T == target_length:
        return data
    elif T > target_length:
        # Uniform sampling
        if mode == 'uniform':
            indices = np.linspace(0, T - 1, target_length, dtype=int)
        else:
            # Random crop
            start = np.random.randint(0, T - target_length)
            indices = np.arange(start, start + target_length)
        return data[indices]
    else:
        # Pad with zeros or repeat last frame
        pad_shape = (target_length - T,) + data.shape[1:]
        padding = np.zeros(pad_shape, dtype=data.dtype)
        return np.concatenate([data, padding], axis=0)


def select_top_k_persons(poses: List[np.ndarray], k: int = 4) -> np.ndarray:
    """
    Select top-K persons across all frames based on visibility/confidence.
    Uses tracking consistency when possible.
    
    Args:
        poses: List of (M_t, V, C) arrays per frame
        k: Maximum number of persons to keep
        
    Returns:
        (T, K, V, C) array with consistent person ordering
    """
    T = len(poses)
    V = NUM_KEYPOINTS
    C = 2
    
    result = np.zeros((T, k, V, C))
    
    for t, frame_poses in enumerate(poses):
        if len(frame_poses) == 0:
            continue
        
        # Score each person by number of visible joints
        scores = []
        for m in range(len(frame_poses)):
            visible = np.sum(np.any(frame_poses[m] > 0, axis=1))
            # Also consider position (prefer persons closer to center)
            center_dist = np.abs(frame_poses[m].mean() - 0.5) if visible > 0 else 1.0
            scores.append(visible - center_dist * 0.1)
        
        # Select top-k persons
        top_indices = np.argsort(scores)[::-1][:k]
        
        for i, idx in enumerate(top_indices):
            result[t, i] = frame_poses[idx]
    
    return result


def load_object_centroids(objects_yaml: str, camera: str) -> np.ndarray:
    """
    Load object centroids for a specific camera from YAML config.
    
    Args:
        objects_yaml: Path to objects.yaml file
        camera: Camera name (e.g., 'columpios_cam4')
        
    Returns:
        (num_objects, 2) array of normalized centroids
    """
    with open(objects_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    # Try different camera name formats
    camera_key = camera.lower().replace('-', '_').replace(' ', '_')
    
    camera_config = None
    for key in config.keys():
        if key.lower().replace('-', '_') == camera_key:
            camera_config = config[key]
            break
    
    if camera_config is None:
        return np.zeros((0, 2))
    
    objects = camera_config.get('objects', [])
    if not objects:
        return np.zeros((0, 2))
    
    centroids = np.array([obj['centroid'] for obj in objects])
    return centroids


def create_panoramic_data(poses: np.ndarray, object_centroids: np.ndarray,
                          normalize: bool = True) -> np.ndarray:
    """
    Create panoramic graph data by combining person skeletons with object centroids.
    
    Args:
        poses: (T, M, V, C) array of person poses
        object_centroids: (num_objects, 2) array of object positions
        normalize: Whether to normalize skeletons
        
    Returns:
        (T, M, V', C) array where V' = V + num_objects
    """
    T, M, V, C = poses.shape
    num_objects = len(object_centroids)
    V_prime = V + num_objects
    
    # Initialize expanded data
    panoramic = np.zeros((T, M, V_prime, C))
    
    # Copy person poses
    panoramic[:, :, :V, :] = poses
    
    # Add object centroids (replicated for all persons and frames)
    if num_objects > 0:
        for t in range(T):
            for m in range(M):
                panoramic[t, m, V:, :] = object_centroids
    
    return panoramic


def compute_bone_features(joints: np.ndarray, connect_joint: np.ndarray) -> np.ndarray:
    """
    Compute bone vectors from joint positions.
    
    Args:
        joints: (C, T, V, M) array
        connect_joint: (V,) array of parent joint indices
        
    Returns:
        (C, T, V, M) array of bone vectors
    """
    C, T, V, M = joints.shape
    bones = np.zeros_like(joints)
    
    for v in range(V):
        parent = connect_joint[v] if v < len(connect_joint) else -1
        if parent >= 0 and parent < V:
            bones[:, :, v, :] = joints[:, :, v, :] - joints[:, :, parent, :]
    
    return bones


def compute_motion_features(data: np.ndarray) -> np.ndarray:
    """
    Compute temporal motion (velocity) features.
    
    Args:
        data: (C, T, V, M) array
        
    Returns:
        (C, T, V, M) array of motion vectors
    """
    C, T, V, M = data.shape
    motion = np.zeros_like(data)
    
    # First-order difference
    motion[:, :-1, :, :] = data[:, 1:, :, :] - data[:, :-1, :, :]
    
    return motion


def prepare_multi_stream_input(data: np.ndarray, connect_joint: np.ndarray,
                               streams: str = 'JBVM') -> np.ndarray:
    """
    Prepare multi-stream input features (Joint, Bone, Velocity, Motion).
    
    Args:
        data: (C, T, V, M) array of joint positions
        connect_joint: Parent joint connections
        streams: Which streams to include ('J', 'B', 'V', 'M' or combinations)
        
    Returns:
        (I, C, T, V, M) array where I = number of streams
    """
    C, T, V, M = data.shape
    
    # Compute all possible features
    joints = data  # (C, T, V, M)
    bones = compute_bone_features(joints, connect_joint)
    joint_motion = compute_motion_features(joints)
    bone_motion = compute_motion_features(bones)
    
    # Select requested streams
    stream_data = []
    for s in streams.upper():
        if s == 'J':
            stream_data.append(joints)
        elif s == 'B':
            stream_data.append(bones)
        elif s == 'V':
            stream_data.append(joint_motion)
        elif s == 'M':
            stream_data.append(bone_motion)
    
    if not stream_data:
        stream_data = [joints]
    
    return np.stack(stream_data, axis=0)


class PlaygroundDataProcessor:
    """
    Complete data processing pipeline for playground activity recognition.
    """
    
    def __init__(self, config: ModelConfig = None, objects_yaml: str = None):
        self.config = config or ModelConfig()
        self.objects_yaml = objects_yaml
        
    def process_video(self, labels_dir: str, camera: str = None) -> np.ndarray:
        """
        Process a single video's pose data into model-ready format.
        
        Args:
            labels_dir: Path to YOLO labels directory
            camera: Camera name for object lookup
            
        Returns:
            (I, C, T, V', M) array ready for model input
        """
        # Load poses
        poses_list, confs_list = load_video_poses(labels_dir)
        
        if not poses_list:
            return None
        
        # Select top-K persons
        poses = select_top_k_persons(poses_list, k=self.config.max_persons)
        
        # Normalize skeletons
        confs = np.ones((len(poses_list), self.config.max_persons, NUM_KEYPOINTS))
        poses = normalize_skeleton(poses, confs)
        
        # Load object centroids
        if self.objects_yaml and camera:
            objects = load_object_centroids(self.objects_yaml, camera)
        else:
            objects = np.zeros((0, 2))
        
        # Create panoramic data
        panoramic = create_panoramic_data(poses, objects, normalize=True)
        
        # Temporal sampling
        panoramic = temporal_sample(panoramic, self.config.sequence_length)
        
        # Convert to (C, T, V, M) format
        # Current: (T, M, V, C) -> (C, T, V, M)
        data = panoramic.transpose(3, 0, 2, 1)
        
        # Create extended connect_joint for objects
        V_prime = data.shape[2]
        connect_joint = np.array(CONNECT_JOINT)
        if V_prime > NUM_KEYPOINTS:
            # Objects connect to wrists (hands)
            obj_connections = np.full(V_prime - NUM_KEYPOINTS, 9)  # Left wrist
            connect_joint = np.concatenate([connect_joint, obj_connections])
        
        # Prepare multi-stream input (Joint, Bone, JointMotion, BoneMotion)
        multi_stream = prepare_multi_stream_input(data, connect_joint, streams='JBVM')
        
        return multi_stream


def parse_annotations(annotations_path: str) -> Dict:
    """
    Parse annotations JSON file to extract video labels.
    Excludes labels in EXCLUDED_LABELS (no_activity, Play_Object_Risk).
    
    Returns:
        Dictionary mapping video_id -> label
    """
    from .config import EXCLUDED_LABELS
    
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    video_labels = {}
    
    for item in data:
        video_data = item.get('data', {})
        video_id = video_data.get('video_id', '')
        camera = video_data.get('camera', '')
        video_name = video_data.get('video_name', '')
        
        # Get annotation
        annotations = item.get('annotations', [])
        if not annotations:
            continue
        
        # Get first annotation result
        result = annotations[0].get('result', [])
        if not result:
            continue
        
        # Extract choice
        for r in result:
            if r.get('type') == 'choices':
                choices = r.get('value', {}).get('choices', [])
                if choices:
                    label = choices[0]
                    # Skip excluded labels
                    if label in EXCLUDED_LABELS:
                        continue
                    if label in ACTIVITY_LABELS:
                        video_labels[video_name] = {
                            'label': label,
                            'label_id': ACTIVITY_LABELS[label],
                            'camera': camera,
                            'video_id': video_id
                        }
                    break
    
    return video_labels
