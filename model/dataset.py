"""
PyTorch Dataset for Playground Activity Recognition
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml

from .config import (
    ACTIVITY_LABELS, NUM_KEYPOINTS, CONNECT_JOINT,
    ModelConfig, DEFAULT_CONFIG
)
from .preprocessing import (
    load_video_poses, normalize_skeleton, temporal_sample,
    select_top_k_persons, load_object_centroids, create_panoramic_data,
    prepare_multi_stream_input, parse_annotations
)


class PlaygroundDataset(Dataset):
    """
    Dataset for playground activity recognition from YOLO poses.
    """
    
    def __init__(self,
                 poses_root: str,
                 annotations_path: str,
                 objects_yaml: str,
                 split: str = 'train',
                 config: ModelConfig = None,
                 transform=None,
                 streams: str = 'JBVM',
                 max_objects: int = 8):  # Max objects to pad to
        """
        Args:
            poses_root: Root directory containing pose data (runs/pose_batch)
            annotations_path: Path to annotations JSON
            objects_yaml: Path to objects.yaml
            split: 'train' or 'val'
            config: Model configuration
            transform: Optional data augmentation
            streams: Which streams to use ('J', 'B', 'JB', 'JBVM')
            max_objects: Maximum number of objects (for padding)
        """
        self.poses_root = Path(poses_root)
        self.objects_yaml = objects_yaml
        self.split = split
        self.config = config or DEFAULT_CONFIG
        self.transform = transform
        self.streams = streams
        self.max_objects = max_objects
        
        # Fixed number of nodes for consistent tensor sizes
        self.num_nodes = NUM_KEYPOINTS + max_objects
        
        # Parse annotations
        self.video_labels = parse_annotations(annotations_path)
        
        # Find available videos
        self.samples = self._find_samples()
        
        # Split data
        self._split_data()
        
        # Load object centroids for each camera
        self.camera_objects = self._load_all_objects()
        
    def _find_samples(self) -> List[Dict]:
        """Find all videos that have both poses and labels."""
        samples = []
        
        # Iterate through camera folders
        for camera_dir in self.poses_root.iterdir():
            if not camera_dir.is_dir():
                continue
            
            camera_name = camera_dir.name
            
            # Iterate through video folders
            for video_dir in camera_dir.iterdir():
                if not video_dir.is_dir():
                    continue
                
                labels_dir = video_dir / 'labels'
                if not labels_dir.exists():
                    continue
                
                # Extract video name from folder name
                # Format: camera_name-YYYY-MM-DD_HH-MM-SS
                video_name = video_dir.name
                
                # Try to match with annotations
                # Convert folder name to video_name format used in annotations
                possible_names = [
                    video_name,
                    video_name.replace('_', ' ').replace('-', ' '),
                    video_name.replace('_', '-'),
                ]
                
                label_info = None
                for name in possible_names:
                    if name in self.video_labels:
                        label_info = self.video_labels[name]
                        break
                
                # Also try matching by camera and partial name
                if label_info is None:
                    for vname, info in self.video_labels.items():
                        if camera_name.lower() in info.get('camera', '').lower():
                            # Check if timestamps match
                            if any(part in vname for part in video_name.split('-')[1:3]):
                                label_info = info
                                break
                
                if label_info is not None:
                    samples.append({
                        'video_name': video_name,
                        'camera': camera_name,
                        'labels_dir': str(labels_dir),
                        'label': label_info['label'],
                        'label_id': label_info['label_id']
                    })
        
        return samples
    
    def _split_data(self, train_ratio: float = 0.8):
        """Split samples into train/val sets."""
        np.random.seed(42)  # Reproducibility
        indices = np.random.permutation(len(self.samples))
        
        split_idx = int(len(indices) * train_ratio)
        
        if self.split == 'train':
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]
    
    def _load_all_objects(self) -> Dict[str, np.ndarray]:
        """Load object centroids for all cameras."""
        objects = {}
        
        if not os.path.exists(self.objects_yaml):
            return objects
        
        with open(self.objects_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        for camera, cam_config in config.items():
            if 'objects' in cam_config:
                centroids = np.array([obj['centroid'] for obj in cam_config['objects']])
                objects[camera.lower()] = centroids
        
        return objects
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        sample_idx = self.indices[idx]
        sample = self.samples[sample_idx]
        
        # Load poses
        poses_list, confs_list = load_video_poses(sample['labels_dir'])
        
        if not poses_list or len(poses_list) < 5:
            # Return dummy data for invalid samples
            return self._get_dummy_sample(sample['label_id'], sample['video_name'])
        
        # Select top-K persons
        poses = select_top_k_persons(poses_list, k=self.config.max_persons)
        
        # Normalize skeletons (center at hip, scale by torso)
        poses = normalize_skeleton(poses, None)
        
        # Get object centroids for this camera
        camera_key = sample['camera'].lower().replace('-', '_')
        objects = self.camera_objects.get(camera_key, np.zeros((0, 2)))
        
        # Pad objects to max_objects
        if len(objects) < self.max_objects:
            padding = np.zeros((self.max_objects - len(objects), 2))
            objects = np.vstack([objects, padding]) if len(objects) > 0 else padding
        else:
            objects = objects[:self.max_objects]
        
        # Create panoramic data (add objects to skeleton)
        panoramic = create_panoramic_data(poses, objects)
        
        # Temporal sampling
        panoramic = temporal_sample(panoramic, self.config.sequence_length)
        
        # Convert to (C, T, V, M) format: (T, M, V, C) -> (C, T, V, M)
        data = panoramic.transpose(3, 0, 2, 1)
        
        # Data augmentation
        if self.transform is not None and self.split == 'train':
            data = self.transform(data)
        
        # Build extended connect_joint
        V = data.shape[2]
        connect_joint = np.array(CONNECT_JOINT)
        if V > NUM_KEYPOINTS:
            obj_connections = np.full(V - NUM_KEYPOINTS, 9)  # Connect to wrist
            connect_joint = np.concatenate([connect_joint, obj_connections])
        
        # Prepare multi-stream input
        multi_stream = prepare_multi_stream_input(data, connect_joint, streams=self.streams)
        
        # Convert to tensor
        tensor = torch.FloatTensor(multi_stream)
        label = sample['label_id']
        name = sample['video_name']
        
        return tensor, label, name
    
    def _get_dummy_sample(self, label: int, name: str) -> Tuple[torch.Tensor, int, str]:
        """Return dummy sample for invalid/missing data."""
        num_streams = len(self.streams) if self.streams.isupper() else 1
        dummy = torch.zeros(
            num_streams,
            self.config.input_channels,
            self.config.sequence_length,
            self.num_nodes,  # Use fixed num_nodes
            self.config.max_persons
        )
        return dummy, label, name
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced data."""
        labels = [self.samples[i]['label_id'] for i in self.indices]
        class_counts = np.bincount(labels, minlength=len(ACTIVITY_LABELS))
        
        # Inverse frequency weighting
        weights = 1.0 / (class_counts + 1)
        weights = weights / weights.sum() * len(ACTIVITY_LABELS)
        
        return torch.FloatTensor(weights)


class SkeletonAugmentation:
    """Data augmentation for skeleton sequences."""
    
    def __init__(self, rotate: bool = True, scale: bool = True, 
                 flip: bool = True, noise: bool = True):
        self.rotate = rotate
        self.scale = scale
        self.flip = flip
        self.noise = noise
    
    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Args:
            data: (C, T, V, M) skeleton data
        """
        C, T, V, M = data.shape
        
        # Random rotation (small angles)
        if self.rotate and np.random.rand() > 0.5:
            angle = np.random.uniform(-15, 15) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            for t in range(T):
                for m in range(M):
                    coords = data[:2, t, :, m]  # (2, V)
                    data[:2, t, :, m] = rot_matrix @ coords
        
        # Random scale
        if self.scale and np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            data[:2] *= scale
        
        # Random horizontal flip
        if self.flip and np.random.rand() > 0.5:
            data[0] = -data[0]  # Flip x coordinates
            # Swap left/right joints
            left_joints = [1, 3, 5, 7, 9, 11, 13, 15]
            right_joints = [2, 4, 6, 8, 10, 12, 14, 16]
            for l, r in zip(left_joints, right_joints):
                if l < V and r < V:
                    data[:, :, [l, r], :] = data[:, :, [r, l], :]
        
        # Random noise
        if self.noise and np.random.rand() > 0.5:
            noise = np.random.randn(*data.shape) * 0.01
            data = data + noise
        
        return data


def create_data_loaders(
    poses_root: str,
    annotations_path: str,
    objects_yaml: str,
    config: ModelConfig = None,
    batch_size: int = 16,
    num_workers: int = 4,
    streams: str = 'JBVM'
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders."""
    
    config = config or DEFAULT_CONFIG
    
    # Augmentation for training
    train_transform = SkeletonAugmentation(
        rotate=config.augment_rotate,
        scale=config.augment_scale,
        flip=config.augment_flip
    )
    
    # Create datasets
    train_dataset = PlaygroundDataset(
        poses_root=poses_root,
        annotations_path=annotations_path,
        objects_yaml=objects_yaml,
        split='train',
        config=config,
        transform=train_transform,
        streams=streams
    )
    
    val_dataset = PlaygroundDataset(
        poses_root=poses_root,
        annotations_path=annotations_path,
        objects_yaml=objects_yaml,
        split='val',
        config=config,
        transform=None,
        streams=streams
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
