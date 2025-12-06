#!/usr/bin/env python3
"""
Main training script for Playground Activity Recognition

Usage:
    python train.py --poses_root runs/pose_batch --annotations annotationsv2.json --objects objects.yaml
    
    # With custom config
    python train.py --epochs 50 --batch_size 8 --lr 0.0005 --lite
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from model import (
    train_model, 
    ModelConfig, 
    create_panoramic_graph,
    ACTIVITY_LABELS
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train Playground Activity Recognition Model')
    
    # Data paths
    parser.add_argument('--poses_root', type=str, default='runs/pose_batch',
                        help='Root directory containing pose data')
    parser.add_argument('--annotations', type=str, default='annotationsv2.json',
                        help='Path to annotations JSON file')
    parser.add_argument('--objects', type=str, default='objects.yaml',
                        help='Path to objects YAML file')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    
    # Model configuration
    parser.add_argument('--lite', action='store_true',
                        help='Use lightweight model')
    parser.add_argument('--max_persons', type=int, default=6,
                        help='Maximum number of persons per frame')
    parser.add_argument('--seq_length', type=int, default=48,
                        help='Sequence length (number of frames)')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='Base number of channels')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--use_attention', action='store_true', default=True,
                        help='Use attention mechanism')
    
    # Training configuration
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    
    # Regularization
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor')
    parser.add_argument('--mixup_alpha', type=float, default=0.2,
                        help='Mixup alpha (0 to disable)')
    
    # Augmentation
    parser.add_argument('--no_augment', action='store_true',
                        help='Disable data augmentation')
    
    # Other
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create config
    config = ModelConfig()
    config.max_persons = args.max_persons
    config.sequence_length = args.seq_length
    config.base_channels = args.base_channels
    config.dropout = args.dropout
    config.use_attention = args.use_attention
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.weight_decay = args.weight_decay
    config.warmup_epochs = args.warmup_epochs
    config.label_smoothing = args.label_smoothing
    config.mixup_alpha = args.mixup_alpha
    
    if args.no_augment:
        config.augment_rotate = False
        config.augment_scale = False
        config.augment_flip = False
    
    # Resolve paths
    base_dir = Path(__file__).parent.parent
    poses_root = base_dir / args.poses_root
    annotations_path = base_dir / args.annotations
    objects_yaml = base_dir / args.objects
    save_dir = base_dir / args.save_dir
    
    # Check paths exist
    if not poses_root.exists():
        print(f"Error: Poses directory not found: {poses_root}")
        sys.exit(1)
    if not annotations_path.exists():
        print(f"Error: Annotations file not found: {annotations_path}")
        sys.exit(1)
    
    # Print configuration
    print("=" * 60)
    print("Playground Activity Recognition Training")
    print("=" * 60)
    print(f"Poses root: {poses_root}")
    print(f"Annotations: {annotations_path}")
    print(f"Objects YAML: {objects_yaml}")
    print(f"Save directory: {save_dir}")
    print(f"Device: {args.device}")
    print(f"Model: {'Lite' if args.lite else 'Full'}")
    print(f"Classes: {list(ACTIVITY_LABELS.keys())}")
    print("-" * 60)
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Max persons: {config.max_persons}")
    print(f"Sequence length: {config.sequence_length}")
    print(f"Label smoothing: {config.label_smoothing}")
    print(f"Mixup alpha: {config.mixup_alpha}")
    print("=" * 60)
    
    # Count objects from YAML
    num_objects = 0
    if objects_yaml.exists():
        import yaml
        with open(objects_yaml) as f:
            obj_config = yaml.safe_load(f)
        # Count max objects across all cameras
        for cam_config in obj_config.values():
            if 'objects' in cam_config:
                num_objects = max(num_objects, len(cam_config['objects']))
        print(f"Max objects per camera: {num_objects}")
    
    # Train
    trainer = train_model(
        poses_root=str(poses_root),
        annotations_path=str(annotations_path),
        objects_yaml=str(objects_yaml),
        save_dir=str(save_dir),
        config=config,
        num_objects=num_objects,
        lite=args.lite
    )
    
    print("Training completed!")
    print(f"Best accuracy: {trainer.best_acc:.4f}")
    print(f"Best F1 score: {trainer.best_f1:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


if __name__ == '__main__':
    main()
