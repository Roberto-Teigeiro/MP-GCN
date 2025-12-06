"""
Playground Activity Recognition Model

A skeleton-based action recognition system using Graph Convolutional Networks,
inspired by MP-GCN (Multi-Person Graph Convolutional Network).

Features:
- Multi-stream input (Joint, Bone, Velocity, Motion)
- Panoramic graph with person-object and person-person interactions  
- Spatio-temporal attention mechanism
- Label smoothing and mixup regularization
"""

from .config import (
    ACTIVITY_LABELS, 
    LABEL_TO_ACTIVITY,
    NUM_CLASSES,
    NUM_KEYPOINTS,
    SKELETON_EDGES,
    BODY_PARTS,
    EXCLUDED_LABELS,
    ModelConfig,
    DEFAULT_CONFIG
)

from .graphs import Graph, create_graph
from .panoramic_graph import PanoramicGraph, create_panoramic_graph
from .networks import PlaygroundGCN, PlaygroundGCNLite, create_model
from .dataset import PlaygroundDataset, create_data_loaders
from .trainer import Trainer, train_model, evaluate_model

__all__ = [
    # Config
    'ACTIVITY_LABELS',
    'LABEL_TO_ACTIVITY', 
    'NUM_CLASSES',
    'NUM_KEYPOINTS',
    'SKELETON_EDGES',
    'BODY_PARTS',
    'EXCLUDED_LABELS',
    'ModelConfig',
    'DEFAULT_CONFIG',
    
    # Graph
    'Graph',
    'create_graph',
    'PanoramicGraph',
    'create_panoramic_graph',
    
    # Model
    'PlaygroundGCN',
    'PlaygroundGCNLite',
    'create_model',
    
    # Data
    'PlaygroundDataset',
    'create_data_loaders',
    
    # Training
    'Trainer',
    'train_model',
    'evaluate_model',
]

__version__ = '0.1.0'
