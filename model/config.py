"""
Configuration for Playground Activity Recognition Model
"""
import os

# Activity labels mapping (excluding no_activity and Play_Object_Risk)
ACTIVITY_LABELS = {
    'Social_People': 0,
    'Transit': 1,
    'Play_Object_Normal': 2,
    'Adult_Assisting': 3,
    'sliding': 4,
}

# Labels to exclude from training
EXCLUDED_LABELS = {'no_activity', 'Play_Object_Risk'}

LABEL_TO_ACTIVITY = {v: k for k, v in ACTIVITY_LABELS.items()}

NUM_CLASSES = len(ACTIVITY_LABELS)

# YOLO pose keypoints (COCO format - 17 keypoints)
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

NUM_KEYPOINTS = 17

# Skeleton connections for COCO format
SKELETON_EDGES = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 5), (12, 6),
    (9, 7), (7, 5), (10, 8), (8, 6), (5, 0), (6, 0),
    (1, 0), (3, 1), (2, 0), (4, 2), (5, 6), (11, 12)
]

# Body parts for attention mechanism
BODY_PARTS = [
    [5, 7, 9],      # left_arm
    [6, 8, 10],     # right_arm
    [11, 13, 15],   # left_leg
    [12, 14, 16],   # right_leg
    [0, 1, 2, 3, 4] # head
]

# Parent joint for each joint (for bone computation)
CONNECT_JOINT = [0, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 0, 0, 11, 12, 13, 14]

# Center joint index (typically hip or nose)
CENTER_JOINT = 0

# Model configurations
class ModelConfig:
    def __init__(self):
        # Data parameters
        self.num_keypoints = NUM_KEYPOINTS
        self.num_classes = NUM_CLASSES
        self.input_channels = 2  # x, y coordinates (normalized)
        self.max_persons = 6  # Maximum number of persons per frame
        self.sequence_length = 48  # Number of frames per sequence
        
        # Graph parameters
        self.max_hop = 2
        self.graph_labeling = 'spatial'  # 'spatial', 'distance', 'zeros'
        
        # Model architecture
        self.base_channels = 64
        self.num_stages = 3
        self.dropout = 0.3
        self.use_attention = True
        # Streams config: default to 4 streams (Joint, Bone, JointMotion, BoneMotion)
        self.streams = 'JBVM'  # supported letters: J,B,V,M (here V==JM, M==BM)
        self.num_streams = 4
        
        # Training parameters
        self.batch_size = 16
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.epochs = 100
        self.warmup_epochs = 5
        
        # Data augmentation
        self.augment_rotate = True
        self.augment_scale = True
        self.augment_flip = True
        
        # Regularization
        self.label_smoothing = 0.1
        self.mixup_alpha = 0.2

# Default config
DEFAULT_CONFIG = ModelConfig()
