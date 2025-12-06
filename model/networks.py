"""
Playground-GCN: Multi-Person Graph Convolutional Network for Activity Recognition
Based on MP-GCN architecture adapted for playground activity recognition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from .blocks import STGCNBlock, SpatialGraphConv, MultiScaleTemporalConv


class PlaygroundGCN(nn.Module):
    """
    Multi-stream Graph Convolutional Network for playground activity recognition.
    
    Architecture:
    - Separate streams for different inputs (J, B, JM, BM)
    - Per-stream ST-GCN processing (like lite model)
    - Late fusion of stream features
    - Global pooling and classification head
    
    This uses per-person skeleton processing (not panoramic), 
    matching the data format of our preprocessing pipeline.
    """
    
    def __init__(self,
                 num_classes: int,
                 num_persons: int = 4,
                 num_joints: int = 17,
                 num_objects: int = 0,
                 in_channels: int = 2,
                 base_channels: int = 64,
                 num_streams: int = 2,  # J, B by default
                 A: Optional[torch.Tensor] = None,
                 dropout: float = 0.3,
                 use_attention: bool = True,
                 parts: Optional[List[List[int]]] = None):
        """
        Args:
            num_classes: Number of activity classes
            num_persons: Maximum number of persons
            num_joints: Joints per person (including objects)
            num_objects: Number of object nodes (already in num_joints)
            in_channels: Input channels (2 for x,y coords)
            base_channels: Base number of channels
            num_streams: Number of input streams (1-4)
            A: Adjacency matrix tensor (for per-person skeleton)
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
            parts: Body part groupings for attention
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.num_persons = num_persons
        self.num_joints = num_joints  # Actually num_nodes = joints + objects
        self.num_objects = num_objects
        self.num_nodes = num_joints  # Per-person nodes
        self.num_streams = num_streams
        
        # Create per-person adjacency if not provided
        if A is None or A.size(-1) != self.num_nodes:
            A = self._create_per_person_adjacency()
        self.register_buffer('A', A)
        
        # Per-stream backbone (each stream gets its own ST-GCN stack)
        self.stream_encoders = nn.ModuleList()
        for _ in range(num_streams):
            encoder = nn.ModuleList([
                nn.BatchNorm2d(in_channels),
                STGCNBlock(in_channels, base_channels, self.A, use_attention=use_attention),
                STGCNBlock(base_channels, base_channels, self.A, stride=2, use_attention=use_attention),
                STGCNBlock(base_channels, base_channels * 2, self.A, use_attention=use_attention),
                STGCNBlock(base_channels * 2, base_channels * 2, self.A, stride=2, use_attention=use_attention),
            ])
            self.stream_encoders.append(encoder)
        
        # Fusion layer to combine stream features
        fused_channels = base_channels * 2 * num_streams
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_channels, base_channels * 4, 1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Final backbone after fusion
        self.backbone_final = nn.ModuleList([
            STGCNBlock(base_channels * 4, base_channels * 4, self.A, use_attention=use_attention),
            STGCNBlock(base_channels * 4, base_channels * 4, self.A, stride=2, use_attention=use_attention),
        ])
        
        # Classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 4, num_classes)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _create_per_person_adjacency(self) -> torch.Tensor:
        """Create adjacency matrix for per-person skeleton with objects."""
        from .graphs import Graph
        
        V = self.num_nodes
        num_objects = V - 17 if V > 17 else 0
        
        # Create basic skeleton graph
        graph = Graph(max_hop=2)
        A_np = graph.get_adjacency()  # NumPy array
        A = torch.from_numpy(A_np).float()
        
        if num_objects > 0:
            # Expand for objects
            num_scales = A.size(0)
            A_expanded = torch.zeros(num_scales, V, V)
            A_expanded[:, :17, :17] = A[:, :17, :17]
            
            # Self-loops for objects
            for i in range(17, V):
                A_expanded[:, i, i] = 1
            
            # Connect objects to hands (indices 9, 10)
            for obj_idx in range(17, V):
                A_expanded[1, 9, obj_idx] = 1
                A_expanded[1, obj_idx, 9] = 1
                A_expanded[1, 10, obj_idx] = 1
                A_expanded[1, obj_idx, 10] = 1
            
            A = A_expanded
        
        # Normalize
        for k in range(A.size(0)):
            d = A[k].sum(1)
            d[d == 0] = 1
            A[k] = A[k] / d.unsqueeze(1)
        
        return A
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (N, I, C, T, V, M)
               N: batch size
               I: number of streams
               C: input channels
               T: temporal length
               V: number of nodes (joints + objects per person)
               M: number of persons
               
        Returns:
            logits: Classification logits (N, num_classes)
            features: Feature representation (N, C', T', V)
        """
        N, I, C, T, V, M = x.size()
        
        # Process each stream separately
        stream_features = []
        for i in range(min(I, self.num_streams)):
            stream_x = x[:, i]  # (N, C, T, V, M)
            # Reshape for per-person processing: (N*M, C, T, V)
            stream_x = stream_x.permute(0, 4, 1, 2, 3).contiguous()  # (N, M, C, T, V)
            stream_x = stream_x.view(N * M, C, T, V)
            
            # Apply stream encoder
            for layer in self.stream_encoders[i]:
                if isinstance(layer, nn.BatchNorm2d):
                    stream_x = layer(stream_x)
                else:
                    stream_x = layer(stream_x)
            
            stream_features.append(stream_x)
        
        # Concatenate stream features
        x = torch.cat(stream_features, dim=1)  # (N*M, C'*I, T', V)
        
        # Fusion
        x = self.fusion(x)
        
        # Final backbone
        for layer in self.backbone_final:
            x = layer(x)
        
        # Extract features before pooling
        _, C_out, T_out, V_out = x.size()
        
        # Pool over nodes and time
        x = self.global_pool(x)  # (N*M, C_out, 1, 1)
        x = x.view(N, M, -1)  # (N, M, C_out)
        x = x.mean(dim=1)  # Average over persons: (N, C_out)
        
        # Features for visualization (before final pooling)
        features = x.view(N, -1)
        
        # Classification
        x = self.dropout(x)
        logits = self.fc(x)
        
        return logits, features
    
    def get_attention_maps(self, x: torch.Tensor) -> dict:
        """Get attention maps for visualization."""
        # TODO: Implement attention visualization
        pass


class PlaygroundGCNLite(nn.Module):
    """
    Lightweight version of PlaygroundGCN for faster training/inference.
    Uses per-person processing with simple adjacency.
    """
    
    def __init__(self,
                 num_classes: int,
                 num_persons: int = 4,
                 num_joints: int = 17,
                 num_objects: int = 0,
                 in_channels: int = 2,
                 base_channels: int = 32,
                 A: Optional[torch.Tensor] = None,
                 dropout: float = 0.3):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_persons = num_persons
        self.num_joints = num_joints
        self.num_objects = num_objects
        self.num_nodes = num_joints + num_objects  # Per-person nodes
        
        # Create per-person adjacency (not multi-person panoramic)
        if A is None or A.size(-1) != self.num_nodes:
            A = self._create_per_person_adjacency()
        self.register_buffer('A', A)
        
        # Simplified architecture
        self.bn_input = nn.BatchNorm2d(in_channels)
        
        self.layers = nn.ModuleList([
            STGCNBlock(in_channels, base_channels, self.A, use_attention=False),
            STGCNBlock(base_channels, base_channels, self.A, stride=2, use_attention=False),
            STGCNBlock(base_channels, base_channels * 2, self.A, use_attention=False),
            STGCNBlock(base_channels * 2, base_channels * 2, self.A, stride=2, use_attention=False),
            STGCNBlock(base_channels * 2, base_channels * 4, self.A, use_attention=False),
        ])
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_channels * 4, num_classes)
        
    def _create_per_person_adjacency(self) -> torch.Tensor:
        """Create adjacency matrix for a single person."""
        from .config import SKELETON_EDGES
        
        V = self.num_nodes
        
        # Build adjacency from skeleton edges
        adj = torch.zeros(V, V)
        for (i, j) in SKELETON_EDGES:
            if i < V and j < V:
                adj[i, j] = 1
                adj[j, i] = 1
        
        # Add self-loops
        adj = adj + torch.eye(V)
        
        # Normalize
        degree = adj.sum(dim=0, keepdim=True)
        degree = torch.clamp(degree, min=1)
        adj = adj / degree
        
        # Stack for multiple scales (3 scales)
        A = adj.unsqueeze(0).repeat(3, 1, 1)
        
        return A
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (N, C, T, V, M) tensor
            
        Returns:
            logits, features
        """
        N, C, T, V, M = x.size()
        
        # Merge person dimension into batch
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (N, M, C, T, V)
        x = x.view(N * M, C, T, V)
        
        x = self.bn_input(x)
        
        for layer in self.layers:
            x = layer(x)
        
        # Features
        _, C_out, T_out, V_out = x.size()
        features = x.view(N, M, C_out, T_out, V_out).mean(dim=1)
        
        # Pool and classify
        x = self.global_pool(x)
        x = x.view(N, M, -1).mean(dim=1)  # Average over persons
        x = self.dropout(x)
        logits = self.fc(x)
        
        return logits, features


def create_model(config, graph=None) -> nn.Module:
    """Factory function to create model from config."""
    from .config import ModelConfig
    
    if isinstance(config, dict):
        cfg = ModelConfig()
        for k, v in config.items():
            setattr(cfg, k, v)
    else:
        cfg = config
    
    # Get adjacency from graph
    A = None
    parts = None
    if graph is not None:
        A = torch.FloatTensor(graph.A)
        parts = graph.parts if hasattr(graph, 'parts') else None
    
    model = PlaygroundGCN(
        num_classes=cfg.num_classes,
        num_persons=cfg.max_persons,
        num_joints=cfg.num_keypoints,
        in_channels=cfg.input_channels,
        base_channels=cfg.base_channels,
        dropout=cfg.dropout,
        use_attention=cfg.use_attention,
        num_streams=getattr(cfg, 'num_streams', 2),
        A=A,
        parts=parts
    )
    
    return model
