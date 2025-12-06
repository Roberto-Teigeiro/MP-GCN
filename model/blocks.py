"""
GCN Building Blocks for Skeleton-based Action Recognition
Based on MP-GCN architecture with modifications for playground activities
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class SpatialGraphConv(nn.Module):
    """
    Spatial Graph Convolution layer.
    Applies graph convolution with multi-scale adjacency matrices.
    """
    
    def __init__(self, in_channels: int, out_channels: int, num_scales: int = 3):
        """
        Args:
            in_channels: Input feature channels
            out_channels: Output feature channels  
            num_scales: Number of adjacency matrix scales (hop distances)
        """
        super().__init__()
        self.num_scales = num_scales
        
        # Separate convolution for each scale
        self.conv = nn.Conv2d(in_channels, out_channels * num_scales, kernel_size=1)
        
    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (N, C, T, V)
            A: Adjacency matrix (num_scales, V, V)
            
        Returns:
            Output features (N, C_out, T, V)
        """
        N, C, T, V = x.size()
        
        # Apply convolution
        x = self.conv(x)  # (N, C_out * num_scales, T, V)
        
        # Reshape for graph multiplication
        n, kc, t, v = x.size()
        x = x.view(n, self.num_scales, kc // self.num_scales, t, v)
        
        # Apply graph convolution: sum over scales
        # x: (N, K, C, T, V), A: (K, V, V)
        x = torch.einsum('nkctv,kvw->nctw', x, A[:self.num_scales]).contiguous()
        
        return x


class TemporalConv(nn.Module):
    """
    Temporal convolution with configurable kernel size and dilation.
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 9, stride: int = 1, dilation: int = 1):
        super().__init__()
        
        padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0),
            dilation=(dilation, 1)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class MultiScaleTemporalConv(nn.Module):
    """
    Multi-scale temporal convolution with multiple branches.
    Inspired by MS-TCN design in MP-GCN.
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 9, stride: int = 1,
                 dilations: List[int] = [1, 2]):
        super().__init__()
        
        self.num_branches = len(dilations) + 2  # + maxpool + 1x1
        
        # Ensure divisibility
        assert out_channels % self.num_branches == 0, \
            f"out_channels ({out_channels}) must be divisible by num_branches ({self.num_branches})"
        
        branch_channels = out_channels // self.num_branches
        
        # Temporal conv branches with different dilations
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(branch_channels, branch_channels, 
                            kernel_size=kernel_size, stride=stride, dilation=dilation),
            ))
        
        # Max pooling branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))
        
        # 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))
        
        # Residual connection
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        
        branch_outputs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outputs, dim=1)
        
        out = self.relu(out + res)
        return out


class SpatioTemporalAttention(nn.Module):
    """
    Spatio-temporal attention module for person-level features.
    Applies attention over both spatial (joints) and temporal dimensions.
    """
    
    def __init__(self, channels: int, parts: List[List[int]], reduction: int = 4):
        """
        Args:
            channels: Number of input channels
            parts: List of body parts (joint indices)
            reduction: Channel reduction ratio for attention
        """
        super().__init__()
        
        self.parts = parts
        self.num_parts = len(parts)
        
        # Register part indices
        num_joints = sum(len(p) for p in parts)
        joints_to_parts = torch.zeros(num_joints, dtype=torch.long)
        for p_idx, part in enumerate(parts):
            for j in part:
                if j < num_joints:
                    joints_to_parts[j] = p_idx
        self.register_buffer('joints_to_parts', joints_to_parts)
        
        # Attention layers
        inner_channels = channels // reduction
        self.fcn = nn.Sequential(
            nn.Conv2d(channels, inner_channels, kernel_size=1),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_t = nn.Conv2d(inner_channels, channels, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channels, channels, kernel_size=1)
        
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V = x.size()
        res = x
        
        # Temporal attention
        x_t = x.mean(dim=3, keepdim=True)  # (N, C, T, 1)
        
        # Spatial (part-level) attention
        x_v = x.mean(dim=2, keepdim=True)  # (N, C, 1, V)
        
        # Combine for attention
        x_att = self.fcn(x_t)
        x_t_att = self.conv_t(x_att).sigmoid()  # (N, C, T, 1)
        
        x_v_att = self.fcn(x_v.transpose(2, 3))  # (N, C, V, 1)
        x_v_att = self.conv_v(x_v_att).sigmoid().transpose(2, 3)  # (N, C, 1, V)
        
        # Apply attention
        x_att = x_t_att * x_v_att
        out = x * x_att
        
        return self.relu(self.bn(out) + res)


class STGCNBlock(nn.Module):
    """
    Spatio-Temporal GCN Block.
    Combines spatial graph conv, temporal conv, and optional attention.
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 A: torch.Tensor,
                 stride: int = 1,
                 kernel_size: int = 9,
                 use_attention: bool = True,
                 dropout: float = 0.0,
                 residual: bool = True,
                 parts: Optional[List[List[int]]] = None):
        super().__init__()
        
        num_scales = A.size(0)
        
        # Spatial convolution
        self.sgcn = SpatialGraphConv(in_channels, out_channels, num_scales)
        self.bn_s = nn.BatchNorm2d(out_channels)
        
        # Temporal convolution
        self.tgcn = MultiScaleTemporalConv(out_channels, out_channels, 
                                           kernel_size=kernel_size, stride=stride)
        
        # Attention
        if use_attention and parts is not None:
            self.attention = SpatioTemporalAttention(out_channels, parts)
        else:
            self.attention = None
        
        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        
        # Learnable edge importance
        self.register_buffer('A', A)
        self.edge_importance = nn.Parameter(torch.ones_like(A))
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        
        # Spatial
        x = self.sgcn(x, self.A * self.edge_importance)
        x = self.bn_s(x)
        x = self.relu(x)
        
        # Temporal
        x = self.tgcn(x)
        x = self.dropout(x)
        
        # Residual
        x = x + res
        x = self.relu(x)
        
        # Attention
        if self.attention is not None:
            x = self.attention(x)
        
        return x


class InputBranch(nn.Module):
    """
    Input branch for processing a single stream (J, B, V, or M).
    """
    
    def __init__(self, in_channels: int, base_channels: int, A: torch.Tensor,
                 use_attention: bool = True, parts: Optional[List[List[int]]] = None):
        super().__init__()
        
        self.bn = nn.BatchNorm2d(in_channels)
        
        self.layers = nn.ModuleList([
            STGCNBlock(in_channels, base_channels, A, use_attention=use_attention, parts=parts),
            STGCNBlock(base_channels, base_channels, A, use_attention=use_attention, parts=parts),
            STGCNBlock(base_channels, base_channels // 2, A, use_attention=False, parts=parts),
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, V, M = x.size()
        
        # Merge person dimension into batch
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (N, M, C, T, V)
        x = x.view(N * M, C, T, V)
        
        x = self.bn(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return x
