"""
Panoramic Graph Construction for Multi-Person Activity Recognition

Based on MP-GCN: Spatial-Temporal Panoramic Graph
Supports:
- Intra-person edges (body topology)
- Person-object edges (hands to objects)
- Inter-person edges (hip-to-hip connections)
"""
import numpy as np
import torch
from typing import List, Tuple, Optional
from .config import NUM_KEYPOINTS, SKELETON_EDGES, BODY_PARTS, CENTER_JOINT


class PanoramicGraph:
    """
    Panoramic graph with person-person and person-object interactions.
    
    Node layout: [Person1_joints, Person2_joints, ..., Object1, Object2, ...]
    """
    
    def __init__(self, 
                 num_persons: int = 4,
                 num_objects: int = 0,
                 num_joints: int = NUM_KEYPOINTS,
                 max_hop: int = 2,
                 inter_person_link: str = 'pelvis',
                 object_link_joints: List[int] = [9, 10]):  # Wrists
        """
        Args:
            num_persons: Maximum number of persons
            num_objects: Number of object nodes
            num_joints: Joints per person (17 for COCO)
            max_hop: Maximum hop distance for adjacency
            inter_person_link: Type of person-person connection ('pelvis', 'all', 'none')
            object_link_joints: Which joints connect to objects (default: wrists)
        """
        self.num_persons = num_persons
        self.num_objects = num_objects
        self.num_joints = num_joints
        self.max_hop = max_hop
        self.inter_person_link = inter_person_link
        self.object_link_joints = object_link_joints
        
        # Total nodes: persons * joints + objects
        self.num_nodes = num_persons * num_joints + num_objects
        
        # Build graph components
        self.edges = self._build_edges()
        self.parts = self._build_parts()
        
        # Build adjacency matrices
        self.A_intra, self.A_inter, self.A_self = self._build_adjacency_matrices()
        
        # Combined adjacency for model
        self.A = self._combine_adjacency()
    
    def _build_edges(self) -> dict:
        """Build all edge types."""
        edges = {
            'self': [],
            'intra': [],
            'inter': [],
            'object': []
        }
        
        # Self-loops
        for i in range(self.num_nodes):
            edges['self'].append((i, i))
        
        # Intra-person edges (body topology)
        for p in range(self.num_persons):
            offset = p * self.num_joints
            for (i, j) in SKELETON_EDGES:
                edges['intra'].append((i + offset, j + offset))
                edges['intra'].append((j + offset, i + offset))  # Bidirectional
        
        # Inter-person edges (pelvis-to-pelvis for adjacent persons)
        if self.inter_person_link == 'pelvis':
            # Connect hip midpoints between adjacent persons
            for p1 in range(self.num_persons):
                for p2 in range(p1 + 1, self.num_persons):
                    # Left hip (11) and right hip (12) connections
                    for hip in [11, 12]:
                        n1 = p1 * self.num_joints + hip
                        n2 = p2 * self.num_joints + hip
                        edges['inter'].append((n1, n2))
                        edges['inter'].append((n2, n1))
        elif self.inter_person_link == 'all':
            # Connect all corresponding joints between persons
            for p1 in range(self.num_persons):
                for p2 in range(p1 + 1, self.num_persons):
                    for j in range(self.num_joints):
                        n1 = p1 * self.num_joints + j
                        n2 = p2 * self.num_joints + j
                        edges['inter'].append((n1, n2))
                        edges['inter'].append((n2, n1))
        
        # Person-object edges (hands to objects)
        if self.num_objects > 0:
            object_start = self.num_persons * self.num_joints
            for p in range(self.num_persons):
                for link_joint in self.object_link_joints:
                    person_node = p * self.num_joints + link_joint
                    for obj_idx in range(self.num_objects):
                        obj_node = object_start + obj_idx
                        edges['object'].append((person_node, obj_node))
                        edges['object'].append((obj_node, person_node))
        
        return edges
    
    def _build_parts(self) -> List[np.ndarray]:
        """Build body part groupings for all persons."""
        parts = []
        for p in range(self.num_persons):
            offset = p * self.num_joints
            for body_part in BODY_PARTS:
                parts.append(np.array([j + offset for j in body_part]))
        
        # Add object parts
        if self.num_objects > 0:
            object_start = self.num_persons * self.num_joints
            parts.append(np.arange(object_start, self.num_nodes))
        
        return parts
    
    def _compute_hop_distance(self, edges: List[Tuple[int, int]]) -> np.ndarray:
        """Compute shortest path distances between all node pairs."""
        # Build adjacency from edges
        adj = np.zeros((self.num_nodes, self.num_nodes))
        for (i, j) in edges:
            adj[i, j] = 1
            adj[j, i] = 1
        
        # Add self-loops
        np.fill_diagonal(adj, 1)
        
        # Compute hop distances using BFS-style matrix powers
        hop_dis = np.full((self.num_nodes, self.num_nodes), np.inf)
        np.fill_diagonal(hop_dis, 0)
        
        transfer_mat = [np.eye(self.num_nodes)]
        for d in range(1, self.max_hop + 1):
            transfer_mat.append(np.linalg.matrix_power(adj, d))
        
        for d in range(self.max_hop + 1):
            reachable = transfer_mat[d] > 0
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if reachable[i, j] and hop_dis[i, j] > d:
                        hop_dis[i, j] = d
        
        return hop_dis
    
    def _normalize_adjacency(self, A: np.ndarray) -> np.ndarray:
        """Normalize adjacency matrix (column-wise)."""
        Dl = np.sum(A, axis=0)
        Dn = np.zeros_like(A)
        for i in range(A.shape[0]):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-0.5)
        # Symmetric normalization: D^(-1/2) A D^(-1/2)
        return Dn @ A @ Dn
    
    def _build_adjacency_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build separate adjacency matrices for different edge types."""
        # Self-loop adjacency
        A_self = np.eye(self.num_nodes)
        
        # Intra-person adjacency (includes body edges + object edges)
        A_intra = np.zeros((self.num_nodes, self.num_nodes))
        for (i, j) in self.edges['intra']:
            A_intra[i, j] = 1
        for (i, j) in self.edges['object']:
            A_intra[i, j] = 1
        
        # Inter-person adjacency
        A_inter = np.zeros((self.num_nodes, self.num_nodes))
        for (i, j) in self.edges['inter']:
            A_inter[i, j] = 1
        
        # Normalize
        A_self = self._normalize_adjacency(A_self)
        A_intra = self._normalize_adjacency(A_intra + np.eye(self.num_nodes))
        A_inter = self._normalize_adjacency(A_inter + np.eye(self.num_nodes))
        
        return A_intra, A_inter, A_self
    
    def _combine_adjacency(self) -> np.ndarray:
        """Combine adjacency matrices into multi-scale format."""
        # Stack: [A_self, A_intra, A_inter]
        A = np.stack([self.A_self, self.A_intra, self.A_inter], axis=0)
        return A
    
    def get_adjacency_tensor(self) -> torch.Tensor:
        """Get adjacency as PyTorch tensor."""
        return torch.FloatTensor(self.A)
    
    def get_connect_joint(self) -> np.ndarray:
        """Get parent joint connections for bone computation."""
        from .config import CONNECT_JOINT
        
        connect = []
        for p in range(self.num_persons):
            offset = p * self.num_joints
            connect.extend([c + offset if c >= 0 else -1 for c in CONNECT_JOINT])
        
        # Objects connect to first wrist
        if self.num_objects > 0:
            first_wrist = self.object_link_joints[0]
            connect.extend([first_wrist] * self.num_objects)
        
        return np.array(connect)
    
    def __repr__(self):
        return (f"PanoramicGraph(persons={self.num_persons}, objects={self.num_objects}, "
                f"nodes={self.num_nodes}, adjacency_shape={self.A.shape})")


def create_panoramic_graph(num_persons: int = 4, 
                           num_objects: int = 0,
                           max_hop: int = 2) -> PanoramicGraph:
    """Factory function to create a panoramic graph."""
    return PanoramicGraph(
        num_persons=num_persons,
        num_objects=num_objects,
        max_hop=max_hop
    )


class DynamicPanoramicGraph:
    """
    Dynamic panoramic graph that adjusts based on actual detections.
    Useful when number of persons varies per sample.
    """
    
    def __init__(self, max_persons: int = 6, max_objects: int = 4, max_hop: int = 2):
        self.max_persons = max_persons
        self.max_objects = max_objects
        self.max_hop = max_hop
        
        # Pre-compute graphs for different configurations
        self._graph_cache = {}
    
    def get_graph(self, num_persons: int, num_objects: int) -> PanoramicGraph:
        """Get or create graph for specific configuration."""
        key = (num_persons, num_objects)
        if key not in self._graph_cache:
            self._graph_cache[key] = PanoramicGraph(
                num_persons=num_persons,
                num_objects=num_objects,
                max_hop=self.max_hop
            )
        return self._graph_cache[key]
    
    def get_max_graph(self) -> PanoramicGraph:
        """Get graph for maximum configuration."""
        return self.get_graph(self.max_persons, self.max_objects)
