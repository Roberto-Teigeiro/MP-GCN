"""
Graph Construction for Skeleton-based Action Recognition
Based on MP-GCN graph structure with multi-person support
"""
import numpy as np
from .config import SKELETON_EDGES, CONNECT_JOINT, BODY_PARTS, CENTER_JOINT, NUM_KEYPOINTS


class Graph:
    """
    Skeleton graph with adjacency matrix construction.
    Supports single and multi-person graphs.
    """
    
    def __init__(self, num_nodes=NUM_KEYPOINTS, num_persons=1, max_hop=2, 
                 labeling='spatial', inter_link='linear'):
        """
        Args:
            num_nodes: Number of joints per person
            num_persons: Number of persons in the graph
            max_hop: Maximum hop distance for adjacency
            labeling: Type of graph labeling ('spatial', 'distance', 'zeros')
            inter_link: Type of inter-person connections ('linear', 'full', 'none')
        """
        self.num_nodes_per_person = num_nodes
        self.num_persons = num_persons
        self.num_nodes = num_nodes * num_persons
        self.max_hop = max_hop
        self.labeling = labeling
        self.inter_link = inter_link
        
        # Build graph components
        self.edges = self._build_edges()
        self.connect_joint = self._build_connect_joint()
        self.parts = self._build_parts()
        self.center = self._build_center()
        
        # Build adjacency matrix
        self.A = self._build_adjacency()
        
    def _build_edges(self):
        """Build edge list for the skeleton graph."""
        edges = []
        
        # Self-loops
        for i in range(self.num_nodes):
            edges.append((i, i))
        
        # Intra-person edges
        for p in range(self.num_persons):
            offset = p * self.num_nodes_per_person
            for (i, j) in SKELETON_EDGES:
                edges.append((i + offset, j + offset))
        
        # Inter-person edges
        if self.num_persons > 1 and self.inter_link != 'none':
            if self.inter_link == 'linear':
                # Connect center joints of adjacent persons
                for p in range(1, self.num_persons):
                    curr_center = p * self.num_nodes_per_person + CENTER_JOINT
                    prev_center = (p - 1) * self.num_nodes_per_person + CENTER_JOINT
                    edges.append((curr_center, prev_center))
            elif self.inter_link == 'full':
                # Connect center joints of all persons
                for p1 in range(self.num_persons):
                    for p2 in range(p1 + 1, self.num_persons):
                        c1 = p1 * self.num_nodes_per_person + CENTER_JOINT
                        c2 = p2 * self.num_nodes_per_person + CENTER_JOINT
                        edges.append((c1, c2))
        
        return edges
    
    def _build_connect_joint(self):
        """Build parent joint connections for bone computation."""
        connect = []
        for p in range(self.num_persons):
            offset = p * self.num_nodes_per_person
            connect.extend([c + offset if c >= 0 else -1 for c in CONNECT_JOINT])
        return np.array(connect)
    
    def _build_parts(self):
        """Build body part groupings for attention."""
        parts = []
        for p in range(self.num_persons):
            offset = p * self.num_nodes_per_person
            for part in BODY_PARTS:
                parts.append(np.array([j + offset for j in part]))
        return parts
    
    def _build_center(self):
        """Build center joint indices for each node."""
        centers = []
        for p in range(self.num_persons):
            center = p * self.num_nodes_per_person + CENTER_JOINT
            centers.extend([center] * self.num_nodes_per_person)
        return centers
    
    def _compute_hop_distance(self):
        """Compute hop distances between all pairs of nodes."""
        # Build adjacency matrix from edges
        adj = np.zeros((self.num_nodes, self.num_nodes))
        for (i, j) in self.edges:
            adj[i, j] = 1
            adj[j, i] = 1
        
        # Compute hop distances using matrix powers
        hop_dis = np.full((self.num_nodes, self.num_nodes), np.inf)
        transfer_mat = [np.linalg.matrix_power(adj, d) for d in range(self.max_hop + 1)]
        arrive_mat = np.stack(transfer_mat) > 0
        
        for d in range(self.max_hop, -1, -1):
            hop_dis[arrive_mat[d]] = d
            
        return hop_dis, adj
    
    def _normalize_adjacency(self, A):
        """Normalize adjacency matrix by column sum (out-degree)."""
        Dl = np.sum(A, axis=0)
        Dn = np.zeros_like(A)
        for i in range(A.shape[0]):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        return np.dot(A, Dn)
    
    def _build_adjacency(self):
        """Build multi-scale adjacency matrices based on labeling strategy."""
        hop_dis, raw_adj = self._compute_hop_distance()
        
        # Valid hops to consider
        valid_hops = range(0, self.max_hop + 1)
        
        # Full adjacency matrix
        adjacency = np.zeros((self.num_nodes, self.num_nodes))
        for hop in valid_hops:
            adjacency[hop_dis == hop] = 1
        
        # Normalize
        norm_adj = self._normalize_adjacency(adjacency)
        
        if self.labeling == 'distance':
            # Simple distance-based: one matrix per hop
            A = np.zeros((len(valid_hops), self.num_nodes, self.num_nodes))
            for i, hop in enumerate(valid_hops):
                A[i][hop_dis == hop] = norm_adj[hop_dis == hop]
                
        elif self.labeling == 'spatial':
            # Spatial labeling: partition by centripetal/centrifugal
            A = []
            for hop in valid_hops:
                a_root = np.zeros((self.num_nodes, self.num_nodes))
                a_close = np.zeros((self.num_nodes, self.num_nodes))
                a_further = np.zeros((self.num_nodes, self.num_nodes))
                
                for i in range(self.num_nodes):
                    for j in range(self.num_nodes):
                        if hop_dis[j, i] == hop:
                            center_i = self.center[i]
                            center_j = self.center[j]
                            
                            # Check for infinite distances
                            dist_j_to_center = hop_dis[j, center_j] if center_j < self.num_nodes else np.inf
                            dist_i_to_center = hop_dis[i, center_i] if center_i < self.num_nodes else np.inf
                            
                            if dist_j_to_center == dist_i_to_center:
                                a_root[j, i] = norm_adj[j, i]
                            elif dist_j_to_center > dist_i_to_center:
                                a_close[j, i] = norm_adj[j, i]
                            else:
                                a_further[j, i] = norm_adj[j, i]
                
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            
            A = np.stack(A)
            
        elif self.labeling == 'zeros':
            # All zeros - let model learn completely
            A = np.zeros((len(valid_hops), self.num_nodes, self.num_nodes))
            
        else:
            # Default: identity-like
            A = np.zeros((len(valid_hops), self.num_nodes, self.num_nodes))
            for i in range(len(valid_hops)):
                A[i] = self._normalize_adjacency(np.eye(self.num_nodes))
        
        return A
    
    def get_adjacency(self):
        """Return the adjacency tensor."""
        return self.A
    
    def get_edge_list(self):
        """Return edge list."""
        return self.edges
    
    def __repr__(self):
        return (f"Graph(num_nodes={self.num_nodes}, num_persons={self.num_persons}, "
                f"max_hop={self.max_hop}, labeling={self.labeling}, "
                f"adjacency_shape={self.A.shape})")


def create_graph(num_persons=1, max_hop=2, labeling='spatial', inter_link='linear'):
    """Factory function to create a skeleton graph."""
    return Graph(
        num_nodes=NUM_KEYPOINTS,
        num_persons=num_persons,
        max_hop=max_hop,
        labeling=labeling,
        inter_link=inter_link
    )
