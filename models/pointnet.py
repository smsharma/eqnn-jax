from typing import Optional, Callable, List
from jax.experimental.sparse import BCOO
import dataclasses

import numpy as np

import jax
import jax.numpy as jnp
from jax.lax import while_loop, fori_loop, scan
import flax.linen as nn
import jraph

import sys
sys.path.append('../')

from models.gnn import GNN
from models.mlp import MLP
from models.utils.graph_utils import ball_query, get_apply_pbc


def farthest_point_sampling(x, num_samples, apply_pbc):
    farthest_points_idx = jnp.zeros(num_samples, dtype=jnp.int32)
    # Arbitrarily choose the first point of the point cloud as the first farthest point
    farthest_points_idx = farthest_points_idx.at[0].set(0)
    
    distances = jnp.full(x.shape[0], jnp.inf)
    
    def sampling_fn(i, val):
        farthest_points_idx, distances = val
        # Get the latest point added to the farthest points
        latest_point_idx = farthest_points_idx[i-1]
        latest_point = x[latest_point_idx]

        # Compute the squared distances from the latest point to all other points
        new_dr = x - latest_point
        if apply_pbc:
            new_dr = apply_pbc(new_dr)
        new_distances = jnp.sum(new_dr ** 2, axis=-1)
        
        # Update the distances to maintain the minimum distance to the farthest points set
        distances = jnp.minimum(distances, new_distances)
        
        # Select the point that is farthest to all previous selections
        farthest_point_idx = jnp.argmax(distances)
        farthest_points_idx = farthest_points_idx.at[i].set(farthest_point_idx)
        
        return farthest_points_idx, distances

    # Iterate over the number of samples to select
    _, distances = fori_loop(1, num_samples, sampling_fn, (farthest_points_idx, distances))
    
    return x[farthest_points_idx]


class PointNet(nn.Module):
    n_downsamples: int = 2  # Number of downsample layers
    d_downsampling_factor: int = 4  # Downsampling factor at each layer
    k: int = 20  # Number of nearest neighbors to consider after downsampling
    radius: float = 0.2  # Radial cutoff to consider after downsampling
    gnn_kwargs: dict = dataclasses.field(default_factory=lambda: {"d_hidden":64, "n_layers":3})
    symmetric: bool = True  # Symmetrize the adjacency matrix
    task: str = "node"  # Node or graph task
    combine_hierarchies_method: str = "mean"  # How to aggregate hierarchical embeddings; TODO: impl attention
    use_edge_features: bool = False  # Whether to use edge features in adjacency matrix
    mlp_readout_widths: List[int] = (8, 2)  # Factor of d_hidden for global readout MLPs
    d_hidden: int = 64
    message_passing_steps: int = 2
    n_outputs: int = 1  # Number of outputs for graph-level readout
    apply_pbc: Callable = None

    @nn.compact
    def __call__(self, x, return_assignments=True): 
        # If graph prediction task, collect pooled embeddings at each hierarchy level
        if self.task == "graph":
            x_pool = jnp.zeros((self.n_downsamples, self.gnn_kwargs['n_outputs']))

        assignments = []
        node_pos = x.nodes[..., :3]

        for i in range(self.n_downsamples):
            
            # Original and downsampled number of nodes
            n_nodes = node_pos.shape[0]
            n_nodes_downsampled = n_nodes // self.d_downsampling_factor

            # Eq. (5), graph embedding layer
            z = GNN(**self.gnn_kwargs)(x) 

            # Create assignment matrix
            centroids, s = self.sample_and_group(node_pos, n_nodes_downsampled, self.radius, self.apply_pbc) 
            s = jax.nn.softmax(s, axis=1)  # Row-wise softmax
            assignments.append(s)
            node_pos = centroids # Pool node positions

            # Sparse adjacency matrix
            # If edge features, use them as weights, otherwise use 1 to indicate connectivity
            edge_index = jnp.array([x.senders, x.receivers])
            if self.use_edge_features:  
                edge_weight = nn.Dense(1)(x.edges)[..., 0]  # Edges might have more than one feature; project down
            else:
                edge_weight = jnp.ones((x.edges.shape[0],))

            a = BCOO((edge_weight, edge_index.T), shape=(n_nodes, n_nodes))
            
            # Eq. (3), coarsened node features
            x = s.T @ z.nodes  
            
            # Eq. (4), coarsened adjacency matrix)
            # Sparse matmul S^T @ A @ S
            a = s.T @ a @ s  

            # Make adj symmetric
            if self.symmetric:
                a = (a + a.T) / 2 #check if already symm

            # Take the coarsened adjacency matrix and make a KNN graph of it
            indices = np.argsort(a, axis=-1)[:, :self.k]
            
            sources = indices[:, 0].repeat(self.k)
            targets = indices.reshape(n_nodes_downsampled * (self.k))

            # Create new graph
            x = jraph.GraphsTuple(
                nodes=x,
                edges=a[sources, targets][..., None],
                senders=sources,
                receivers=targets,
                globals=z.globals,
                n_node=n_nodes_downsampled,
                n_edge=self.k,
            )

            # If graph prediction task, get hierarchical embeddings
            if self.task == "graph":
                x_pool = x_pool.at[i].set(jnp.mean(x.nodes, axis=0))

        if self.task == "graph":
            if self.combine_hierarchies_method == "mean":  # Mean over hierarchy levels
                x_pool = jnp.mean(x_pool, axis=0)
            elif self.combine_hierarchies_method == "concat":  # Max over hierarchy levels
                x_pool = jnp.concatenate(x_pool, axis=0)
            else:
                raise ValueError(f"Unknown combine_hierarchies_method: {self.combine_hierarchies_method}")
            
            if x.globals is not None:
                x_pool = jnp.concatenate([x_pool, processed_graphs.globals]) #use tpcf
                
                norm = nn.LayerNorm()
                x_pool = norm(x_pool)
                
            # Readout and return
            mlp = MLP([
                self.mlp_readout_widths[0] * x_pool.shape[-1]] + \
                [w * self.d_hidden for w in self.mlp_readout_widths[1:]] + \
                [self.n_outputs,]
            )                                                                        
            out = mlp(x_pool)    
            
            if return_assignments:
                return out, assignments
            return out
           
        return x

    def sample_and_group(self, x, n_points_downsampled, radius, apply_pbc):
        centroids = farthest_point_sampling(x, n_points_downsampled, apply_pbc)
        group_matrix = ball_query(x, centroids, radius, apply_pbc)

        return centroids, group_matrix.astype(float)