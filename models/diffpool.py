from typing import Optional, Callable, List
from jax.experimental.sparse import BCOO
import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import jraph

import sys
sys.path.append('../')

from models.gnn import GNN
from models.mlp import MLP


class DiffPool(nn.Module):
    n_downsamples: int = 2  # Number of downsample layers
    d_downsampling_factor: int = 4  # Downsampling factor at each layer
    k: int = 20  # Number of nearest neighbors to consider after downsampling
    gnn_kwargs: dict = dataclasses.field(default_factory=lambda: {"d_hidden":64, "n_layers":3})
    symmetric: bool = True  # Symmetrize the adjacency matrix
    task: str = "node"  # Node or graph task
    combine_hierarchies_method: str = "mean"  # How to aggregate hierarchical embeddings; TODO: impl attention
    use_edge_features: bool = False  # Whether to use edge features in adjacency matrix
    mlp_readout_widths: List[int] = (8, 2)  # Factor of d_hidden for global readout MLPs
    d_hidden: int = 64
    n_outputs: int = 1  # Number of outputs for graph-level readout

    @nn.compact
    def __call__(self, x):
        
        # If graph prediction task, collect pooled embeddings at each hierarchy level
        if self.task == "graph":
            x_pool = jnp.zeros((self.n_downsamples, self.gnn_kwargs['d_output']))

        for i in range(self.n_downsamples):
            
            # Original and downsampled number of nodes
            n_nodes = x.nodes.shape[0]
            n_nodes_downsampled = n_nodes // self.d_downsampling_factor

             # Eq. (5), graph embedding layer
            z = GNN(**self.gnn_kwargs)(x) 

            # Eq. (6), generate assignment matrix
            # Remove d_hidden from gnn_kwargs and replace it with n_nodes_downsampled
            gnn_kwargs = dict(self.gnn_kwargs.copy())
            gnn_kwargs['d_output'] = n_nodes_downsampled

            s = GNN(**gnn_kwargs,)(x).nodes  
            s = jax.nn.softmax(s, axis=1)  # Row-wise softmax
            
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
                globals=None,
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
            
            mlp = MLP([self.mlp_readout_widths[0] * x_pool.shape[-1]] + [w * self.d_hidden for w in self.mlp_readout_widths[1:]] + [self.n_outputs,])                                                         
            out = mlp(x_pool)                                                             
            return out
        
        return x