from typing import Optional, Callable, List
import flax.linen as nn
import jax.numpy as jnp
import jraph
from jraph._src import utils

from models.mlp import MLP


def get_node_mlp_updates(d_hidden, n_layers, activation):
    """Get a node MLP update  function

    Args:
        mlp_feature_sizes (int): number of features in the MLP
        name (str, optional): name of the update function. Defaults to None.

    Returns:
        Callable: update function
    """

    def update_fn(
        nodes: jnp.ndarray,
        sent_attributes: jnp.ndarray,
        received_attributes: jnp.ndarray,
        globals: jnp.ndarray,
    ) -> jnp.ndarray:
        """update node features

        Args:
            nodes (jnp.ndarray): node features
            sent_attributes (jnp.ndarray): attributes sent to neighbors
            received_attributes (jnp.ndarray): attributes received from neighbors
            globals (jnp.ndarray): global features

        Returns:
            jnp.ndarray: updated node features
        """

        if received_attributes is not None:
            inputs = jnp.concatenate([nodes, received_attributes], axis=1)
        else:  # If lone node
            inputs = jnp.concatenate([nodes], axis=1)
        return MLP([d_hidden] * n_layers, activation=activation)(inputs)

    return update_fn


def get_edge_mlp_updates(d_hidden, n_layers, activation) -> Callable:
    """Get an edge MLP update function

    Args:
        mlp_feature_sizes (int): number of features in the MLP
        name (str, optional): name of the update function. Defaults to None.

    Returns:
        Callable: update function
    """

    def update_fn(
        edges: jnp.array,
        senders: jnp.array,
        receivers: jnp.array,
        globals: jnp.array,
    ) -> jnp.array:
        """update edge features

        Args:
            edges (jnp.ndarray): edge attributes
            senders (jnp.ndarray): senders node attributes
            receivers (jnp.ndarray): receivers node attributes
            globals (jnp.ndarray): global features

        Returns:
            jnp.ndarray: updated edge features
        """

        # If there are no edges in the initial layer
        if edges is not None:
            inputs = jnp.concatenate([edges, senders, receivers], axis=1)
        else:
            inputs = jnp.concatenate([senders, receivers], axis=1)
        return MLP([d_hidden] * n_layers, activation=activation)(inputs)

    return update_fn


class Identity(nn.Module):
    """Identity module"""
    @nn.compact
    def __call__(self, x):
        return x


class GNN(nn.Module):
    """Standard Graph Neural Network"""

    d_hidden: int = 64
    n_layers: int = 3
    message_passing_steps: int = 3
    message_passing_agg: str = "sum"  # "sum", "mean", "max"
    activation: str = "gelu"
    norm: str = "layer"
    task: str = "graph"  # "graph" or "node"
    n_outputs: int = 1  # Number of outputs for graph-level readout
    readout_agg: str = "mean"
    mlp_readout_widths: List[int] = (4, 2, 2)  # Factor of d_hidden for global readout MLPs
    position_features: bool = True  # Use absolute positions as node features

    @nn.compact
    def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Apply graph convolutional layers to graph

        Args:
            graphs (jraph.GraphsTuple): Input graph

        Returns:
            jraph.GraphsTuple: Updated graph
        """
        processed_graphs = graphs

        activation = getattr(nn, self.activation)

        if self.message_passing_agg not in ["sum", "mean", "max"]:
            raise ValueError(
                f"Invalid message passing aggregation function {self.message_passing_agg}"
            )

        aggregate_edges_for_nodes_fn = getattr(
            utils, f"segment_{self.message_passing_agg}"
        )
        if not self.position_features:
            processed_graphs = processed_graphs._replace(
                    nodes=processed_graphs.nodes[..., 3:],
                    )
            
        # Project node features into d_hidden
        processed_graphs = processed_graphs._replace(
            nodes=nn.Dense(self.d_hidden)(processed_graphs.nodes)
        )

        # Apply message-passing rounds
        for _ in range(self.message_passing_steps):
            # Node and edge update functions
            update_node_fn = get_node_mlp_updates(
                self.d_hidden, self.n_layers, activation
            )
            update_edge_fn = get_edge_mlp_updates(
                self.d_hidden, self.n_layers, activation
            )

            # Instantiate graph network and apply GCL
            graph_net = jraph.GraphNetwork(
                update_node_fn=update_node_fn, 
                update_edge_fn=update_edge_fn,
                aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn
            )
            processed_graphs = graph_net(processed_graphs)

            # Optional normalization
            if self.norm == "layer":
                norm = nn.LayerNorm()
            else:
                norm = Identity()  # No normalization
            processed_graphs = processed_graphs._replace(
                nodes=norm(processed_graphs.nodes), edges=norm(processed_graphs.edges)
            )

        if self.readout_agg not in ["sum", "mean", "max", "attn"]:
            raise ValueError(
                f"Invalid global aggregation function {self.readout_agg}"
            )

        if self.task == "node":
            if self.d_output is not None:
                nodes = MLP(
                    [self.d_hidden] * (self.n_layers - 1) + [self.n_outputs],
                    activation=activation,
                )(processed_graphs.nodes)
                processed_graphs = processed_graphs._replace(
                    nodes=nodes,
                )
            return processed_graphs

        elif self.task == "graph":  # Aggregate residual node features
            
            if self.readout_agg == "attn":
                q_agg = self.param("q_qgg", nn.initializers.xavier_uniform(), (1, processed_graphs.nodes.shape[-1]))
                agg_nodes = nn.MultiHeadDotProductAttention(num_heads=2,)(q_agg, processed_graphs.nodes,)[0, :]
            else:
                readout_agg_fn = getattr(jnp, f"{self.readout_agg}")
                agg_nodes = readout_agg_fn(processed_graphs.nodes, axis=0)

            if processed_graphs.globals is not None:
                agg_nodes = jnp.concatenate([agg_nodes, processed_graphs.globals]) # Use tpcf
                
                norm = nn.LayerNorm()
                agg_nodes = norm(agg_nodes)
                
            # Readout and return
            mlp = MLP([
                self.mlp_readout_widths[0] * agg_nodes.shape[-1]] + \
                [w * self.d_hidden for w in self.mlp_readout_widths[1:]] + \
                [self.n_outputs,]
            )                                                                        
            out = mlp(agg_nodes)                                                             
            return out

        else:
            raise ValueError(f"Invalid task {self.task}")
