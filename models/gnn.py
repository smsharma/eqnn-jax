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
        senders: jnp.ndarray,
        receivers: jnp.ndarray,
        globals: jnp.ndarray,
    ) -> jnp.ndarray:
        """update node features

        Args:
            nodes (jnp.ndarray): node features
            senders (jnp.ndarray): attributes sent to neighbors
            received_attributes (jnp.ndarray): attributes received from neighbors
            globals (jnp.ndarray): global features

        Returns:
            jnp.ndarray: updated node features
        """
        m_i = nodes
        if receivers is not None:
            m_i = jnp.concatenate([m_i, receivers], axis=-1)
        return MLP([d_hidden] * n_layers, activation=activation)(m_i)

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
            m_ij = jnp.concatenate([edges, senders, receivers], axis=-1)
        else:
            m_ij = jnp.concatenate([senders, receivers], axis=-1)
        return MLP([d_hidden] * n_layers, activation=activation)(m_ij)

    return update_fn


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
    residual: bool = False  # Residual connections

    @nn.compact
    def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Apply graph convolutional layers to graph

        Args:
            graphs (jraph.GraphsTuple): Input graph

        Returns:
            jraph.GraphsTuple: Updated graph
        """

        activation = getattr(nn, self.activation)

        if self.message_passing_agg not in ["sum", "mean", "max"]:
            raise ValueError(
                f"Invalid message passing aggregation function {self.message_passing_agg}"
            )

        aggregate_edges_for_nodes_fn = getattr(
            utils, f"segment_{self.message_passing_agg}"
        )
        if not self.position_features:
            graphs = graphs._replace(
                    nodes=graphs.nodes[..., 3:],
                    )
            
        # Project node features into d_hidden
        graphs = graphs._replace(
            nodes=nn.Dense(self.d_hidden)(graphs.nodes)
        )

        # Apply message-passing rounds
        for _ in range(self.message_passing_steps):
            # Node and edge update functions

            update_edge_fn = get_edge_mlp_updates(
                self.d_hidden, self.n_layers, activation
            )
            update_node_fn = get_node_mlp_updates(
                self.d_hidden, self.n_layers, activation
            )

            # Instantiate graph network and apply GCL
            graph_net = jraph.GraphNetwork(
                update_node_fn=update_node_fn, 
                update_edge_fn=update_edge_fn,
                aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn
            )
            processed_graphs = graph_net(graphs)

            # Skip connection
            if self.residual:
                graphs = processed_graphs._replace(
                    nodes=processed_graphs.nodes + graphs.nodes
                )
            else:
                graphs = processed_graphs

            # Optional normalization
            norm = nn.LayerNorm() if self.norm == "layer" else lambda x: x

            graphs = graphs._replace(
                nodes=norm(graphs.nodes), edges=norm(graphs.edges)
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
                )(graphs.nodes)
                graphs = graphs._replace(
                    nodes=nodes,
                )
            return graphs

        elif self.task == "graph":  # Aggregate residual node features
            
            if self.readout_agg == "attn":
                q_agg = self.param("q_qgg", nn.initializers.xavier_uniform(), (1, graphs.nodes.shape[-1]))
                agg_nodes = nn.MultiHeadDotProductAttention(num_heads=2,)(q_agg, graphs.nodes,)[0, :]
            else:
                readout_agg_fn = getattr(jnp, f"{self.readout_agg}")
                agg_nodes = readout_agg_fn(graphs.nodes, axis=0)

            if graphs.globals is not None:
                agg_nodes = jnp.concatenate([agg_nodes, graphs.globals]) # Use tpcf
                
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
