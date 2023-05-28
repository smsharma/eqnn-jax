import jax
from typing import Callable
import flax.linen as nn
import jax.numpy as jnp
import jraph
from jraph._src import utils

from models.graph_utils import add_graphs_tuples
from models.mlp import MLP


def fourier_features(x, num_encodings=16, include_self=True):
    """Add Fourier features to a set of coordinates

    Args:
        x (jnp.array): Coordinates
        num_encodings (int, optional): Number of Fourier feature encodings. Defaults to 16.
        include_self (bool, optional): Whether to include original coordinates in output. Defaults to True.

    Returns:
        jnp.array: Fourier features of input coordinates
    """

    dtype, orig_x = x.dtype, x
    scales = 2 ** jnp.arange(num_encodings, dtype=dtype)
    x = x / scales
    x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)
    x = jnp.concatenate((x, orig_x), axis=-1) if include_self else x
    return x


def get_edge_mlp_updates(d_hidden, n_layers, activation, position_only=False, use_fourier_features=False) -> Callable:
    """Get an edge MLP update function

    Args:
        mlp_feature_sizes (int): number of features in the MLP

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

        # Split senders and receivers into coordinates, velocities, and scalar attrs
        x_i, v_i, h_i = senders[:, :3], senders[:, 3:6], senders[:, 6:]
        x_j, v_j, h_j = receivers[:, :3], receivers[:, 3:6], receivers[:, 6:]

        # Messages from Eqs. (3) and (4)/(7)
        phi_e = MLP([d_hidden] * (n_layers), activation=activation)
        phi_x = MLP([d_hidden] * (n_layers - 1) + [1], activation=activation)

        m_ij = phi_e(
            jnp.concatenate(
                [
                    h_i,
                    h_j,
                    jnp.linalg.norm(x_i - x_j, axis=1, keepdims=True) ** 2,
                    globals,
                ],
                axis=-1,
            )
        )
        return (x_i - x_j) * phi_x(m_ij), m_ij

    def update_fn_position_only(
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

        # Split senders and receivers into coordinates, velocities, and scalar attrs
        x_i = senders
        x_j = receivers

        if x_i.shape[-1] == 3:
            concats = globals
        else:
            x_i, h_i = x_i[:, :3], x_i[:, 3:]
            x_j, h_j = x_j[:, :3], x_j[:, 3:]
            concats = jnp.concatenate([h_i, h_j, globals], -1)

        # Messages from Eqs. (3) and (4)/(7)
        phi_e = MLP([d_hidden] * (n_layers), activation=activation)
        phi_x = MLP([d_hidden] * (n_layers - 1) + [1], activation=activation)

        d_ij2 = jnp.sum((x_i - x_j) ** 2, axis=1, keepdims=True)
        if use_fourier_features:
            d_ij2 = fourier_features(d_ij2)
        print(d_ij2.shape)

        # Get invariants
        message_scalars = jnp.concatenate([d_ij2, concats], axis=-1)
        m_ij = phi_e(message_scalars)
        return (x_i - x_j) * phi_x(jnp.concatenate([m_ij, concats], -1)), m_ij

    return update_fn if not position_only else update_fn_position_only


def get_node_mlp_updates(d_hidden, n_layers, activation, n_edge, position_only=False) -> Callable:
    """Get an node MLP update function

    Args:
        mlp_feature_sizes (int): number of features in the MLP

    Returns:
        Callable: update function
    """

    def update_fn(
        nodes: jnp.array,
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

        sum_x_ij, m_i = receivers  # Get aggregated messages
        x_i, v_i, h_i = nodes[:, :3], nodes[:, 3:6], nodes[:, 6:]  # Split node attrs

        # From Eqs. (6) and (7)
        phi_v = MLP([d_hidden] * (n_layers - 1) + [1], activation=activation)
        phi_h = MLP([d_hidden] * (n_layers - 1) + [h_i.shape[-1]], activation=activation)

        # Apply updates
        v_i_p = sum_x_ij / (n_edge - 1) + phi_v(h_i) * v_i
        x_i_p = x_i + v_i_p
        h_i_p = phi_h(jnp.concatenate([h_i, m_i], -1)) + h_i  # Skip connection

        return jnp.concatenate([x_i_p, v_i_p, h_i_p], -1)

    def update_fn_position_only(
        nodes: jnp.array,
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

        sum_x_ij, m_i = receivers  # Get aggregated messages
        x_i = nodes

        # Apply updates
        if x_i.shape[-1] == 3:
            phi_h = MLP([d_hidden] * (n_layers - 1) + [int(d_hidden / 2)], activation=activation)
            x_i_p = x_i + sum_x_ij / (n_edge - 1)
            h_i_p = phi_h(jnp.concatenate([m_i], -1))
            return jnp.concatenate([x_i_p, h_i_p], -1)
        else:
            x_i_p = x_i[..., :3] + sum_x_ij / (n_edge - 1)
            h_i = x_i[..., 3:]
            phi_h = MLP([d_hidden] * (n_layers - 1) + [h_i.shape[-1]], activation=activation)
            h_i_p = phi_h(jnp.concatenate([h_i, m_i], -1))
            return x_i_p

    return update_fn if not position_only else update_fn_position_only


class EGNN(nn.Module):
    """Equivariant Graph Neural Network following Satorras et al (2021)."""

    message_passing_steps: int = 3
    d_hidden: int = 64
    n_layers: int = 3
    skip_connections: bool = False
    activation: str = "swish"
    use_fourier_features: bool = True
    position_only: bool = False

    @nn.compact
    def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Do message passing on graph

        Args:
            graphs (jraph.GraphsTuple): graph object

        Returns:
            jraph.GraphsTuple: updated graph object
        """

        if graphs.nodes is None:
            raise ValueError("`GraphsTuple` must have nodes")
        
        in_features = graphs.nodes.shape[-1]
        processed_graphs = graphs


        if processed_graphs.globals is not None:
            processed_graphs = processed_graphs._replace(globals=processed_graphs.globals.reshape(1, -1))

        activation = getattr(nn, self.activation)

        # Switch for whether to use positions-only version of edge/node updates
        if (graphs.nodes.shape[-1] > 3) & (graphs.nodes.shape[-1] < 6):
            raise NotImplementedError("Number of features should be either 3 (just positions) or >= 6 (positions, velocities, and scalars)")

        positions_only = True if graphs.nodes.shape[-1] == 3 else False

        # Apply message-passing rounds
        for _ in range(self.message_passing_steps):

            # Get node and edge update functions
            update_node_fn = get_node_mlp_updates(self.d_hidden, self.n_layers, activation, n_edge=processed_graphs.n_edge, position_only=positions_only)
            update_edge_fn = get_edge_mlp_updates(self.d_hidden, self.n_layers, activation, position_only=positions_only, use_fourier_features=self.use_fourier_features)

            # Get jraph graph network
            graph_net = jraph.GraphNetwork(update_node_fn=update_node_fn, update_edge_fn=update_edge_fn)
            
            # Apply updates, optionally with skip connections
            if self.skip_connections:
                processed_graphs = add_graphs_tuples(graph_net(processed_graphs), processed_graphs)
            else:
                processed_graphs = graph_net(processed_graphs)

        # Return updated graph
        return processed_graphs