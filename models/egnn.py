import jax
from typing import Callable
import flax.linen as nn
import jax.numpy as jnp
import jraph
from einops import repeat, rearrange

from models.graph_utils import add_graphs_tuples
from models.mlp import MLP


def fourier_features(x, num_encodings=8, include_self=True):
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


def get_edge_mlp_updates(d_hidden, n_layers, activation, position_only=False, use_fourier_features=False, fourier_feature_kwargs={"num_encodings": 16, "include_self": True}, tanh_out=False) -> Callable:
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

        # If we have additional scalar features

        if position_only:  # Positions only; no velocities
            if senders.shape[-1] == 3:
                x_i = senders
                x_j = receivers
                concats = globals
            else:
                x_i, h_i = senders[:, :3], senders[:, 3:]
                x_j, h_j = receivers[:, :3], receivers[:, 3:]
                concats = jnp.concatenate([h_i, h_j, globals], -1)
        else:  # Positions and velocities
            if senders.shape[-1] == 6:
                x_i, v_i = senders[:, :3], senders[:, 3:6]
                x_j, v_j = receivers[:, :3], receivers[:, 3:6]
                concats = globals
            else:
                x_i, v_i, h_i = senders[:, :3], senders[:, 3:6], senders[:, 6:]
                x_j, v_j, h_j = receivers[:, :3], receivers[:, 3:6], receivers[:, 6:]
                concats = jnp.concatenate([h_i, h_j, globals], -1)

        # Messages from Eqs. (3) and (4)/(7)
        phi_e = MLP([d_hidden] * (n_layers), activation=activation, activate_final=True)

        # Super special init for last layer of position MLP
        phi_x = MLP([d_hidden] * (n_layers - 1), activation=activation, activate_final=True)
        phi_x_last_layer = nn.Dense(1, use_bias=False, kernel_init=jax.nn.initializers.variance_scaling(scale=1e-3, mode="fan_in", distribution="truncated_normal"))

        d_ij2 = jnp.sum((x_i - x_j) ** 2, axis=1, keepdims=True)
        d_ij2 = fourier_features(d_ij2, **fourier_feature_kwargs) if use_fourier_features else d_ij2

        # ## Multi-channel experiments ##
        # n_channels = 5
        # X_i = repeat(x_i, "n d -> n d c", c=n_channels)
        # X_j = repeat(x_j, "n d -> n d c", c=n_channels)
        # D_ij2 = jnp.sum((X_i - X_j) ** 2, axis=1, keepdims=True)
        # Phi_e = MLP([d_hidden] * (n_layers), activation=activation, activate_final=True)
        # Phi_x_last_layer = nn.Dense(n_channels * n_channels, use_bias=False, kernel_init=jax.nn.initializers.variance_scaling(scale=1e-3, mode="fan_in", distribution="truncated_normal"))
        # D_ij2 = jax.vmap(fourier_features, in_axes=(2), out_axes=(2))(D_ij2) if use_fourier_features else D_ij2  # FF
        # D_ij2 = D_ij2.reshape(D_ij2.shape[0], -1)  # Flatten
        # message_scalars = jnp.concatenate([D_ij2, concats], axis=-1)
        # m_ij = Phi_e(message_scalars)
        # trans = Phi_x_last_layer(phi_x(m_ij))
        # trans = rearrange(trans, "n (c1 c2) -> n c1 c2", c1=n_channels, c2=n_channels)
        # out = jnp.matmul((X_i - X_j), trans)  # n_dim x n_channels
        # ## End multi-channel experiments ##

        # Get invariants
        message_scalars = jnp.concatenate([d_ij2, concats], axis=-1)
        m_ij = phi_e(message_scalars)

        trans = phi_x_last_layer(phi_x(m_ij))

        # Optionally apply tanh to pre-factor to stabilize
        if tanh_out:
            trans = jax.nn.tanh(trans)

        return (x_i - x_j) * trans, m_ij

    return update_fn


def get_node_mlp_updates(d_hidden, n_layers, activation, n_edge, position_only=False) -> Callable:
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

        if position_only:
            if nodes.shape[-1] == 3:
                x_i = nodes
                x_i_p = x_i + sum_x_ij / (n_edge - 1)
                return x_i_p
            else:
                x_i, h_i = nodes[..., :3], nodes[..., 3:]
                x_i_p = x_i + sum_x_ij / (n_edge - 1)
                phi_h = MLP([d_hidden] * (n_layers - 1) + [h_i.shape[-1]], activation=activation)
                concats = jnp.concatenate([h_i, m_i, globals], -1)
                h_i_p = phi_h(concats)
                return jnp.concatenate([x_i_p, h_i_p], -1)

        else:
            if nodes.shape[-1] == 6:
                x_i, v_i = nodes[:, :3], nodes[:, 3:6]  # Split node attrs

                # From Eqs. (6) and (7)
                phi_v = MLP([d_hidden] * (n_layers - 1) + [1], activation=activation)

                concats = jnp.concatenate([m_i, globals], -1)

                # Apply updates
                v_i_p = sum_x_ij / (n_edge - 1) + phi_v(concats) * v_i
                x_i_p = x_i + v_i_p

                return jnp.concatenate([x_i_p, v_i_p], -1)
            else:
                x_i, v_i, h_i = nodes[:, :3], nodes[:, 3:6], nodes[:, 6:]  # Split node attrs

                # From Eqs. (6) and (7)
                phi_v = MLP([d_hidden] * (n_layers - 1) + [1], activation=activation)
                phi_h = MLP([d_hidden] * (n_layers - 1) + [h_i.shape[-1]], activation=activation)

                concats = jnp.concatenate([h_i, m_i, globals], -1)

                # Apply updates
                v_i_p = sum_x_ij / (n_edge - 1) + phi_v(concats) * v_i
                x_i_p = x_i + v_i_p
                h_i_p = phi_h(concats) + h_i  # Skip connection

                return jnp.concatenate([x_i_p, v_i_p, h_i_p], -1)

    return update_fn


class EGNN(nn.Module):
    """E(n) Equivariant Graph Neural Network (EGNN) following Satorras et al (2021; https://arxiv.org/abs/2102.09844)."""

    message_passing_steps: int = 3
    d_hidden: int = 64
    n_layers: int = 3
    skip_connections: bool = False
    activation: str = "gelu"
    use_fourier_features: bool = True
    fourier_feature_kwargs = {"num_encodings": 16, "include_self": True}
    positions_only: bool = False

    @nn.compact
    def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Do message passing on graph

        Args:
            graphs (jraph.GraphsTuple): graph object

        Returns:
            jraph.GraphsTuple: updated graph object
        """

        processed_graphs = graphs

        if processed_graphs.globals is not None:
            processed_graphs = processed_graphs._replace(globals=processed_graphs.globals.reshape(1, -1))

        activation = getattr(nn, self.activation)

        # Apply message-passing rounds
        for _ in range(self.message_passing_steps):
            # Node update function
            update_node_fn = get_node_mlp_updates(self.d_hidden, self.n_layers, activation, n_edge=processed_graphs.n_edge, position_only=self.positions_only)

            # Edge update function
            update_edge_fn = get_edge_mlp_updates(self.d_hidden, self.n_layers, activation, position_only=self.positions_only, use_fourier_features=self.use_fourier_features, fourier_feature_kwargs=self.fourier_feature_kwargs)

            # Get jraph graph network
            graph_net = jraph.GraphNetwork(update_node_fn=update_node_fn, update_edge_fn=update_edge_fn)

            # Apply updates, optionally with skip connections
            if self.skip_connections:
                processed_graphs = add_graphs_tuples(graph_net(processed_graphs), processed_graphs)
            else:
                processed_graphs = graph_net(processed_graphs)

        # Return updated graph
        return processed_graphs
