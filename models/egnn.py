import jax
from typing import Callable, List
import flax.linen as nn
import jax.numpy as jnp
import jraph
from jraph._src import utils

from utils.graph_utils import fourier_features
from models.mlp import MLP


def get_edge_mlp_updates(d_hidden, n_layers, activation, position_only=False, use_fourier_features=False, fourier_feature_kwargs={"num_encodings": 16, "include_self": True}, tanh_out=False, soft_edges=False) -> Callable:

    def update_fn(edges: jnp.array, senders: jnp.array, receivers: jnp.array, globals: jnp.array) -> jnp.array:
        """update edge features

        Args:
            edges (jnp.ndarray): edge attributes
            senders (jnp.ndarray): senders node attributes
            receivers (jnp.ndarray): receivers node attributes
            globals (jnp.ndarray): global features

        Returns:
            jnp.ndarray: updated edge features
        """

        if position_only:  # Positions only; no velocities
            if senders.shape[-1] == 3:
                x_i = senders
                x_j = receivers
                concats = globals
            else:
                x_i, h_i = senders[:, :3], senders[:, 3:]
                x_j, h_j = receivers[:, :3], receivers[:, 3:]
                concats = jnp.concatenate([h_i, h_j], -1)
                if globals is not None:
                    concats = jnp.concatenate([concats, globals], -1)

        else:  # Positions and velocities
            if senders.shape[-1] == 6:
                x_i, v_i = senders[:, :3], senders[:, 3:6]
                x_j, v_j = receivers[:, :3], receivers[:, 3:6]
                concats = globals
            else:
                x_i, v_i, h_i = senders[:, :3], senders[:, 3:6], senders[:, 6:]
                x_j, v_j, h_j = receivers[:, :3], receivers[:, 3:6], receivers[:, 6:]
                concats = jnp.concatenate([h_i, h_j], -1)
                if globals is not None:
                    concats = jnp.concatenate([concats, globals], -1)

        # Messages from Eqs. (3) and (4)/(7)
        phi_e = MLP([d_hidden] * (n_layers - 1) + [d_hidden], activation=activation, activate_final=True)

        # Super special init for last layer of position MLP (from https://github.com/lucidrains/egnn-pytorch)
        phi_x = MLP([d_hidden] * (n_layers - 1), activation=activation, activate_final=True)
        phi_x_last_layer = nn.Dense(1, use_bias=False, kernel_init=jax.nn.initializers.variance_scaling(scale=1e-2, mode="fan_in", distribution="uniform"))

        # Relative distances, optionally with Fourier features
        d_ij2 = jnp.sum((x_i - x_j) ** 2, axis=1, keepdims=True)
        d_ij2 = fourier_features(d_ij2, **fourier_feature_kwargs) if use_fourier_features else d_ij2

        # Get invariants
        message_scalars = d_ij2

        if concats is not None:
            message_scalars = jnp.concatenate([message_scalars, concats], axis=-1)

        m_ij = phi_e(message_scalars)

        # Optionally apply soft_edges
        phi_a = MLP([d_hidden], activation=activation)
        if soft_edges:
            a_ij = phi_a(m_ij)
            a_ij = nn.sigmoid(a_ij)
            m_ij = m_ij * a_ij

        trans = phi_x_last_layer(phi_x(m_ij))

        # Optionally apply tanh to pre-factor to stabilize
        if tanh_out:
            trans = jax.nn.tanh(trans)

        x_ij_trans = (x_i - x_j) * trans
        x_ij_trans = jnp.clip(x_ij_trans, -100.0, 100.0)  # From original code

        return x_ij_trans, m_ij

    return update_fn


def get_node_mlp_updates(d_hidden, n_layers, activation, n_edges, position_only=False, normalize_messages=False, decouple_pos_vel_updates=False) -> Callable:

    def update_fn(nodes: jnp.array, senders: jnp.array, receivers: jnp.array, globals: jnp.array) -> jnp.array:
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

        if normalize_messages:
            sum_x_ij = sum_x_ij / (n_edges - 1)  # C = M - 1 as in original paper

        if position_only:  # Positions only; no velocities
            if nodes.shape[-1] == 3:  # No scalar attributes
                x_i = nodes
                x_i_p = x_i + sum_x_ij
                return x_i_p
            else:  # Additional scalar attributes
                x_i, h_i = nodes[..., :3], nodes[..., 3:]
                x_i_p = x_i + sum_x_ij
                phi_h = MLP([d_hidden] * (n_layers - 1) + [h_i.shape[-1]], activation=activation)
                concats = jnp.concatenate([h_i, m_i], -1)
                if globals is not None:
                    concats = jnp.concatenate([concats, globals], -1)
                h_i_p = h_i + phi_h(concats)
                return jnp.concatenate([x_i_p, h_i_p], -1)

        else:  # Positions and velocities
            if nodes.shape[-1] == 6:  # No scalar attributes
                x_i, v_i = nodes[:, :3], nodes[:, 3:6]  # Split node attrs

                # From Eqs. (6) and (7)
                phi_v = MLP([d_hidden] * (n_layers - 1) + [1], activation=activation)

                # Create some scalar attributes for use in future rounds
                # Necessary for stability
                concats = m_i
                if globals is not None:
                    concats = jnp.concatenate([concats, globals], -1)

                # Apply updates
                v_i_p = phi_v(concats) * v_i + sum_x_ij

                if decouple_pos_vel_updates:
                    # Decouple position and velocity updates
                    phi_xx = MLP([d_hidden] * (n_layers - 1) + [1], activation=activation)
                    x_i_p = phi_xx(concats) * x_i + v_i_p
                else:
                    # Assumes dynamical system with coupled position and velocity updates, as in original paper!
                    x_i_p = x_i + v_i_p

                return jnp.concatenate([x_i_p, v_i, concats], -1)
            else:  # Additional scalar attributes
                x_i, v_i, h_i = nodes[:, :3], nodes[:, 3:6], nodes[:, 6:]  # Split node attrs

                # From Eqs. (6) and (7)
                phi_v = MLP([d_hidden] * (n_layers - 1) + [1], activation=activation)
                phi_h = MLP([d_hidden] * (n_layers - 1) + [h_i.shape[-1]], activation=activation)

                concats = h_i
                if globals is not None:
                    concats = jnp.concatenate([concats, globals], -1)

                # Apply updates
                v_i_p = phi_v(concats) * v_i + sum_x_ij

                if decouple_pos_vel_updates:
                    # Decouple position and velocity updates
                    phi_xx = MLP([d_hidden] * (n_layers - 1) + [1], activation=activation)
                    x_i_p = phi_xx(concats) * x_i + v_i_p
                else:
                    # Assumes dynamical system with coupled position and velocity updates, as in original paper!
                    x_i_p = x_i + v_i_p

                h_i_p = h_i + phi_h(concats)  # Skip connection, as in original paper

                return jnp.concatenate([x_i_p, v_i, h_i_p], -1)

    return update_fn


class EGNN(nn.Module):
    """E(n) Equivariant Graph Neural Network (EGNN) following Satorras et al (2021; https://arxiv.org/abs/2102.09844)."""

    # Attributes for all MLPs
    message_passing_steps: int = 3
    d_hidden: int = 64
    n_layers: int = 3
    activation: str = "gelu"

    soft_edges: bool = False  # Scale edges by a learnable function
    use_fourier_features: bool = True
    fourier_feature_kwargs = {"num_encodings": 16, "include_self": True}
    positions_only: bool = False  # (pos, vel, scalars) vs (pos, scalars)
    tanh_out: bool = False
    normalize_messages: bool = True  # Divide sum_x_ij by (n_edges - 1)
    decouple_pos_vel_updates: bool = False  # Use extra MLP to decouple position and velocity updates

    message_passing_agg: str = "sum"  # "sum", "mean", "max"
    readout_agg: str = "mean"
    mlp_readout_widths: List[int] = (8, 2)  # Factor of d_hidden for global readout MLPs
    task: str = "graph"  # "graph" or "node"
    readout_only_positions: bool = False  # Graph-level readout only uses positions; otherwise use all features
    n_outputs: int = 1  # Number of outputs for graph-level readout

    @nn.compact
    def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Apply equivariant graph convolutional layers to graph

        Args:
            graphs (jraph.GraphsTuple): Input graph

        Returns:
            jraph.GraphsTuple: Updated graph
        """

        processed_graphs = graphs

        if processed_graphs.globals is not None:
            processed_graphs = processed_graphs._replace(globals=processed_graphs.globals.reshape(1, -1))

        activation = getattr(nn, self.activation)

        if self.message_passing_agg not in ["sum", "mean", "max"]:
            raise ValueError(f"Invalid message passing aggregation function {self.message_passing_agg}")

        aggregate_edges_for_nodes_fn = getattr(utils, f"segment_{self.message_passing_agg}")

        # Apply message-passing rounds
        for _ in range(self.message_passing_steps):
            # Node and edge update functions
            update_node_fn = get_node_mlp_updates(
                self.d_hidden, self.n_layers, activation, n_edges=processed_graphs.n_edge, position_only=self.positions_only, normalize_messages=self.normalize_messages, decouple_pos_vel_updates=self.decouple_pos_vel_updates
            )
            update_edge_fn = get_edge_mlp_updates(
                self.d_hidden, self.n_layers, activation, position_only=self.positions_only, use_fourier_features=self.use_fourier_features, fourier_feature_kwargs=self.fourier_feature_kwargs, tanh_out=self.tanh_out, soft_edges=self.soft_edges
            )

            # Instantiate graph network and apply EGCL
            graph_net = jraph.GraphNetwork(update_node_fn=update_node_fn, update_edge_fn=update_edge_fn, aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn)

            processed_graphs = graph_net(processed_graphs)

        if self.task == "node":
            return processed_graphs

        elif self.task == "graph":
            # Aggregate residual node features; only use positions, optionally

            if self.readout_agg not in ["sum", "mean", "max"]:
                raise ValueError(f"Invalid global aggregation function {self.message_passing_agg}")

            readout_agg_fn = getattr(jnp, f"{self.readout_agg}")
            if self.readout_only_positions:
                agg_nodes = readout_agg_fn(processed_graphs.nodes[:, :3], axis=0)
            else:
                agg_nodes = readout_agg_fn(processed_graphs.nodes, axis=0)

            # Readout and return
            out = MLP([w * self.d_hidden for w in self.mlp_readout_widths] + [self.n_outputs])(agg_nodes)
            return out

        else:
            raise ValueError(f"Invalid task {self.task}")
