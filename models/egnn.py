import jax
from typing import Callable, List
import flax.linen as nn
import jax.numpy as jnp
import jraph
from jraph._src import utils
import e3nn_jax as e3nn

from models.mlp import MLP


def get_edge_mlp_updates(
    d_hidden,
    n_layers,
    activation,
    position_only=False,
    tanh_out=False,
    soft_edges=False,
    apply_pbc=None,
    n_radial_basis=32,
    r_max=0.3,
) -> Callable:
    def update_fn(
        edges: jnp.array, senders: jnp.array, receivers: jnp.array, globals: jnp.array
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
        phi_e = MLP(
            [d_hidden] * (n_layers - 1) + [d_hidden],
            activation=activation,
            activate_final=True,
        )

        # Super special init for last layer of position MLP (from https://github.com/lucidrains/egnn-pytorch)
        phi_x = MLP(
            [d_hidden] * (n_layers - 1), activation=activation, activate_final=True
        )
        phi_x_last_layer = nn.Dense(
            1,
            use_bias=False,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=1e-2, mode="fan_in", distribution="uniform"
            ),
        )

        # Relative distances, optionally with Fourier features
        EPS = 1e-7
        r_ij = x_i - x_j + EPS
        if apply_pbc:
            r_ij = apply_pbc(r_ij)

        d_ij = jnp.sqrt(jnp.sum(r_ij ** 2, axis=1, keepdims=False))
        if n_radial_basis > 0:
            d_ij = e3nn.bessel(d_ij, n_radial_basis, r_max)
        d_ij = d_ij.reshape(d_ij.shape[0], -1)
                
        # Get invariants
        message_scalars = d_ij

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

        x_ij_trans = r_ij * trans
        x_ij_trans = jnp.clip(x_ij_trans, -100.0, 100.0)  # From original code

        return x_ij_trans, m_ij

    return update_fn


def get_node_mlp_updates(
    d_hidden,
    n_layers,
    activation,
    position_only=False,
    decouple_pos_vel_updates=False,
    apply_pbc=None
) -> Callable:
    def update_fn(
        nodes: jnp.array, senders: jnp.array, receivers: jnp.array, globals: jnp.array
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

        if position_only:  # Positions only; no velocities
            if nodes.shape[-1] == 3:  # No scalar attributes
                x_i = nodes
                x_i_p = x_i + sum_x_ij
                if apply_pbc:
                    x_i_p = apply_pbc(x_i_p)
                return x_i_p
            else:  # Additional scalar attributes
                x_i, h_i = nodes[..., :3], nodes[..., 3:]
                x_i_p = x_i + sum_x_ij
                if apply_pbc:
                    x_i_p = apply_pbc(x_i_p)
                phi_h = MLP(
                    [d_hidden] * (n_layers - 1) + [h_i.shape[-1]], activation=activation
                )
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
                    phi_xx = MLP(
                        [d_hidden] * (n_layers - 1) + [1], activation=activation
                    )
                    x_i_p = phi_xx(concats) * x_i + v_i_p
                else:
                    # Assumes dynamical system with coupled position and velocity updates, as in original paper!
                    x_i_p = x_i + v_i_p

                if apply_pbc:
                    x_i_p = apply_pbc(x_i_p)

                return jnp.concatenate([x_i_p, v_i, concats], -1)
            else:  # Additional scalar attributes
                x_i, v_i, h_i = (
                    nodes[:, :3],
                    nodes[:, 3:6],
                    nodes[:, 6:],
                )  # Split node attrs

                # From Eqs. (6) and (7)
                phi_v = MLP([d_hidden] * (n_layers - 1) + [1], activation=activation)
                phi_h = MLP(
                    [d_hidden] * (n_layers - 1) + [h_i.shape[-1]], activation=activation
                )

                concats = h_i
                if globals is not None:
                    concats = jnp.concatenate([concats, globals], -1)

                # Apply updates
                v_i_p = phi_v(concats) * v_i + sum_x_ij

                if decouple_pos_vel_updates:
                    # Decouple position and velocity updates
                    phi_xx = MLP(
                        [d_hidden] * (n_layers - 1) + [1], activation=activation
                    )
                    x_i_p = phi_xx(concats) * x_i + v_i_p
                else:
                    # Assumes dynamical system with coupled position and velocity updates, as in original paper!
                    x_i_p = x_i + v_i_p

                if apply_pbc:
                    x_i_p = apply_pbc(x_i_p)
                h_i_p = h_i + phi_h(concats)  # Skip connection, as in original paper

                return jnp.concatenate([x_i_p, v_i, h_i_p], -1)

    return update_fn


class EGNN(nn.Module):
    """E(n) Equivariant Graph Neural Network (EGNN) following Satorras et al (2021; https://arxiv.org/abs/2102.09844)."""

    # Attributes for all MLPs
    message_passing_steps: int = 3
    d_hidden: int = 128
    n_layers: int = 3
    activation: str = "gelu"
    soft_edges: bool = True  # Scale edges by a learnable function
    positions_only: bool = False  # (pos, vel, scalars) vs (pos, scalars)
    tanh_out: bool = False
    n_radial_basis: int = 16  # Number of radial basis functions for distance
    r_max: float = 0.3  # Maximum distance for radial basis functions
    decouple_pos_vel_updates: bool = False  # Use extra MLP to decouple position and velocity updates
    message_passing_agg: str = "sum"  # "sum", "mean", "max"
    readout_agg: str = "mean"
    mlp_readout_widths: List[int] = (4, 2, 2)  # Factor of d_hidden for global readout MLPs
    task: str = "graph"  # "graph" or "node"
    n_outputs: int = 1  # Number of outputs for graph-level readout
    apply_pbc: Callable = None

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
            processed_graphs = processed_graphs._replace(
                globals=processed_graphs.globals.reshape(1, -1)
            )

        activation = getattr(nn, self.activation)

        if self.message_passing_agg not in ["sum", "mean", "max"]:
            raise ValueError(
                f"Invalid message passing aggregation function {self.message_passing_agg}"
            )

        aggregate_edges_for_nodes_fn = getattr(
            utils, f"segment_{self.message_passing_agg}"
        )

        # Apply message-passing rounds
        for _ in range(self.message_passing_steps):
            # Node and edge update functions
            update_node_fn = get_node_mlp_updates(
                self.d_hidden,
                self.n_layers,
                activation,
                position_only=self.positions_only,
                decouple_pos_vel_updates=self.decouple_pos_vel_updates,
                apply_pbc=self.apply_pbc
            )
            update_edge_fn = get_edge_mlp_updates(
                self.d_hidden,
                self.n_layers,
                activation,
                position_only=self.positions_only,
                tanh_out=self.tanh_out,
                soft_edges=self.soft_edges,
                apply_pbc=self.apply_pbc,
                n_radial_basis=self.n_radial_basis,
                r_max=self.r_max
            )

            # Instantiate graph network and apply EGCL
            graph_net = jraph.GraphNetwork(
                update_node_fn=update_node_fn,
                update_edge_fn=update_edge_fn,
                aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn,
            )
            processed_graphs = graph_net(processed_graphs)

        if self.task == "node":
            return processed_graphs

        elif self.task == "graph":
            # Aggregate residual node features; only use positions, optionally

            if self.readout_agg not in ["sum", "mean", "max"]:
                raise ValueError(
                    f"Invalid global aggregation function {self.message_passing_agg}"
                )

            readout_agg_fn = getattr(jnp, f"{self.readout_agg}")
            agg_nodes = readout_agg_fn(processed_graphs.nodes, axis=0)
                
            if processed_graphs.globals is not None:
                agg_nodes = jnp.concatenate([agg_nodes, processed_graphs.globals]) #use tpcf
                
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
