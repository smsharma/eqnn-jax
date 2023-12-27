from typing import Optional, Tuple, Union, List

import jax.numpy as jnp
import flax.linen as nn

import jraph
from jraph._src import utils

import e3nn_jax as e3nn
from e3nn_jax import Irreps
from e3nn_jax import IrrepsArray
from e3nn_jax import tensor_product
from e3nn_jax.flax import Linear, BatchNorm

from models.mlp import MLP

EPS = 1e-4


class TensorProductLinearGate(nn.Module):
    output_irreps: Irreps = None
    bias: bool = True
    gradient_normalization: Optional[Union[str, float]] = "element"
    path_normalization: Optional[Union[str, float]] = "element"
    gate_activation: bool = True

    @nn.compact
    def __call__(self, x: IrrepsArray, y: IrrepsArray) -> IrrepsArray:
        output_irreps = self.output_irreps
        if not isinstance(output_irreps, Irreps):
            output_irreps = Irreps(output_irreps)

        # Predict extra scalars for gating \ell > 0 irreps
        if self.gate_activation:
            gate_irreps = Irreps(f"{output_irreps.num_irreps - output_irreps.count('0e')}x0e")
            output_irreps = (gate_irreps + output_irreps).regroup()

        linear = Linear(output_irreps, biases=self.bias, gradient_normalization=self.gradient_normalization, path_normalization=self.path_normalization)
        out = linear(tensor_product(x, y))

        if self.gate_activation:
            out = e3nn.gate(out)  # Default activations

        return out

def get_edge_mlp_updates(irreps_out: Irreps = None, n_layers: int = 2, irreps_sh: Irreps = None):
    def update_fn(
        edges: jnp.array,
        senders: jnp.array,
        receivers: jnp.array,
        globals: jnp.array,
    ) -> jnp.array:
        x_i, x_j = senders.slice_by_mul[:1], receivers.slice_by_mul[:1]  # Get position coordinates
        # m_i, m_j = senders.slice_by_mul[2:].array, receivers.slice_by_mul[2:].array  # Get masses
        r_ij = x_i - x_j  # Relative position vector
        d_ij = jnp.linalg.norm(r_ij.array, axis=-1)

        # F_ij = m_i * m_j / d_ij**3 * r_ij  # Coulomb force

        # F_tilde_ij = e3nn.spherical_harmonics(irreps_out=irreps_sh, input=F_ij, normalize=True, normalization="component")  # Project onto spherical harmonic basis
        a_ij = e3nn.spherical_harmonics(irreps_out=irreps_sh, input=r_ij, normalize=True, normalization="component")  # Project onto spherical harmonic basis
        # a_ij = e3nn.concatenate([a_ij, F_tilde_ij], axis=-1)  # Concatenate with Coulomb force

        # cutoff = .1
        # d_ij = d_ij / cutoff
        # a_ij = e3nn.concatenate([e3nn.bessel(d_ij, 4)  * e3nn.soft_envelope(d_ij)[:, None], a_ij], axis=-1)  # Concatenate with radial basis

        m_ij = e3nn.concatenate([senders, receivers], axis=-1)  # Messages

        # Gated tensor product steered by geometric features
        for _ in range(n_layers):
            m_ij = TensorProductLinearGate(irreps_out)(m_ij, a_ij)

        return m_ij, a_ij

    return update_fn


def get_node_mlp_updates(irreps_out: Irreps = None, n_layers: int = 2, irreps_sh: Irreps = None, n_edges: int = 1, normalize_messages: bool = True):
    def update_fn(
        nodes: jnp.array,
        senders: jnp.array,
        receivers: jnp.array,
        globals: jnp.array,
    ) -> jnp.array:
        m_i, a_i = receivers
        if normalize_messages:
            m_i, a_i = m_i / (n_edges - 1), a_i / (n_edges - 1)

        m_i = e3nn.concatenate([m_i, a_i], axis=-1)

        # # Include velocity as steerable feature
        # v_i = nodes.slice_by_mul[1:2]
        # v_tilde_i = e3nn.spherical_harmonics(irreps_out=irreps_sh, input=v_i, normalize=True, normalization="component")
        # m_i = e3nn.concatenate([m_i, v_tilde_i], axis=-1)

        for _ in range(n_layers):
            nodes = TensorProductLinearGate(irreps_out)(nodes, m_i)
        return nodes

    return update_fn


class SEGNN(nn.Module):
    num_message_passing_steps: int = 3
    num_blocks: int = 3
    irreps_hidden: Irreps = Irreps("0e")
    irreps_sh: Irreps = Irreps("0e")
    normalize_messages: bool = True

    message_passing_agg: str = "mean"  # "sum", "mean", "max"
    readout_agg: str = "mean"
    mlp_readout_widths: List[int] = (4, 2)  # Factor of d_hidden for global readout MLPs
    task: str = "node"  # "graph" or "node"
    readout_only_positions: bool = True  # Graph-level readout only uses positions; otherwise use all features
    n_outputs: int = 1  # Number of outputs for graph-level readout

    @nn.compact
    def __call__(self, graphs: jraph.GraphsTuple) -> jraph.GraphsTuple:
        aggregate_edges_for_nodes_fn = getattr(utils, f"segment_{self.message_passing_agg}")

        irreps_in = graphs.nodes.irreps
        for _ in range(self.num_message_passing_steps):

            update_edge_fn = get_edge_mlp_updates(irreps_out=self.irreps_hidden, n_layers=self.num_blocks, irreps_sh=self.irreps_sh)
            update_node_fn = get_node_mlp_updates(irreps_out=irreps_in, n_layers=self.num_blocks, irreps_sh=self.irreps_sh, normalize_messages=self.normalize_messages, n_edges=graphs.n_edge)

            graph_net = jraph.GraphNetwork(update_node_fn=update_node_fn, update_edge_fn=update_edge_fn, aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn)
            processed_graphs = graph_net(graphs)

            nodes = Linear(irreps_in)(processed_graphs.nodes)
            # nodes = BatchNorm(instance=True)(nodes)

            # Project to input irreps for good measure
            graphs = processed_graphs._replace(nodes=nodes)



        if self.task == "node":
            return graphs

        elif self.task == "graph":
            # Aggregate residual node features; only use positions, optionally

            if self.readout_agg not in ["sum", "mean", "max"]:
                raise ValueError(f"Invalid global aggregation function {self.message_passing_agg}")

            readout_agg_fn = getattr(jnp, f"{self.readout_agg}")
            if self.readout_only_positions:
                agg_nodes = readout_agg_fn(graphs.nodes.slice_by_mul[:1].array, axis=0)
            else:
                agg_nodes = readout_agg_fn(graphs.nodes.array, axis=0)

            # Readout and return
            out = MLP([w * Irreps(self.irreps_hidden).num_irreps for w in self.mlp_readout_widths] + [self.n_outputs])(agg_nodes)
            return out

        else:
            raise ValueError(f"Invalid task {self.task}")
