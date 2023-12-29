# Linear on nodes
# TP + aggregate
# divide by average number of neighbors
# Concatenation
# Linear on nodes
# Self-connection
# Gate

from typing import Optional, Tuple, Union, List
import dataclasses

import jax.numpy as jnp
import flax.linen as nn

import jraph
from jraph._src import utils

import e3nn_jax as e3nn
from e3nn_jax import Irreps
from e3nn_jax import IrrepsArray
from e3nn_jax import tensor_product
from e3nn_jax.flax import Linear, MultiLayerPerceptron

from models.mlp import MLP
from models.utils.irreps_utils import balanced_irreps
from models.utils.graph_utils import apply_pbc


def get_edge_mlp_updates(
    irreps_out: Irreps = None,
    irreps_attr: Irreps = None,
    sphharm_norm: str = "component",
    n_layers: int = 2,
    d_hidden: int = 64,
    n_radial_basis: int = 4,
):
    # irreps_out = irreps_out.regroup()

    def update_fn(edges: jnp.array, senders: jnp.array, receivers: jnp.array, globals: jnp.array) -> jnp.array:
        x_i, x_j = senders.slice_by_mul[:1], receivers.slice_by_mul[:1]  # Get position coordinates
        r_ij = x_i - x_j  # Relative position vector

        d_ij = jnp.linalg.norm(r_ij.array, axis=-1)  # Distance

        m_ij = Linear(senders.irreps)(senders)

        # Angular
        a_ij = e3nn.spherical_harmonics(irreps_out=irreps_attr, input=r_ij, normalize=True, normalization=sphharm_norm)
        m_ij = e3nn.concatenate([m_ij, a_ij])

        tp = tensor_product(m_ij, a_ij)
        R = e3nn.bessel(d_ij, n_radial_basis)

        mix = MultiLayerPerceptron(n_layers * (d_hidden,) + (tp.irreps.num_irreps,), nn.gelu, output_activation=False)(R)

        mix = mix * tp

        # Return messages and also relative positions (to use as node attributes)
        return mix

    return update_fn


def get_node_mlp_updates(irreps_out: Irreps = None):
    def update_fn(nodes: jnp.array, senders: jnp.array, receivers: jnp.array, globals: jnp.array) -> jnp.array:
        m_i = receivers
        irreps = irreps_out.filter(keep=m_i.irreps)
        gate_irreps = Irreps(f"{irreps.num_irreps - irreps.count('0e')}x0e")
        _irreps_out = (gate_irreps + irreps).regroup()
        m_i = Linear(_irreps_out)(m_i) + Linear(_irreps_out)(nodes)  # Skip
        nodes = e3nn.gate(m_i)
        return nodes

    return update_fn


class NequIP(nn.Module):
    d_hidden: int = 64  # Hidden dimension
    l_max_hidden: int = 1  # Maximum spherical harmonic degree for hidden features
    l_max_attr: int = 1  # Maximum spherical harmonic degree for steerable geometric features
    sphharm_norm: str = "component"  # Normalization for spherical harmonics; "component", "integral", "norm"
    irreps_out: Optional[Irreps] = None  # Output irreps; defaults to input irreps
    normalize_messages: bool = True  # Normalize messages by number of edges
    num_message_passing_steps: int = 3  # Number of message passing steps
    n_layers: int = 3  # Number of gated tensor products in each message passing step
    message_passing_agg: str = "mean"  # "sum", "mean", "max"
    readout_agg: str = "mean"  # "sum", "mean", "max"
    mlp_readout_widths: List[int] = (4, 2)  # Factor of d_hidden for global readout MLPs
    task: str = "node"  # "graph" or "node"
    readout_only_positions: bool = True  # Graph-level readout only uses positions; otherwise use all features
    n_outputs: int = 1  # Number of outputs for graph-level readout
    n_radial_basis: int = 4  # Number of radial basis functions

    @nn.compact
    def __call__(self, graphs: jraph.GraphsTuple, node_attrs: Optional[Irreps] = None, edge_attrs: Optional[Irreps] = None) -> jraph.GraphsTuple:
        aggregate_edges_for_nodes_fn = getattr(utils, f"segment_{self.message_passing_agg}")

        # Compute irreps
        irreps_attr = Irreps.spherical_harmonics(self.l_max_attr)  # Steerable geometric features
        irreps_hidden = balanced_irreps(lmax=self.l_max_hidden, feature_size=self.d_hidden, use_sh=True)  # Hidden features
        irreps_in = graphs.nodes.irreps  # Input irreps
        irreps_out = self.irreps_out if self.irreps_out is not None else irreps_in  # Output irreps, if different from input irreps

        # Message passing rounds
        for _ in range(self.num_message_passing_steps):
            update_edge_fn = get_edge_mlp_updates(
                irreps_out=irreps_hidden,
                d_hidden=self.d_hidden,
                n_layers=self.n_layers,
                irreps_attr=irreps_attr,
                sphharm_norm=self.sphharm_norm,
                n_radial_basis=self.n_radial_basis,
            )
            update_node_fn = get_node_mlp_updates(
                irreps_out=irreps_in,
            )

            # Apply steerable EGCL
            graph_net = jraph.GraphNetwork(update_node_fn=update_node_fn, update_edge_fn=update_edge_fn, aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn)
            processed_graphs = graph_net(graphs)

            # Project to original irreps
            nodes = Linear(irreps_in)(processed_graphs.nodes)

            # Update graph
            graphs = processed_graphs._replace(nodes=nodes)

        # If output irreps differ from input irreps, project to output irreps
        if irreps_out != irreps_in:
            graphs = graphs._replace(nodes=Linear(irreps_out)(graphs.nodes))

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
            out = MLP([w * Irreps(irreps_hidden).num_irreps for w in self.mlp_readout_widths] + [self.n_outputs])(agg_nodes)
            return out

        else:
            raise ValueError(f"Invalid task {self.task}")