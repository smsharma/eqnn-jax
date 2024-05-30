# Linear on nodes
# TP + aggregate
# Divide by average number of neighbors
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


def get_edge_mlp_updates(
    rel_distance: IrrepsArray,
    irreps_attr: Irreps = None,
    sphharm_norm: str = "component",
    n_layers: int = 2,
    d_hidden: int = 64,
    n_radial_basis: int = 4,
    r_cutoff: float = 1.0,
    activation: str = "gelu",
):
    
    def update_fn(
        edges: jnp.array, senders: IrrepsArray, receivers: IrrepsArray, globals: jnp.array
    ) -> jnp.array:
        
        m_ij = Linear(senders.irreps)(senders)

        # Angular
        a_ij = e3nn.spherical_harmonics(
            irreps_out=irreps_attr,
            input=rel_distance,
            normalize=True,
            normalization=sphharm_norm,
        )
        m_ij = e3nn.concatenate([m_ij, a_ij])
        m_ij = tensor_product(m_ij, a_ij)

        # Radial
        d_ij = jnp.linalg.norm(rel_distance.array, axis=-1)
        if n_radial_basis > 0:
            R_ij = e3nn.bessel(d_ij, n_radial_basis, r_cutoff)
        else:
            R_ij = d_ij[...,None]

        activation_nn = getattr(nn, activation)
        W_R_ij = MultiLayerPerceptron(
            n_layers * (d_hidden,) + (m_ij.irreps.num_irreps,),
            activation_nn,
            output_activation=False,
        )(R_ij)

        m_ij = W_R_ij * m_ij

        # Return messages and also relative positions (to use as node attributes)
        return m_ij

    return update_fn


def get_node_mlp_updates(irreps_out: Irreps = None, n_edges: int = 20, residual: bool = True):
    def update_fn(
        nodes: jnp.array, senders: jnp.array, receivers: jnp.array, globals: jnp.array
    ) -> jnp.array:
        m_i = receivers / jnp.sqrt(n_edges)
        irreps = irreps_out.filter(keep=m_i.irreps)
        gate_irreps = Irreps(f"{irreps.num_irreps - irreps.count('0e')}x0e")
        _irreps_out = (gate_irreps + irreps).regroup()
        if residual:
            m_i = Linear(_irreps_out)(m_i) + Linear(_irreps_out, force_irreps_out=True,)(nodes)  # Skip
        else:
            m_i = Linear(_irreps_out)(m_i) 
        nodes = e3nn.gate(
            m_i,
        )
        return nodes

    return update_fn


class NequIP(nn.Module):
    d_hidden: int = 128  # Hidden dimension
    n_layers: int = 3  # Number of gated tensor products in each message passing step
    message_passing_steps: int = 3  # Number of message passing steps
    message_passing_agg: str = "mean"  # "sum", "mean", "max"
    activation: str = "gelu"  # Activation function for MLPs
    task: str = "graph"  # "graph" or "node"
    n_outputs: int = 1  # Number of outputs for graph-level readout
    readout_agg: str = "mean"  # "sum", "mean", "max"
    mlp_readout_widths: List[int] = (4, 2, 2)  # Factor of d_hidden for global readout MLPs
    n_radial_basis: int = 4  # Number of radial basis (Bessel) functions
    r_cutoff: float = 1.0  # Cutoff radius for radial basis (Bessel) functions
    l_max: int = 1  # Maximum spherical harmonic degree for steerable geometric features and hidden features
    sphharm_norm: str = "component"  # Normalization for spherical harmonics; "component", "integral", "norm"
    irreps_out: Optional[Irreps] = None  # Output irreps; defaults to input irreps
    residual: bool = True

    @nn.compact
    def __call__(
        self,
        graphs: jraph.GraphsTuple,
    ) -> jraph.GraphsTuple:
        aggregate_edges_for_nodes_fn = getattr(
            utils, f"segment_{self.message_passing_agg}"
        )

        # Compute irreps
        irreps_attr = Irreps.spherical_harmonics(self.l_max)  # Steerable geometric features
        irreps_hidden = balanced_irreps(lmax=self.l_max, feature_size=self.d_hidden, use_sh=True)  # Hidden features # NOT USED
        irreps_in = graphs.nodes.irreps  # Input irreps
        irreps_out = (self.irreps_out if self.irreps_out is not None else irreps_in)  # Output irreps, if different from input irreps

        # Distance vector; distance embedding is always used as edge attribute
        x_i, x_j = (
            graphs.nodes.slice_by_mul[:1][graphs.senders],
            graphs.nodes.slice_by_mul[:1][graphs.receivers],
        )  # Get position coordinates

        r_ij = x_i - x_j  # Relative position vector

        # Message passing rounds
        for _ in range(self.message_passing_steps):
            update_edge_fn = get_edge_mlp_updates(
                rel_distance=r_ij,
                d_hidden=self.d_hidden,
                n_layers=self.n_layers,
                irreps_attr=irreps_attr,
                sphharm_norm=self.sphharm_norm,
                n_radial_basis=self.n_radial_basis,
                r_cutoff=self.r_cutoff
            )
            update_node_fn = get_node_mlp_updates(
                irreps_out=irreps_hidden,
                n_edges=graphs.n_edge,
                residual=self.residual,
            )

            # Apply steerable EGCL
            graph_net = jraph.GraphNetwork(
                update_node_fn=update_node_fn,
                update_edge_fn=update_edge_fn,
                aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn,
            )
            processed_graphs = graph_net(graphs)

            # Project to original irreps
            nodes = Linear(irreps_in)(processed_graphs.nodes)

            # Update graph
            graphs = processed_graphs._replace(nodes=nodes)

        if self.task == "node":
            # If output irreps differ from input irreps, project to output irreps
            if irreps_out != irreps_in:
                graphs = graphs._replace(nodes=Linear(irreps_out)(graphs.nodes))
            return graphs
        elif self.task == "graph":
            # Aggregate residual node features
            if self.readout_agg not in ["sum", "mean", "max"]:
                raise ValueError(
                    f"Invalid global aggregation function {self.message_passing_agg}"
                )

            # Aggregate distance embedding over neighbours to use as node attribute for readout
            a_ij = e3nn.spherical_harmonics(
                irreps_out=irreps_attr,
                input=r_ij,
                normalize=True,
                normalization=self.sphharm_norm,
            )
            node_attrs = (
                e3nn.scatter_sum(
                    a_ij, dst=graphs.receivers, output_size=graphs.nodes.shape[0]
                )
                / graphs.n_edge
            )

            # Steerable linear layer conditioned on node attributes; output scalars for invariant readout
            irreps_pre_pool = Irreps(f"{self.d_hidden}x0e")
            readout_agg_fn = getattr(jnp, f"{self.readout_agg}")
            agg_nodes = readout_agg_fn(
                Linear(irreps_pre_pool)(tensor_product(graphs.nodes, node_attrs)).array,
                axis=0,
            )

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
