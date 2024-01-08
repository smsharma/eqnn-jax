from typing import Optional, Tuple, Union, List
import dataclasses

import jax
import jax.numpy as jnp
import flax.linen as nn

import jraph
from jraph._src import utils

import e3nn_jax as e3nn
from e3nn_jax import Irreps
from e3nn_jax import IrrepsArray
from e3nn_jax import tensor_product
from e3nn_jax.flax import Linear

from models.mlp import MLP
from models.utils.irreps_utils import balanced_irreps
from models.utils.graph_utils import apply_pbc


class TensorProductLinearGate(nn.Module):
    output_irreps: Irreps = None
    bias: bool = True
    gradient_normalization: Optional[Union[str, float]] = "element"
    path_normalization: Optional[Union[str, float]] = "element"
    gate_activation: bool = True
    act_scalars: str = "gelu"
    act_gates: str = "sigmoid"

    @nn.compact
    def __call__(self, x: IrrepsArray, y: IrrepsArray) -> IrrepsArray:
        output_irreps = self.output_irreps
        if not isinstance(output_irreps, Irreps):
            output_irreps = Irreps(output_irreps)

        # Predict extra scalars for gating \ell > 0 irreps
        if self.gate_activation:
            gate_irreps = Irreps(f"{output_irreps.num_irreps - output_irreps.count('0e')}x0e")
            output_irreps = (gate_irreps + output_irreps).regroup()  # Contains extra scalars for gating

        # Linear + TP
        linear = Linear(output_irreps, biases=self.bias, gradient_normalization=self.gradient_normalization, path_normalization=self.path_normalization)
        out = linear(tensor_product(x, y))

        act_scalars = getattr(jax.nn, self.act_scalars)
        act_gates = getattr(jax.nn, self.act_gates)

        if self.gate_activation:
            out = e3nn.gate(out, even_act=act_scalars, even_gate_act=act_gates)  # Default activations

        return out


def get_edge_mlp_updates(
    irreps_out: Irreps = None,
    n_layers: int = 2,
    edge_attrs: Optional[Irreps] = None,
    act_scalars: str = "gelu",
    act_gates: str = "sigmoid",
):
    def update_fn(edges: jnp.array, senders: jnp.array, receivers: jnp.array, globals: jnp.array) -> jnp.array:
        m_ij = e3nn.concatenate([senders, receivers], axis=-1)  # Messages
        a_ij = edge_attrs  # Attributes

        # Gated tensor product steered by geometric features attributes
        for _ in range(n_layers - 1):
            m_ij = TensorProductLinearGate(irreps_out, act_scalars=act_scalars, act_gates=act_gates)(m_ij, a_ij)
        m_ij = TensorProductLinearGate(irreps_out, gate_activation=False)(m_ij, a_ij)  # No activation

        # Return messages
        return m_ij

    return update_fn


def get_node_mlp_updates(
    irreps_out: Irreps = None,
    n_layers: int = 2,
    n_edges: int = 1,
    normalize_messages: bool = True,
    node_attrs: Optional[Irreps] = None,
):
    def update_fn(nodes: jnp.array, senders: jnp.array, receivers: jnp.array, globals: jnp.array) -> jnp.array:
        m_i = receivers
        a_i = node_attrs
        if normalize_messages:
            m_i /= n_edges - 1
            a_i /= n_edges - 1

        m_i = e3nn.concatenate([m_i, a_i], axis=-1)  # Eq. 8 of 2110.02905

        # Gated tensor product steered by geometric feature messages
        for _ in range(n_layers - 1):
            nodes = TensorProductLinearGate(irreps_out)(nodes, m_i)
        nodes = TensorProductLinearGate(irreps_out, gate_activation=False)(nodes, m_i)  # No activation

        return nodes

    return update_fn


class SEGNN(nn.Module):
    d_hidden: int = 64  # Hidden dimension
    l_max_hidden: int = 1  # Maximum spherical harmonic degree for hidden features
    l_max_attr: int = 1  # Maximum spherical harmonic degree for steerable geometric features
    sphharm_norm: str = "component"  # Normalization for spherical harmonics; "component", "integral", "norm"
    irreps_out: Optional[Irreps] = None  # Output irreps for node-wise task; defaults to input irreps
    use_vel_attrs: bool = False  # Use velocity as steerable attribute
    normalize_messages: bool = True  # Normalize messages by number of edges
    num_message_passing_steps: int = 3  # Number of message passing steps
    num_blocks: int = 3  # Number of gated tensor products in each message passing step
    residual: bool = True  # Residual connections
    use_pbc: bool = False  # Use periodic boundary conditions when computing relative position vectors
    message_passing_agg: str = "mean"  # "sum", "mean", "max"
    readout_agg: str = "mean"  # "sum", "mean", "max"
    mlp_readout_widths: List[int] = (4, 2)  # Factor of d_hidden for global readout MLPs
    task: str = "node"  # "graph" or "node"
    n_outputs: int = 1  # Number of outputs for graph-level readout
    norm_dict: dict = dataclasses.field(default_factory=lambda: {"mean": 0.0, "std": 1.0})  # Normalization dictionary for relative position vectors
    unit_cell: Optional[jnp.array] = None  # Unit cell for applying periodic boundary conditions; should be compatible with norm_dict
    intermediate_hidden_irreps: bool = False  # Use hidden irreps for intermediate message passing steps; otherwise use input irreps
    act_scalars: str = "gelu"  # Activation function for scalars
    act_gates: str = "sigmoid"  # Activation function for gate scalars

    @nn.compact
    def __call__(self, graphs: jraph.GraphsTuple, node_attrs: Optional[Irreps] = None, edge_attrs: Optional[Irreps] = None) -> jraph.GraphsTuple:
        # Compute irreps
        irreps_attr = Irreps.spherical_harmonics(self.l_max_attr)  # For steerable geometric features
        irreps_hidden = balanced_irreps(lmax=self.l_max_hidden, feature_size=self.d_hidden, use_sh=True)  # For hidden features
        irreps_in = graphs.nodes.irreps  # Input irreps
        irreps_out = self.irreps_out if self.irreps_out is not None else irreps_in  # Output irreps desired, if different from input irreps

        # Distance vector; distance embedding is always used as edge attribute, and doesn't change between message-passing steps
        x_i, x_j = graphs.nodes.slice_by_mul[:1][graphs.senders], graphs.nodes.slice_by_mul[:1][graphs.receivers]  # Get position coordinates
        r_ij = x_i - x_j  # Relative position vector

        # Optionally apply PBC; unnormalize, PBC, then normalize back
        if self.use_pbc:
            r_ij = r_ij.array * self.norm_dict["std"][None, :3]
            r_ij = apply_pbc(r_ij, self.unit_cell)
            r_ij = IrrepsArray("1o", r_ij / self.norm_dict["std"][None, :3])

        # Project relative distance vectors onto spherical harmonic basis and include as edge attribute
        a_r_ij = e3nn.spherical_harmonics(irreps_out=irreps_attr, input=r_ij, normalize=True, normalization=self.sphharm_norm)
        edge_attrs = a_r_ij

        # Aggregate distance embedding over neighbours to use as node attribute
        a_i = e3nn.scatter_sum(a_r_ij, dst=graphs.receivers, output_size=graphs.nodes.shape[0])

        # Append to any existing node attributes
        node_attrs = a_i if node_attrs is None else e3nn.concatenate([a_i, node_attrs], axis=-1)

        # Optionally use velocity as steerable node attribute
        if self.use_vel_attrs:
            v_i = graphs.nodes.slice_by_mul[1:2]
            a_v_i = e3nn.spherical_harmonics(irreps_out=irreps_attr, input=v_i, normalize=True, normalization=self.sphharm_norm)
            node_attrs = e3nn.concatenate([node_attrs, a_v_i], axis=-1)

        # If specified, use hidden irreps between message passing steps; otherwise, use input irreps (bottleneck and fewer parameters)
        irreps_intermediate = irreps_hidden if self.intermediate_hidden_irreps else irreps_in

        # Neighborhood aggregation function
        aggregate_edges_for_nodes_fn = getattr(utils, f"segment_{self.message_passing_agg}")

        # Message passing rounds
        for step in range(self.num_message_passing_steps):
            update_edge_fn = get_edge_mlp_updates(
                irreps_out=irreps_hidden,
                n_layers=self.num_blocks,
                edge_attrs=edge_attrs,
                act_scalars=self.act_scalars,
                act_gates=self.act_gates,
            )
            update_node_fn = get_node_mlp_updates(
                irreps_out=irreps_intermediate,  # Node update outputs to `irreps_intermediate`
                n_layers=self.num_blocks,
                normalize_messages=self.normalize_messages,
                n_edges=graphs.n_edge,
                node_attrs=node_attrs,
            )

            # Apply steerable EGCL
            graph_net = jraph.GraphNetwork(update_node_fn=update_node_fn, update_edge_fn=update_edge_fn, aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn)
            processed_graphs = graph_net(graphs)

            # Skip connection
            if self.residual:
                graphs = processed_graphs._replace(nodes=processed_graphs.nodes + Linear(irreps_intermediate)(graphs.nodes))
            else:
                graphs = processed_graphs

        if self.task == "node":
            # If output irreps differ from input irreps, project to output irreps
            if irreps_out != irreps_in:
                graphs = graphs._replace(nodes=Linear(irreps_out)(graphs.nodes))
            return graphs
        elif self.task == "graph":
            # Aggregate residual node features
            if self.readout_agg not in ["sum", "mean", "max"]:
                raise ValueError(f"Invalid global aggregation function {self.message_passing_agg}")

            # Steerable linear layer conditioned on node attributes; output scalars for invariant readout
            irreps_pre_pool = Irreps(f"{self.d_hidden}x0e")
            readout_agg_fn = getattr(jnp, f"{self.readout_agg}")
            nodes_pre_pool = nn.Dense(self.d_hidden)(TensorProductLinearGate(irreps_pre_pool, gate_activation=False)(graphs.nodes, node_attrs).array)
            agg_nodes = readout_agg_fn(nodes_pre_pool, axis=0)

            # Readout and return
            out = MLP([w * self.d_hidden for w in self.mlp_readout_widths] + [self.n_outputs])(agg_nodes)
            return out

        else:
            raise ValueError(f"Invalid task {self.task}")
