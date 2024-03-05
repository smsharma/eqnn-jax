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
from models.utils.equivariant_graph_utils import SteerableGraphsTuple


class TensorProductLinearGate(nn.Module):
    output_irreps: Irreps = None
    bias: bool = True
    gradient_normalization: Optional[Union[str, float]] = "element"
    path_normalization: Optional[Union[str, float]] = "element"
    activation: bool = True
    scalar_activation: str = "silu"
    gate_activation: str = "sigmoid"

    @nn.compact
    def __call__(self, x: IrrepsArray, y: IrrepsArray) -> IrrepsArray:
        output_irreps = self.output_irreps
        if not isinstance(output_irreps, Irreps):
            output_irreps = Irreps(output_irreps)

        # Predict extra scalars for gating \ell > 0 irreps
        if self.activation: 
            gate_irreps = Irreps(
                f"{output_irreps.num_irreps - output_irreps.count('0e')}x0e"
            )
            output_irreps = (
                gate_irreps + output_irreps
            ).regroup()  # Contains extra scalars for gating

        # Linear + TP
        linear = Linear(
            output_irreps,
            biases=self.bias,
            gradient_normalization=self.gradient_normalization,
            path_normalization=self.path_normalization,
        )
        out = linear(tensor_product(x, y))
        if self.activation:
            scalar_activation = getattr(jax.nn, self.scalar_activation)
            gate_activation = getattr(jax.nn, self.gate_activation)
            # TODO: make sure even / odd resolved here
            out = e3nn.gate(
                out,
                scalar_activation,
                odd_gate_act=gate_activation,
            )  # Default activations
        return out


def get_edge_mlp_updates(
    output_irreps: Irreps = None,
    n_layers: int = 2,
    steerable_edge_attrs: Optional[Irreps] = None,
    additional_messages: Optional[jnp.array] = None,
    scalar_activation: str = "silu",
    gate_activation: str = "sigmoid",
):
    def update_fn(
        edges: jnp.array, senders: jnp.array, receivers: jnp.array, globals: jnp.array
    ) -> jnp.array:
        to_concat = [senders, receivers]
        if additional_messages is not None:
            to_concat.append(additional_messages)
        m_ij = e3nn.concatenate(to_concat, axis=-1)  # Messages
        # Gated tensor product steered by geometric features attributes
        for _ in range(n_layers - 1):
            m_ij = TensorProductLinearGate(
                output_irreps,
                scalar_activation=scalar_activation,
                gate_activation=gate_activation,
            )(m_ij, steerable_edge_attrs)
        m_ij = TensorProductLinearGate(output_irreps, activation=False)(
            m_ij, steerable_edge_attrs
        )  # No activation
        return m_ij

    return update_fn


def get_node_mlp_updates(
    output_irreps: Irreps = None,
    n_layers: int = 2,
    n_edges: int = 1,
    normalize_messages: bool = True,
    steerable_node_attrs: Optional[Irreps] = None,
    scalar_activation: str = "silu",
    gate_activation: str = "sigmoid",
):
    def update_fn(
        nodes: jnp.array, senders: jnp.array, receivers: jnp.array, globals: jnp.array
    ) -> jnp.array:
        m_i = receivers
        if normalize_messages:
            m_i /= n_edges - 1
            # steerable_node_attrs /= n_edges - 1

        m_i = e3nn.concatenate([m_i, nodes], axis=-1)  # Eq. 8 of 2110.02905
        # Gated tensor product steered by geometric feature messages
        for _ in range(n_layers - 1):
            nodes = TensorProductLinearGate(
                output_irreps,
                scalar_activation=scalar_activation,
                gate_activation=gate_activation,
            )(m_i, steerable_node_attrs)
        nodes = TensorProductLinearGate(output_irreps, activation=False)(
            m_i, steerable_node_attrs
        )  # No activation
        return nodes

    return update_fn


def wrap_graph_tuple(graph):
    """Remove additional attributes from the graph tuple."""
    # Assuming 'steerable_node_attrs' is the extra attribute.
    basic_graph = jraph.GraphsTuple(
        nodes=graph.nodes,
        edges=graph.edges,
        receivers=graph.receivers,
        senders=graph.senders,
        globals=graph.globals,
        n_node=graph.n_node,
        n_edge=graph.n_edge,
    )
    equivariant_attrs = {
        'steerable_node_attrs': graph.steerable_node_attrs,
        'steerable_edge_attrs': graph.steerable_edge_attrs,
        'additional_messages': graph.steerable_node_attrs,
    }
    return basic_graph, equivariant_attrs 

class SEGNN(nn.Module):
    d_hidden: int = 64  # Hidden dimension
    l_max_hidden: int = 1  # Maximum spherical harmonic degree for hidden features
    sphharm_norm: str = None  # "component"  # Normalization for spherical harmonics; "component", "integral", "norm"
    irreps_out: Optional[
        Irreps
    ] = None  # Output irreps for node-wise task; defaults to input irreps
    use_vel_attrs: bool = False  # Use velocity as steerable attribute
    normalize_messages: bool = True  # Normalize messages by number of edges
    num_message_passing_steps: int = 3  # Number of message passing steps
    num_blocks: int = 3  # Number of gated tensor products in each message passing step
    residual: bool = True  # Residual connections
    use_pbc: bool = False  # Use periodic boundary conditions when computing relative position vectors
    message_passing_agg: str = "sum"  # "sum", "mean", "max"
    readout_agg: str = "mean"  # "sum", "mean", "max"
    mlp_readout_widths: List[int] = (4, 2)  # Factor of d_hidden for global readout MLPs
    task: str = "node"  # "graph" or "node"
    n_outputs: int = 1  # Number of outputs for graph-level readout
    norm_dict: dict = dataclasses.field(
        default_factory=lambda: {"mean": 0.0, "std": 1.0}
    )  # Normalization dictionary for relative position vectors
    unit_cell: Optional[
        jnp.array
    ] = None  # Unit cell for applying periodic boundary conditions; should be compatible with norm_dict
    #TODO: This makes it much smaller, do we want it?
    intermediate_hidden_irreps: bool = False  # Use hidden irreps for intermediate message passing steps; otherwise use input irreps
    scalar_activation: str = "silu"  # Activation function for scalars
    gate_activation: str = "sigmoid"  # Activation function for gate scalars

    def _embed(
        self,
        embed_irreps: e3nn.Irreps,
        graph: jraph.GraphsTuple, 
        steerable_node_attrs,
    ):
        nodes =  TensorProductLinearGate(embed_irreps, activation=False)(graph.nodes, steerable_node_attrs)
        graph = graph._replace(nodes=nodes)
        return graph 

    def _decode(self, output_irreps: e3nn.Irreps, graph: jraph.GraphsTuple, steerable_node_attrs):
        nodes =  TensorProductLinearGate(output_irreps, activation=False)(graph.nodes, steerable_node_attrs)
        graph = graph._replace(nodes=nodes)
        return graph

    @nn.compact
    def __call__(
        self,
        st_graphs: SteerableGraphsTuple,
    ) -> jraph.GraphsTuple:
        irreps_hidden = balanced_irreps(
            lmax=self.l_max_hidden, feature_size=self.d_hidden, use_sh=True
        )  # For hidden features
        irreps_in = st_graphs.nodes.irreps  # Input irreps
        irreps_out = (
            self.irreps_out if self.irreps_out is not None else irreps_in
        )  # Output irreps desired, if different from input irreps
        additional_messages = st_graphs.additional_messages
        steerable_node_attrs = st_graphs.steerable_node_attrs
        steerable_edge_attrs = st_graphs.steerable_edge_attrs
        graphs, _ = wrap_graph_tuple(st_graphs)
        # Compute irreps
        # If specified, use hidden irreps between message passing steps; otherwise, use input irreps (bottleneck and fewer parameters)
        irreps_intermediate = (
            irreps_hidden if self.intermediate_hidden_irreps else irreps_in
        )
        # Neighborhood aggregation function
        aggregate_edges_for_nodes_fn = getattr(
            utils, f"segment_{self.message_passing_agg}"
        )
        graphs = self._embed(irreps_intermediate,graphs,steerable_node_attrs )
        # Message passing rounds
        for _ in range(self.num_message_passing_steps):
            update_edge_fn = get_edge_mlp_updates(
                output_irreps=irreps_hidden,
                n_layers=self.num_blocks,
                steerable_edge_attrs=steerable_edge_attrs,
                additional_messages=additional_messages,
                scalar_activation=self.scalar_activation,
                gate_activation=self.gate_activation,
            )
            update_node_fn = get_node_mlp_updates(
                output_irreps=irreps_intermediate,  # Node update outputs to `irreps_intermediate`
                n_layers=self.num_blocks,
                normalize_messages=self.normalize_messages,
                n_edges=graphs.n_edge,
                steerable_node_attrs=steerable_node_attrs,
                scalar_activation=self.scalar_activation,
                gate_activation=self.gate_activation,
            )
            # Apply steerable EGCL
            graph_net = jraph.GraphNetwork(
                update_node_fn=update_node_fn,
                update_edge_fn=update_edge_fn,
                aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn,
            )
            processed_graphs = graph_net(graphs)
            # Skip connection
            if self.residual:
                graphs = processed_graphs._replace(
                    nodes=processed_graphs.nodes + graphs.nodes
                )
            else:
                graphs = processed_graphs

        if self.task == "node":
            # If output irreps differ from input irreps, project to output irreps
            if irreps_out != irreps_in:
                graphs = self._decode(irreps_out, graphs, steerable_node_attrs=steerable_node_attrs)
            return graphs
        elif self.task == "graph":
            # TODO: check if eq graph aggregation sensible and flexible

            # Aggregate residual node features
            if self.readout_agg not in ["sum", "mean", "max"]:
                raise ValueError(
                    f"Invalid global aggregation function {self.message_passing_agg}"
                )

            # Steerable linear layer conditioned on node attributes; output scalars for invariant readout
            irreps_pre_pool = Irreps(f"{self.d_hidden}x0e")
            readout_agg_fn = getattr(jnp, f"{self.readout_agg}")
            nodes_pre_pool = nn.Dense(self.d_hidden)(
                TensorProductLinearGate(irreps_pre_pool, gate_activation=False)(
                    graphs.nodes, node_attrs
                ).array
            )
            agg_nodes = readout_agg_fn(nodes_pre_pool, axis=0)

            # Readout and return
            out = MLP(
                [w * self.d_hidden for w in self.mlp_readout_widths] + [self.n_outputs]
            )(agg_nodes)
            return out

        else:
            raise ValueError(f"Invalid task {self.task}")
