from typing import Optional, Union, List, Callable

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
    def __call__(self, x: IrrepsArray, y: IrrepsArray = None) -> IrrepsArray:
        output_irreps = self.output_irreps
        if not isinstance(output_irreps, Irreps):
            output_irreps = Irreps(output_irreps)
        if not y:
            y = IrrepsArray("1x0e", jnp.ones((1, 1), dtype=x.dtype))

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

def get_node_mlp_updates(
    output_irreps: Irreps = None,
    n_layers: int = 2,
    steerable_node_attrs: Optional[Irreps] = None,
    scalar_activation: str = "silu",
    gate_activation: str = "sigmoid",
):
    def update_fn(
        nodes: jnp.array, 
        senders: jnp.array, 
        receivers: jnp.array, 
        globals: jnp.array
    ) -> jnp.array:
        m_i = nodes
        if receivers is not None:
            m_i = e3nn.concatenate([m_i, receivers], axis=-1)  # Eq. 8 of 2110.02905
        
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

def get_edge_mlp_updates(
    output_irreps: Irreps = None,
    n_layers: int = 2,
    steerable_edge_attrs: Optional[Irreps] = None,
    additional_messages: Optional[jnp.array] = None,
    scalar_activation: str = "silu",
    gate_activation: str = "sigmoid",
):
    def update_fn(
        edges: jnp.array, 
        senders: jnp.array, 
        receivers: jnp.array, 
        globals: jnp.array
    ) -> jnp.array:
        if additional_messages is not None:
            m_ij = e3nn.concatenate([additional_messages, senders, receivers], axis=-1)
        else:
            m_ij = e3nn.concatenate([senders, receivers], axis=-1)  # Messages

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
        "steerable_node_attrs": graph.steerable_node_attrs,
        "steerable_edge_attrs": graph.steerable_edge_attrs,
        "additional_messages": graph.steerable_node_attrs,
    }
    return basic_graph, equivariant_attrs


class SEGNN(nn.Module):

    d_hidden: int = 64  # Hidden dimension
    n_layers: int = 3  # Number of gated tensor products in each message passing step
    message_passing_steps: int = 3  # Number of message passing steps
    message_passing_agg: str = "sum"  # "sum", "mean", "max"
    scalar_activation: str = "gelu"  # Activation function for scalars
    gate_activation: str = "sigmoid"  # Activation function for gate scalars
    task: str = "graph"  # "graph" or "node"
    d_output: int = 1
    output_irreps: Optional[Irreps] = None  # Output irreps for node-wise task; defaults to input irreps
    readout_agg: str = "mean"  # "sum", "mean", "max"
    mlp_readout_widths: List[int] = (4, 2, 2)  # Factor of d_hidden for global readout MLPs
    l_max_hidden: int = 1  # Maximum spherical harmonic degree for hidden features
    hidden_irreps: Optional[Irreps] = None
    residual: bool = True  # Residual connections

    def _embed(
        self,
        embed_irreps: e3nn.Irreps,
        graph: jraph.GraphsTuple,
        steerable_node_attrs,
    ):
        nodes = TensorProductLinearGate(embed_irreps, activation=False)(
            graph.nodes, steerable_node_attrs
        )
        graph = graph._replace(nodes=nodes)
        return graph

    def _decode(
        self, hidden_irreps: e3nn.Irreps, graph: jraph.GraphsTuple, steerable_node_attrs
    ):
        nodes = graph.nodes
        for _ in range(self.n_layers):
            nodes = TensorProductLinearGate(
                hidden_irreps,
                activation=True,
                scalar_activation=self.scalar_activation,
                gate_activation=self.gate_activation,
            )(
                nodes,
            )
        nodes = TensorProductLinearGate(self.output_irreps, activation=False)(
            nodes, steerable_node_attrs
        )
        graph = graph._replace(nodes=nodes)
        return graph

    @nn.compact
    def __call__(
        self,
        st_graphs: SteerableGraphsTuple,
    ) -> jraph.GraphsTuple:
        
        if self.hidden_irreps is None:
            hidden_irreps = balanced_irreps(lmax=self.l_max_hidden, feature_size=self.d_hidden, use_sh=True)  # For hidden features
        else:
            hidden_irreps = self.hidden_irreps

        irreps_in = st_graphs.nodes.irreps  # Input irreps
        output_irreps = self.output_irreps if self.output_irreps is not None else irreps_in  # Output irreps desired, if different from input irreps

        additional_messages = st_graphs.additional_messages
        steerable_node_attrs = st_graphs.steerable_node_attrs
        steerable_edge_attrs = st_graphs.steerable_edge_attrs
        graphs, _ = wrap_graph_tuple(st_graphs)

        # Neighborhood aggregation function
        aggregate_edges_for_nodes_fn = getattr(utils, f"segment_{self.message_passing_agg}")

        graphs = self._embed(hidden_irreps, graphs, steerable_node_attrs)

        # Apply message-passing rounds
        for _ in range(self.message_passing_steps):

            update_edge_fn = get_edge_mlp_updates(
                output_irreps=hidden_irreps,
                n_layers=self.n_layers,
                steerable_edge_attrs=steerable_edge_attrs,
                additional_messages=additional_messages,
                scalar_activation=self.scalar_activation,
                gate_activation=self.gate_activation,
            )
            update_node_fn = get_node_mlp_updates(
                output_irreps=hidden_irreps,
                n_layers=self.n_layers,
                steerable_node_attrs=steerable_node_attrs,
                scalar_activation=self.scalar_activation,
                gate_activation=self.gate_activation,
            )

            # Instantiate graph network and apply steerable EGCL
            graph_net = jraph.GraphNetwork(
                update_node_fn=update_node_fn,
                update_edge_fn=update_edge_fn,
                aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn,
            )
            processed_graphs = graph_net(graphs)

            # Skip connection
            if self.residual:
                graphs = processed_graphs._replace(nodes=processed_graphs.nodes + graphs.nodes)
            else:
                graphs = processed_graphs

            # print(graphs.nodes.array.std())

        if self.task == "node":  # If output irreps differ from input irreps, project to output irreps
            if output_irreps != irreps_in:
                graphs = self._decode(
                    hidden_irreps=hidden_irreps,
                    graph=graphs,
                    steerable_node_attrs=steerable_node_attrs,
                )
            return graphs
        elif self.task == "graph":  # Aggregate residual node features
            
            if self.readout_agg not in ["sum", "mean", "max"]:
                raise ValueError(
                    f"Invalid global aggregation function {self.readout_agg}"
                )

            # Steerable linear layer conditioned on node attributes; output scalars for invariant readout
            irreps_pre_pool = Irreps(f"{self.d_hidden}x0e")
            readout_agg_fn = getattr(jnp, f"{self.readout_agg}")
            nodes_pre_pool = nn.Dense(self.d_hidden)(
                TensorProductLinearGate(irreps_pre_pool, activation=False)(
                    graphs.nodes, steerable_node_attrs
                ).array
            )
            agg_nodes = readout_agg_fn(nodes_pre_pool, axis=0)

            if processed_graphs.globals is not None:
                agg_nodes = jnp.concatenate([agg_nodes, processed_graphs.globals]) # Use tpcf
                
                norm = nn.LayerNorm()
                agg_nodes = norm(agg_nodes)
                
            # Readout and return
            mlp = MLP([
                self.mlp_readout_widths[0] * agg_nodes.shape[-1]] + \
                [w * self.d_hidden for w in self.mlp_readout_widths[1:]] + \
                [self.d_output,]
            )                                                                        
            out = mlp(agg_nodes)                                                             
            return out
        else:
            raise ValueError(f"Invalid task {self.task}")