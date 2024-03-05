import sys

sys.path.append("../models")
sys.path.append("../")

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import jraph
from models.gnn import GNN
from models.egnn import EGNN
from models.segnn import SEGNN, TensorProductLinearGate
from e3nn_jax import IrrepsArray
from e3nn_jax import Irreps
from utils.graph_utils import nearest_neighbors, rotate_representation
from utils.equivariant_graph_utils import get_equivariant_graph

import pytest


class GraphWrapper(nn.Module):
    model: nn.Module  # Specify the model as a class attribute

    @nn.compact
    def __call__(self, x):
        model = jax.vmap(self.model)
        return model(x)


def create_dummy_graph(
    node_features,
    k,
    use_irreps=False,
):
    sources, targets = jax.vmap(nearest_neighbors, in_axes=(0, None))(
        node_features[..., :3], k
    )
    n_node = np.array([len(node_feat) for node_feat in node_features])
    if use_irreps:
        # subtract mean position
        # node_features = node_features.at[...,:2].set(node_features[...,:2] - np.mean(node_features[...,:2], axis=1, keepdims=True))
        node_features = IrrepsArray("1o + 1o + 1x0e", node_features)
        positions, velocities = node_features[..., :3], node_features[..., 3:6]
        additional_messages = None
        edges = None
        return get_equivariant_graph(
            node_features=node_features,
            positions=positions,
            velocities=velocities,
            senders=sources,
            receivers=targets,
            n_node=n_node,
            n_edge=np.array(len(node_features) * [[k]]),
            edges=edges,
            globals=None,
            lmax_attributes=1,
            additional_messages=additional_messages,
        )
    else:
        nodes = node_features
        edges = None
        return jraph.GraphsTuple(
            n_node=n_node,
            n_edge=np.array(len(node_features) * [[k]]),
            nodes=nodes,
            edges=edges,
            globals=None,
            senders=sources,
            receivers=targets,
        )

@pytest.fixture
def node_features():
    key = jax.random.PRNGKey(0)
    return jax.random.normal(key, (2, 5, 7))

def apply_transformation(
    x, angle_deg=45.0, axis=np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2)])
):
    return jax.vmap(rotate_representation, in_axes=(0, None, None, None))(
        x, angle_deg, axis, False
    )


def transform_graph(
    nodes,
    k=5,
    use_irreps=False,
):
    nodes = apply_transformation(
        nodes,
    )
    return create_dummy_graph(
        node_features=nodes,
        k=k,
        use_irreps=use_irreps,
    )


def is_model_equivariant(
    data, model, params, should_be_equivariant=True, use_irreps=False, rtol=0.05
):
    transformed_data = transform_graph(
        data.nodes if not use_irreps else data.nodes.array,
        use_irreps=use_irreps,
    )
    output_original = model.apply(params, data).nodes
    output_transformed = model.apply(params, transformed_data).nodes
    output_original_transformed = apply_transformation(
        output_original if not use_irreps else output_original.array
    )
    if use_irreps:
        output_original = output_original.array
        output_transformed = output_transformed.array
    # Make sure output original sufficiently different from output transformed
    assert not np.allclose(output_original, output_transformed, rtol=rtol)
    if should_be_equivariant:
        assert np.allclose(
            np.array(output_transformed),
            np.array(output_original_transformed),
            rtol=rtol,
        )
    else:
        assert not np.allclose(
            output_transformed, output_original_transformed, rtol=rtol
        )


def is_model_invariant(
    data, model, params, should_be_invariant=True, use_irreps=False, rtol=0.05
):
    transformed_data = transform_graph(
        data.nodes if not use_irreps else data.nodes.array,
        use_irreps=use_irreps,
    )
    output_original = model.apply(params, data)
    output_transformed = model.apply(params, transformed_data)
    if use_irreps:
        output_original = output_original.array
        output_transformed = output_transformed.array
    # Make sure output original sufficiently different from output transformed
    if should_be_invariant:
        assert np.allclose(output_original, output_transformed, rtol=rtol)
    else:
        assert not np.allclose(output_original, output_transformed, rtol=rtol)

def test_not_equivariant_gnn(
    node_features,
):
    dummy_graph = create_dummy_graph(
        node_features=node_features,
        k=5,
        use_irreps=False,
    )
    model = GraphWrapper(
        GNN(
            message_passing_steps=3,
            d_hidden=32,
            n_layers=3,
            activation="gelu",
            task="node",
        )
    )
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, dummy_graph)
    is_model_equivariant(
        dummy_graph,
        model,
        params,
        should_be_equivariant=False,
        use_irreps=False,
    )

def test_not_invariant_gnn(
    node_features,
):
    dummy_graph = create_dummy_graph(
        node_features=node_features,
        k=5,
        use_irreps=False,
    )
    model = GraphWrapper(
        GNN(
            message_passing_steps=3,
            d_hidden=32,
            n_layers=3,
            activation="gelu",
            task="graph",
        )
    )
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, dummy_graph)
    is_model_invariant(
        dummy_graph,
        model,
        params,
        should_be_invariant=False,
        use_irreps=False,
    )

def test_equivariant_egnn(node_features):
    dummy_graph = create_dummy_graph(
        node_features=node_features,
        k=5,
        use_irreps=False,
    )
    model = GraphWrapper(
        EGNN(
            message_passing_steps=3,
            d_hidden=32,
            n_layers=3,
            activation="gelu",
            tanh_out=True,
            soft_edges=True,
            positions_only=False,
            task="node",
            decouple_pos_vel_updates=True,
        )
    )
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, dummy_graph)
    is_model_equivariant(
        dummy_graph,
        model,
        params,
        use_irreps=False,
    )


def test_equivariant_embedding_for_segnn(node_features):
    dummy_graph = create_dummy_graph(
        node_features=node_features,
        k=5,
        use_irreps=True,
    )
    class Embedding(nn.Module):
        irreps_out: Irreps("1o")

        @nn.compact
        def __call__(
            self,
            graphs,
        ):
            nodes = TensorProductLinearGate(
                output_irreps=self.irreps_out, activation=False
            )(graphs.nodes, graphs.steerable_node_attrs)
            graphs = graphs._replace(nodes=nodes)
            return graphs

    model = GraphWrapper(
        Embedding(
            irreps_out="1o + 1o + 1x0e",
        )
    )
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, dummy_graph)
    is_model_equivariant(
        dummy_graph,
        model,
        params,
        use_irreps=True,
    )


def test_equivariant_segnn(node_features):
    dummy_graph = create_dummy_graph(
        node_features=node_features,
        k=5,
        use_irreps=True,
    )
    model = GraphWrapper(
        SEGNN(
            num_message_passing_steps=2,
            d_hidden=32,
            task="node",
            intermediate_hidden_irreps=True,
            residual=True,
            irreps_out="1o + 1o + 1x0e",
        )
    )
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, dummy_graph)
    is_model_equivariant(
        dummy_graph,
        model,
        params,
        use_irreps=True,
    )


def test_invariant_segnn(node_features):
    dummy_graph = create_dummy_graph(
        node_features=node_features,
        k=5,
        use_irreps=True,
    )
    model = GraphWrapper(
        SEGNN(
            num_message_passing_steps=3,
            d_hidden=32,
            task="graph",
            intermediate_hidden_irreps=True,
            residual=True,
            irreps_out="1x0e",
        )
    )
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, dummy_graph)
    is_model_invariant(
        dummy_graph,
        model,
        params,
        use_irreps=True,
    )