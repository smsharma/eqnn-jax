import sys
sys.path.append("../models")
sys.path.append("../")
import jax
import numpy as np
import flax.linen as nn
import jraph
from models.egnn import EGNN
from utils.graph_utils import nearest_neighbors, rotate_representation
from scipy.spatial import distance_matrix

import pytest

class GraphWrapper(nn.Module):
    model: nn.Module  # Specify the model as a class attribute

    @nn.compact
    def __call__(self, x):
        model = jax.vmap(self.model)
        return model(x)

def create_dummy_graph(node_features, k,):
    sources, targets = jax.vmap(nearest_neighbors, in_axes=(0, None))(node_features[...,:3], k)
    n_node = np.array([len(node_feat) for node_feat in node_features])
    return jraph.GraphsTuple(
        n_node=n_node,
        n_edge=np.array(len(node_features)* [[k]]),
        nodes=node_features,
        edges=None,
        globals=None,
        senders=sources,
        receivers=targets,
    )



@pytest.fixture
def node_features():
    return np.random.randn(2,5,7)

@pytest.fixture
def dummy_graph(node_features):
    return create_dummy_graph(
        node_features=node_features,
        k = 5,
    )

def apply_transformation(
        x, angle_deg=45.,axis = np.array([0, 1 / np.sqrt(2), 1 / np.sqrt(2)])
    ):
    return jax.vmap(
        rotate_representation, 
        in_axes=(0,None,None,None))(
            x, angle_deg, axis, False
        )

def transform_graph(
        nodes,
    ):
    nodes = apply_transformation(nodes,)
    return create_dummy_graph(
        node_features=nodes,
        k = 5,
    )

def is_model_equivariant(data, model, params):
    transformed_data = transform_graph(data.nodes)
    output_original = model.apply(params, data)
    output_original_transformed = apply_transformation(output_original.nodes)
    output_transformed = model.apply(params, transformed_data).nodes
    assert np.allclose(output_transformed, output_original_transformed, rtol=5.e-2)


def test_equivariant_segnn(dummy_graph):
    model = GraphWrapper(EGNN(
        message_passing_steps=3, d_hidden=32, n_layers=3, activation='gelu', tanh_out=True, soft_edges=True,
        positions_only=False, task='node', decouple_pos_vel_updates=True,
    ))
    rng = jax.random.PRNGKey(0)  
    params = model.init(rng, dummy_graph)
    is_model_equivariant(dummy_graph, model, params)
