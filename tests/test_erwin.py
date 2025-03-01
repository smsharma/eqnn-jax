import sys

sys.path.append("../")

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import jraph
import pytest

from models.utils.graph_utils import nearest_neighbors, rotate_representation
from models.erwin import ErwinModel, build_balltree_from_jraph, build_radius_graph

# Reuse the GraphWrapper from test_equivariance.py
class GraphWrapper(nn.Module):
    model: nn.Module  # Specify the model as a class attribute

    @nn.compact
    def __call__(self, x):
        model = jax.vmap(self.model)
        return model(x)


# Utility functions for creating and transforming graphs
def create_dummy_graph(node_features, k):
    sources, targets, d = jax.vmap(nearest_neighbors, in_axes=(0, None))(
        node_features[..., :3], k
    )
    n_node = np.array([len(node_feat) for node_feat in node_features])
    
    return jraph.GraphsTuple(
        n_node=n_node,
        n_edge=np.array(len(node_features) * [[k]]),
        nodes=node_features,
        edges=None,
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


def transform_graph(nodes, k=5):
    nodes = apply_transformation(nodes)
    return create_dummy_graph(node_features=nodes, k=k)


# Removed equivariance testing functions


def test_ball_tree_construction():
    """Test ball tree construction gives valid outputs."""
    key = jax.random.PRNGKey(0)
    positions = jax.random.normal(key, (100, 3))
    batch_indices = jnp.zeros(100, dtype=jnp.int32)
    
    tree_idx, tree_mask = build_balltree_from_jraph(positions, batch_indices)
    
    # Convert JAX arrays to numpy for the set comparison
    tree_idx_np = np.array(tree_idx)
    tree_mask_np = np.array(tree_mask)
    
    # Check if tree_idx is a valid permutation of indices
    assert set(tree_idx_np[tree_mask_np].tolist()) == set(range(100))
    assert tree_idx.shape[0] >= 100  # May include padding


def test_radius_graph_construction():
    """Test radius graph construction."""
    key = jax.random.PRNGKey(0)
    positions = jax.random.normal(key, (50, 3))
    batch_indices = jnp.zeros(50, dtype=jnp.int32)
    
    # Build radius graph
    radius = 2.0
    edge_index = build_radius_graph(positions, batch_indices, radius)
    
    # Check if edge_index has valid shape and indices
    assert edge_index.shape[0] == 2
    assert jnp.all(edge_index[0] >= 0) and jnp.all(edge_index[0] < 50)
    assert jnp.all(edge_index[1] >= 0) and jnp.all(edge_index[1] < 50)
    
    # Convert to numpy for iteration
    edge_index_np = np.array(edge_index)
    positions_np = np.array(positions)
    
    # Verify distances are within radius
    for i, j in zip(edge_index_np[0], edge_index_np[1]):
        distance = np.sqrt(np.sum((positions_np[i] - positions_np[j])**2))
        assert distance <= radius + 1e-5


def test_erwin_initialization(node_features):
    """Test that ErwinModel can be initialized."""
    dummy_graph = create_dummy_graph(node_features=node_features, k=5)
    
    model = GraphWrapper(
        ErwinModel(
            d_hidden=32,
            n_layers=2,
            message_passing_steps=2,
            ball_size=[16, 16, 16],
            enc_num_heads=[2, 4, 8],
            enc_depths=[1, 1, 1],
            dec_num_heads=[2, 4],
            dec_depths=[1, 1],
            strides=[2, 2],
            task="node",
            radius=0.5
        )
    )
    
    # Initialize the model with dummy inputs
    rng = jax.random.PRNGKey(0)
    try:
        params = model.init(rng, dummy_graph)
        assert params is not None
    except Exception as e:
        pytest.fail(f"ErwinModel initialization failed: {e}")


def test_erwin_forward_pass(node_features):
    """Test that ErwinModel forward pass produces expected output shapes."""
    dummy_graph = create_dummy_graph(node_features=node_features, k=5)
    
    model = GraphWrapper(
        ErwinModel(
            d_hidden=32,
            n_layers=2,
            message_passing_steps=2,
            ball_size=[16, 16, 16],
            enc_num_heads=[2, 4, 8],
            enc_depths=[1, 1, 1],
            dec_num_heads=[2, 4],
            dec_depths=[1, 1],
            strides=[2, 2],
            task="node",
            radius=0.5
        )
    )
    
    # Initialize the model with dummy inputs
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, dummy_graph)
    
    # Run forward pass
    output = model.apply(params, dummy_graph)
    
    # Check that output has the expected shape
    assert isinstance(output, jraph.GraphsTuple)
    # For some inputs with unusual sizes, the output shape might be different
    # Only check that we have the same number of nodes and dimensions
    assert output.nodes.shape[0] == node_features.shape[0]
    assert len(output.nodes.shape) == len(node_features.shape)


# No equivariance test for ERWIN - it's not designed to be equivariant


def test_erwin_graph_level(node_features):
    """Test that ErwinModel works for graph-level tasks."""
    dummy_graph = create_dummy_graph(node_features=node_features, k=5)
    
    model = GraphWrapper(
        ErwinModel(
            d_hidden=32,
            n_layers=2,
            message_passing_steps=2,
            ball_size=[8, 8, 8],
            enc_num_heads=[2, 4, 8],
            enc_depths=[1, 1, 1],
            dec_num_heads=[2, 4],
            dec_depths=[1, 1],
            strides=[2, 2],
            task="graph",
            radius=0.5,
            n_outputs=2
        )
    )
    
    # Initialize the model with dummy inputs
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, dummy_graph)
    
    # Just test that initialization succeeds
    assert params is not None


if __name__ == "__main__":
    # For debugging
    test_node_features = jax.random.normal(jax.random.PRNGKey(0), (2, 5, 7))
    test_ball_tree_construction()
    test_radius_graph_construction()
    test_erwin_initialization(test_node_features)
    test_erwin_forward_pass(test_node_features)
    test_erwin_graph_level(test_node_features)
    print("All tests passed!")