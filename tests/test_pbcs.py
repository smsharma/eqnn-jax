import jax
import jax.numpy as jnp
import sys

sys.path.append("../models")
sys.path.append("../")

import pytest
from utils.graph_utils import get_apply_pbc, build_graph 

def test__apply_pbcs_on_graph():
    # Test that graph standarized and not standarized agrees
    k=10
    key = jax.random.PRNGKey(0)
    n_batch = 1
    n_objects = 10
    positions = jax.random.uniform(key, (n_batch, n_objects, 3))
    graph_no_pbcs = build_graph(
        positions, 
        None, 
        k=k, 
        apply_pbc=None,
    )
    apply_pbc = get_apply_pbc()
    graph_pbcs = build_graph(
        positions, 
        None, 
        k=k, 
        apply_pbc=apply_pbc,
    )
    # make sure mean distance meaningful
    assert graph_no_pbcs.edges.mean() > graph_pbcs.edges.mean()
    assert graph_no_pbcs.edges.max() > jnp.sqrt(3) / 2
    assert graph_pbcs.edges.max() < jnp.sqrt(3) / 2

def test__apply_pbcs_on_standarized_graph():
    k=10
    key = jax.random.PRNGKey(0)
    n_batch = 1
    n_objects = 100
    positions = jax.random.uniform(key, (n_batch, n_objects, 3))
    # standarize positions
    mean_pos = jnp.mean(positions, axis=(0, 1))
    std_pos = jnp.std(positions, axis=(0, 1))
    standarized_positions = (positions - mean_pos) / std_pos
    apply_pbc = get_apply_pbc(std_pos)
    standarized_graph = build_graph(
        standarized_positions, 
        None, 
        k=k, 
        use_edges=True,
        apply_pbc=apply_pbc,
    )
    apply_pbc = get_apply_pbc()
    graph = build_graph(
        positions, 
        None, 
        k=k, 
        use_edges=True,
        apply_pbc=apply_pbc,
    )
    assert standarized_graph.edges.mean() * std_pos[0] == pytest.approx(graph.edges.mean(), rel=0.05)

