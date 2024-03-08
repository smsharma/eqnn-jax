from typing import Any, NamedTuple, Iterable, Mapping, Union, Optional

import e3nn_jax as e3nn
from e3nn_jax import IrrepsArray

import jax
from jraph import segment_mean
import jax.numpy as jnp
from .graph_utils import apply_pbc

ArrayTree = Union[jnp.ndarray, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]


class SteerableGraphsTuple(NamedTuple):
    nodes: Optional[ArrayTree]
    edges: Optional[ArrayTree]
    receivers: Optional[jnp.ndarray]  # with integer dtype
    senders: Optional[jnp.ndarray]  # with integer dtype
    additional_messages: Optional[jnp.ndarray]
    steerable_node_attrs: Optional[jnp.ndarray]
    steerable_edge_attrs: Optional[jnp.ndarray]
    globals: Optional[ArrayTree]
    n_node: jnp.ndarray  # with integer dtype
    n_edge: jnp.ndarray  # with integer dtype


def get_equivariant_graph(
    node_features,
    positions,
    velocities,
    senders,
    receivers,
    n_node,
    n_edge,
    edges,
    globals,
    lmax_attributes,
    additional_messages: Optional[jnp.ndarray] = None,
    steerable_velocities: bool = False,
    spherical_harmonics_norm="integral",
    periodic_boundaries: bool = False,
    norm_dict=None,
    unit_cell=None,
):
    attribute_irreps = e3nn.Irreps.spherical_harmonics(lmax_attributes)
    x_i = positions[jnp.arange(positions.shape[0])[:, None], senders, :]
    x_j = positions[jnp.arange(positions.shape[0])[:, None], receivers, :]
    r_ij = x_i - x_j
    if periodic_boundaries:
        r_ij = r_ij.array * norm_dict["std"][None, :3]
        r_ij = apply_pbc(r_ij, unit_cell)
        r_ij = IrrepsArray("1o", r_ij / norm_dict["std"][None, :3])
    steerable_edge_attrs = e3nn.spherical_harmonics(
        irreps_out=attribute_irreps,
        input=r_ij,
        normalize=True,
        normalization=spherical_harmonics_norm,
    )

    # TODO: clean this up (should the entire function be part of vmap?)
    def scatter_mean_wrapper(steerable_edge_attrs, receivers, output_size):
        return e3nn.scatter_mean(
            steerable_edge_attrs, dst=receivers, output_size=output_size
        )

    # TODO: Why not learn this?
    steerable_node_attrs = jax.vmap(scatter_mean_wrapper, in_axes=(0, 0, None))(
        steerable_edge_attrs,
        receivers,
        positions.shape[1],
    )
    if steerable_velocities:
        vel_sph = e3nn.spherical_harmonics(
            attribute_irreps,
            input=velocities,
            normalize=True,
            normalization=spherical_harmonics_norm,
        )
        steerable_node_attrs += vel_sph
    return SteerableGraphsTuple(
        nodes=node_features,
        edges=edges,
        receivers=receivers,
        senders=senders,
        additional_messages=additional_messages,
        steerable_edge_attrs=steerable_edge_attrs,
        steerable_node_attrs=steerable_node_attrs,
        globals=globals,
        n_node=n_node,
        n_edge=n_edge,
    )
