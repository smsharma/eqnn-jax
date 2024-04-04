import jax
import jax.numpy as jnp
import jraph

from functools import partial
from e3nn_jax import Irreps, IrrepsArray

EPS = 1e-5


def apply_pbc(dr: jnp.array, cell: jnp.array) -> jnp.array:
    """Apply periodic boundary conditions to a displacement vector, dr, given a cell.

    Args:
        dr (np.array): An array of shape (N,3) containing the displacement vector
        cell_matrix (np.array): A 3x3 matrix describing the box dimensions and orientation.

    Returns:
        np.array: displacement vector with periodic boundary conditions applied
    """
    return dr - jnp.round(dr.dot(jnp.linalg.inv(cell))).dot(cell)


@partial(jax.jit, static_argnums=(1, 3))
def nearest_neighbors(
    x: jnp.array,
    k: int,
    cell: jnp.array = None,
    pbc: bool = False,
):
    """Returns the nearest neighbors of each node in x.

    Args:
        x (np.array): positions of nodes
        k (int): number of nearest neighbors to find
        mask (np.array, optional): node mask. Defaults to None.

    Returns:
        sources, targets: pairs of neighbors
    """
    n_nodes = x.shape[0]
    # Compute the vector difference between positions
    dr = (x[:, None, :] - x[None, :, :]) + EPS
    if pbc:
        dr = apply_pbc(
            dr=dr,
            cell=cell,
        )

    # Calculate the distance matrix
    distance_matrix = jnp.sum(dr**2, axis=-1)

    # Get indices of nearest neighbors
    indices = jnp.argsort(distance_matrix, axis=-1)[:, :k]

    # Create sources and targets arrays
    sources = jnp.repeat(jnp.arange(n_nodes), k)
    targets = indices.ravel()

    # return sources, targets, distance_matrix[sources, targets]
    return sources, targets, dr[sources, targets]


def build_graph(
    halo_pos,
    k,
    use_pbc=True,
    use_edges=True,
    unit_cell=jnp.array(
        [
            [
                1.0,
                0.0,
                0.0,
            ],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
):

    n_batch = len(halo_pos)
    sources, targets, distances = jax.vmap(
        partial(nearest_neighbors, pbc=use_pbc), in_axes=(0, None, None)
    )(halo_pos[..., :3], k, unit_cell)

    return jraph.GraphsTuple(
        n_node=jnp.array([[halo_pos.shape[1]]] * n_batch),
        n_edge=jnp.array(n_batch * [[k]]),
        nodes=halo_pos,
        edges=(
            jnp.sqrt(jnp.sum(distances**2, axis=-1, keepdims=True))
            if use_edges
            else None
        ),
        globals=None,
        senders=sources,
        receivers=targets,
    )


def build_graph_irreps(
    halo_pos,
    k,
    use_pbc=True,
    use_edges=True,
    unit_cell=jnp.array(
        [
            [
                1.0,
                0.0,
                0.0,
            ],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
    **kwargs
):

    n_batch = len(halo_pos)
    sources, targets, distances = jax.vmap(
        partial(nearest_neighbors, pbc=use_pbc), in_axes=(0, None, None)
    )(halo_pos[..., :3], k, unit_cell)

    return jraph.GraphsTuple(
        n_node=jnp.array([[halo_pos.shape[1]]] * n_batch),
        n_edge=jnp.array(n_batch * [[k]]),
        # nodes=IrrepsArray("1o + 1o + 1x0e", halo_pos),
        nodes=IrrepsArray("1o", halo_pos),
        edges=(
            jnp.sqrt(jnp.sum(distances**2, axis=-1, keepdims=True))
            if use_edges
            else None
        ),
        globals=None,
        senders=sources,
        receivers=targets,
    )


def fourier_features(x, num_encodings=8, include_self=True):
    """Add Fourier features to a set of coordinates

    Args:
        x (jnp.array): Coordinates
        num_encodings (int, optional): Number of Fourier feature encodings. Defaults to 16.
        include_self (bool, optional): Whether to include original coordinates in output. Defaults to True.

    Returns:
        jnp.array: Fourier features of input coordinates
    """

    dtype, orig_x = x.dtype, x
    scales = 2 ** jnp.arange(num_encodings, dtype=dtype)
    x = x / scales
    x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)
    x = jnp.concatenate((x, orig_x), axis=-1) if include_self else x
    return x
