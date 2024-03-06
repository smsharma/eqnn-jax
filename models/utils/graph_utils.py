import jax
import jax.numpy as np
import jraph

from functools import partial


def apply_pbc(
    dr: np.array,
    cell: np.array = np.array(
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
) -> np.array:
    """Apply periodic boundary conditions to a displacement vector, dr, given a cell.

    Args:
        dr (np.array): An array of shape (N,3) containing the displacement vector
        cell_matrix (np.array): A 3x3 matrix describing the box dimensions and orientation.

    Returns:
        np.array: displacement vector with periodic boundary conditions applied
    """
    return dr - np.round(dr.dot(np.linalg.inv(cell))).dot(cell)


@partial(jax.jit, static_argnums=(1,))
def nearest_neighbors(
    x: np.array,
    k: int,
    mask: np.array = None,
):
    """Returns the nearest neighbors of each node in x.

    Args:
        x (np.array): positions of nodes
        k (int): number of nearest neighbors to find
        boxsize (float, optional): size of box if perdioc boundary conditions. Defaults to None.
        unit_cell (np.array, optional): unit cell for applying periodic boundary conditions. Defaults to None.
        mask (np.array, optional): node mask. Defaults to None.

    Returns:
        sources, targets: pairs of neighbors
    """
    if mask is None:
        mask = np.ones((x.shape[0],), dtype=np.int32)

    n_nodes = x.shape[0]

    # Compute the vector difference between positions
    dr = x[:, None, :] - x[None, :, :]

    # Calculate the distance matrix
    distance_matrix = np.sum(dr**2, axis=-1)

    distance_matrix = np.where(mask[:, None], distance_matrix, np.inf)
    distance_matrix = np.where(mask[None, :], distance_matrix, np.inf)

    indices = np.argsort(distance_matrix, axis=-1)[:, :k]

    sources = indices[:, 0].repeat(k)
    targets = indices.reshape(n_nodes * (k))

    return (sources, targets)


def add_graphs_tuples(
    graphs: jraph.GraphsTuple, other_graphs: jraph.GraphsTuple
) -> jraph.GraphsTuple:
    """Adds the nodes, edges and global features from other_graphs to graphs."""
    return graphs._replace(nodes=graphs.nodes + other_graphs.nodes)


def rotation_matrix(angle_deg, axis):
    """Return the rotation matrix associated with counterclockwise rotation of `angle_deg` degrees around the given axis."""
    angle_rad = np.radians(angle_deg)
    axis = axis / np.linalg.norm(axis)

    a = np.cos(angle_rad / 2)
    b, c, d = -axis * np.sin(angle_rad / 2)

    return np.array(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c],
        ]
    )


def rotate_representation(data, angle_deg, axis, positions_only=False):
    """Rotate `data` by `angle_deg` degrees around `axis`."""
    rot_mat = rotation_matrix(angle_deg, axis)

    if not positions_only:
        positions, velocities, scalars = data[:, :3], data[:, 3:6], data[:, 6:]
        rotated_positions = np.matmul(rot_mat, positions.T).T
        rotated_velocities = np.matmul(rot_mat, velocities.T).T
        return np.concatenate([rotated_positions, rotated_velocities, scalars], axis=1)

    else:
        positions, scalars = data[:, :3], data[:, 3:]
        rotated_positions = np.matmul(rot_mat, positions.T).T
        return np.concatenate([rotated_positions, scalars], axis=1)


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
    scales = 2 ** np.arange(num_encodings, dtype=dtype)
    x = x / scales
    x = np.concatenate([np.sin(x), np.cos(x)], axis=-1)
    x = np.concatenate((x, orig_x), axis=-1) if include_self else x
    return x
