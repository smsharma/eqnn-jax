import jax
import jax.numpy as np
import jraph

import e3nn_jax as e3nn

from typing import Callable, Optional
from functools import partial

EPS = 1e-7

def get_apply_pbc(std: np.array=None, cell: np.array = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    ),
):
    if std is not None:
        cell = cell / std[:3]
        # cell -= 0.5 / std[:3]

    def apply_pbc(
        dr: np.array,
    ) -> np.array:
        """Apply periodic boundary conditions to a displacement vector, dr, given a cell.

        Args:
            dr (np.array): An array of shape (N,3) containing the displacement vector
            cell_matrix (np.array): A 3x3 matrix describing the box dimensions and orientation.

        Returns:
            np.array: displacement vector with periodic boundary conditions applied
        """
        return (dr - np.round(dr.dot(np.linalg.inv(cell))).dot(cell)) 
    return apply_pbc


@partial(jax.jit, static_argnums=(1,2))
def nearest_neighbors(
    x: np.array,
    k: int,
    apply_pbc: Optional[Callable] = None,
):
    """Returns the nearest neighbors of each node in x.

    Args:
        x (jnp.array): positions of nodes
        k (int): number of nearest neighbors to find
        unit_cell (jnp.array, optional): unit cell for applying periodic boundary conditions. Defaults to None.
        mask (jnp.array, optional): node mask. Defaults to None.

    Returns:
        sources, targets, distances: pairs of neighbors and their distances
    """
    n_nodes = x.shape[0]
    # Compute the vector difference between positions
    dr = (x[:, None, :] - x[None, :, :]) + EPS # (n_nodes, n_nodes, 3)
    if apply_pbc:
        dr = apply_pbc(dr)

    # Calculate the distance matrix
    distance_matrix = np.sum(dr**2, axis=-1)
    
    # Get indices of nearest neighbors
    indices = np.argsort(distance_matrix, axis=-1)[:, :k]

    # Create sources and targets arrays
    sources = np.repeat(np.arange(n_nodes), k)
    targets = indices.ravel()
    return sources, targets, dr[sources, targets]


@partial(jax.jit, static_argnums=(1,))
def nearest_neighbors_centroids(
    x: np.array,
    centroids: np.array,
    k: int,
    apply_pbc: Optional[Callable] = None,
):
    """Returns the k nearest neighbors of each centroid in point_cloud.

    Args:
        point_cloud (jnp.array): positions of all points
        centroids (jnp.array): indices of centroid points
        k (int): number of nearest neighbors to find
        apply_pbc (Callable, optional): function to apply periodic boundary conditions

    Returns:
        sources, targets, distances: pairs of neighbors and their distances
    """
    n_centroids = centroids.shape[0]
    n_nodes = x.shape[0]

    # Select the centroid coordinates
    centroids = x[centroids]

    # Compute the vector difference between centroids and all points
    dr = (centroids[:, None, :] - x[None, :, :]) + EPS
    if apply_pbc:
        dr = apply_pbc(dr, unit_cell)

    # Calculate the squared distance matrix
    distance_matrix = np.sum(dr ** 2, axis=-1)

    # Get indices of nearest neighbors for each centroid
    indices = np.argsort(distance_matrix, axis=-1)[:, :k]

    # Create sources and targets arrays
    sources = np.repeat(centroids, k)
    targets = indices.ravel()

    # Compute distances for selected pairs
    distances = distance_matrix[np.arange(n_centroids)[:, None], indices].ravel()

    return sources, targets, distances


def ball_query(x, centroids, radius, apply_pbc):
    """
    Performs a ball query around each centroid.

    Args:
        x(jnp.ndarray): The entire point cloud data. Shape (N, D), where N is the number of points and D is the dimensionality.
        centroids (jnp.ndarray): The coordinates of the centroids. Shape (M, D), where M is the number of centroids.
        radius (float): The radius within which to search for neighbors around each centroid.
        apply_pbc (Callable, optional): function to apply periodic boundary conditions

    Returns:
        jnp.ndarray: A list of boolean masks indicating which points are within the radius for each centroid.
    """
    # Compute the vector difference between positions
    dr = (centroids[:, None, :] - x[None, :, :]) + EPS
    if apply_pbc:
        dr = apply_pbc(dr)

    # Calculate the squared distance matrix
    distance_matrix = np.sum(dr**2, axis=-1)

    cutoff_dists = distance_matrix < radius**2
    return cutoff_dists.T


def update_if_mask_true(i, j, idx, sources, targets, distances, mask, distance_squared):
    """ Update the sources, targets, and distances arrays conditionally. """
    def true_fun(_):
        return (
            sources.at[idx].set(i),
            targets.at[idx].set(j),
            distances.at[idx].set(np.sqrt(distance_squared[i, j])),
            idx + 1
        )
    
    def false_fun(_):
        return (sources, targets, distances, idx)
    
    # Conditionally update
    return jax.lax.cond(mask[i, j], true_fun, false_fun, None)


@partial(jax.jit, static_argnums=(1,2))
def neighbors_within_radius(
    x: np.array,
    r: float,
    apply_pbc: Optional[Callable] = None,
):
    """Returns the neighbors within radius r of each node in x.

    Args:
        x (jnp.array): positions of nodes
        r (float): radial cutoff
        mask (jnp.array, optixonal): node mask. Defaults to None.

    Returns:
        sources, targets, distances: pairs of neighbors and their distances
    """
    n_nodes = x.shape[0]
    # Compute the vector difference between positions
    dr = (x[:, None, :] - x[None, :, :]) + EPS
    if apply_pbc:
        dr = apply_pbc(dr)
    # Calculate the distance matrix
    distance_matrix = np.sum(dr**2, axis=-1)
    # adj = distance_matrix <= r**2

    # return distance_matrix, adj.astype(np.int32)

    # Get adjacency matrix based on radial cutoff
    mask = (distance_matrix <= r**2) & (distance_matrix > 0)  # Excludes self-loops
    num_edges = np.sum(mask)
    
     # Initialize fixed size output arrays (pre-specify maximum possible edges)
    max_edges =  n_nodes * (n_nodes - 1) // 2  # Maximum edges if fully connected minus diagonal
    sources = np.zeros(max_edges, dtype=np.int32)
    targets = np.zeros(max_edges, dtype=np.int32)
    distances = np.zeros(max_edges, dtype=dr.dtype)

    idx = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):  # Avoid double-counting and diagonal
            sources, targets, distances, idx = update_if_mask_true(
                i, j, idx, sources, targets, distances, mask, distance_matrix
            )

    num_edges = idx
    return sources[:num_edges], targets[:num_edges], distances[:num_edges], num_edges



@partial(jax.jit, static_argnums=(1,))
def compute_distances(
    x: np.array,
    apply_pbc: Optional[Callable] = None,
):
    """Returns the pairwise distances of nodes in x.

    Args:
        x (jnp.array): positions of nodes
        mask (jnp.array, optixonal): node mask. Defaults to None.

    Returns:
        distances (jnp.array): distances between pairs of neighbors
    """
    n_nodes = x.shape[0]
    dr = x[:, None, :] - x[None, :, :]
    if apply_pbc:
        dr = apply_pbc(dr)
    return np.sqrt(np.sum(dr**2, axis=-1))


def build_graph(
    node_feats,
    global_feats,
    k,
    use_edges=True,
    apply_pbc: Optional[Callable] = None,
    n_radial_basis=0,
    r_max=1.,
    radius=None,
    use_3d_distances=False,
):
    n_batch = len(node_feats)
    n_nodes = node_feats.shape[1]
    n_node = np.array([[n_nodes]] * n_batch)

    if radius is not None:
        distances = jax.vmap(
            partial(compute_distances), in_axes=(0, None)
        )(node_feats[..., :3], apply_pbc)

        # print(distances.shape)
        mask = (distances < radius) & (distances > 0)  # Exclude self-distance
        
        sources, targets = np.where(mask)
        distances = distances[sources, targets]

        n_edge = np.array([np.sum(mask[i]) for i in range(x.shape[0])])  

    else:
        sources, targets, distances = jax.vmap(
            partial(nearest_neighbors), in_axes=(0, None, None)
        )(node_feats[..., :3], k, apply_pbc)

        n_edge = np.array(n_batch * [[k]])
    
    if use_edges:
        if use_3d_distances:
            edges = distances
        else:
            edges = np.sqrt(np.sum(distances ** 2, axis=-1, keepdims=True))

        if n_radial_basis > 0:
            edges = e3nn.bessel(edges, n=n_radial_basis, x_max=r_max)
            edges = np.squeeze(edges)

        # Concat last 2 dims if use_3d_distances
        if use_3d_distances:
            edges = edges.reshape(edges.shape[0], edges.shape[1], -1)

        if radius is not None:
            edges = np.concatenate(edges, axis=0)
    else:
        edges = None

    return jraph.GraphsTuple(
        n_node=n_node,
        n_edge=n_edge,
        nodes=node_feats,
        edges=edges,
        globals=global_feats,
        senders=sources,
        receivers=targets,
    )


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