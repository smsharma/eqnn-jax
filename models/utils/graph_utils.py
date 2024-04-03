import jax
import jax.numpy as np
import jraph

from functools import partial

EPS = 1e-7

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
    cell: np.array = None,
    mask: np.array = None,
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
    dr = (x[:, None, :] - x[None, :, :]) + EPS
    if cell != None:
        dr = apply_pbc(
            dr=dr,
            cell=cell,
        )

    # Calculate the distance matrix 
    distance_matrix = np.sum(dr**2, axis=-1)

    # Get indices of nearest neighbors
    indices = np.argsort(distance_matrix, axis=-1)[:, :k]

    # Create sources and targets arrays
    sources = np.repeat(np.arange(n_nodes), k)
    targets = indices.ravel()

    return sources, targets, dr[sources, targets]

def build_graph(halos, 
                tpcfs, 
                k, 
                use_pbc=True, 
                use_edges=True, 
                boxsize=1000., 
                unit_cell = np.array([[1.,0.,0.,],[0.,1.,0.], [0.,0.,1.]]), 
                mean=None, 
                std=None,
                use_rbf=False,
                sigma_num=8):
    
    # if mean is not None and std is not None:
    #     halos = halos * std + mean
    #     halos /= boxsize

    n_batch = len(halos)
    
    if use_pbc:
        if std is not None:
            halos += std[:3]
            cell = np.diag(boxsize/std[:3])
        else:
            cell = np.array([[1.,0.,0.,],[0.,1.,0.], [0.,0.,1.]])
    else:
        cell = None
        
    sources, targets, distances = jax.vmap(partial(nearest_neighbors), in_axes=(0, None, None))(halos[..., :3], k, cell)
    if use_pbc and std is not None:
        halos -= std[:3]
    
    if use_edges:
        edges = np.sqrt(np.sum(distances **2, axis=-1, keepdims=True))
        if use_rbf:
            min_sigma = np.min(edges)
            max_sigma = np.mean(edges)
            sigma_vals = np.linspace(min_sigma, max_sigma, num=sigma_num) 
            edges = [np.exp(- edges**2 / (2 * sigma**2)) for sigma in sigma_vals]
            edges = np.concatenate(edges, axis=-1)
    else:
        edges = None

    print('node features')
    print(halos)
    print('edge features')
    print(edges.shape)
    
    return jraph.GraphsTuple(
            n_node=np.array([[halos.shape[1]]]*n_batch),
            n_edge=np.array(n_batch * [[k]]),
            nodes=halos,
            edges=edges,
            globals=tpcfs,
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
