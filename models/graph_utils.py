import jax
import jax.numpy as np
import jraph

from functools import partial


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