import jax.numpy as jnp
import jax
import flax.linen as nn
import jraph

from utils.graph_utils import nearest_neighbors
from models.egnn import EGNN

# Config
k = 10
n_batch = 2
n_nodes = 2000
positions_only = False
n_feat = 7  # 3 position + 3 velocity + 1 scalar

# Select data; not loaded in minimal example
x = x[:n_batch, :n_nodes, :n_feat]

# Compute adjacency
senders, receivers = jax.vmap(nearest_neighbors, in_axes=(0, None))(x[:n_batch], k)

# Init graph
graph = jraph.GraphsTuple(
    n_node=jnp.array(n_batch * [[n_nodes]]),
    n_edge=jnp.array(n_batch * [[k]]),
    nodes=x,
    edges=None,
    globals=jnp.ones((n_batch, 7)),
    senders=senders,
    receivers=receivers,
)


# Define model
class GraphWrapper(nn.Module):
    @nn.compact
    def __call__(self, x):
        model = jax.vmap(EGNN(message_passing_steps=3, d_hidden=32, n_layers=3, skip_connections=False, activation="gelu", positions_only=positions_only, use_fourier_features=True))
        return model(x)


model = GraphWrapper()
rng = jax.random.PRNGKey(42)

# Init model
graph_out, _ = model.init_with_output(rng, graph)
