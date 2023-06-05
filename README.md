# E(n) Equivariant Graph Neural Networks (EGNN)

Jax implementation of E(n) Equivariant Graph Neural Network (EGNN) following [Satorras et al (2021)](https://arxiv.org/abs/2102.09844).

## Basic usage

```py
from models.egnn import EGNN

model = EGNN(message_passing_steps=3,  # Number of message-passing rounds
    d_hidden=32, n_layers=3, activation="gelu",  # Edge/position/velocity/scalar-update MLP attributes 
    positions_only=True,  # Position-only (3 + scalar features) or including velocities (3 + 3 + scalar features) 
    use_fourier_features=True,  # Whether to use a Fourier feature projection of input relative coordinates
    tanh_out=False)  # Tanh-activate the position-update scalars, i.e. (x_i - x_j) * Tanh(scalars) which sometimes helps with stability

rng = jax.random.PRNGKey(42)
graph_out, params = model.init_with_output(rng, graph)  # graph is a jraph.GraphsTuple 
```
## Examples

Minimal example in `minimal.py`. Full example with equivariance test in `notebooks/equivariance_test.ipynb`.

## Implementation notes

- The model takes either 3-D coordinates (`positions_only=True`) or 3-D coordinates and velocities (`positions_only=False`). Additional scalar nodes features and global graph attributes are handled separately.
- Experimented with a "multi-channel vector" version of position updates in `egnn.py`, but this hasn't been tested yet
- Can optionally use Fourier projections of relative distances with `use_fourier_features=True`

## TODO

- [ ] Fold in optional edge information
- [ ] Decouple position and velocity updates
- [ ] Add attention
- [ ] Benchmark