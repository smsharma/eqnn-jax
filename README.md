# Equivariant Neural Networks in Jax

[![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)

> [!NOTE]  
> Work in progress. So far, implements EGNN, SEGNN, NequIP.

## TODO

- [ ] Double check regroup/simplify and gate scalars spec
- [ ] Move distance vector to privileged position as attribute

> [!CAUTION]  
> Old README below.

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
- The position and velocity updates (for the velocity-included version) are coupled at the moment as in the original paper (Eq. 7)
- Can optionally use Fourier projections of relative distances with `use_fourier_features=True`
