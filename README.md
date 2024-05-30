# $E(3)$ Equivariant Graph Neural Networks in Jax

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC--4.0--BY-red.svg)](https://creativecommons.org/licenses/by/4.0/deed.en)

Implementation of $E(3)$ equivariant graph neural networks in Jax.

## Models

The following equivariant models are implemented:

- [EGNN](./models/egnn.py) ([Satorras et al 2021](https://arxiv.org/abs/2102.09844))
- [SEGNN](./models/segnn.py) ([Brandstetter et al 2021](https://arxiv.org/abs/2110.02905))
- [NequIP](./models/nequip.py) ([Batzner et al 2021](https://arxiv.org/abs/2101.03164))

Additionally, the following non-equivariant models are implemented:

- [Graph Network](./models/gnn.py) ([Battaglia et al 2018](https://arxiv.org/abs/1806.01261))
- [PointNet++](./models/pointnet.py) ([Qi et al 2017](https://arxiv.org/abs/1706.02413))
- [DiffPool](./models/diffpool.py) ([Ying et al 2018](https://arxiv.org/abs/1806.08804))
- [Set Transformer](./models/transformer.py) ([Lee et al 2019](https://arxiv.org/abs/1810.00825))

## Requirements

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

## Attribution

[segnn-jax](https://github.com/gerkone/segnn-jax)