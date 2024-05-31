# $E(3)$ Equivariant Graph Neural Networks in Jax

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC--4.0--BY-red.svg)](https://creativecommons.org/licenses/by/4.0/deed.en)
[![Run Tests](https://github.com/smsharma/eqnn-jax/actions/workflows/tests.yml/badge.svg)](https://github.com/smsharma/eqnn-jax/actions/workflows/tests.yml)

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

## Requirements and tests

To install requirements:
```
pip install -r requirements.txt
```

To run tests (testing equivariance and periodic boundary conditions):
```
cd tests
pytest .
``` 

## Basic usage and examples

See [`notebooks/examples.ipynb`](./notebooks/examples.ipynb) for example usage of GNN, SEGNN, NequIP, and EGNN.

## Attribution

See [CITATION.cff](./CITATION.cff) for citation information. The implementation of SEGNN was partially inspired by [segnn-jax](https://github.com/gerkone/segnn-jax).