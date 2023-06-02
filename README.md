# E(n) Equivariant Graph Neural Networks (EGNN)

Jax implementation of E(n) Equivariant Graph Neural Network (EGNN) following [Satorras et al (2021)](tps://arxiv.org/abs/2102.09844).

## Examples

Minimal example in `minimal.py`. Full example with equivariance test in `notebooks/equivariance_test.py`.

## Implementation notes

- Experimented with a "multi-channel vector" version of position updates in `egnn.py`, but this hasn't been tested yet
- Can optionally use Fourier projections of relative distances with `use_fourier_features=True`

## TODO

- [ ] Fold in optional edge information
- [ ] Decouple position and velocity updates