# From https://github.com/gerkone/segnn-jax/blob/main/segnn_jax/irreps_computer.py

from math import prod

from e3nn_jax import Irreps


def balanced_irreps(lmax: int, feature_size: int, use_sh: bool = True) -> Irreps:
    """Allocates irreps uniformely up until level lmax with budget feature_size."""
    irreps = ["0e"]
    n_irreps = 1 + (lmax if use_sh else lmax * 2)
    total_dim = 0
    for level in range(1, lmax + 1):
        dim = 2 * level + 1
        multi = int(feature_size / dim / n_irreps)
        if multi == 0:
            break
        if use_sh:
            irreps.append(f"{multi}x{level}{'e' if (level % 2) == 0 else 'o'}")
            total_dim = multi * dim
        else:
            irreps.append(f"{multi}x{level}e+{multi}x{level}o")
            total_dim = multi * dim * 2

    # add scalars to fill missing dimensions
    irreps[0] = f"{feature_size - total_dim}x{irreps[0]}"

    return Irreps("+".join(irreps))


def weight_balanced_irreps(
    scalar_units: int, irreps_right: Irreps, lmax: int = None
) -> Irreps:
    """
    Determines irreps_left such that the parametrized tensor product
        Linear(tensor_product(irreps_left, irreps_right))
    has (at least) scalar_units weights.

    Args:
        scalar_units: number of desired weights
        irreps_right: irreps of the right tensor
        lmax: maximum level of spherical harmonics
    """
    # irrep order
    if lmax is None:
        lmax = irreps_right.lmax
    # linear layer with squdare weight matrix
    linear_weights = scalar_units**2
    # raise hidden features until enough weigths
    n = 0
    while True:
        n += 1
        irreps_left = (Irreps.spherical_harmonics(lmax) * n).sort().irreps.simplify()
        # number of paths
        tp_weights = sum(
            prod([irreps_left[i_1].mul ** 2, irreps_right[i_2].mul])
            for i_1, (_, ir_1) in enumerate(irreps_left)
            for i_2, (_, ir_2) in enumerate(irreps_right)
            for _, (_, ir_out) in enumerate(irreps_left)
            if ir_out in ir_1 * ir_2
        )
        if tp_weights >= linear_weights:
            break
    return Irreps(irreps_left)