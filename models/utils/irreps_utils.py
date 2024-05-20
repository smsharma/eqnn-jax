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