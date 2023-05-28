import jax
import jax.numpy as np
import flax.linen as nn

from typing import Sequence, Callable



class MLP(nn.Module):
    """A simple MLP."""

    feature_sizes: Sequence[int]
    activation: Callable[[np.array], np.array] = nn.gelu

    @nn.compact
    def __call__(self, x):
        for features in self.feature_sizes[:-1]:
            x = nn.Dense(features)(x)
            x = self.activation(x)

        # No activation on final layer
        x = nn.Dense(self.feature_sizes[-1])(x)
        return x