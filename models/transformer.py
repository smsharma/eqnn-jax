import jax
import jax.numpy as np
from flax import linen as nn
from typing import Optional, Callable, List

from models.mlp import MLP


class MultiHeadAttentionBlock(nn.Module):
    """Multi-head attention. Uses pre-LN configuration (LN within residual stream), which seems to work much better than post-LN."""

    n_heads: int
    d_model: int
    d_mlp: int

    @nn.compact
    def __call__(self, x, y, mask=None):
        mask = None if mask is None else mask[..., None, :, :]

        # Multi-head attention
        if x is y:  # Self-attention
            x_sa = nn.LayerNorm()(x)  # pre-LN
            x_sa = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros,
            )(x_sa, x_sa, mask=mask)
        else:  # Cross-attention
            x_sa, y_sa = nn.LayerNorm()(x), nn.LayerNorm()(y)
            x_sa = nn.MultiHeadDotProductAttention(
                num_heads=self.n_heads,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros,
            )(x_sa, y_sa, mask=mask)

        # Add into residual stream
        x += x_sa

        # MLP
        x_mlp = nn.LayerNorm()(x)  # pre-LN
        x_mlp = nn.gelu(nn.Dense(self.d_mlp)(x_mlp))
        x_mlp = nn.Dense(self.d_model)(x_mlp)

        # Add into residual stream
        x += x_mlp

        return x


class PoolingByMultiHeadAttention(nn.Module):
    """PMA block from the Set Transformer paper."""

    n_seed_vectors: int
    n_heads: int
    d_model: int
    d_mlp: int

    @nn.compact
    def __call__(self, z, mask=None):
        seed_vectors = self.param(
            "seed_vectors",
            nn.linear.default_embed_init,
            (self.n_seed_vectors, z.shape[-1]),
        )
        seed_vectors = np.broadcast_to(seed_vectors, z.shape[:-2] + seed_vectors.shape)
        mask = None if mask is None else mask[..., None, :]
        return MultiHeadAttentionBlock(
            n_heads=self.n_heads, d_model=self.d_model, d_mlp=self.d_mlp
        )(seed_vectors, z, mask)


class Transformer(nn.Module):
    """Simple decoder-only transformer for set modeling.
    Attributes:
      d_model: The dimension of the model embedding space.
      d_mlp: The dimension of the multi-layer perceptron (MLP) used in the feed-forward network.
      n_layers: Number of transformer layers.
      n_heads: The number of attention heads.
      induced_attention: Whether to use induced attention.
      n_inducing_points: The number of inducing points for induced attention.
      n_outputs: The number of outputs for graph-level readout.
      readout_agg: Aggregation function for readout, "sum", "mean", or "max".
      mlp_readout_widths: Widths of the MLPs used in the readout.
      task: The task to perform, either 'graph' or 'node'.
    """

    d_model: int = 128
    d_mlp: int = 512
    n_layers: int = 4
    n_heads: int = 4
    induced_attention: bool = False
    n_inducing_points: int = 32
    n_outputs: int = 2
    readout_agg: str = "mean"
    mlp_readout_widths: List[int] = (2, 1)  # Factor of d_hidden for global readout MLPs
    task: str = "graph"  # "graph" or "node"

    @nn.compact
    def __call__(self, x: np.ndarray, mask=None):

        # Input embedding
        x = nn.Dense(int(self.d_model))(x)  # (batch, seq_len, d_model)

        # Transformer layers
        for _ in range(self.n_layers):
            if not self.induced_attention:  # Vanilla self-attention
                mask_attn = (
                    None if mask is None else mask[..., None] * mask[..., None, :]
                )
                x = MultiHeadAttentionBlock(
                    n_heads=self.n_heads,
                    d_model=self.d_model,
                    d_mlp=self.d_mlp,
                )(x, x, mask_attn,)
            else:  # Induced attention from Set Transformer paper
                h = PoolingByMultiHeadAttention(
                    self.n_inducing_points,
                    self.n_heads,
                    d_model=self.d_model,
                    d_mlp=self.d_mlp,
                )(x, mask)
                mask_attn = None if mask is None else mask[..., None]
                x = MultiHeadAttentionBlock(
                    n_heads=self.n_heads, d_model=self.d_model, d_mlp=self.d_mlp
                )(x, h, mask_attn)

        # Final LN as in pre-LN configuration
        x = nn.LayerNorm()(x)

        if self.task == "node":  # Node-level prediction
            x = MLP([int(self.d_mlp * w) for w in self.mlp_readout_widths] + [self.n_outputs], activation=nn.gelu)(x)

        elif self.task == "graph":  # Graph-level prediction

            if self.readout_agg not in ["sum", "mean", "max"]:
                raise ValueError(
                    f"Invalid message passing aggregation function {self.message_passing_agg}"
                )

            aggregate_fn = getattr(np, self.readout_agg)
            x = aggregate_fn(x, axis=-2)  # Aggregate along seq dim; (batch, d_model)

            # Graph-level MLP
            x = MLP([int(self.d_mlp * w) for w in self.mlp_readout_widths] + [self.n_outputs], activation=nn.gelu)(x)

        return x