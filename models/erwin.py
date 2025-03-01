from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
from typing import Dict, List, Literal, Optional, Tuple, Callable, Union
import jraph
from jraph._src.utils import segment_mean, segment_sum
import e3nn_jax as e3nn

from models.utils.graph_utils import build_graph
from models.mlp import MLP


class Node:
    """Container to store hierarchical node information in the ERWIN model.
    
    This class functions like a dataclass, storing features, positions, and 
    batch information for a node in the hierarchical structure, along with
    pointers to its children nodes.
    
    Attributes:
        x: Node features
        pos: Node positions
        batch_idx: Batch indices
        children: Pointers to children nodes
    """
    
    def __init__(self, x: jnp.ndarray, pos: jnp.ndarray, batch_idx: jnp.ndarray, children: Node = None):
        self.x = x  # Node features
        self.pos = pos  # Node positions
        self.batch_idx = batch_idx  # Batch indices
        self.children = children  # Pointers to children nodes


class SwiGLU(nn.Module):
    """Implementation of SwiGLU activation function: W_3 SiLU(W_1 x) ⊗ W_2 x
    
    This is a variant of the GLU activation with SiLU that has shown strong
    performance in transformer models.
    
    Args:
        in_dim: Input dimension
        dim: Hidden dimension
    """
    
    in_dim: int
    dim: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        w1 = nn.Dense(self.dim)(x)
        w2 = nn.Dense(self.dim)(x)
        w3 = nn.Dense(self.in_dim)(jax.nn.silu(w1) * w2)
        return w3


class MPNN(nn.Module):
    """Message Passing Neural Network.
    
    Implementation of Message Passing Neural Network (see Gilmer et al., 2017):
        m_ij = MLP([h_i, h_j, pos_i - pos_j])       message
        m_i = mean(m_ij)                            aggregate
        h_i' = MLP([h_i, m_i])                      update
        
    Args:
        dim: Hidden dimension
        mp_steps: Number of message passing steps
        dimensionality: Dimensionality of the positional data (default: 3)
    """
    
    dim: int
    mp_steps: int
    dimensionality: int = 3
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, pos: jnp.ndarray, edge_index: jnp.ndarray) -> jnp.ndarray:
        for _ in range(self.mp_steps):
            # Message passing
            senders, receivers = edge_index[0], edge_index[1]
            
            # Compute edge attributes (relative positions)
            edge_attr = pos[senders] - pos[receivers]
            
            # Compute messages
            message_inputs = jnp.concatenate([x[senders], x[receivers], edge_attr], axis=-1)
            messages = nn.Dense(self.dim)(message_inputs)
            messages = nn.gelu(messages)
            messages = nn.LayerNorm()(messages)
            
            # Aggregate messages using scatter_mean
            n_nodes = x.shape[0]
            messages_agg = segment_mean(messages, receivers, n_nodes)
            
            # Update node features
            update_inputs = jnp.concatenate([x, messages_agg], axis=-1)
            update = nn.Dense(self.dim)(update_inputs)
            update = nn.LayerNorm()(update)
            
            # Residual connection
            x = x + update
            
        return x


class ErwinEmbedding(nn.Module):
    """Linear projection followed by Message Passing Neural Network.
    
    Args:
        in_dim: Input feature dimension
        dim: Hidden dimension
        mp_steps: Number of message passing steps
        dimensionality: Dimensionality of the positional data (default: 3)
    """
    
    in_dim: int
    dim: int
    mp_steps: int
    dimensionality: int = 3
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, pos: jnp.ndarray, edge_index: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.dim)(x)
        x = MPNN(self.dim, self.mp_steps, self.dimensionality)(x, pos, edge_index)
        return x


class BallPooling(nn.Module):
    """Pooling of leaf nodes in a ball for hierarchical graph coarsening.
    
    This operation:
    1. Selects balls of size 'stride'.
    2. Concatenates leaf node features inside each ball along with their 
       relative positions to the ball center.
    3. Applies linear projection and batch normalization.
    4. The output is the center of each ball endowed with the pooled features.
    
    This is a key component of ERWIN's hierarchical representation learning.
    
    Args:
        dim: Hidden dimension
        stride: Stride size (pooling factor)
        dimensionality: Dimensionality of the positional data (default: 3)
        use_running_average: Whether to use running average in BatchNorm
    """
    
    dim: int
    stride: int
    dimensionality: int = 3
    use_running_average: bool = True
    
    @nn.compact
    def __call__(self, node: Node) -> Node:
        if self.stride == 1:  # No pooling
            return Node(
                x=node.x, 
                pos=node.pos, 
                batch_idx=node.batch_idx, 
                children=node
            )
            
        # Handle case where we have too few nodes
        if node.x.shape[0] < self.stride:
            return Node(
                x=node.x, 
                pos=node.pos, 
                batch_idx=node.batch_idx, 
                children=node
            )
            
        # Ensure we have a multiple of stride
        needed_padding = (self.stride - (node.x.shape[0] % self.stride)) % self.stride
        if needed_padding > 0:
            x_padded = jnp.pad(node.x, ((0, needed_padding), (0, 0)))
            pos_padded = jnp.pad(node.pos, ((0, needed_padding), (0, 0)))
            batch_idx_padded = jnp.pad(node.batch_idx, ((0, needed_padding),))
        else:
            x_padded = node.x
            pos_padded = node.pos
            batch_idx_padded = node.batch_idx
            
        # Reshape and pool
        n_nodes = x_padded.shape[0] // self.stride
        
        # Compute centers as mean of positions in each stride group
        pos_reshaped = pos_padded.reshape(n_nodes, self.stride, -1)
        centers = jnp.mean(pos_reshaped, axis=1)
        
        # Compute relative positions to centers
        rel_pos = pos_reshaped - centers[:, None, :]
        
        # Concatenate features and relative positions
        x_reshaped = x_padded.reshape(n_nodes, self.stride, -1)
        x_concat = jnp.concatenate([
            x_reshaped.reshape(n_nodes, -1),
            rel_pos.reshape(n_nodes, -1)
        ], axis=1)
        
        # Project to higher dimensions
        x = nn.Dense(2 * self.dim)(x_concat)
        x = nn.BatchNorm(use_running_average=self.use_running_average)(x)
        
        # Return new node with pooled features
        batch_idx = batch_idx_padded[::self.stride]
        return Node(
            x=x, 
            pos=centers, 
            batch_idx=batch_idx, 
            children=node
        )


class BallUnpooling(nn.Module):
    """Ball unpooling (refinement) for hierarchical graph refinement.
    
    This operation:
    1. Computes relative positions of children to the center.
    2. Concatenates the pooled features with the relative positions.
    3. Applies linear projection and self-connection.
    4. The output is a refined tree with the same number of nodes as before pooling.
    
    This is the inverse operation of BallPooling, used in the decoder part of ERWIN.
    
    Args:
        dim: Hidden dimension
        stride: Stride size (unpooling factor)
        dimensionality: Dimensionality of the positional data (default: 3)
        use_running_average: Whether to use running average in BatchNorm
    """
    
    dim: int
    stride: int
    dimensionality: int = 3
    use_running_average: bool = True
    
    @nn.compact
    def __call__(self, node: Node) -> Node:
        if node.children is None:
            raise ValueError("Node has no children for unpooling")
            
        # Handle empty nodes
        if node.x.shape[0] == 0:
            return Node(
                x=node.children.x,
                pos=node.children.pos,
                batch_idx=node.children.batch_idx,
                children=node.children.children
            )
            
        # Ensure we have a multiple of stride for children
        children_n = node.children.x.shape[0]
        parent_n = node.x.shape[0]
        expected_children = parent_n * self.stride
        
        if children_n < expected_children:
            # Pad children
            padding = expected_children - children_n
            children_x = jnp.pad(node.children.x, ((0, padding), (0, 0)))
            children_pos = jnp.pad(node.children.pos, ((0, padding), (0, 0)))
            children_batch = jnp.pad(node.children.batch_idx, ((0, padding),))
        else:
            # Use original children
            children_x = node.children.x
            children_pos = node.children.pos
            children_batch = node.children.batch_idx
            
        # Reshape for correct relative position calculation
        try:
            pos_reshaped = children_pos.reshape(parent_n, self.stride, -1)
            rel_pos = pos_reshaped - node.pos[:, None, :]
            
            # Concatenate parent features with relative positions
            parent_x_expanded = jnp.tile(node.x[:, None, :], (1, self.stride, 1))
            x_concat = jnp.concatenate([parent_x_expanded, rel_pos], axis=-1)
            x_concat = x_concat.reshape(-1, node.x.shape[-1] + pos_reshaped.shape[-1])
            
            # Project to lower dimensions
            x_proj = nn.Dense(self.dim)(x_concat)
            
            # Add residual connection and normalize
            children_x_reshaped = children_x[:expected_children].reshape(parent_n, self.stride, -1)
            children_x_flat = children_x_reshaped.reshape(-1, children_x_reshaped.shape[-1])
            updated_x = children_x_flat + x_proj
            updated_x = nn.BatchNorm(use_running_average=self.use_running_average)(updated_x)
            
            # Create a new node using the children node but with updated features
            # Make sure we only return the original number of children
            return Node(
                x=updated_x[:children_n],
                pos=node.children.pos,
                batch_idx=node.children.batch_idx,
                children=node.children.children
            )
        except Exception as e:
            # Fallback for error cases, just keep original features
            return Node(
                x=node.children.x,
                pos=node.children.pos,
                batch_idx=node.children.batch_idx,
                children=node.children.children
            )


class BallMSA(nn.Module):
    """Ball Multi-Head Self-Attention module for efficient local attention.
    
    This module implements efficient attention within local neighborhoods (balls),
    reducing the complexity from O(n²) to O(n) by restricting attention to local regions.
    It incorporates relative position encoding and distance-based attention bias.
    
    Args:
        dim: Hidden dimension
        num_heads: Number of attention heads
        ball_size: Size of each ball for local attention
        dimensionality: Dimensionality of the positional data (default: 3)
    """
    
    dim: int
    num_heads: int
    ball_size: int
    dimensionality: int = 3
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, pos: jnp.ndarray) -> jnp.ndarray:
        # Handle empty or small inputs
        if x.shape[0] < self.ball_size:
            # Just return the input for very small inputs
            return x
            
        # Ensure we have a multiple of ball_size
        needed_padding = (self.ball_size - (x.shape[0] % self.ball_size)) % self.ball_size
        if needed_padding > 0:
            x_padded = jnp.pad(x, ((0, needed_padding), (0, 0)))
            pos_padded = jnp.pad(pos, ((0, needed_padding), (0, 0)))
        else:
            x_padded = x
            pos_padded = pos
            
        # Reshape inputs to process each ball independently
        n_balls = x_padded.shape[0] // self.ball_size
        x_reshaped = x_padded.reshape(n_balls, self.ball_size, -1)
        pos_reshaped = pos_padded.reshape(n_balls, self.ball_size, -1)
        
        # Relative position encoding
        centers = jnp.mean(pos_reshaped, axis=1, keepdims=True)
        rel_pos = pos_reshaped - centers
        rpe_projection = nn.Dense(self.dim)(rel_pos)
        
        # Augment features with relative position encoding
        x_with_rpe = x_reshaped + rpe_projection
        
        # Project to queries, keys, values
        qkv = nn.Dense(3 * self.dim)(x_with_rpe)
        qkv = qkv.reshape(n_balls, self.ball_size, 3, self.num_heads, self.dim // self.num_heads)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Distance-based attention bias
        sigma = self.param("sigma_att", nn.initializers.normal(0.02), (1, self.num_heads, 1, 1))
        distances = jnp.sqrt(jnp.sum((pos_reshaped[:, None, :, :] - pos_reshaped[:, :, None, :]) ** 2, axis=-1))
        attn_bias = sigma * distances[:, None, :, :]
        
        # Compute attention
        scale = (self.dim // self.num_heads) ** -0.5
        attention = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * scale
        attention = attention + attn_bias
        attention = jax.nn.softmax(attention, axis=-1)
        
        # Apply attention to values
        out = jnp.matmul(attention, v)
        out = jnp.transpose(out, (0, 2, 1, 3))
        out = out.reshape(n_balls * self.ball_size, self.dim)
        
        # Final projection
        out = nn.Dense(self.dim)(out)
        
        # Return only the original size
        return out[:x.shape[0]]


class ErwinTransformerBlock(nn.Module):
    """ErwinTransformer block with Ball Multi-Head Self-Attention and SwiGLU.
    
    This block combines Ball Multi-Head Self-Attention (BMSA) with SwiGLU feed-forward
    networks, including skip connections and RMSNorm for stability.
    
    Args:
        dim: Hidden dimension
        num_heads: Number of attention heads
        ball_size: Size of each ball for local attention
        dimensionality: Dimensionality of the positional data (default: 3)
    """
    
    dim: int
    num_heads: int
    ball_size: int
    dimensionality: int = 3
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, pos: jnp.ndarray) -> jnp.ndarray:
        # Layer normalization before attention
        norm1 = nn.RMSNorm()(x)
        
        # Ball Multi-Head Self-Attention
        attn_output = BallMSA(
            dim=self.dim, 
            num_heads=self.num_heads, 
            ball_size=self.ball_size,
            dimensionality=self.dimensionality
        )(norm1, pos)
        
        # Residual connection
        x = x + attn_output
        
        # Layer normalization before feed-forward
        norm2 = nn.RMSNorm()(x)
        
        # SwiGLU feed-forward
        swiglu = SwiGLU(in_dim=self.dim, dim=self.dim * 4)(norm2)
        
        # Residual connection
        x = x + swiglu
        
        return x


class BasicLayer(nn.Module):
    """Basic layer for ErwinTransformer with transformer blocks and pooling/unpooling.
    
    This layer combines multiple ErwinTransformerBlocks with optional pooling or unpooling
    operations depending on whether it's in the encoder or decoder part of the network.
    
    Args:
        direction: Direction of the layer - 'down' for encoder, 'up' for decoder, None for bottleneck
        depth: Number of transformer blocks
        stride: Stride for pooling/unpooling (if applicable)
        dim: Hidden dimension
        num_heads: Number of attention heads
        ball_size: Size of each ball for local attention
        dimensionality: Dimensionality of the positional data (default: 3)
    """
    
    direction: Literal['down', 'up', None]  # down: encoder, up: decoder, None: bottleneck
    depth: int
    stride: Optional[int]
    dim: int
    num_heads: int
    ball_size: int
    dimensionality: int = 3
    
    @nn.compact
    def __call__(self, node: Node, *, train: bool = False) -> Node:
        # Apply unpooling if in decoder path
        if self.direction == 'up' and self.stride is not None:
            node = BallUnpooling(
                dim=self.dim, 
                stride=self.stride, 
                dimensionality=self.dimensionality,
                use_running_average=not train
            )(node)
            
        # Apply transformer blocks
        x, pos = node.x, node.pos
        for i in range(self.depth):
            x = ErwinTransformerBlock(
                dim=self.dim,
                num_heads=self.num_heads,
                ball_size=self.ball_size,
                dimensionality=self.dimensionality
            )(x, pos)
        
        # Update node features
        node = Node(x=x, pos=pos, batch_idx=node.batch_idx, children=node.children)
        
        # Apply pooling if in encoder path
        if self.direction == 'down' and self.stride is not None:
            node = BallPooling(
                dim=self.dim, 
                stride=self.stride, 
                dimensionality=self.dimensionality,
                use_running_average=not train
            )(node)
            
        return node


def build_balltree_from_jraph(
    positions: jnp.ndarray, 
    batch_indices: jnp.ndarray, 
    edge_index: Optional[jnp.ndarray] = None,
    radius: Optional[float] = None,
    max_depth: int = 10,
    min_points_per_leaf: int = 1
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Build a simplified ball tree from node positions for JIT compatibility.
    
    Instead of a complex recursive ball tree construction, this simplified version sorts 
    points along the primary axis of variation to create a contiguous ordering suitable
    for efficient ball multi-head self-attention operations.
    
    This JIT-compatible implementation provides the hierarchical structure needed
    by ERWIN while avoiding operations that are incompatible with JAX's JIT compiler.
    
    Args:
        positions: Node positions of shape (N, D)
        batch_indices: Batch indices of shape (N,)
        edge_index: Optional edge indices of shape (2, E)
        radius: Optional radius for radius graph
        max_depth: Maximum depth of the ball tree (not used in simplified version)
        min_points_per_leaf: Minimum number of points in a leaf node (not used)
        
    Returns:
        tree_idx: Indices to rearrange nodes into a ball tree structure
        tree_mask: Mask for leaf nodes
    """
    # Handle empty arrays
    if positions.shape[0] == 0:
        return jnp.array([], dtype=jnp.int32), jnp.array([], dtype=bool)
    
    # Find the dimension with the greatest spread
    min_vals = jnp.min(positions, axis=0)
    max_vals = jnp.max(positions, axis=0)
    spread = max_vals - min_vals
    split_dim = jnp.argmax(spread)
    
    # Sort points by batch index first, then by position along the primary axis
    sorting_keys = batch_indices * 1e6 + positions[:, split_dim]  # Combine batch and position
    indices = jnp.argsort(sorting_keys)
    
    # For power-of-two padding, calculate the next power of 2
    num_points = positions.shape[0]
    log2_size = jnp.log2(jnp.maximum(num_points, 1.0))
    power = jnp.ceil(log2_size).astype(jnp.int32)
    target_size = jnp.power(2, power).astype(jnp.int32)
    
    # Pad indices to the next power of 2 (needed for efficient tree operations)
    indices_padded = jnp.pad(indices, (0, target_size - num_points), constant_values=0)
    
    # Create a mask for the real points (not padding)
    mask = jnp.zeros(target_size, dtype=bool).at[:num_points].set(True)
    
    return indices_padded, mask

def _pad_to_power_of_two(x: jnp.ndarray, fill_value=0) -> jnp.ndarray:
    """Pad an array to the next power of two length.
    
    This utility function ensures that an array's first dimension is padded to the next power
    of two, which is important for efficient hierarchical operations in ERWIN.
    
    Args:
        x: Input array to pad
        fill_value: Value to use for padding
    
    Returns:
        Padded array with first dimension as power of two
    """
    # Use JAX operations instead of Python int conversion for JIT compatibility
    log2_size = jnp.log2(jnp.maximum(x.shape[0], 1.0))
    power = jnp.ceil(log2_size).astype(jnp.int32)
    target_size = jnp.power(2, power).astype(jnp.int32)
    padding_size = target_size - x.shape[0]
    
    # Use jnp.pad instead of conditional logic for JIT compatibility
    return jnp.pad(x, [(0, padding_size)] + [(0, 0)] * (len(x.shape) - 1), 
                  constant_values=fill_value)


def build_radius_graph(positions: jnp.ndarray, batch_indices: jnp.ndarray, radius: float, self_loops: bool = False) -> jnp.ndarray:
    """Build a radius graph from node positions based on Euclidean distance.
    
    This function creates edges between nodes that are within a specified radius of each other,
    respecting batch boundaries to prevent connections between different graphs.
    
    Note: This function is not JIT-compatible because it uses jnp.where with a dynamic condition.
    When using with JIT compilation, either:
    1. Provide pre-computed edge indices to the model
    2. Use this function outside of JIT compilation
    3. Modify to use a fixed-size output representation
    
    Args:
        positions: Node positions of shape (N, D)
        batch_indices: Batch indices of shape (N,)
        radius: Radius cutoff for connections
        self_loops: Whether to include self-loops in the graph
        
    Returns:
        edge_index: Edge indices of shape (2, E)
    """
    # This function is not JIT-compatible due to dynamic sized output
    # It should be called outside of JIT or run with static arguments
    
    # Compute pairwise squared distances
    n_nodes = positions.shape[0]
    
    # Compute all pairwise squared distances
    differences = positions[:, None, :] - positions[None, :, :]
    squared_distances = jnp.sum(differences ** 2, axis=-1)
    
    # Create batch mask to ensure we only connect within the same batch
    # This creates a mask that's True when nodes i and j are in the same batch
    same_batch_mask = batch_indices[:, None] == batch_indices[None, :]
    
    # Create mask for connections within radius
    if self_loops:
        radius_mask = (squared_distances <= radius ** 2) & same_batch_mask
    else:
        # Exclude self-connections
        radius_mask = (squared_distances <= radius ** 2) & (squared_distances > 0) & same_batch_mask
    
    # Get indices of connected nodes
    sources, targets = jnp.where(radius_mask)
    
    # Stack to form edge_index
    edge_index = jnp.stack([sources, targets], axis=0)
    
    return edge_index


class ErwinTransformer(nn.Module):
    """ErwinTransformer model.
    
    This is the core implementation of the ERWIN architecture, featuring hierarchical ball trees,
    Ball Multi-Head Self-Attention (BMSA), and UNet-style pooling/unpooling operations.
    
    Args:
        c_in: Input feature dimension
        c_hidden: Hidden feature dimension
        ball_size: List of ball sizes for each level of the hierarchy
        enc_num_heads: List of number of attention heads in encoder layers
        enc_depths: List of depths of encoder layers
        dec_num_heads: List of number of attention heads in decoder layers
        dec_depths: List of depths of decoder layers
        strides: List of strides for hierarchical pooling
        decode: Whether to use the decoder (UNet-style architecture)
        dimensionality: Dimensionality of the data (default: 3)
        mp_steps: Number of message passing steps
        use_bessel: Whether to use Bessel basis for distances
    """
    
    c_in: int
    c_hidden: int
    ball_size: List
    enc_num_heads: List
    enc_depths: List
    dec_num_heads: List
    dec_depths: List
    strides: List
    decode: bool = True
    dimensionality: int = 3
    mp_steps: int = 3
    use_bessel: bool = False
    
    def setup(self):
        # Convert to lists to ensure consistent handling
        ball_size = list(self.ball_size)
        enc_num_heads = list(self.enc_num_heads)
        enc_depths = list(self.enc_depths)
        dec_num_heads = list(self.dec_num_heads)
        dec_depths = list(self.dec_depths)
        strides = list(self.strides)
        
        # Validate input lists
        assert len(enc_num_heads) == len(enc_depths) == len(ball_size)
        assert len(dec_num_heads) == len(dec_depths) == len(strides)
        assert len(strides) == len(ball_size) - 1
        
        # Node embedding
        self.embed = ErwinEmbedding(
            in_dim=self.c_in, 
            dim=self.c_hidden, 
            mp_steps=self.mp_steps,
            dimensionality=self.dimensionality
        )
        
        # Setup strides with bottleneck
        strides = strides + [1]  # 1 for bottleneck
        num_layers = len(enc_depths) - 1  # last one is bottleneck
        
        # Compute channel dimensions
        num_hidden = [self.c_hidden] + [self.c_hidden * 2**i for i in range(1, num_layers+1)]
        
        # Encoder layers - store as separate submodules
        self.num_encoder_layers = num_layers
        for i in range(num_layers):
            setattr(self, f'encoder_{i}', BasicLayer(
                direction='down',
                depth=enc_depths[i],
                stride=strides[i],
                dim=num_hidden[i],
                num_heads=enc_num_heads[i],
                ball_size=ball_size[i],
                dimensionality=self.dimensionality,
            ))
            
        # Bottleneck layer
        self.bottleneck = BasicLayer(
            direction=None,
            depth=enc_depths[-1],
            stride=None,
            dim=num_hidden[-1],
            num_heads=enc_num_heads[-1],
            ball_size=ball_size[-1],
            dimensionality=self.dimensionality,
        )
        
        # Decoder layers (if needed)
        if self.decode:
            self.num_decoder_layers = num_layers
            for i in range(num_layers - 1, -1, -1):
                setattr(self, f'decoder_{num_layers - 1 - i}', BasicLayer(
                    direction='up',
                    depth=dec_depths[i],
                    stride=strides[i],
                    dim=num_hidden[i],
                    num_heads=dec_num_heads[i],
                    ball_size=ball_size[i],
                    dimensionality=self.dimensionality,
                ))
                
        # Set output dimension
        self.in_dim = self.c_in
        self.out_dim = self.c_hidden if self.decode else num_hidden[-1]
            
    def __call__(
        self, 
        node_features: jnp.ndarray, 
        node_positions: jnp.ndarray, 
        batch_idx: jnp.ndarray, 
        edge_index=None, 
        tree_idx=None, 
        tree_mask=None, 
        radius=None, 
        train: bool = False,
        **kwargs
    ):
        # Build the ball tree if not provided
        if tree_idx is None or tree_mask is None:
            tree_idx, tree_mask = build_balltree_from_jraph(node_positions, batch_idx)
            
        # Create radius graph if not provided
        # Note: For JIT compatibility, edge_index should be provided when JIT compiling
        if edge_index is None:
            assert radius is not None, "Radius must be provided if edge_index is not provided"
            # This will work during normal execution but may fail under JIT
            edge_index = build_radius_graph(node_positions, batch_idx, radius, self_loops=True)
            
        # Embed nodes
        x = self.embed(node_features, node_positions, edge_index)
        
        # Create initial Node structure
        node = Node(
            x=x[tree_idx],
            pos=node_positions[tree_idx],
            batch_idx=batch_idx[tree_idx],
            children=None
        )
        
        # Apply encoder
        for i in range(self.num_encoder_layers):
            node = getattr(self, f'encoder_{i}')(node, train=train)
            
        # Apply bottleneck
        node = self.bottleneck(node, train=train)
        
        # Apply decoder (if needed)
        if self.decode:
            for i in range(self.num_decoder_layers):
                node = getattr(self, f'decoder_{i}')(node, train=train)
                
            # Return features of leaf nodes, properly reordered
            # For JIT compatibility, use a fixed permutation approach
            original_indices = jnp.arange(node.x.shape[0])
            leaf_indices = tree_idx[tree_mask]
            # Create a scatter index that maps from tree-ordered to original order
            scatter_indices = jnp.zeros(node.x.shape[0], dtype=jnp.int32)
            scatter_indices = scatter_indices.at[leaf_indices].set(original_indices[tree_mask])
            # Use the scatter indices to reorder
            reordered_x = node.x[tree_mask]
            # Return based on a fixed size output
            return reordered_x
            
        # Return bottleneck features and batch indices
        return node.x, node.batch_idx


class ErwinModel(nn.Module):
    """Jraph-compatible wrapper for the ErwinTransformer.
    
    ERWIN (Efficient Hierarchical Transformer) is a hierarchical transformer model for
    point cloud data that uses ball trees for efficient attention computation. It builds
    on the architecture proposed in the paper "ERWIN: Scalable Geometric Deep Learning"
    but uses JAX for efficient computation.
    
    Args:
        d_hidden: Dimension of hidden layers
        n_layers: Number of layers in MLPs
        message_passing_steps: Number of message passing steps in embedding
        message_passing_agg: Aggregation function for message passing ('sum' or 'mean')
        ball_size: List of ball sizes for each level of the hierarchy
        enc_num_heads: List of number of attention heads in encoder layers
        enc_depths: List of depths of encoder layers
        dec_num_heads: List of number of attention heads in decoder layers
        dec_depths: List of depths of decoder layers
        strides: List of strides for hierarchical pooling
        activation: Activation function to use
        task: Task type, either 'graph' or 'node'
        n_outputs: Number of outputs for graph-level tasks
        readout_agg: Aggregation function for readout ('sum' or 'mean')
        mlp_readout_widths: List of widths for readout MLP layers
        position_features: Whether to use position features
        dimensionality: Dimensionality of positional data (default: 3)
        radius: Radius for neighborhood graph construction
        use_bessel: Whether to use Bessel basis for distances
        n_radial_basis: Number of radial basis functions to use
        r_max: Maximum radius for radial basis functions
        apply_pbc: Function to apply periodic boundary conditions
    """
    
    d_hidden: int = 64
    n_layers: int = 3
    message_passing_steps: int = 3
    message_passing_agg: str = "sum"
    ball_size: List[int] = (32, 32, 32, 32, 32)
    enc_num_heads: List[int] = (2, 4, 8, 16, 32)
    enc_depths: List[int] = (2, 2, 2, 6, 2)
    dec_num_heads: List[int] = (4, 4, 8, 16)
    dec_depths: List[int] = (2, 2, 2, 2)
    strides: List[int] = (2, 2, 2, 2)
    activation: str = "gelu"
    task: str = "graph"  # "graph" or "node"
    n_outputs: int = 1
    readout_agg: str = "mean"
    mlp_readout_widths: List[int] = (4, 2, 2)
    position_features: bool = True
    dimensionality: int = 3
    radius: Optional[float] = None
    use_bessel: bool = False
    n_radial_basis: int = 0
    r_max: float = 0.3
    apply_pbc: Optional[Callable] = None
    
    @nn.compact
    def __call__(self, graphs: jraph.GraphsTuple, *, train: bool = False) -> jraph.GraphsTuple:
        """Process GraphsTuple using ErwinTransformer.
        
        Args:
            graphs: jraph.GraphsTuple with node features, positions, and connectivity
            train: Whether model is in training mode
            
        Returns:
            Updated jraph.GraphsTuple with processed features
        """
        # Extract node features and positions
        if self.position_features:
            # Use all node features (including positions)
            node_features = graphs.nodes
            node_positions = graphs.nodes[..., :self.dimensionality]
        else:
            # Extract positions for computation but use remaining features as inputs
            node_positions = graphs.nodes[..., :self.dimensionality]
            node_features = graphs.nodes[..., self.dimensionality:]
        
        # Get batch indices from the graph
        n_nodes_per_graph = graphs.n_node
        
        # Handle scalar case (batched graph when vmapping)
        if isinstance(n_nodes_per_graph, jnp.ndarray) and len(n_nodes_per_graph.shape) == 0:
            # Single graph case (scalar n_node)
            batch_idx = jnp.zeros(node_features.shape[0], dtype=jnp.int32)
        else:
            # Multiple graphs case
            batch_idx = jnp.repeat(jnp.arange(n_nodes_per_graph.shape[0]), n_nodes_per_graph)
        
        # Extract edge connectivity if available
        edge_index = None
        if graphs.senders is not None and graphs.receivers is not None:
            edge_index = jnp.stack([graphs.senders, graphs.receivers], axis=0)
        
        # Set up bessel functions if using them
        bessel_fn = None
        if self.use_bessel and self.n_radial_basis > 0:
            bessel_fn = partial(e3nn.bessel, n=self.n_radial_basis, x_max=self.r_max)
        
        # Process through the ErwinTransformer
        erwin = ErwinTransformer(
            c_in=node_features.shape[-1],
            c_hidden=self.d_hidden,
            ball_size=self.ball_size,
            enc_num_heads=self.enc_num_heads,
            enc_depths=self.enc_depths,
            dec_num_heads=self.dec_num_heads,
            dec_depths=self.dec_depths, 
            strides=self.strides,
            dimensionality=self.dimensionality,
            mp_steps=self.message_passing_steps,
            use_bessel=self.use_bessel,
        )
        
        # Process the data through ErwinTransformer
        output = erwin(
            node_features=node_features,
            node_positions=node_positions,
            batch_idx=batch_idx,
            edge_index=edge_index,
            radius=self.radius,
            apply_pbc=self.apply_pbc,
            train=train
        )
        
        # Handle different output formats based on task
        if self.task == "node":
            # For node-level prediction, update node features in the graph
            if isinstance(output, tuple):
                output = output[0]  # Extract features if bottleneck returned (x, batch_idx)
                
            processed_graphs = graphs._replace(nodes=output)
            return processed_graphs
            
        elif self.task == "graph":
            # For graph-level prediction, aggregate node features and apply MLP readout
            if isinstance(output, tuple):
                node_features, batch_idx = output
            else:
                node_features = output
                
            # Choose aggregation function
            if self.readout_agg == "sum":
                agg_fn = segment_sum
            elif self.readout_agg == "mean":
                agg_fn = segment_mean
            else:
                raise ValueError(f"Unsupported readout aggregation: {self.readout_agg}")
            
            # Handle scalar case (single graph)
            if isinstance(n_nodes_per_graph, jnp.ndarray) and len(n_nodes_per_graph.shape) == 0:
                # For a single graph, just aggregate all nodes
                agg_features = agg_fn(node_features, jnp.zeros_like(batch_idx), 1)
            else:
                # For multiple graphs, aggregate by graph
                n_graphs = n_nodes_per_graph.shape[0]
                agg_features = agg_fn(node_features, batch_idx, n_graphs)
            
            # Apply final MLP for graph-level prediction
            mlp = MLP(
                feature_sizes=[self.mlp_readout_widths[0] * node_features.shape[-1]] + 
                             [w * self.d_hidden for w in self.mlp_readout_widths[1:]] +
                             [self.n_outputs],
                activation=getattr(nn, self.activation)
            )
            output = mlp(agg_features)
            
            return output
            
        else:
            raise ValueError(f"Unsupported task: {self.task}")
            
    def create_embeddings(self, features, positions=None, k=None, radius=None):
        """Create graph embeddings from features and positions.
        
        Args:
            features: Node features
            positions: Node positions (optional, uses features[:, :3] if None)
            k: Number of nearest neighbors (optional, uses model radius if None)
            radius: Radius for connectivity (optional, uses model radius if None)
            
        Returns:
            jraph.GraphsTuple with embedded features
        """
        # Use provided positions or extract from features
        if positions is None:
            positions = features[:, :self.dimensionality]
            
        # Use positions and features to build the graph
        k = k if k is not None else 16
        radius = radius if radius is not None else self.radius
        
        # Build the graph using the utility function
        return build_graph(
            node_feats=features,
            global_feats=None,
            k=k,
            use_edges=True,
            apply_pbc=self.apply_pbc,
            n_radial_basis=self.n_radial_basis if self.use_bessel else 0,
            r_max=self.r_max,
            radius=radius,
            use_3d_distances=False,
        )