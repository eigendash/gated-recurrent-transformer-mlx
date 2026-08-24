"""Pre-norm transformer blocks, and small accounting helpers.

The block is the nanoGPT block: layer norm, causal multi-head self-attention
with a residual, layer norm, MLP with a residual. The paper builds on nanoGPT
so this keeps the comparison against the non-recurrent baseline fair -- both
models use the exact same block class.
"""

import math

import mlx.core as mx
import mlx.nn as nn


def causal_mask(seq_len: int) -> mx.array:
    """Additive mask of shape ``(seq_len, seq_len)``, ``-inf`` above the diagonal."""
    rows = mx.arange(seq_len)[:, None]
    cols = mx.arange(seq_len)[None, :]
    return mx.where(cols > rows, -mx.inf, 0.0).astype(mx.float32)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention, no positional encoding of its own."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"dim {dim} is not divisible by n_heads {n_heads}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        # (B, T, H, Dh) -> (B, H, T, Dh)
        shape = (batch, seq_len, self.n_heads, self.head_dim)
        q = q.reshape(shape).transpose(0, 2, 1, 3)
        k = k.reshape(shape).transpose(0, 2, 1, 3)
        v = v.reshape(shape).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(self.head_dim), mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, self.dim)
        return self.out_proj(out)


class MLP(nn.Module):
    """Token-wise feed-forward with a GELU, ``mlp_ratio`` times wider."""

    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        hidden = mlp_ratio * dim
        self.fc_in = nn.Linear(dim, hidden, bias=True)
        self.fc_out = nn.Linear(hidden, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc_out(nn.gelu(self.fc_in(x)))


class Block(nn.Module):
    """Pre-norm transformer block: ``x + attn(ln(x))`` then ``x + mlp(ln(x))``."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.ln_attn = nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads)
        self.ln_mlp = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        x = x + self.attn(self.ln_attn(x), mask)
        return x + self.mlp(self.ln_mlp(x))


def default_block_factory(dim: int, n_heads: int, mlp_ratio: int) -> Block:
    return Block(dim, n_heads, mlp_ratio)


class CountingBlock(nn.Module):
    """Wrap a block and count how many times it is executed.

    Used to measure the compute proxy (block applications per forward pass)
    instead of asserting it from a formula.
    """

    def __init__(self, block: nn.Module, counter: list):
        super().__init__()
        self.block = block
        self.counter = counter

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        self.counter[0] += 1
        return self.block(x, mask)


def counting_block_factory(counter: list):
    """Return a block factory that wraps every block it builds."""

    def factory(dim: int, n_heads: int, mlp_ratio: int) -> CountingBlock:
        return CountingBlock(Block(dim, n_heads, mlp_ratio), counter)

    return factory


def count_parameters(module: nn.Module) -> int:
    """Total number of scalar parameters in a module tree."""
    return int(sum(p.size for _, p in nn.utils.tree_flatten(module.parameters())))
