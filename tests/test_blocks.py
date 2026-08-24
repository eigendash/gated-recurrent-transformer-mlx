"""Tests for the transformer block itself.

The interesting checks here are causality (a block must not let position t see
position t+1) and an exact hand-computed attention pattern for a tiny case.
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from grt.blocks import (
    Block,
    CausalSelfAttention,
    CountingBlock,
    MLP,
    causal_mask,
    count_parameters,
    counting_block_factory,
    default_block_factory,
)


def test_causal_mask_is_lower_triangular():
    mask = np.array(causal_mask(4))
    expected = np.triu(np.full((4, 4), -np.inf), k=1)
    np.testing.assert_allclose(np.where(np.isfinite(expected), mask, 0.0), np.where(np.isfinite(expected), expected, 0.0))
    # Upper triangle is -inf and the diagonal plus below is exactly zero.
    assert np.all(np.isneginf(mask[np.triu_indices(4, k=1)]))
    assert np.all(mask[np.tril_indices(4)] == 0.0)


def test_attention_at_first_position_is_value_of_first_token():
    """Position 0 may only attend to itself, so its output is v_0 exactly."""
    mx.random.seed(0)
    attn = CausalSelfAttention(dim=8, n_heads=2)
    x = mx.random.normal((2, 5, 8))
    y = attn(x, causal_mask(5))

    # Recompute v_0 by hand from the qkv projection and compare.
    qkv = attn.qkv(x)
    _, _, v = mx.split(qkv, 3, axis=-1)
    expected = attn.out_proj(v[:, 0:1, :])[:, 0, :]
    np.testing.assert_allclose(np.array(y[:, 0, :]), np.array(expected), atol=1e-5)


def test_attention_matches_hand_computed_uniform_prefix_mean():
    """With q = k = 0 all allowed positions tie, so the output is the prefix mean of v.

    We set the q and k blocks of the fused projection to zero and the v block to
    the identity, so v = x, and the output projection to the identity.
    """
    dim = 4
    attn = CausalSelfAttention(dim=dim, n_heads=1)
    zero = mx.zeros((dim, dim))
    ident = mx.eye(dim)
    attn.qkv.weight = mx.concatenate([zero, zero, ident], axis=0)
    attn.qkv.bias = mx.zeros((3 * dim,))
    attn.out_proj.weight = ident
    attn.out_proj.bias = mx.zeros((dim,))

    x = mx.array(np.arange(1 * 5 * dim, dtype=np.float32).reshape(1, 5, dim))
    y = np.array(attn(x, causal_mask(5)))
    xn = np.array(x)
    for t in range(5):
        np.testing.assert_allclose(y[:, t, :], xn[:, : t + 1, :].mean(axis=1), atol=1e-5)


def test_block_is_causal():
    """Changing the input at position t must not change outputs before t."""
    mx.random.seed(1)
    block = Block(dim=16, n_heads=4)
    x = mx.random.normal((1, 6, 16))
    y = block(x, causal_mask(6))

    delta = mx.zeros((1, 6, 16)).at[:, 4, :].add(3.0)
    y_perturbed = block(x + delta, causal_mask(6))

    np.testing.assert_allclose(
        np.array(y[:, :4, :]), np.array(y_perturbed[:, :4, :]), atol=1e-6
    )
    assert not np.allclose(np.array(y[:, 4:, :]), np.array(y_perturbed[:, 4:, :]))


def test_block_and_mlp_shape_dtype_contracts():
    mx.random.seed(2)
    block = default_block_factory(dim=32, n_heads=4, mlp_ratio=2)
    x = mx.random.normal((3, 7, 32))
    y = block(x, causal_mask(7))
    assert y.shape == (3, 7, 32)
    assert y.dtype == mx.float32

    mlp = MLP(dim=32, mlp_ratio=2)
    assert mlp(x).shape == (3, 7, 32)


def test_block_parameter_count_matches_formula():
    """A pre-norm block with ratio 4 has 12 d^2 + 13 d parameters.

    attn: 3d^2 + 3d (qkv) + d^2 + d (out) ; mlp: 8d^2 + 5d ; norms: 4d.
    """
    for dim in (16, 64):
        block = Block(dim=dim, n_heads=4)
        expected = 12 * dim * dim + 13 * dim
        assert count_parameters(block) == expected


def test_counting_block_wraps_and_counts():
    counter = [0]
    factory = counting_block_factory(counter)
    block = factory(8, 2, 4)
    assert isinstance(block, CountingBlock)
    x = mx.zeros((1, 3, 8))
    block(x, causal_mask(3))
    block(x, causal_mask(3))
    assert counter[0] == 2


def test_attention_rejects_bad_head_split():
    with pytest.raises(ValueError):
        CausalSelfAttention(dim=8, n_heads=3)


def test_mask_matches_dense_reference_for_small_case():
    """SDPA with the additive mask equals a dense numpy softmax attention."""
    mx.random.seed(3)
    dim, heads, seq = 8, 2, 4
    attn = CausalSelfAttention(dim=dim, n_heads=heads)
    x = mx.random.normal((1, seq, dim))
    y = np.array(attn(x, causal_mask(seq)))

    qkv = np.array(attn.qkv(x))[0]
    q, k, v = np.split(qkv, 3, axis=-1)
    head_dim = dim // heads
    ref = np.zeros((seq, dim), dtype=np.float64)
    for h in range(heads):
        sl = slice(h * head_dim, (h + 1) * head_dim)
        scores = q[:, sl] @ k[:, sl].T / math.sqrt(head_dim)
        scores = np.where(np.triu(np.ones((seq, seq)), k=1) > 0, -np.inf, scores)
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        ref[:, sl] = weights @ v[:, sl]
    ref = ref @ np.array(attn.out_proj.weight).T + np.array(attn.out_proj.bias)
    np.testing.assert_allclose(y[0], ref, atol=1e-5)
