"""Tests for the recurrent modulation: projection, gate limits, noise, gradients."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from grt.modulation import (
    ElementwiseGate,
    RecurrentProjection,
    gated_update,
    sample_gate_noise,
    sample_state_noise,
    split_key,
)


def test_projection_is_a_linear_function_of_both_inputs():
    """Hand-computed: with identity-ish weights, W_proj concatenates and scales."""
    dim = 3
    proj = RecurrentProjection(dim)
    weight = np.zeros((dim, 2 * dim), dtype=np.float32)
    weight[0, 0] = 2.0  # read state[0]
    weight[1, dim + 1] = 3.0  # read anchor[1]
    weight[2, 0] = 1.0
    weight[2, dim + 2] = 1.0
    proj.linear.weight = mx.array(weight)

    state = mx.array(np.arange(1 * 2 * dim, dtype=np.float32).reshape(1, 2, dim))
    anchor = mx.array((np.arange(1 * 2 * dim, dtype=np.float32) + 10.0).reshape(1, 2, dim))
    out = np.array(proj(state, anchor))

    s = np.array(state)
    a = np.array(anchor)
    np.testing.assert_allclose(out[:, :, 0], 2.0 * s[:, :, 0], atol=1e-6)
    np.testing.assert_allclose(out[:, :, 1], 3.0 * a[:, :, 1], atol=1e-6)
    np.testing.assert_allclose(out[:, :, 2], s[:, :, 0] + a[:, :, 2], atol=1e-6)
    # The unused anchor slot must not leak in anywhere.
    out_other = np.array(proj(state, anchor * 0.0 + mx.array(np.zeros_like(a) - 5.0)))
    np.testing.assert_allclose(out_other[:, :, 0], out[:, :, 0], atol=1e-6)


def test_projection_shape_dtype():
    proj = RecurrentProjection(16)
    state = mx.random.normal((2, 7, 16))
    anchor = mx.random.normal((2, 7, 16))
    out = proj(state, anchor)
    assert out.shape == (2, 7, 16)
    assert out.dtype == mx.float32


def test_gate_limits_are_bit_exact():
    """Eq. 4 at the two ends: g = 1 keeps the state, g = 0 takes the proposal."""
    state = mx.random.normal((2, 3, 8))
    proposal = mx.random.normal((2, 3, 8))

    kept = gated_update(state, proposal, mx.ones((2, 3, 8)))
    assert mx.array_equal(kept, state)

    written = gated_update(state, proposal, mx.zeros((2, 3, 8)))
    assert mx.array_equal(written, proposal)

    # And a half-open gate really does interpolate both.
    half = gated_update(state, proposal, mx.full((2, 3, 8), 0.5))
    np.testing.assert_allclose(np.array(half), 0.5 * (np.array(state) + np.array(proposal)), atol=1e-6)


def test_gate_is_elementwise_and_bounded():
    mx.random.seed(4)
    gate = ElementwiseGate(8)
    state = mx.random.normal((2, 5, 8))
    anchor = mx.random.normal((2, 5, 8))
    g = gate(state, anchor)
    assert g.shape == (2, 5, 8)
    assert bool(mx.all(g >= 0.0)) and bool(mx.all(g <= 1.0))

    # Perturbing a single element of the state changes that element's gate and,
    # through the MLP, others: the gate is elementwise but not diagonal.
    delta = mx.zeros_like(state).at[:, 3, 2].add(5.0)
    g2 = gate(state + delta, anchor)
    assert not np.allclose(np.array(g), np.array(g2))


def test_gate_init_bias_opens_the_gate():
    """With the MLP zeroed out the gate is exactly sigmoid(bias); default +4 -> 0.982."""
    gate = ElementwiseGate(4, init_bias=4.0)
    gate.fc_in.weight = mx.zeros_like(gate.fc_in.weight)
    gate.fc_in.bias = mx.zeros_like(gate.fc_in.bias)
    gate.fc_out.weight = mx.zeros_like(gate.fc_out.weight)
    g = gate(mx.random.normal((1, 2, 4)), mx.random.normal((1, 2, 4)))
    np.testing.assert_allclose(np.array(g), 1.0 / (1.0 + np.exp(-4.0)), atol=1e-6)

    # And with the random init the gate still starts nearly closed (i.e. g ~ 1).
    mx.random.seed(5)
    random_gate = ElementwiseGate(64, init_bias=4.0)
    g0 = random_gate(mx.random.normal((1, 4, 64)), mx.random.normal((1, 4, 64)))
    assert float(mx.mean(g0)) > 0.9


def test_gate_conditioned_on_the_anchor():
    """Perturbing only the anchor moves the gate: the prelude conditions it.

    The perturbation has to break the anchor's mean/variance profile, because
    the gate reads ``LN(anchor)`` and layer norm is invariant to shifting and
    positive rescaling. A constant shift therefore leaves the gate unchanged,
    which is a property of Eq. 5 and not a bug.
    """
    mx.random.seed(6)
    gate = ElementwiseGate(16)
    state = mx.random.normal((1, 3, 16))
    anchor = mx.random.normal((1, 3, 16))
    base = np.array(gate(state, anchor))

    np.testing.assert_allclose(
        base, np.array(gate(state, anchor + 0.5)), atol=1e-6
    )  # shift invariance through LN
    moved = np.array(gate(state, anchor + mx.random.normal(anchor.shape)))
    assert not np.allclose(base, moved, atol=1e-6)
    # Zeroing the anchor must also move it: it is read, not just carried along.
    zeroed = np.array(gate(state, mx.zeros_like(anchor)))
    assert not np.allclose(base, zeroed, atol=1e-6)


def test_temperature_flattens_the_gate():
    mx.random.seed(7)
    gate = ElementwiseGate(16)
    state = mx.random.normal((1, 4, 16))
    anchor = mx.random.normal((1, 4, 16))
    sharp = np.array(gate(state, anchor, temperature=1.0))
    flat = np.array(gate(state, anchor, temperature=1e6))
    assert np.all(np.abs(flat - 0.5) < 1e-3)
    assert np.mean(np.abs(sharp - 0.5)) > np.mean(np.abs(flat - 0.5))


def test_gate_noise_shifts_the_logits_by_a_scalar():
    mx.random.seed(8)
    gate = ElementwiseGate(16)
    state = mx.random.normal((1, 4, 16))
    anchor = mx.random.normal((1, 4, 16))
    key = mx.random.key(0)
    eps = float(np.array(sample_gate_noise(0.1, key))[0])
    noisy = np.array(gate(state, anchor, key=key, noise_std=0.1))
    clean = np.array(gate(state, anchor))
    logit_shift = np.log(noisy / (1 - noisy)) - np.log(clean / (1 - clean))
    np.testing.assert_allclose(logit_shift, eps, atol=1e-5)


def test_noise_sampling_is_seeded_and_optional():
    key = mx.random.key(0)
    a = sample_state_noise((2, 3), 0.1, key)
    b = sample_state_noise((2, 3), 0.1, key)
    c = sample_state_noise((2, 3), 0.1, split_key(key)[0])
    assert mx.array_equal(a, b)
    assert not mx.array_equal(a, c)
    assert float(mx.std(c)) > 0.0

    # Zero standard deviation is exactly no noise, even with a key present.
    assert mx.array_equal(sample_state_noise((2, 3), 0.0, key), mx.zeros((2, 3)))
    assert mx.array_equal(sample_gate_noise(0.0, key), mx.zeros((1,)))

    # A split key gives independent noise streams.
    k0, k1 = split_key(key)
    assert not mx.array_equal(sample_state_noise((4,), 1.0, k0), sample_state_noise((4,), 1.0, k1))


def test_gradients_flow_through_projection_and_gate():
    mx.random.seed(9)
    proj = RecurrentProjection(8)
    gate = ElementwiseGate(8)
    state = mx.random.normal((1, 3, 8))
    anchor = mx.random.normal((1, 3, 8))
    proposal = mx.random.normal((1, 3, 8))

    def loss(params):
        proj.update(params[0])
        gate.update(params[1])
        h = proj(state, anchor)
        g = gate(state, anchor, key=mx.random.key(0), noise_std=0.1)
        return mx.sum(gated_update(h, proposal, g) ** 2)

    params = [proj.parameters(), gate.parameters()]
    grads = mx.grad(loss)(params)
    for tree in grads:
        for name, grad in nn.utils.tree_flatten(tree):
            assert bool(mx.all(mx.isfinite(grad))), name
    gnorms = [float(mx.sqrt(sum(float(mx.sum(g * g)) for _, g in nn.utils.tree_flatten(t)))) for t in grads]
    assert all(n > 0.0 for n in gnorms)


def test_zero_temperature_saturates_the_gate():
    """tau = 0 divides by zero and saturates every gate to 0 or 1, so GRT rejects it."""
    mx.random.seed(10)
    gate = ElementwiseGate(4)
    state = mx.random.normal((1, 2, 4))
    anchor = mx.random.normal((1, 2, 4))
    g = np.array(gate(state, anchor, temperature=0.0))
    assert np.all((g == 0.0) | (g == 1.0))
