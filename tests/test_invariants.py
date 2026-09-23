"""The invariants the architecture is supposed to guarantee.

These are the tests that would catch a broken recurrence: an R = 1 clean limit,
parameters that do not move with R, compute that does, the two gate limits, the
noise contract, prelude conditioning, and gradients reaching the shared core
from every iteration.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from grt.blocks import causal_mask, count_parameters, counting_block_factory
from grt.config import GRTConfig
from grt.model import GRT, PlainStack
from grt.modulation import gated_update


def cfg(**kwargs) -> GRTConfig:
    base = dict(vocab_size=32, max_seq_len=16, dim=16, n_heads=2, mlp_ratio=2)
    base.update(kwargs)
    return GRTConfig(**base)


def tokens(n=6, batch=2):
    mx.random.seed(1234)
    return mx.random.randint(0, 32, (batch, n))


# --------------------------------------------------------------------------- #
# 1. R = 1 is the clean limit
# --------------------------------------------------------------------------- #


def test_r1_with_modulation_ablated_is_bit_identical_to_the_plain_stack():
    mx.random.seed(0)
    full = GRT(cfg(n_prelude=1, n_core=1, n_coda=1))
    plain = PlainStack.from_grt(full)

    ablated = GRT(cfg(n_prelude=1, n_core=1, n_coda=1, use_projection=False, use_gate=False))
    ablated.update(full.parameters())
    x = tokens(7)
    assert bool(mx.array_equal(ablated(x, R=1), plain(x)))

    # gate_override = 0 also writes the core output through, so it must land on
    # the same logits as the plain stack bit for bit.
    forced = GRT(cfg(n_prelude=1, n_core=1, n_coda=1, use_projection=False))
    forced.update(full.parameters())
    assert bool(mx.array_equal(forced(x, R=1, gate_override=0.0), plain(x)))

    # And the same blocks in the same order: prelude, core, coda.
    assert [b for b in plain.blocks] == [full.prelude[0], full.core[0], full.coda[0]]


def test_r1_is_one_pass_over_the_blocks_for_a_deeper_core():
    """n_pre + n_core * 1 + n_coda blocks are visited at R = 1."""
    counter = [0]
    mx.random.seed(1)
    model = GRT(cfg(n_prelude=2, n_core=3, n_coda=1), block_factory=counting_block_factory(counter))
    model(tokens(5), R=1)
    assert counter[0] == 6 == model.layer_applications(1)


# --------------------------------------------------------------------------- #
# 2. Parameters are independent of R
# --------------------------------------------------------------------------- #


def test_parameter_count_and_shapes_do_not_depend_on_R():
    mx.random.seed(2)
    model = GRT(cfg(n_prelude=1, n_core=2, n_coda=1))
    before = count_parameters(model)
    shapes_before = {k: v.shape for k, v in nn.utils.tree_flatten(model.parameters())}

    x = tokens(6)
    for R in (1, 2, 3, 8, 16):
        model(x, R=R)
        assert count_parameters(model) == before
        assert {k: v.shape for k, v in nn.utils.tree_flatten(model.parameters())} == shapes_before

    # Two models that differ only in the default recurrence depth hold exactly
    # the same parameter tree.
    a = GRT(cfg(n_prelude=1, n_core=2, n_coda=1, recurrence=1))
    b = GRT(cfg(n_prelude=1, n_core=2, n_coda=1, recurrence=12))
    assert [k for k, _ in nn.utils.tree_flatten(a.parameters())] == [
        k for k, _ in nn.utils.tree_flatten(b.parameters())
    ]
    assert count_parameters(a) == count_parameters(b)

    # The parameter count only grows with unique blocks, not with R.
    counts = [GRT(cfg(n_prelude=1, n_core=c, n_coda=1)).parameter_count() for c in (1, 2, 3)]
    assert counts[0] < counts[1] < counts[2]


def test_parameters_are_shared_across_iterations_not_duplicated():
    """There is exactly one core block, however many times it runs."""
    mx.random.seed(3)
    model = GRT(cfg(n_prelude=1, n_core=2, n_coda=1))
    assert len(model.core) == 2
    names = [k for k, _ in nn.utils.tree_flatten(model.parameters())]
    # 12 tensors per block (two norms, attention, MLP), and exactly two blocks
    # however many times they are applied.
    assert sum(1 for k in names if k.startswith("core.")) == 2 * 12
    # One set of modulation parameters, not one per recurrence step.
    assert sum(1 for k in names if k.startswith("proj.")) == 1
    assert sum(1 for k in names if k.startswith("gate.")) == 8


# --------------------------------------------------------------------------- #
# 3. Compute scales linearly with R
# --------------------------------------------------------------------------- #


def test_block_applications_scale_linearly_with_R():
    counter = [0]
    mx.random.seed(4)
    n_pre, n_core, n_coda = 1, 2, 1
    model = GRT(
        cfg(n_prelude=n_pre, n_core=n_core, n_coda=n_coda),
        block_factory=counting_block_factory(counter),
    )
    x = tokens(6)
    measured, formula = [], []
    for R in (1, 2, 3, 4, 5):
        counter[0] = 0
        model(x, R=R)
        measured.append(counter[0])
        formula.append(model.layer_applications(R))
    assert measured == formula == [n_pre + n_core * R + n_coda for R in (1, 2, 3, 4, 5)]

    # Linear: equal increments of one core block per extra recurrence.
    increments = [measured[i + 1] - measured[i] for i in range(len(measured) - 1)]
    assert increments == [n_core] * len(increments)


def test_plain_stack_compute_is_flat_in_R():
    counter = [0]
    mx.random.seed(5)
    model = PlainStack(cfg(), n_layers=4, block_factory=counting_block_factory(counter))
    x = tokens(5)
    for R in (1, 3, 9):
        counter[0] = 0
        model(x, R=R)
        assert counter[0] == 4


# --------------------------------------------------------------------------- #
# 4. The gate really is a gate
# --------------------------------------------------------------------------- #


def test_gate_limits_pass_through_or_replace_the_state():
    """The brief's two ends, in the paper's notation.

    Eq. 4 is ``h = g * h + (1 - g) * o`` where ``g`` is the copy gate. So
    ``g = 1`` ignores the core output and passes the state through, and
    ``g = 0`` takes the core output -- the brief's "gate = 1 means taken" is
    the complement ``1 - g``.
    """
    mx.random.seed(6)
    model = GRT(cfg(n_prelude=1, n_core=1, n_coda=1, use_projection=False))
    x = tokens(5)
    R = 3

    _, kept = model(x, R=R, gate_override=1.0, return_trace=True)
    for state in kept["states"]:
        assert bool(mx.array_equal(state, kept["h_pre"]))
    assert bool(mx.array_equal(kept["states"][-1], kept["h_pre"]))

    # g = 0: the state after the last step is exactly the shared block output.
    logits, written = model(x, R=R, gate_override=0.0, return_trace=True)
    last = written["states"][-1]
    # Recompute the core output by hand from the state that entered the step.
    previous = written["h_pre"] if R == 1 else model(x, R=R - 1, gate_override=0.0, return_trace=True)[1]["states"][-1]
    manual = GRT._run_blocks(previous, model.core, causal_mask(x.shape[1]))
    assert bool(mx.array_equal(last, manual))
    assert not bool(mx.array_equal(last, written["h_pre"]))

    # The two ends give different logits: the gate is not a no-op.
    assert not bool(mx.array_equal(logits, model(x, R=R, gate_override=1.0)))


def test_gated_update_matches_eq4_elementwise():
    mx.random.seed(7)
    state = mx.random.normal((2, 3, 4))
    proposal = mx.random.normal((2, 3, 4))
    g = mx.random.uniform(shape=(2, 3, 4))
    np.testing.assert_allclose(
        np.array(gated_update(state, proposal, g)),
        np.array(g) * np.array(state) + (1.0 - np.array(g)) * np.array(proposal),
        atol=1e-6,
    )


def test_gate_starts_nearly_closed_so_the_state_survives_the_recurrence():
    """At initialisation the copy branch dominates: h^(R) stays near h^(0)."""
    mx.random.seed(8)
    model = GRT(cfg(n_prelude=1, n_core=1, n_coda=1))
    x = tokens(6)
    _, trace = model(x, R=4, return_trace=True)
    assert all(float(mx.mean(g)) > 0.9 for g in trace["gates"])
    drift = float(mx.max(mx.abs(trace["states"][-1] - trace["h_pre"])))
    scale = float(mx.max(mx.abs(trace["h_pre"])))
    assert drift < 0.5 * scale


# --------------------------------------------------------------------------- #
# 5. Noise
# --------------------------------------------------------------------------- #


def test_without_a_key_the_model_is_exactly_deterministic():
    mx.random.seed(9)
    model = GRT(cfg(n_prelude=1, n_core=1, n_coda=1))
    x = tokens(6)
    assert bool(mx.array_equal(model(x, R=3), model(x, R=3)))
    # ... whatever the configured noise scale is, because no key means no noise.
    noisy_cfg = GRT(cfg(n_prelude=1, n_core=1, n_coda=1, state_noise_std=1.0, gate_noise_std=1.0))
    noisy_cfg.update(model.parameters())
    assert bool(mx.array_equal(noisy_cfg(x, R=3), noisy_cfg(x, R=3)))


def test_the_same_key_reproduces_the_same_noise():
    mx.random.seed(10)
    model = GRT(cfg(n_prelude=1, n_core=1, n_coda=1))
    x = tokens(6)
    key = mx.random.key(0)
    assert bool(mx.array_equal(model(x, R=3, key=key), model(x, R=3, key=key)))


def test_different_keys_give_different_forwards():
    mx.random.seed(11)
    model = GRT(cfg(n_prelude=1, n_core=1, n_coda=1))
    x = tokens(6)
    a = np.array(model(x, R=3, key=mx.random.key(0)))
    b = np.array(model(x, R=3, key=mx.random.key(1)))
    assert not np.allclose(a, b, atol=1e-6)

    # Resampling at every step is visible: the states move within one forward.
    _, trace = model(x, R=3, key=mx.random.key(0), return_trace=True)
    assert not np.allclose(np.array(trace["states"][0]), np.array(trace["states"][1]), atol=1e-6)


def test_zero_noise_scale_reproduces_the_noiseless_forward():
    mx.random.seed(12)
    model = GRT(cfg(n_prelude=1, n_core=1, n_coda=1))
    x = tokens(6)
    keyed = model(x, R=3, key=mx.random.key(5), state_noise_std=0.0, gate_noise_std=0.0)
    assert bool(mx.array_equal(keyed, model(x, R=3)))


def test_gate_noise_alone_changes_the_result():
    """Isolate the two noise terms: state noise off, gate noise on."""
    mx.random.seed(13)
    model = GRT(cfg(n_prelude=1, n_core=1, n_coda=1))
    x = tokens(6)
    a = model(x, R=3, key=mx.random.key(2), state_noise_std=0.0, gate_noise_std=0.1)
    b = model(x, R=3, state_noise_std=0.0, gate_noise_std=0.1)
    assert not np.allclose(np.array(a), np.array(b), atol=1e-6)


# --------------------------------------------------------------------------- #
# 6. The prelude conditions every recurrence
# --------------------------------------------------------------------------- #


def test_the_anchor_changes_every_steps_gate_and_projection():
    mx.random.seed(14)
    model = GRT(cfg(n_prelude=1, n_core=1, n_coda=1))
    x = tokens(6)
    _, trace = model(x, R=4, return_trace=True)
    anchor = trace["anchor"]
    # A perturbation that survives layer norm (a constant shift does not).
    perturbed = anchor + mx.random.normal(anchor.shape)

    for step, state in enumerate(trace["states"]):
        base_gate = np.array(model.gate(state, anchor))
        moved_gate = np.array(model.gate(state, perturbed))
        assert not np.allclose(base_gate, moved_gate, atol=1e-6), f"gate frozen at step {step}"

        base_proj = np.array(model.proj(state, anchor))
        moved_proj = np.array(model.proj(state, perturbed))
        assert not np.allclose(base_proj, moved_proj, atol=1e-6), f"projection ignores anchor at {step}"


def test_perturbing_the_prelude_changes_every_recurrence_step():
    """End to end: nudge a prelude weight and watch all R states move."""
    mx.random.seed(15)
    model = GRT(cfg(n_prelude=1, n_core=1, n_coda=1))
    x = tokens(6)
    R = 4

    logits_before, trace_before = model(x, R=R, return_trace=True)

    nudged = GRT(cfg(n_prelude=1, n_core=1, n_coda=1))
    nudged.update(model.parameters())
    nudged.prelude[0].ln_attn.weight = nudged.prelude[0].ln_attn.weight * 1.05
    logits_after, trace_after = nudged(x, R=R, return_trace=True)

    assert not np.allclose(np.array(logits_before), np.array(logits_after), atol=1e-6)
    assert not np.allclose(np.array(trace_before["h_pre"]), np.array(trace_after["h_pre"]), atol=1e-6)
    for step in range(R):
        assert not np.allclose(
            np.array(trace_before["states"][step]), np.array(trace_after["states"][step]), atol=1e-6
        ), f"prelude perturbation did not reach step {step}"


def test_removing_the_prelude_injection_changes_every_step():
    mx.random.seed(16)
    x = tokens(6)
    on = GRT(cfg(n_prelude=1, n_core=1, n_coda=1))
    off = GRT(cfg(n_prelude=1, n_core=1, n_coda=1, prelude_injection=False))
    off.update(on.parameters())
    _, a = on(x, R=3, return_trace=True)
    _, b = off(x, R=3, return_trace=True)
    assert float(mx.max(mx.abs(a["anchor"]))) > 0.0
    assert bool(mx.array_equal(b["anchor"], mx.zeros_like(b["anchor"])))
    for step in range(3):
        assert not np.allclose(np.array(a["states"][step]), np.array(b["states"][step]), atol=1e-6)


# --------------------------------------------------------------------------- #
# 7. Gradients
# --------------------------------------------------------------------------- #


def _grad_norms_for(grads):
    flat = dict(nn.utils.tree_flatten(grads))
    core = {k: v for k, v in flat.items() if k.startswith("core.")}
    all_finite = all(bool(mx.all(mx.isfinite(v))) for v in flat.values())
    core_norm = float(
        mx.sqrt(sum(float(mx.sum(v * v)) for v in core.values()))
    )
    return all_finite, core_norm, core


def test_gradients_reach_the_shared_core_from_every_iteration():
    mx.random.seed(17)
    R = 4
    model = GRT(cfg(n_prelude=1, n_core=2, n_coda=1))
    x = tokens(6)

    def state_energy(model, x, step, R):
        _, trace = model(x, R=R, return_trace=True)
        return mx.sum(trace["states"][step] ** 2)

    for step in range(R):
        grad_fn = nn.value_and_grad(model, lambda m, x, s, r: state_energy(m, x, s, r))
        value, grads = grad_fn(model, x, step, R)
        assert bool(mx.isfinite(value))
        finite, core_norm, core = _grad_norms_for(grads)
        assert finite
        assert core, "no core gradients at all"
        assert core_norm > 0.0, f"no gradient reached the shared core at step {step}"


def test_gradients_of_the_lm_loss_are_finite_for_every_R():
    mx.random.seed(18)
    model = GRT(cfg(n_prelude=1, n_core=1, n_coda=1))
    x = tokens(6, batch=2)

    def loss_fn(model, x, R):
        logits = model(x, R=R, key=mx.random.key(0))
        return mx.sum(nn.log_softmax(logits, axis=-1) ** 2)

    for R in (1, 2, 5):
        grad_fn = nn.value_and_grad(model, loss_fn)
        value, grads = grad_fn(model, x, R)
        assert bool(mx.isfinite(value))
        for name, grad in nn.utils.tree_flatten(grads):
            assert bool(mx.all(mx.isfinite(grad))), f"{name} at R={R}"
        _, core_norm, _ = _grad_norms_for(grads)
        assert core_norm > 0.0


# --------------------------------------------------------------------------- #
# 8. Shapes and dtypes
# --------------------------------------------------------------------------- #


def test_shapes_and_dtypes_across_R():
    mx.random.seed(19)
    model = GRT(cfg(n_prelude=1, n_core=2, n_coda=1))
    x = tokens(7, batch=3)
    for R in (1, 4):
        logits, trace = model(x, R=R, return_trace=True)
        assert logits.shape == (3, 7, 32) and logits.dtype == mx.float32
        assert len(trace["states"]) == len(trace["gates"]) == R
        for state in trace["states"]:
            assert state.shape == (3, 7, 16) and state.dtype == mx.float32
        for gate in trace["gates"]:
            assert gate.shape == (3, 7, 16)
            assert float(mx.min(gate)) >= 0.0 and float(mx.max(gate)) <= 1.0
        assert trace["h_pre"].shape == (3, 7, 16)


def test_recurrence_must_be_positive_everywhere():
    model = GRT(cfg())
    x = tokens(4)
    for fn in (lambda: model(x, R=0), lambda: model.layer_applications(0)):
        with pytest.raises(ValueError):
            fn()
    with pytest.raises(ValueError):
        cfg().with_recurrence(0)
