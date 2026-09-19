"""Model-level tests for the recurrent stack: shapes, validation, accounting."""

import mlx.core as mx
import numpy as np
import pytest

from grt.blocks import count_parameters
from grt.config import GRTConfig
from grt.model import GRT


def tiny_config(**kwargs) -> GRTConfig:
    base = dict(vocab_size=32, max_seq_len=16, dim=16, n_heads=2, mlp_ratio=2)
    base.update(kwargs)
    return GRTConfig(**base)


def test_logits_shape_and_dtype():
    mx.random.seed(0)
    model = GRT(tiny_config(n_prelude=1, n_core=1, n_coda=1))
    tokens = mx.random.randint(0, 32, (2, 9))
    logits = model(tokens, R=1)
    assert logits.shape == (2, 9, 32)
    assert logits.dtype == mx.float32
    assert bool(mx.all(mx.isfinite(logits)))


def test_recurrence_defaults_to_config_and_can_be_overridden():
    mx.random.seed(1)
    model = GRT(tiny_config(recurrence=3))
    tokens = mx.random.randint(0, 32, (1, 5))
    assert model(tokens).shape == (1, 5, 32)
    assert model(tokens, R=1).shape == (1, 5, 32)


def test_rejects_bad_recurrence_and_bad_temperature():
    model = GRT(tiny_config())
    tokens = mx.random.randint(0, 32, (1, 5))
    with pytest.raises(ValueError):
        model(tokens, R=0)
    with pytest.raises(ValueError):
        model(tokens, R=-2)
    with pytest.raises(ValueError):
        model(tokens, R=1, temperature=0.0)
    with pytest.raises(ValueError):
        model(mx.zeros((1, 17), dtype=mx.int32), R=1)


def test_parameter_count_is_a_formula_of_unique_blocks_not_R():
    """params = n_unique * P_block + embedding/head/norm + P_modulation."""
    dim = 16
    for n_pre, n_core, n_coda in [(1, 1, 1), (1, 2, 1), (2, 3, 2)]:
        cfg = tiny_config(n_prelude=n_pre, n_core=n_core, n_coda=n_coda)
        model = GRT(cfg)
        ratio = cfg.mlp_ratio
        p_block = (4 + 2 * ratio) * dim * dim + (9 + ratio) * dim
        p_mod = 5 * dim * dim + 6 * dim
        p_embed = cfg.vocab_size * dim + cfg.max_seq_len * dim
        p_head = cfg.vocab_size * dim + 2 * dim
        expected = (n_pre + n_core + n_coda) * p_block + p_mod + p_embed + p_head
        assert model.parameter_count() == expected
        assert model.parameter_count() == count_parameters(model)
        assert model.unique_blocks == n_pre + n_core + n_coda


def test_parameter_count_does_not_move_with_recurrence():
    cfg = tiny_config(n_prelude=1, n_core=2, n_coda=1)
    a = GRT(cfg)
    b = GRT(cfg.with_recurrence(8))
    assert count_parameters(a) == count_parameters(b)
    # Same weights, different depth: R is only an argument.
    b.update(a.parameters())
    tokens = mx.random.randint(0, 32, (1, 6))
    assert count_parameters(b) == count_parameters(a)
    assert b(tokens, R=8).shape == a(tokens, R=1).shape


def test_layer_application_accounting():
    model = GRT(tiny_config(n_prelude=2, n_core=3, n_coda=1))
    assert model.layer_applications(1) == 6
    assert model.layer_applications(4) == 15
    with pytest.raises(ValueError):
        model.layer_applications(0)


def test_trace_reports_every_recurrence_step():
    mx.random.seed(2)
    model = GRT(tiny_config(n_prelude=1, n_core=1, n_coda=1))
    tokens = mx.random.randint(0, 32, (2, 6))
    logits, trace = model(tokens, R=4, return_trace=True)
    assert logits.shape == (2, 6, 32)
    assert len(trace["states"]) == 4
    assert len(trace["gates"]) == 4
    assert trace["h_pre"].shape == (2, 6, 16)
    # Step 0 moves the state away from the anchor.
    assert not bool(mx.array_equal(trace["states"][0], trace["anchor"]))
    for g in trace["gates"]:
        assert bool(mx.all(g >= 0.0)) and bool(mx.all(g <= 1.0))
    assert trace["block_applications"] == model.layer_applications(4)


def test_model_is_causal():
    """Logits at position t must not depend on tokens after t."""
    mx.random.seed(3)
    model = GRT(tiny_config(recurrence=3))
    tokens = mx.random.randint(0, 32, (1, 8))
    base = np.array(model(tokens, R=3))
    changed = np.concatenate([np.array(tokens)[:, :5], np.full((1, 3), 7)], axis=1)
    perturbed = np.array(model(mx.array(changed.astype(np.int32)), R=3))
    np.testing.assert_allclose(base[:, :5, :], perturbed[:, :5, :], atol=1e-5)


def test_prelude_injection_ablation_changes_the_result():
    mx.random.seed(4)
    tokens = mx.random.randint(0, 32, (1, 6))
    on = GRT(tiny_config(recurrence=2))
    off = GRT(tiny_config(recurrence=2, prelude_injection=False))
    off.update(on.parameters())
    a = np.array(on(tokens, R=2))
    b = np.array(off(tokens, R=2))
    assert not np.allclose(a, b, atol=1e-6)


def test_zero_block_counts_are_allowed_where_sensible():
    """A model with no prelude or no coda still runs; a zero-depth core cannot recur."""
    mx.random.seed(5)
    tokens = mx.random.randint(0, 32, (1, 4))
    model = GRT(tiny_config(n_prelude=0, n_core=1, n_coda=0))
    assert model(tokens, R=2).shape == (1, 4, 32)
    with pytest.raises(ValueError):
        GRT(tiny_config(n_prelude=0, n_core=0, n_coda=0))
