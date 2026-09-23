"""Tests for the synthetic recall task and the training helpers."""

import math

import mlx.core as mx
import numpy as np
import pytest

from grt.config import GRTConfig
from grt.model import GRT
from grt.tasks import MQARConfig, batch_accuracy, batch_loss, make_batch, to_mlx
from grt.train import cosine_schedule, evaluate, train


def test_batch_shapes_and_dtypes():
    cfg = MQARConfig(n_pairs=6, n_keys=8, batch_size=5)
    tokens, targets, mask = make_batch(np.random.default_rng(0), cfg)
    assert tokens.shape == (5, 18)
    assert targets.shape == (5, 18) and mask.shape == (5, 18)
    assert targets.dtype == np.int32 and mask.dtype == bool
    assert tokens.min() >= 0 and tokens.max() < cfg.vocab_size
    # Only the last n_pairs positions are scored.
    assert mask[:, : 2 * cfg.n_pairs].sum() == 0
    assert mask[:, 2 * cfg.n_pairs :].all()
    assert (targets[:, : 2 * cfg.n_pairs] == -1).all()


def test_query_targets_are_the_pairs_from_the_first_pass():
    """Reconstruct every query answer straight from the sequence."""
    cfg = MQARConfig(n_pairs=8, n_keys=16, batch_size=4)
    tokens, targets, mask = make_batch(np.random.default_rng(1), cfg)
    for row in range(cfg.batch_size):
        binding = {}
        for i in range(cfg.n_pairs):
            binding[int(tokens[row, 2 * i])] = int(tokens[row, 2 * i + 1])
        assert len(binding) == cfg.n_pairs, "keys in a sequence are distinct"
        for j in range(cfg.n_pairs):
            position = 2 * cfg.n_pairs + j
            assert targets[row, position] == binding[int(tokens[row, position])]
            assert mask[row, position]


def test_batch_loss_matches_hand_computation():
    logits = mx.array([[[0.0, 1.0, 0.0], [0.0, 0.0, 5.0]]])
    targets = mx.array([[-1, 2]], dtype=mx.int32)
    mask = mx.array([[False, True]])

    expected = -(5.0 - math.log(math.exp(0.0) + math.exp(0.0) + math.exp(5.0)))
    assert abs(float(batch_loss(logits, targets, mask)) - expected) < 1e-5


def test_batch_loss_is_the_mean_over_masked_positions_only():
    """Padding the batch with unscored positions must not change the loss."""
    logits = mx.random.normal((2, 4, 6))
    targets = mx.array([[1, 2, 3, 4], [5, 0, 1, 2]], dtype=mx.int32)
    mask = mx.array([[True, True, False, False], [True, True, False, False]])
    full = float(batch_loss(logits, targets, mask))

    only_first = float(batch_loss(logits[:, :2], targets[:, :2], mask[:, :2]))
    assert abs(full - only_first) < 1e-6

    # A different target at an unscored position must not matter either.
    moved = mx.where(mask, targets, mx.array([[0, 0, 5, 5], [0, 0, 4, 4]], dtype=mx.int32))
    assert abs(float(batch_loss(logits, moved, mask)) - full) < 1e-6


def test_batch_accuracy_hand_computed():
    logits = mx.array([[[3.0, 0.0], [0.0, 3.0], [0.0, 3.0]]])
    targets = mx.array([[0, 0, 1]], dtype=mx.int32)
    mask = mx.array([[True, True, True]])
    assert abs(float(batch_accuracy(logits, targets, mask)) - 2 / 3) < 1e-6


def test_uniform_logits_give_chance_accuracy():
    cfg = MQARConfig(n_pairs=8, n_keys=16, batch_size=64)
    rng = np.random.default_rng(3)
    tokens, targets, mask = to_mlx(make_batch(rng, cfg))
    logits = mx.random.normal((64, cfg.seq_len, cfg.vocab_size))
    accuracy = float(batch_accuracy(logits, targets, mask))
    assert abs(accuracy - cfg.chance_accuracy) < 0.02


def test_cosine_schedule_shape():
    peak, steps, warmup = 1e-3, 100, 10
    assert 0.0 < cosine_schedule(0, steps, peak=peak, warmup=warmup) < peak
    assert abs(cosine_schedule(warmup, steps, peak=peak, warmup=warmup) - peak) < 1e-12
    values = [cosine_schedule(s, steps, peak=peak, warmup=warmup) for s in range(steps)]
    assert values[warmup] == max(values)
    assert values[-1] < 0.2 * peak
    with pytest.raises(ValueError):
        cosine_schedule(0, steps, peak=-1.0, warmup=0)


def test_evaluate_is_deterministic_and_noise_free():
    mx.random.seed(0)
    cfg = MQARConfig(n_pairs=4, n_keys=8, batch_size=8)
    model = GRT(GRTConfig(vocab_size=cfg.vocab_size, max_seq_len=cfg.seq_len,
                          dim=16, n_heads=2, n_core=2, recurrence=3))
    a = evaluate(model, cfg, R=3, n_batches=2, seed=5)
    b = evaluate(model, cfg, R=3, n_batches=2, seed=5)
    assert a == b
    assert set(a) == {"loss", "accuracy"}
    assert 0.0 <= a["accuracy"] <= 1.0 and a["loss"] > 0.0


def test_training_reduces_loss_on_a_tiny_task():
    mx.random.seed(1)
    cfg = MQARConfig(n_pairs=2, n_keys=4, batch_size=16)
    model = GRT(GRTConfig(vocab_size=cfg.vocab_size, max_seq_len=cfg.seq_len,
                          dim=32, n_heads=2, n_core=1, recurrence=2))
    started = evaluate(model, cfg, R=2, n_batches=2, seed=11)
    history = train(model, cfg, R=2, steps=60, peak_lr=1e-2, warmup=10,
                    eval_every=0, log_every=10**6, log=lambda *_: None)
    finished = evaluate(model, cfg, R=2, n_batches=2, seed=11)
    assert finished["loss"] < started["loss"]
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert history["wall_time_s"] > 0.0
