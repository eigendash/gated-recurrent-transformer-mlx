"""A synthetic multi-query associative recall task.

The sequence is a list of key/value pairs followed by a shuffled list of query
keys:

    k1 v1 k2 v2 ... kK vK   q1 q2 ... qK

and the model has to output the value bound to each query key. The values are
drawn independently of the keys, so the only way to answer is to look the key
up in the context: the task measures induction-like in-context recall rather
than a memorised key -> value table. Chance accuracy over the whole vocabulary
is ``1 / vocab_size`` (half of the vocabulary is keys), or ``1 / n_keys`` for a
model that already emits only values.

Keys are tokens ``0 .. n_keys - 1``, values are ``n_keys .. 2 * n_keys - 1``.
The loss and the accuracy are computed only at the K query positions.
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class MQARConfig:
    n_pairs: int = 8
    n_keys: int = 16
    batch_size: int = 32

    @property
    def seq_len(self) -> int:
        return 3 * self.n_pairs

    @property
    def vocab_size(self) -> int:
        return 2 * self.n_keys

    @property
    def chance_accuracy(self) -> float:
        """Accuracy of an untrained model that spreads mass over the whole vocabulary.

        Half the vocabulary is keys, so this is ``1 / vocab_size``. A model that
        had already learned to emit only value tokens would score ``1 / n_keys``.
        """
        return 1.0 / self.vocab_size

    @property
    def chance_accuracy_over_values(self) -> float:
        return 1.0 / self.n_keys


def make_batch(rng: np.random.Generator, cfg: MQARConfig):
    """Return ``(tokens, targets, mask)`` as int32/bool numpy arrays.

    ``mask`` is true exactly at the K query positions, where ``targets`` holds
    the value that the corresponding key was bound to earlier in the sequence.
    """
    batch, pairs, n_keys = cfg.batch_size, cfg.n_pairs, cfg.n_keys
    keys = np.stack([rng.permutation(n_keys)[:pairs] for _ in range(batch)])
    values = rng.integers(0, n_keys, size=(batch, pairs)) + n_keys
    order = np.stack([rng.permutation(pairs) for _ in range(batch)])

    tokens = np.zeros((batch, cfg.seq_len), dtype=np.int32)
    tokens[:, 0 : 2 * pairs : 2] = keys
    tokens[:, 1 : 2 * pairs : 2] = values
    rows = np.arange(batch)[:, None]
    tokens[:, 2 * pairs :] = keys[rows, order]

    targets = np.full((batch, cfg.seq_len), -1, dtype=np.int32)
    targets[:, 2 * pairs :] = values[rows, order]
    mask = np.zeros((batch, cfg.seq_len), dtype=bool)
    mask[:, 2 * pairs :] = True
    return tokens, targets, mask


def to_mlx(batch, dtype=mx.int32):
    tokens, targets, mask = batch
    return (
        mx.array(tokens.astype(np.int32)),
        mx.array(targets.astype(np.int32)),
        mx.array(mask),
    )


def batch_loss(logits: mx.array, targets: mx.array, mask: mx.array) -> mx.array:
    """Cross-entropy averaged over the masked query positions only."""
    log_probs = nn.log_softmax(logits.astype(mx.float32), axis=-1)
    safe = mx.where(mask, targets, mx.zeros_like(targets))
    picked = mx.take_along_axis(log_probs, safe[..., None], axis=-1)[..., 0]
    denom = mx.maximum(mask.sum().astype(mx.float32), 1.0)
    return -(picked * mask.astype(mx.float32)).sum() / denom


def batch_accuracy(logits: mx.array, targets: mx.array, mask: mx.array) -> mx.array:
    """Fraction of query positions whose argmax is the right value."""
    predicted = mx.argmax(logits, axis=-1).astype(mx.int32)
    correct = (predicted == targets).astype(mx.float32) * mask.astype(mx.float32)
    denom = mx.maximum(mask.sum().astype(mx.float32), 1.0)
    return correct.sum() / denom
