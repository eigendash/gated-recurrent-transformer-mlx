"""Training and evaluation helpers.

Small and explicit on purpose: AdamW, a warmup + cosine schedule, gradient
norm clipping, and depth sampling (Section 3.4 of the paper: at every training
step the recurrence depth is drawn uniformly from ``{1, ..., R}``, which trains
every exit point without an auxiliary loss).
"""

import math
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from grt.tasks import MQARConfig, batch_accuracy, batch_loss, make_batch, to_mlx


def cosine_schedule(
    step: int, total: int, *, peak: float, warmup: int = 0, floor_ratio: float = 0.1
) -> float:
    """Linear warmup then cosine decay from ``peak`` to ``floor_ratio * peak``."""
    if peak < 0.0 or not 0.0 <= floor_ratio <= 1.0:
        raise ValueError(f"bad schedule: peak={peak}, floor_ratio={floor_ratio}")
    step = float(step)
    floor = floor_ratio * peak
    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def evaluate(
    model,
    cfg: MQARConfig,
    *,
    R: int | None = None,
    n_batches: int = 8,
    seed: int = 1234,
) -> dict:
    """Held-out loss and query accuracy, noise-free and deterministic.

    Evaluation never passes a PRNG key, so no noise is injected and the numbers
    do not depend on the evaluation seed.
    """
    rng = np.random.default_rng(seed)
    total_loss, total_acc = 0.0, 0.0
    for _ in range(n_batches):
        tokens, targets, mask = to_mlx(make_batch(rng, cfg))
        logits = model(tokens, R=R)
        total_loss += float(batch_loss(logits, targets, mask))
        total_acc += float(batch_accuracy(logits, targets, mask))
    return {"loss": total_loss / n_batches, "accuracy": total_acc / n_batches}


def train(
    model,
    cfg: MQARConfig,
    *,
    R: int | None = None,
    steps: int = 1500,
    batch_size: int | None = None,
    peak_lr: float = 3e-3,
    warmup: int = 100,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    seed: int = 0,
    depth_sampling: bool = False,
    log_every: int = 250,
    eval_every: int = 250,
    eval_batches: int = 8,
    log=print,
) -> dict:
    """Train ``model`` on the recall task and return a history dict.

    ``depth_sampling=True`` draws the recurrence depth uniformly from
    ``{1, ..., R}`` at every step instead of using ``R`` always.
    """
    if batch_size is not None:
        cfg = MQARConfig(cfg.n_pairs, cfg.n_keys, batch_size)
    rng = np.random.default_rng(seed)
    key = mx.random.key(seed)

    optimizer = optim.AdamW(
        learning_rate=lambda step: mx.array(
            cosine_schedule(step, steps, peak=peak_lr, warmup=warmup), mx.float32
        ),
        weight_decay=weight_decay,
    )

    def loss_fn(model, tokens, targets, mask, depth, key):
        logits = model(tokens, R=depth, key=key)
        return batch_loss(logits, targets, mask)

    step_fn = nn.value_and_grad(model, loss_fn)

    history = {"step": [], "train_loss": [], "eval": []}
    start = time.time()
    for step in range(steps):
        tokens, targets, mask = to_mlx(make_batch(rng, cfg))
        key, sub = mx.random.split(key)
        if depth_sampling:
            depth = int(rng.integers(1, (R or 1) + 1))
        else:
            depth = R if R is not None else 1
        loss, grads = step_fn(model, tokens, targets, mask, depth, sub)
        grads, _ = optim.clip_grad_norm(grads, grad_clip)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        history["step"].append(step)
        history["train_loss"].append(float(loss))

        if (step + 1) % log_every == 0 or step == 0:
            log(
                f"    step {step + 1:5d}/{steps}  train_loss {float(loss):.4f}"
                f"  ({(time.time() - start) / (step + 1):.3f}s/step)"
            )
        if eval_every and (step + 1) % eval_every == 0:
            metrics = evaluate(model, cfg, R=R, n_batches=eval_batches, seed=seed + 999)
            metrics["step"] = step + 1
            history["eval"].append(metrics)
            log(
                f"    eval @ {step + 1:5d}  loss {metrics['loss']:.4f}"
                f"  acc {metrics['accuracy']:.4f}"
            )

    history["wall_time_s"] = time.time() - start
    return history
