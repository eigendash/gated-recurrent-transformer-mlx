"""The recurrent modulation: projection, elementwise update gate, and noise.

This is Section 3.3 of the paper, equations 2 to 5:

    h_tilde^(r) = W_proj [ h^(r-1) + eps_x , h^(pre) ]            (2)
    o^(r)       = B_shared( h_tilde^(r) )                        (3)
    h^(r)       = g^(r) * h^(r-1) + (1 - g^(r)) * o^(r)          (4)
    g^(r)       = sigmoid( f_g([LN(h^(r-1)), LN(h^(pre))]) / tau + eps_g )   (5)

with eps_x ~ N(0, sigma_x^2) and eps_g ~ N(0, sigma_g^2) resampled at every
recurrence step. Note that g is the *copy* gate: g -> 1 keeps the state and
ignores the shared block, g -> 0 takes the block output and overwrites the
state. The gate is initialised at g ~ 0.98 (bias +4) so training starts with
the residual stream passing through the recurrence almost unchanged.

Noise is never sampled implicitly. A forward pass only injects noise if it is
handed an explicit PRNG key, which makes every test and every evaluation run
deterministic.
"""

import mlx.core as mx
import mlx.nn as nn


def split_key(key: mx.array, n: int = 2) -> list:
    """Split a PRNG key into ``n`` independent keys."""
    return [mx.random.split(key, n)[i] for i in range(n)]


def sample_state_noise(shape, std: float, key: mx.array) -> mx.array:
    """Zero-mean Gaussian state noise eps_x of shape ``shape``, scaled by ``std``."""
    if std == 0.0:
        return mx.zeros(shape)
    return mx.random.normal(shape, key=key) * std


def sample_gate_noise(std: float, key: mx.array, dtype=mx.float32) -> mx.array:
    """The per-scalar gate noise eps_g of Eq. 5, shape ``(1,)``."""
    if std == 0.0:
        return mx.zeros((1,), dtype=dtype)
    return mx.random.normal((1,), key=key).astype(dtype) * std


def gated_update(state: mx.array, proposal: mx.array, gate: mx.array) -> mx.array:
    """Eq. 4: ``gate * state + (1 - gate) * proposal``."""
    return gate * state + (1.0 - gate) * proposal


class RecurrentProjection(nn.Module):
    """``W_proj`` of Eq. 2: concatenate the (noised) state and the anchor, project to d.

    The paper writes ``W_proj in R^{d x 2d}``, i.e. a matrix with no bias, so
    that is what this layer is.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(2 * dim, dim, bias=False)

    def __call__(self, state: mx.array, anchor: mx.array) -> mx.array:
        return self.linear(mx.concatenate([state, anchor], axis=-1))


class ElementwiseGate(nn.Module):
    """``f_g`` of Eq. 5: two-layer MLP with SiLU over ``[LN(state), LN(anchor)]``."""

    def __init__(self, dim: int, init_bias: float = 4.0):
        super().__init__()
        self.dim = dim
        self.ln_state = nn.LayerNorm(dim)
        self.ln_anchor = nn.LayerNorm(dim)
        self.fc_in = nn.Linear(2 * dim, dim, bias=True)
        self.fc_out = nn.Linear(dim, dim, bias=True)
        # Open the gate at initialisation (g ~ 0.98 for the default +4).
        self.fc_out.bias = mx.full((dim,), float(init_bias), dtype=mx.float32)

    def logits(self, state: mx.array, anchor: mx.array) -> mx.array:
        features = mx.concatenate([self.ln_state(state), self.ln_anchor(anchor)], axis=-1)
        return self.fc_out(nn.silu(self.fc_in(features)))

    def __call__(
        self,
        state: mx.array,
        anchor: mx.array,
        key: mx.array | None = None,
        noise_std: float = 0.0,
        temperature: float = 1.0,
    ) -> mx.array:
        logits = self.logits(state, anchor) / temperature
        if key is not None and noise_std > 0.0:
            logits = logits + sample_gate_noise(noise_std, key, dtype=logits.dtype)
        return mx.sigmoid(logits)
