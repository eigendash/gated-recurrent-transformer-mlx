"""The Gated Recurrent Transformer: prelude -> shared core x R -> coda.

Section 3.2 and 3.3 of the paper. The vocabulary/position embeddings and the
prelude are applied once to build a fixed anchor ``h^(pre)``; the shared core
is then applied ``R`` times, each time reading the previous state (plus
resampled noise) concatenated with the anchor through ``W_proj``, and writing
its proposal back through the elementwise gate of Eq. 4; the coda and the head
read the final state.

``R`` is a forward-pass argument, never a weight, so the parameter count is a
function of ``n_prelude + n_core + n_coda`` alone. That is the paper's central
claim and it is asserted in the tests.
"""

import mlx.core as mx
import mlx.nn as nn

from grt.blocks import causal_mask, count_parameters, default_block_factory
from grt.config import GRTConfig
from grt.modulation import (
    ElementwiseGate,
    RecurrentProjection,
    gated_update,
    sample_gate_noise,
    sample_state_noise,
)


class GRT(nn.Module):
    """Gated Recurrent Transformer.

    Args:
        config: a :class:`~grt.config.GRTConfig`.
        block_factory: callable ``(dim, n_heads, mlp_ratio) -> block`` used to
            build every block. Defaults to the plain :class:`~grt.blocks.Block`;
            tests and the experiment pass counting wrappers here.
    """

    def __init__(self, config: GRTConfig, block_factory=None):
        super().__init__()
        self.config = config
        factory = block_factory if block_factory is not None else default_block_factory
        dim, heads, ratio = config.dim, config.n_heads, config.mlp_ratio

        self.embed = nn.Embedding(config.vocab_size, dim)
        self.pos = nn.Embedding(config.max_seq_len, dim)
        self.prelude = [factory(dim, heads, ratio) for _ in range(config.n_prelude)]
        self.core = [factory(dim, heads, ratio) for _ in range(config.n_core)]
        self.coda = [factory(dim, heads, ratio) for _ in range(config.n_coda)]

        self.proj = RecurrentProjection(dim)
        self.gate = ElementwiseGate(dim, init_bias=config.gate_init_bias)

        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, config.vocab_size, bias=False)

    # ------------------------------------------------------------------ helpers

    @property
    def unique_blocks(self) -> int:
        """Number of distinct transformer blocks that hold weights."""
        return self.config.n_prelude + self.config.n_core + self.config.n_coda

    def layer_applications(self, R: int | None = None) -> int:
        """Block executions in one forward pass: ``n_pre + n_core * R + n_coda``."""
        r = self.config.recurrence if R is None else int(R)
        if r < 1:
            raise ValueError(f"recurrence must be >= 1, got {r}")
        return self.config.n_prelude + self.config.n_core * r + self.config.n_coda

    def parameter_count(self) -> int:
        return count_parameters(self)

    @staticmethod
    def _run_blocks(h: mx.array, blocks, mask: mx.array) -> mx.array:
        for block in blocks:
            h = block(h, mask)
        return h

    # ------------------------------------------------------------------ forward

    def __call__(
        self,
        tokens: mx.array,
        R: int | None = None,
        key: mx.array | None = None,
        gate_override: float | None = None,
        state_noise_std: float | None = None,
        gate_noise_std: float | None = None,
        temperature: float | None = None,
        return_trace: bool = False,
    ):
        """Run the model.

        Args:
            tokens: integer array ``(batch, seq_len)``.
            R: recurrence depth for this call. Defaults to ``config.recurrence``.
            key: PRNG key. Noise is injected **only** when a key is given, so a
                call without a key is exactly deterministic; the same key always
                gives the same noise.
            gate_override: use a constant gate instead of the gate MLP. This is
                an analysis knob: 1.0 makes every recurrence a no-op copy and
                0.0 takes the shared block output unchanged.
            state_noise_std / gate_noise_std / temperature: override the config.
            return_trace: also return the anchor, the per-step gates and the
                per-step states.

        Returns:
            ``logits`` or ``(logits, trace)``.
        """
        cfg = self.config
        r = cfg.recurrence if R is None else int(R)
        if r < 1:
            raise ValueError(f"recurrence must be >= 1, got {r}")
        tau = cfg.gate_temperature if temperature is None else float(temperature)
        if tau <= 0.0:
            raise ValueError(f"gate temperature must be > 0, got {tau}")

        batch, seq_len = tokens.shape
        if seq_len > cfg.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len {cfg.max_seq_len}"
            )
        mask = causal_mask(seq_len)

        h = self.embed(tokens) + self.pos(mx.arange(seq_len))
        h = self._run_blocks(h, self.prelude, mask)

        # h^(0) = h^(pre): the anchor is also the initial state.
        h_pre = h
        anchor = h_pre if cfg.prelude_injection else mx.zeros_like(h_pre)

        sigma_x = cfg.state_noise_std if state_noise_std is None else float(state_noise_std)
        sigma_g = cfg.gate_noise_std if gate_noise_std is None else float(gate_noise_std)
        step_keys = None
        if key is None:
            # No key, no noise: an unkeyed forward pass is exactly deterministic.
            sigma_x = sigma_g = 0.0
        else:
            step_keys = mx.random.split(key, r)

        gates, states = [], []
        for step in range(r):
            state_key = gate_key = None
            if step_keys is not None:
                state_key, gate_key = mx.random.split(step_keys[step])

            if cfg.use_projection:
                state_in = h + sample_state_noise(h.shape, sigma_x, state_key)
                proposal = self.proj(state_in, anchor)
            else:
                # Ablation: no projection, feed the state straight to the core.
                proposal = h + sample_state_noise(h.shape, sigma_x, state_key)
            proposal = self._run_blocks(proposal, self.core, mask)

            if gate_override is not None:
                g = mx.full(h.shape, float(gate_override), dtype=h.dtype)
            elif cfg.use_gate:
                g = self.gate(
                    h, anchor, key=gate_key, noise_std=sigma_g, temperature=tau
                )
            else:
                # Ablation: no gate, the shared block output replaces the state.
                g = mx.zeros(h.shape, dtype=h.dtype)

            h = gated_update(h, proposal, g)
            gates.append(g)
            states.append(h)

        h = self._run_blocks(h, self.coda, mask)
        logits = self.lm_head(self.ln_f(h))

        if not return_trace:
            return logits
        trace = {
            "h_pre": h_pre,
            "anchor": anchor,
            "states": states,
            "gates": gates,
            "block_applications": self.layer_applications(r),
        }
        return logits, trace


__all__ = ["GRT", "sample_gate_noise"]
