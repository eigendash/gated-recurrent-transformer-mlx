"""Configuration for the Gated Recurrent Transformer.

Everything here is deliberately small: the paper's models are 12L GPT-2-scale
and up, this package is built to be trained on a laptop in minutes.
"""

from dataclasses import dataclass, replace


@dataclass
class GRTConfig:
    """Shape and recurrence settings for :class:`grt.model.GRT`.

    The recurrence depth ``R`` is *not* fixed by the weights: it is stored here
    only as a default for inference. Every forward pass may override it, which
    is the whole point of the architecture -- the parameter count is a function
    of ``n_prelude + n_core + n_coda`` and not of ``R``.
    """

    vocab_size: int = 64
    max_seq_len: int = 64
    dim: int = 64
    n_heads: int = 4
    mlp_ratio: int = 4

    # Prelude / shared core / coda depths, in unique blocks.
    n_prelude: int = 1
    n_core: int = 1
    n_coda: int = 1

    # Default number of times the shared core is applied at inference.
    recurrence: int = 1

    # Equation 5 of the paper: g = sigmoid(f_g(.) / tau + eps_g).
    gate_temperature: float = 1.0
    # Bias of the last gate linear layer. +4 puts g ~ 0.98 at initialisation.
    gate_init_bias: float = 4.0

    # Standard deviations of the state noise (Eq. 2) and gate noise (Eq. 5).
    state_noise_std: float = 0.1
    gate_noise_std: float = 0.1

    # Ablation switches. With both off, one recurrence step is exactly a plain
    # pre-norm transformer stack.
    prelude_injection: bool = True
    use_projection: bool = True
    use_gate: bool = True

    def __post_init__(self) -> None:
        if self.dim % self.n_heads != 0:
            raise ValueError(f"dim {self.dim} is not divisible by n_heads {self.n_heads}")
        if min(self.n_prelude, self.n_core, self.n_coda) < 0:
            raise ValueError("block counts must be non-negative")
        if self.n_core == 0 and self.recurrence > 0:
            raise ValueError("a zero-depth shared core cannot be applied recurrently")

    def with_recurrence(self, recurrence: int) -> "GRTConfig":
        """Return a copy with a different default recurrence depth."""
        if recurrence < 1:
            raise ValueError("recurrence must be >= 1")
        return replace(self, recurrence=recurrence)
