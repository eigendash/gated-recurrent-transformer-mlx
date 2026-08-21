"""An MLX implementation of the Gated Recurrent Transformer.

Gated Recurrent Transformers: Expressive Depth through Recurrent Modulation,
Amr Hegazy, Amr Alanwar and Mostafa Elhoushi, arXiv:2608.15062.

Public API
----------
``GRTConfig``          configuration dataclass
``GRT``                prelude -> shared core x R (gated) -> coda
``PlainStack``         non-recurrent unique-layer baseline
``Block``              pre-norm attention + MLP transformer block
``RecurrentProjection``/``ElementwiseGate``   the recurrent modulation
``count_parameters``/``layer_applications``   parameter and compute accounting
"""

from grt.config import GRTConfig

__all__ = ["GRTConfig"]
