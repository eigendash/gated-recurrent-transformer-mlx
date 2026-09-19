"""An MLX implementation of the Gated Recurrent Transformer.

Gated Recurrent Transformers: Expressive Depth through Recurrent Modulation,
Amr Hegazy, Amr Alanwar and Mostafa Elhoushi, arXiv:2608.15062.

Public API
----------
``GRTConfig``            configuration dataclass
``GRT``                  prelude -> shared core x R (gated) -> coda
``PlainStack``           the non-recurrent unique-layer baseline
``Block``                the pre-norm transformer block both models use
``RecurrentProjection``  ``W_proj`` of Eq. 2
``ElementwiseGate``      ``f_g`` of Eq. 5
``count_parameters``     parameter accounting
"""

from grt.blocks import Block, count_parameters
from grt.config import GRTConfig
from grt.model import GRT, PlainStack
from grt.modulation import ElementwiseGate, RecurrentProjection, gated_update

__all__ = [
    "Block",
    "ElementwiseGate",
    "GRT",
    "GRTConfig",
    "PlainStack",
    "RecurrentProjection",
    "count_parameters",
    "gated_update",
]
