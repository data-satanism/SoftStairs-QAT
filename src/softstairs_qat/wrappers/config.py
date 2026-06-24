from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Type


@dataclass(frozen=True)
class QuantizationConfig:
    """Configuration for SoftStairs quantization-aware training."""

    rank: int = 4
    r: float = 0.99
    n_bits: int = 32
    safety_factor: float = 0.7
    modified: bool = False
    symmetric: bool = False
    target_modules: Optional[Tuple[Type[Any], ...]] = None
