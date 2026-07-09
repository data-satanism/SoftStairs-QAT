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
    is_lora: bool = False

    r_scheduler_strategy: str = "constant"
    r_start: float = 0.5
    r_end: float = 0.9999
    r_tau: float = 8.0
    r_step: int = 100
