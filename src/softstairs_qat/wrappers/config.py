from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Type


@dataclass(frozen=True)
class QuantizationConfig:
    """Configuration for SoftStairs quantization-aware training."""

    rank: int = 4
    t: float = 0.01
    n_bits: int = 32
    # safety_factor: float = 0.7
    normalized: bool = False
    symmetric: bool = False
    target_modules: Optional[Tuple[Type[Any], ...]] = None
    is_lora: bool = False

    t_scheduler_strategy: str = "constant"
    t_start: float = 0.2
    t_end: float = 0.0001
    t_tau: float = 8.0
    t_step: int = 1000