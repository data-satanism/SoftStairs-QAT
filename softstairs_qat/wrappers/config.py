from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Type, Literal


@dataclass(frozen=True)
class QuantizationConfig:
    """Configuration for SoftStairs quantization-aware training."""

    rank: int = 4
    n_bits: int = 32
    # safety_factor: float = 0.7
    normalized: bool = False
    symmetric: bool = False
    half_shift: bool = False
    target_modules: Optional[Tuple[Type[Any], ...]] = None
    is_lora: bool = False

    type: Literal['naive', 'standard', 'shifted'] = 'naive'

    t_scheduler_strategy: str = "constant"
    t_start: float = 0.2
    t_end: float = 0.05
    t_tau: float = 8.0
    n_steps: int = 1000