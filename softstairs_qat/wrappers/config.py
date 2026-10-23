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
    t_metric_delta: float = 1e-3
    async_t_factor: float = 1.
    early_power: float = 0.7
    min_majorant: float = 0.01

    # Frequency homotopy: anneal the SoftStairs staircase frequency f_start -> f_end
    # linearly in t between (t_start, f_start) and (t_end, f_end). f=1 recovers the
    # original SoftStairs exactly; disabled by default.
    frequency_homotopy: bool = False
    f_start: float = 2.0
    f_end: float = 1.0

    # Clamp the soft staircase output to [q_min, q_max] (code space) before
    # downscaling in the quantizer hook: the training forward then matches the
    # deployed round(clamp(W)) and out-of-range weights stop receiving task
    # gradients instead of drifting freely until a deploy-time clamp.
    clamp_codes: bool = True 
