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
    # Deprecated: no longer read anywhere (the half-shifted upscale was
    # removed). Kept only so old checkpoints/hparams that carry the key still
    # reconstruct; do not set it in new code.
    half_shift: bool = False
    target_modules: Optional[Tuple[Type[Any], ...]] = None
    is_lora: bool = False

    type: Literal['naive', 'standard', 'shifted', 'raw_gradient'] = 'naive'

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

    # Constant weight-scale factor folded into the stored quantization scale:
    # the weights entering quantization are W' = weight_scale * W, so the
    # init-time code window is occupied up to `weight_scale` (90% by default),
    # leaving headroom near the boundary codes. The stored affine map
    # (code - zero_point) / scale remains the exact inverse for training,
    # evaluation, and deployment alike.
    weight_scale: float = 0.9

    # Clamp the soft staircase output to the training-forward window
    # [q_min - 0.5, q_max + 0.5] (code space) before downscaling in the
    # quantizer hook: the half-code "luft" lets saturated weights keep a
    # gradient signal while they still round to a valid boundary code.
    # Deployment paths (fake_quantize, finalize, error estimates) clamp to
    # the exact [q_min, q_max] window, matching round(clamp(W)) at deploy.
    clamp_codes: bool = True

    # Recalculate the scale and zero-point of every quantized parameter from
    # its current values on each quantizer step(), so the code window tracks
    # the weights as they drift during training. Disabled by default: the
    # scale is computed once at initialization.
    recalc_scale_each_step: bool = False
