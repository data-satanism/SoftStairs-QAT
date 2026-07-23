from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuantizationParams:
    """Immutable affine quantization parameters for a weight tensor."""

    scale: torch.Tensor
    zero_point: torch.Tensor
    q_min: int
    q_max: int


class QuantizationParamsCalculator:
    """Computes per-tensor scale and zero-point for integer quantization."""

    def compute(
        self,
        tensor: torch.Tensor,
        n_bits: int,
        symmetric: bool = False,
    ) -> QuantizationParams:
        """Derive quantization parameters from tensor statistics.

        Args:
            tensor: Weight tensor to quantize.
            n_bits: Bit width of the target integer representation.
            symmetric: Whether to use symmetric (signed-only) quantization.

        Returns:
            Quantization parameters for the provided tensor.
        """
        q_min = -2 ** (n_bits - 1)
        q_max = 2 ** (n_bits - 1) - 1

        if symmetric:
            max_abs = torch.max(torch.abs(tensor))
            scale = max_abs / q_max if max_abs > 0 else torch.tensor(1.0)
            zero_point = torch.tensor(0, dtype=tensor.dtype, device=tensor.device)
        else:
            min_val = tensor.min()
            max_val = tensor.max()
            eps = 1e-6
            min_val = min_val - eps
            max_val = max_val + eps
            scale = (max_val - min_val) / (q_max - q_min)
            if scale == 0 or torch.isinf(scale):
                scale = torch.tensor(1.0, dtype=tensor.dtype, device=tensor.device)
            zero_point = q_min - min_val / scale
            zero_point = torch.round(zero_point)
            zero_point = torch.clamp(zero_point, q_min, q_max)

        safety_gap = 0.5

        return QuantizationParams(
            scale=scale * safety_gap,
            zero_point=zero_point,
            q_min=q_min,
            q_max=q_max,
        )
