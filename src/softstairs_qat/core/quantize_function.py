from __future__ import annotations

from typing import Optional

import torch

from softstairs_qat.core.soft_stairs import SoftStairs


class _SoftStairsQuantizeFunction(torch.autograd.Function):
    """Internal autograd function for SoftStairs QAT."""

    @staticmethod
    def forward(
        ctx,
        weight: torch.Tensor,
        adapter_a: Optional[torch.Tensor],
        adapter_b: Optional[torch.Tensor],
        soft_stairs: SoftStairs,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        q_min: int,
        q_max: int,
        sigma_b_max: float,
    ) -> torch.Tensor:
        """Quantize weights with optional low-rank correction."""
        if adapter_a is not None and adapter_b is not None:
            sigma_b = torch.std(adapter_b).item()
            if sigma_b > sigma_b_max:
                adapter_b.mul_(sigma_b_max / sigma_b)
            effective = weight + torch.matmul(adapter_a, adapter_b)
        else:
            effective = weight.clone()

        normalized = effective.div(scale).add(zero_point)
        soft_rounded = soft_stairs.forward(normalized)
        soft_rounded.clamp_(q_min, q_max)
        quantized = soft_rounded.sub(zero_point).mul(scale)

        ctx.save_for_backward(normalized, scale, zero_point)
        ctx.soft_stairs = soft_stairs
        ctx.adapter_a = adapter_a
        ctx.adapter_b = adapter_b
        return quantized

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Propagate gradients through SoftStairs and low-rank adapters."""
        normalized, scale, zero_point = ctx.saved_tensors
        soft_stairs: SoftStairs = ctx.soft_stairs
        adapter_a = ctx.adapter_a
        adapter_b = ctx.adapter_b

        deriv = soft_stairs.derivative(normalized)
        grad_effective = grad_output * deriv
        grad_weight = grad_effective

        if adapter_a is not None and adapter_b is not None:
            grad_a = torch.matmul(grad_effective, adapter_b.T)
            grad_b = torch.matmul(adapter_a.T, grad_effective)
        else:
            grad_a = None
            grad_b = None

        return (
            grad_weight,
            grad_a,
            grad_b,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class SoftStairsQuantizer:
    """Facade over the SoftStairs quantization autograd path."""

    def __init__(self, soft_stairs: SoftStairs | None = None) -> None:
        """Create a quantizer bound to a SoftStairs instance.

        Args:
            soft_stairs: SoftStairs configuration used for forward/backward.
        """
        self._soft_stairs = soft_stairs or SoftStairs()

    @property
    def soft_stairs(self) -> SoftStairs:
        """Return the underlying SoftStairs instance."""
        return self._soft_stairs

    def quantize(
        self,
        weight: torch.Tensor,
        adapter_a: Optional[torch.Tensor],
        adapter_b: Optional[torch.Tensor],
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        q_min: int,
        q_max: int,
        sigma_b_max: float,
    ) -> torch.Tensor:
        """Quantize a weight tensor with optional low-rank adapters.

        Args:
            weight: Full-precision weight matrix.
            adapter_a: Low-rank left adapter or None.
            adapter_b: Low-rank right adapter or None.
            scale: Per-tensor quantization scale.
            zero_point: Per-tensor zero point.
            q_min: Minimum quantized integer value.
            q_max: Maximum quantized integer value.
            sigma_b_max: Maximum allowed standard deviation for adapter B.

        Returns:
            Quantized-dequantized weight tensor suitable for forward pass.
        """
        return _SoftStairsQuantizeFunction.apply(
            weight,
            adapter_a,
            adapter_b,
            self._soft_stairs,
            scale,
            zero_point,
            q_min,
            q_max,
            sigma_b_max,
        )
