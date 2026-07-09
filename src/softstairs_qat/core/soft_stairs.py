from __future__ import annotations

import math
from typing import Optional

import torch


class SoftStairs:
    """Differentiable approximation of a quantization staircase.

    The forward map blends a periodic atan2 correction with an optional linear
  term subtraction (modified variant) to control bias near the origin.
    """

    def __init__(
        self,
        r: float = 0.99,
        modified: bool = False,
    ) -> None:
        """Initialize SoftStairs parameters.

        Args:
            r: Sharpness parameter in (0, 1); values closer to 1 yield
                narrower derivative peaks near quantization boundaries.
            modified: Whether to subtract the linear correction term.
            counter: Optional call counter used for diagnostics.
        """
        self.r = r
        self.modified = modified

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SoftStairs forward map.

        Args:
            x: Input tensor in normalized quantization coordinates.

        Returns:
            Soft-rounded tensor with the same shape as ``x``.
        """
        result = x + (1.0 / math.pi) * torch.atan2(
            -self.r * torch.sin_(2.0 * math.pi * x),
            1.0 + self.r * torch.cos_(2.0 * math.pi * x),
        )
        if self.modified:
            result = result - ((1 - self.r) / (self.r + 1)) * x
        return result

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the SoftStairs derivative used during backpropagation.

        Args:
            x: Input tensor in normalized quantization coordinates.

        Returns:
            Element-wise derivative with the same shape as ``x``.
        """
        deriv = (1.0 - self.r * self.r) * (1.0 - self.r * self.r) / (
            1.0 + 2.0 * self.r * torch.cos_(2.0 * math.pi * x) + self.r * self.r
        )
        if self.modified:
            deriv = deriv - (1 - self.r) / (self.r + 1)
        return deriv
