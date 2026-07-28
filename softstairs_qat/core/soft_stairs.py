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
        t: float = 0.01,
        normalized: bool = False,
    ) -> None:
        """Initialize SoftStairs parameters.

        Args:
            t: Temperature parameter in (0, 1); values closer to 0 yield
                narrower derivative peaks near quantization boundaries.
            modified: Whether to subtract the linear correction term.
            counter: Optional call counter used for diagnostics.
        """
        self.t = t
        self.normalized = normalized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SoftStairs forward map.

        Args:
            x: Input tensor in normalized quantization coordinates.

        Returns:
            Soft-rounded tensor with the same shape as ``x``.
        """
        # result = x + (1.0 / math.pi) * torch.atan2(
        #     -self.t * torch.sin_(2.0 * math.pi * x),
        #     1.0 + self.t * torch.cos_(2.0 * math.pi * x),
        # )
        result = x + (1 / math.pi) * torch.atan2(
            (1 - self.t) * torch.sin(2 * math.pi * x),
            self.t + 2 * (1 - self.t) * torch.square(torch.sin(math.pi * x))
        )
        return result

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the SoftStairs derivative used during backpropagation.

        Args:
            x: Input tensor in normalized quantization coordinates.
        Returns:
            Element-wise derivative with the same shape as ``x``.
        """
        den = (self.t * self.t + 4 * (1 - self.t) * torch.square(torch.cos(math.pi * x)))
        if self.normalized:
            nom = self.t * self.t
        else:
            nom = (2 - self.t) * self.t 
        deriv = nom / den
        # deriv = (1.0 - self.r) * (1.0 - self.r) / (
        #     1.0 + 2.0 * self.r * torch.cos_(2.0 * math.pi * x) + self.r * self.r
        # )
        return deriv



class ScaledSoftStairs(SoftStairs):
    """Differentiable approximation of a quantization staircase.

    The forward map blends a periodic atan2 correction with an optional linear
  term subtraction (modified variant) to control bias near the origin.
    """

    def __init__(
        self,
        t: float = 0.01,
        normalized: bool = False,
        scale: float = 1.,
    ) -> None:
        """Initialize SoftStairs parameters.

        Args:
            t: Temperature parameter in (0, 1); values closer to 0 yield
                narrower derivative peaks near quantization boundaries.
            modified: Whether to subtract the linear correction term.
            counter: Optional call counter used for diagnostics.
        """
        self.t = t
        self.normalized = normalized
        self.scale = scale 

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        return (1 / self.scale) * super().derivative(x * self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (1 / self.scale) * super().derivative(x * self.scale)