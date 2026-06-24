from __future__ import annotations

import math
from typing import Optional

import torch


class SoftStairsCallCounter:
    """Tracks how many SoftStairs forward evaluations were performed."""

    def __init__(self) -> None:
        """Initialize the counter to zero."""
        self._count = 0

    def reset(self) -> None:
        """Reset the call counter to zero."""
        self._count = 0

    @property
    def count(self) -> int:
        """Return the current number of recorded forward calls."""
        return self._count

    def increment(self) -> None:
        """Increment the call counter by one."""
        self._count += 1


class SoftStairs:
    """Differentiable approximation of a quantization staircase.

    The forward map blends a periodic atan2 correction with an optional linear
  term subtraction (modified variant) to control bias near the origin.
    """

    def __init__(
        self,
        r: float = 0.99,
        modified: bool = False,
        counter: Optional[SoftStairsCallCounter] = None,
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
        self._counter = counter or SoftStairsCallCounter()

    @property
    def counter(self) -> SoftStairsCallCounter:
        """Return the diagnostic call counter."""
        return self._counter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SoftStairs forward map.

        Args:
            x: Input tensor in normalized quantization coordinates.

        Returns:
            Soft-rounded tensor with the same shape as ``x``.
        """
        self._counter.increment()
        result = x + (1.0 / math.pi) * torch.atan2(
            -self.r * torch.sin(2.0 * math.pi * x),
            1.0 + self.r * torch.cos(2.0 * math.pi * x),
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
        deriv = (1.0 - self.r * self.r) / (
            1.0 + 2.0 * self.r * torch.cos(2.0 * math.pi * x) + self.r * self.r
        )
        if self.modified:
            deriv = deriv - (1 - self.r) / (self.r + 1)
        return deriv
