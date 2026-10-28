from __future__ import annotations

import math
from typing import Optional

import torch

def softstairs_naive(x, t, normalized=False, async_t_factor=1., frequency=1.):
     """Function which aplies SS according to autograd. normalized is deprecated"""
     return x + (1 / (math.pi * frequency)) * torch.atan2(
            -(1 - t) * torch.sin(2 * math.pi * frequency * x),
            t + 2 * (1 - t) * torch.square(torch.cos(math.pi * frequency * x))
        )


class SoftStairs:
    """Differentiable approximation of a quantization staircase.

    The forward map blends a periodic atan2 correction with an optional linear
    term subtraction (modified variant) to control bias near the origin.
    """

    def __init__(
        self,
        t: float = 0.01,
        normalized: bool = False,
        async_t_factor: float = 1.,
    ) -> None:
        """Initialize SoftStairs parameters.

        Args:
            t: Temperature parameter in (0, 1); values closer to 0 yield
                narrower derivative peaks near quantization boundaries.
            normalized: Whether the backward surrogate uses the normalized
                numerator (see ``derivative``).
            async_t_factor: Multiplier applied to ``t`` in the backward
                derivative only.
        """
        self.t = t
        self.normalized = normalized
        self.async_t_factor = async_t_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SoftStairs forward map.

        Args:
            x: Input tensor in normalized quantization coordinates.

        Returns:
            Soft-rounded tensor with the same shape as ``x``.
        """
        result = x + (1 / math.pi) * torch.atan2(
            -(1 - self.t) * torch.sin(2 * math.pi * x),
            self.t + 2 * (1 - self.t) * torch.square(torch.cos(math.pi * x))
        )
        return result

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the SoftStairs derivative used during backpropagation.

        Args:
            x: Input tensor in normalized quantization coordinates.
        Returns:
            Element-wise derivative with the same shape as ``x``.
        """
        t = self.t * self.async_t_factor
        t = min(t, 1)

        den = (t * t + 4 * (1 - t) * torch.square(torch.cos(math.pi * x)))
        if self.normalized:
            nom = t * t
        else:
            nom = (2 - t) * t 
        deriv = nom / den
        return deriv
    
    def second_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the derivative of the SoftStairs backward derivative.

        This corresponds to the second derivative of the surrogate
        used during backpropagation for normalized=False.
        """
        t = self.t * self.async_t_factor
        t = min(t, 1)

        cos_x = torch.cos(math.pi * x)
        sin_2x = torch.sin(2 * math.pi * x)

        den = t * t + 4 * (1 - t) * torch.square(cos_x)

        nom = (2 - t) * t

        second_deriv = (
            4 * math.pi * nom * (1 - t) * sin_2x
            / torch.square(den)
        )

        return second_deriv
    
    @torch.no_grad()
    def trainable_parameters_proportion(self, threshold: float):
        threshold = torch.scalar_tensor(threshold)
        if self.normalized:
            prop = 1 - torch.acos(((self.t ** 2) / threshold - 1 - (1 - self.t) ** 2) / (2 * (1 - self.t))) / math.pi
        else:
            threshold = torch.exp(threshold)
            prop = 1 - torch.acos(((self.t ** 2) * (2 + self.t) / threshold / self.t - 1 - (1 - self.t) ** 2) / 2 / (1 - self.t)) / math.pi 
        return prop
            


class SoftStairsShifted(SoftStairs):
    """Half-period-shifted staircase: rounds toward the half-integers.

    ``Shifted(x) = SoftStairs(x + 0.5) - 0.5``, so its exact Jacobian swaps the
    cos² phase of the base surrogate for sin²: ≈ ``t/2`` at the half-integer
    attractors (wells) and up to ``(2-t)/t`` (exact) or ``1`` (normalized) at the
    integer boundaries.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shifted SoftStairs forward map.

        Args:
            x: Input tensor in normalized quantization coordinates.

        Returns:
            Soft-rounded tensor with the same shape as ``x``.
        """
        result = x + (1 / math.pi) * torch.atan2(
            (1 - self.t) * torch.sin(2 * math.pi * x),
            self.t + 2 * (1 - self.t) * torch.square(torch.sin(math.pi * x))
        )
        return result

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Exact-phase derivative of the shifted staircase (sin² denominator).

        Args:
            x: Input tensor in normalized quantization coordinates.
        Returns:
            Element-wise derivative with the same shape as ``x``.
        """
        t = self.t * self.async_t_factor
        t = min(t, 1)

        den = (t * t + 4 * (1 - t) * torch.square(torch.sin(math.pi * x)))
        if self.normalized:
            nom = t * t
        else:
            nom = (2 - t) * t
        return nom / den

    def second_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """Second derivative of the shifted backward surrogate.

        Phase-shifted variant of :meth:`SoftStairs.second_derivative`:
        ``sin(2π(x + 1/2)) = -sin(2πx)`` flips the sign.
        """
        t = self.t * self.async_t_factor
        t = min(t, 1)

        sin_x = torch.sin(math.pi * x)
        sin_2x = torch.sin(2 * math.pi * x)

        den = t * t + 4 * (1 - t) * torch.square(sin_x)

        nom = (2 - t) * t

        second_deriv = (
            -4 * math.pi * nom * (1 - t) * sin_2x
            / torch.square(den)
        )

        return second_deriv


class ScaledSoftStairs(SoftStairs):
    """SoftStairs on a rescaled lattice: ``S_{t,s}(x) = S_{t,1}(s x) / s``.

    This is the only place carrying the frequency/lattice-scale logic. The
    scale ``s`` multiplies the staircase frequency: steps of height ``1 / s``
    at spacing ``1 / s``, so ``s = 2`` doubles the quantization level density
    (a frequency-homotopy continuation towards ``s = 1``, which recovers the
    original SoftStairs exactly -- bit-identical, since multiplying and
    dividing by 1 is exact in floating point).

    The backward derivative stays the exact Jacobian of the rescaled forward,
    ``d/dx S_{t,s}(x) = S'_{t,1}(s x)``, so ``derivative`` evaluates the base
    surrogate at ``s x`` with no extra factor, and ``second_derivative`` picks
    up the single chain-rule factor ``s``.
    """

    def __init__(
        self,
        t: float = 0.01,
        normalized: bool = False,
        async_t_factor: float = 1.,
        scale: float = 1.,
    ) -> None:
        """Initialize ScaledSoftStairs parameters.

        Args:
            t: Temperature parameter in (0, 1); values closer to 0 yield
                narrower derivative peaks near quantization boundaries.
            normalized: Whether the backward surrogate uses the normalized
                numerator (see ``derivative``).
            async_t_factor: Multiplier applied to ``t`` in the backward
                derivative only.
            scale: Lattice scale ``s > 0`` (the staircase frequency); ``s = 1``
                is the original SoftStairs.
        """
        super().__init__(t=t, normalized=normalized, async_t_factor=async_t_factor)
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        self.scale = scale

    def derivative(self, x: torch.Tensor) -> torch.Tensor:
        return super().derivative(x * self.scale)

    def second_derivative(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * super().second_derivative(x * self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x * self.scale) / self.scale

    def max_derivative_value(self):
        if self.normalized:
            return 1.

        return 2 / self.t - 1
        
        
