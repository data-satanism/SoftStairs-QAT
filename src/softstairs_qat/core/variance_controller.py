from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from softstairs_qat.core.quantization_params import (
    QuantizationParams,
    QuantizationParamsCalculator,
)


@dataclass
class LowRankAdapterState:
    """Low-rank adapter matrices and associated quantization metadata."""

    adapter_a: torch.Tensor
    adapter_b: torch.Tensor
    sigma: float
    params: QuantizationParams
    sigma_b_max: float


class VarianceController:
    """Initializes low-rank adapters while controlling tail variance."""

    def __init__(
        self,
        params_calculator: QuantizationParamsCalculator | None = None,
        safety_factor: float = 0.7,
    ) -> None:
        """Create a variance controller.

        Args:
            params_calculator: Calculator for affine quantization parameters.
            safety_factor: Conservative multiplier applied to sigma bounds.
        """
        self._params_calculator = params_calculator or QuantizationParamsCalculator()
        self.safety_factor = safety_factor

    def compute_sigma_b_max(
        self,
        l_bound: float,
        sigma_a: float,
        rank: int,
        rows: int,
        cols: int,
    ) -> float:
        """Compute the maximum allowable standard deviation for adapter B.

        Uses the bound derived from low-rank tail control so that ``W + AB``
        remains inside the quantizable range with high probability.

        Args:
            l_bound: Remaining headroom below quantization ceiling T.
            sigma_a: Standard deviation of adapter matrix A.
            rank: Low-rank adapter rank.
            rows: Number of rows in the weight matrix.
            cols: Number of columns in the weight matrix.

        Returns:
            Maximum allowed sigma_B, floored at 1e-10.
        """
        if l_bound <= 0 or sigma_a <= 0:
            return 1e-10
        num_elements = rows * cols
        bound = l_bound / np.sqrt(2 * rank * np.log(num_elements))
        return max(self.safety_factor * bound / sigma_a, 1e-10)

    def estimate_expected_max_w(self, weight: torch.Tensor) -> float:
        """Estimate an upper envelope for weight magnitudes.

        Combines mean absolute value with a Gaussian tail bound over all
        elements, following extreme-value heuristics for large matrices.

        Args:
            weight: Layer weight tensor.

        Returns:
            Estimated maximum absolute weight value.
        """
        mean = weight.mean().item()
        std = weight.std().item()
        num_elements = weight.numel()
        return abs(mean) + std * np.sqrt(2 * np.log(num_elements))

    def compute_sigma_symmetric(
        self,
        l_bound: float,
        rank: int,
        num_elements: int,
    ) -> float:
        """Compute a shared sigma for symmetric adapter initialization.

        Args:
            l_bound: Remaining headroom below quantization ceiling T.
            rank: Low-rank adapter rank.
            num_elements: Total number of elements in the weight matrix.

        Returns:
            Shared sigma for matrices A and B, floored at 1e-10.
        """
        if l_bound <= 0:
            return 1e-10
        sigma = (l_bound ** 2 / (2 * rank * np.log(num_elements))) ** 0.25
        return max(sigma * self.safety_factor, 1e-10)

    def initialize_adapters(
        self,
        weight: torch.Tensor,
        rank: int,
        n_bits: int,
        symmetric: bool = False,
    ) -> LowRankAdapterState:
        """Create low-rank adapters for a layer from weight statistics.

        Args:
            weight: Layer weight tensor of shape (rows, cols).
            rank: Low-rank adapter rank.
            n_bits: Target quantization bit width.
            symmetric: Whether to use symmetric quantization parameters.

        Returns:
            Initialized adapter state with quantization metadata.
        """
        with torch.no_grad():
            params = self._params_calculator.compute(weight, n_bits, symmetric)
            expected_max_w = self.estimate_expected_max_w(weight)
            ceiling = 2 ** (n_bits - 1) - 1
            l_bound = ceiling - expected_max_w

            rows, cols = weight.shape
            num_elements = rows * cols
            sigma = self.compute_sigma_symmetric(l_bound, rank, num_elements)

            adapter_a = torch.randn(rows, rank, device=weight.device) * sigma
            adapter_b = torch.randn(rank, cols, device=weight.device) * sigma
            sigma_b_max = self.compute_sigma_b_max(l_bound, sigma, rank, rows, cols)

            return LowRankAdapterState(
                adapter_a=adapter_a,
                adapter_b=adapter_b,
                sigma=sigma,
                params=params,
                sigma_b_max=sigma_b_max,
            )
