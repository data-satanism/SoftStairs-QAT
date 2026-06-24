from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from softstairs_qat.core.quantize_function import SoftStairsQuantizer
from softstairs_qat.core.soft_stairs import SoftStairs
from softstairs_qat.core.variance_controller import VarianceController
from softstairs_qat.wrappers.config import QuantizationConfig


class QuantizedModelWrapper(nn.Module):
    """Wraps a model to apply SoftStairs QAT during the forward pass."""

    def __init__(
        self,
        model: nn.Module,
        config: QuantizationConfig,
        variance_controller: VarianceController | None = None,
        quantizer: SoftStairsQuantizer | None = None,
    ) -> None:
        """Initialize adapters and quantization metadata for ``model``.

        Args:
            model: Base model whose weights are quantized on each forward pass.
            config: Quantization hyperparameters.
            variance_controller: Controller used to initialize low-rank adapters.
            quantizer: Quantizer facade used during forward passes.
        """
        super().__init__()
        self.model = model
        self.config = config
        self._variance_controller = variance_controller or VarianceController(
            safety_factor=config.safety_factor,
        )
        self._quantizer = quantizer or SoftStairsQuantizer(
            SoftStairs(r=config.r, modified=config.modified),
        )
        self.adapters = nn.ParameterDict()
        self.quant_params: Dict[str, Tuple[Any, ...]] = {}
        self._init_for_model()

    @property
    def soft_stairs_counter(self):
        """Return the SoftStairs diagnostic counter."""
        return self._quantizer.soft_stairs.counter

    def _should_quantize_module(self, module: nn.Module) -> bool:
        """Return whether a module should receive quantization adapters."""
        if self.config.target_modules is None:
            return hasattr(module, "weight") and module.weight is not None
        return isinstance(module, self.config.target_modules)

    def _init_for_model(self) -> None:
        """Attach low-rank adapters to eligible modules."""
        for name, module in self.model.named_modules():
            if not self._should_quantize_module(module):
                continue

            state = self._variance_controller.initialize_adapters(
                module.weight,
                self.config.rank,
                self.config.n_bits,
                self.config.symmetric,
            )
            safe_name = name.replace(".", "_")
            self.adapters[f"{safe_name}_A"] = nn.Parameter(
                state.adapter_a.to(module.weight.device),
            )
            self.adapters[f"{safe_name}_B"] = nn.Parameter(
                state.adapter_b.to(module.weight.device),
            )
            self.quant_params[name] = (
                state.params.scale,
                state.params.zero_point,
                state.params.q_min,
                state.params.q_max,
                state.sigma_b_max,
            )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the wrapped model with temporarily quantized weights."""
        original_weights: Dict[str, torch.Tensor] = {}

        for name, module in self.model.named_modules():
            if name not in self.quant_params:
                continue

            safe_name = name.replace(".", "_")
            adapter_a = self.adapters[f"{safe_name}_A"]
            adapter_b = self.adapters[f"{safe_name}_B"]
            scale, zero_point, q_min, q_max, sigma_b_max = self.quant_params[name]

            original_weights[name] = module.weight.data.clone()
            module.weight.data = self._quantizer.quantize(
                module.weight,
                adapter_a,
                adapter_b,
                scale,
                zero_point,
                q_min,
                q_max,
                sigma_b_max,
            )

        output = self.model(*args, **kwargs)

        for name, module in self.model.named_modules():
            if name in original_weights:
                module.weight.data = original_weights[name]

        return output

    def get_trainable_parameters(self):
        """Return adapter parameters that should be optimized."""
        return list(self.adapters.parameters())
