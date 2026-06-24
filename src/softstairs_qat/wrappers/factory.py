from __future__ import annotations

import torch.nn as nn

from softstairs_qat.wrappers.config import QuantizationConfig
from softstairs_qat.wrappers.quantize_wrapper import QuantizedModelWrapper


class ModelQuantizationFactory:
    """Creates ``QuantizedModelWrapper`` instances from configuration."""

    def wrap(
        self,
        model: nn.Module,
        config: QuantizationConfig | None = None,
        **kwargs,
    ) -> QuantizedModelWrapper:
        """Wrap a model for SoftStairs quantization-aware training.

        Args:
            model: Base PyTorch model.
            config: Quantization configuration. If omitted, keyword arguments
                are used to build a ``QuantizationConfig``.
            **kwargs: Fields accepted by ``QuantizationConfig`` when ``config``
                is not provided.

        Returns:
            A wrapper that quantizes eligible weights on each forward pass.
        """
        if config is None:
            config = QuantizationConfig(**kwargs)
        return QuantizedModelWrapper(model=model, config=config)
