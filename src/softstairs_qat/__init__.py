from softstairs_qat.core.soft_stairs import SoftStairs, SoftStairsCallCounter
from softstairs_qat.wrappers import (
    ModelQuantizationFactory,
    QuantizationConfig,
    QuantizedModelWrapper,
)

_factory = ModelQuantizationFactory()


def wrap_model_for_quantization(model, **kwargs) -> QuantizedModelWrapper:
    """Backward-compatible helper that wraps a model for QAT.

    Args:
        model: Base PyTorch model.
        **kwargs: Fields accepted by ``QuantizationConfig``.

    Returns:
        Quantized model wrapper.
    """
    return _factory.wrap(model, **kwargs)


__all__ = [
    "ModelQuantizationFactory",
    "QuantizationConfig",
    "QuantizedModelWrapper",
    "SoftStairs",
    "SoftStairsCallCounter",
    "wrap_model_for_quantization",
]
