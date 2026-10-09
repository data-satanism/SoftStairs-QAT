from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch

_UNGATED_GRADIENTS: Dict[str, torch.Tensor] = {}


def stash_ungated_gradient(key: str, grad: torch.Tensor) -> None:
    """Store (or accumulate) an ungated task gradient under ``key``.

    Called from the backward pass of the SoftStairs autograd functions with
    the gradient *before* the SoftStairs derivative gate is applied. Repeated
    backward passes (gradient accumulation) add up, mirroring how PyTorch
    accumulates ``param.grad``.

    Args:
        key: Unique stash key (scoped per quantizer and parameter name).
        grad: Ungated gradient tensor as received in ``backward``.
    """
    grad = grad.detach()
    existing = _UNGATED_GRADIENTS.get(key)
    if existing is None:
        _UNGATED_GRADIENTS[key] = grad.clone()
    else:
        existing += grad


def get_ungated_gradient(key: str) -> Optional[torch.Tensor]:
    """Return the stashed ungated gradient for ``key``, or ``None``."""
    return _UNGATED_GRADIENTS.get(key)


def ungated_gradient_keys() -> List[str]:
    """Return a snapshot of all stash keys with stored gradients."""
    return list(_UNGATED_GRADIENTS.keys())


def clear_ungated_gradients(keys: Iterable[str]) -> None:
    """Remove the stashed ungated gradients for the given keys."""
    for key in keys:
        _UNGATED_GRADIENTS.pop(key, None)
