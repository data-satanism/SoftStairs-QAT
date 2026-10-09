"""SoftStairs-aware Adam optimizers (variants C, D, and E).

These optimizers combine the ordinary (ungated) task gradient ``G_t``, Adam's
adaptive normalization, and the SoftStairs derivative ``D(W)`` in different
ways for quantization-aware training:

- ``SoftStairsAdamC``: first moment from ``G_t * D(W_t)``, second moment from
  ``G_t ** 2``.
- ``SoftStairsAdamD``: both moments from ``G_t``; the SoftStairs derivative
  gates the normalized Adam update direction.
- ``SoftStairsAdamE``: ordinary Adam step followed by an additive SoftStairs
  correction that reuses the cached normalized direction and evaluates the
  derivative at the intermediate parameters.

The ungated task gradients are provided by the gradient stash that the
SoftStairs autograd functions populate during the backward pass (see
``softstairs_qat.core.ungated_gradients``), so no extra forward/backward pass
and no division by the derivative are ever performed. Parameters that are not
SoftStairs-quantized (biases, excluded modules, or ``quantizer=None``) fall
back to standard Adam behaviour, since no SoftStairs derivative is defined
for them; for variant E this means the additive correction is simply omitted.

Notes:

- Gradient accumulation: the stash accumulates across backward passes exactly
  like ``param.grad`` and is cleared by ``zero_grad()`` together with it.
- Mixed precision: the stashed gradients carry the same loss-scale factor as
  the backpropagated gradients. Since every variant builds both Adam moments
  from stashed gradients, the normalized direction ``U_t`` is invariant to a
  uniform positive gradient rescaling, so AMP/GradScaler conventions are
  preserved (as with standard Adam, ``eps`` is applied to the unscaled
  denominator).
- ``type='standard'`` (and ``type='shifted'`` with the default
  ``async_t_factor=1``) expose the ungated gradient. ``type='naive'`` routes
  autograd through the SoftStairs forward formula and cannot stash the
  ungated gradient; those optimizers then fall back to standard Adam on the
  (already gated) ``param.grad``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


class _SoftStairsAdamBase(torch.optim.Optimizer):
    """Shared plumbing for the SoftStairs-aware Adam variants."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        quantizer=None,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f'Invalid learning rate: {lr}')
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]}')
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]}')
        if eps < 0.0:
            raise ValueError(f'Invalid epsilon value: {eps}')
        defaults = {'lr': lr, 'betas': tuple(betas), 'eps': eps}
        super().__init__(params, defaults)
        self.quantizer = quantizer
        self._param_names: Dict[int, str] = {}
        if quantizer is not None:
            self._refresh_param_names()

    def _refresh_param_names(self) -> None:
        self._param_names = {
            id(parameter): name
            for name, parameter in self.quantizer.model.named_parameters()
        }

    def _name_for(self, parameter: torch.Tensor) -> Optional[str]:
        name = self._param_names.get(id(parameter))
        if name is None and self.quantizer is not None:
            self._refresh_param_names()
            name = self._param_names.get(id(parameter))
        return name

    def _is_quantized(self, name: Optional[str]) -> bool:
        return (
            name is not None
            and self.quantizer is not None
            and self.quantizer.is_quantized_parameter(name)
        )

    def _task_gradient(self, parameter: torch.Tensor, name: Optional[str]) -> torch.Tensor:
        """Return the ungated task gradient ``G_t`` for ``parameter``."""
        if self._is_quantized(name):
            grad = self.quantizer.task_gradient(name)
            if grad is not None:
                return grad
        return parameter.grad

    def _init_state(self, parameter: torch.Tensor) -> dict:
        state = self.state[parameter]
        if len(state) == 0:
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
            state['exp_avg_sq'] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
        state['step'] += 1
        return state

    def _adam_direction(
        self,
        grad_first_moment: torch.Tensor,
        grad_second_moment: torch.Tensor,
        state: dict,
        group: dict,
    ) -> torch.Tensor:
        """Update both Adam moments and return the normalized direction ``U_t``.

        ``U_t = m_hat / (sqrt(v_hat) + eps)`` with standard bias correction.
        """
        beta1, beta2 = group['betas']
        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        exp_avg.mul_(beta1).add_(grad_first_moment, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad_second_moment, grad_second_moment, value=1 - beta2)
        step = state['step']
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        m_hat = exp_avg / bias_correction1
        v_hat = exp_avg_sq / bias_correction2
        return m_hat / (v_hat.sqrt() + group['eps'])

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear parameter gradients and the matching stashed task gradients."""
        super().zero_grad(set_to_none=set_to_none)
        if self.quantizer is None:
            return
        names = []
        for group in self.param_groups:
            for parameter in group['params']:
                name = self._name_for(parameter)
                if name is not None and self.quantizer.is_quantized_parameter(name):
                    names.append(name)
        if names:
            self.quantizer.clear_task_gradients(names)


class SoftStairsAdamC(_SoftStairsAdamBase):
    """Variant C: SoftStairs-gated first moment, ungated second moment.

    With ``D_t = D(W_t)`` evaluated at the pre-update parameters and ``G_t``
    the ungated task gradient:

    - ``m_t = beta1 * m_{t-1} + (1 - beta1) * G_t * D_t``
    - ``v_t = beta2 * v_{t-1} + (1 - beta2) * G_t ** 2``
    - ``W_{t+1} = W_t - lr * m_hat / (sqrt(v_hat) + eps)``

    SoftStairs affects the accumulated update signal but not the
    second-moment normalization signal.
    """

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                name = self._name_for(parameter)
                grad = self._task_gradient(parameter, name)
                state = self._init_state(parameter)
                if self._is_quantized(name):
                    derivative = self.quantizer.softstairs_derivative(name)
                    grad_first = grad * derivative
                else:
                    grad_first = grad
                direction = self._adam_direction(grad_first, grad, state, group)
                parameter.add_(direction, alpha=-group['lr'])
        return loss


class SoftStairsAdamD(_SoftStairsAdamBase):
    """Variant D: ungated moments, SoftStairs-gated normalized update.

    Both Adam moments use the ungated task gradient ``G_t``; the SoftStairs
    derivative gates the normalized update direction:

    - ``m_t = beta1 * m_{t-1} + (1 - beta1) * G_t``
    - ``v_t = beta2 * v_{t-1} + (1 - beta2) * G_t ** 2``
    - ``W_{t+1} = W_t - lr * D_t * m_hat / (sqrt(v_hat) + eps)``

    with ``D_t = D(W_t)`` evaluated at the pre-update parameters.
    """

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                name = self._name_for(parameter)
                grad = self._task_gradient(parameter, name)
                state = self._init_state(parameter)
                direction = self._adam_direction(grad, grad, state, group)
                if self._is_quantized(name):
                    derivative = self.quantizer.softstairs_derivative(name)
                    direction = direction * derivative
                parameter.add_(direction, alpha=-group['lr'])
        return loss


class SoftStairsAdamE(_SoftStairsAdamBase):
    """Variant E: ordinary Adam step plus additive SoftStairs correction.

    Stage 1 performs the ordinary Adam update with the ungated task gradient
    and computes the normalized direction ``U_t`` exactly once. Stage 2
    evaluates the SoftStairs derivative at the intermediate (post-stage-1)
    parameters and applies an additive correction with the cached ``U_t``:

    - ``W_{t+1} = W_t - lr * U_t``
    - ``W_{t+1} += lr * U_t * D(W_t - lr * U_t)``

    The positive sign of the correction is intentional: it partially offsets
    the first update in proportion to the SoftStairs derivative. Beyond the
    ordinary Adam work, the only extra cost is one derivative evaluation and
    one elementwise update; no additional forward pass, backward pass, or
    gradient computation is performed.
    """

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                name = self._name_for(parameter)
                grad = self._task_gradient(parameter, name)
                state = self._init_state(parameter)
                direction = self._adam_direction(grad, grad, state, group)
                parameter.add_(direction, alpha=-group['lr'])
                if self._is_quantized(name):
                    derivative = self.quantizer.softstairs_derivative(name, values=parameter)
                    parameter.add_(direction * derivative, alpha=group['lr'])
        return loss


__all__ = [
    'SoftStairsAdamC',
    'SoftStairsAdamD',
    'SoftStairsAdamE',
]
