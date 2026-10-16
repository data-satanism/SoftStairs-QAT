"""SoftStairs-aware Adam optimizers: ``SSAGatedM``, ``SSAGatedV``, ``SSAG``.

These optimizers combine the ordinary (ungated) task gradient ``G_t``, Adam's
adaptive normalization, and the SoftStairs derivative ``D(W)`` in different
ways for quantization-aware training:

- ``SSAGatedM`` (variant C): first moment from ``G_t * D(W_t)``, second
  moment from ``G_t ** 2`` — the SoftStairs derivative gates the *moment*
  signal.
- ``SSAGatedV`` (variant D): both moments from ``G_t``; the SoftStairs
  derivative gates the normalized Adam update direction.
- ``SSAG`` (variant E): ordinary Adam step followed by an additive SoftStairs
  correction that reuses the cached normalized direction and evaluates the
  derivative at the intermediate parameters.

Usage
-----
Wrap the model with the quantizer using the ``'raw_gradient'`` type so that
the backward pass of the SoftStairs function passes the raw task gradient
through to ``weight_orig.grad`` unchanged (``G_t``), and pass the quantizer
to the optimizer:

    from softstairs_qat import (
        QuantizationConfig, SoftStairsQuantizer, SSAG,
    )

    quantizer = SoftStairsQuantizer(
        model, QuantizationConfig(..., type='raw_gradient'),
    )
    optimizer = SSAG(model.parameters(), quantizer, lr=1e-3)
    ...
    optimizer.zero_grad()
    loss.backward()          # weight_orig.grad is now the ungated G_t
    optimizer.step()         # applies D via the quantizer where appropriate
    quantizer.step()

The optimizers subclass :class:`torch.optim.Adam` and reuse its parameter
handling: parameter groups, per-group learning rates, ``zero_grad()``,
``state_dict()``/``load_state_dict()``, missing-gradient handling, and lazy
Adam state initialization (``exp_avg``/``exp_avg_sq``/``step``) all come from
PyTorch. The only addition is the SoftStairs derivative, which is evaluated
through ``quantizer.softstairs_derivative`` using the exact conventions of the
quantizer (scale, zero point, half shift, and current temperature). The
derivative computation never builds an autograd graph and no division by the
derivative is ever performed. Parameters that are not SoftStairs-quantized
(biases, excluded modules) follow standard Adam behaviour; for ``SSAG`` the
additive correction is simply omitted for them.

Notes:

- Mixed precision: as with standard Adam, unscale gradients (e.g.
  ``GradScaler.unscale_(optimizer)``) before ``step()``. ``U_t`` is invariant
  to a uniform positive rescaling of the gradients, so loss scaling does not
  change the update direction (up to ``eps`` and bias-correction transients).
- ``foreach``/``fused``/``capturable``/``amsgrad``/``maximize`` are not
  supported by the SoftStairs update paths; the optimizers always use the
  single-tensor math. Weight decay is not applied (the SoftStairs dynamics are
  studied in isolation).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

_RAW_GRADIENT_TYPE = 'raw_gradient'


def _validate_quantizer_type(quantizer) -> None:
    """Require a quantizer configured with the raw-gradient backward."""
    config = getattr(quantizer, 'config', None)
    hook_type = getattr(config, 'type', None)
    if hook_type is not None and hook_type != _RAW_GRADIENT_TYPE:
        raise ValueError(
            f'{type(quantizer).__name__} uses type={hook_type!r}. SoftStairs-aware '
            f"Adam optimizers require type='raw_gradient' so that param.grad is the "
            f'ungated task gradient G_t instead of the SoftStairs-gated G_t * D.'
        )


def _refresh_param_names(optimizer) -> Dict[int, str]:
    return {
        id(parameter): name
        for name, parameter in optimizer.quantizer.model.named_parameters()
    }


def _parameter_name(optimizer, parameter: torch.Tensor) -> Optional[str]:
    """Return the model parameter name for ``parameter`` (or ``None``)."""
    name = optimizer._param_names.get(id(parameter))
    if name is None:
        # The parameter was not known at construction time (e.g. added later
        # through ``add_param_group``); rebuild the mapping once.
        optimizer._param_names = _refresh_param_names(optimizer)
        name = optimizer._param_names.get(id(parameter))
    return name


def _derivative(optimizer, name: Optional[str], values=None) -> Optional[torch.Tensor]:
    """SoftStairs derivative for a quantized parameter, or ``None``."""
    if name is None or not optimizer.quantizer.is_quantized_parameter(name):
        return None
    return optimizer.quantizer.softstairs_derivative(name, values=values)


def _gather_group(self, group):
    """Reuse ``torch.optim.Adam._init_group`` to collect params, grads, state."""
    params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps = (
        [], [], [], [], [], [],
    )
    self._init_group(
        group, params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps,
    )
    return params_with_grad, grads, state_steps


def _normalized_direction(grad: torch.Tensor, state: dict, group: dict,
                          first_moment_grad: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Update both Adam moments and return the normalized direction ``U_t``.

    Implements Adam's standard update: ``m_t = beta1 * m + (1 - beta1) * g_m``,
    ``v_t = beta2 * v + (1 - beta2) * grad ** 2`` with bias correction, and
    ``U_t = m_hat / (sqrt(v_hat) + eps)``. The second moment always uses
    ``grad``; the first moment uses ``first_moment_grad`` when given (the
    SoftStairs-gated ``G * D`` of ``SSAGatedM``) and ``grad`` otherwise.
    """
    beta1, beta2 = group['betas']
    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
    step = state['step']
    if first_moment_grad is None:
        first_moment_grad = grad
    exp_avg.mul_(beta1).add_(first_moment_grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    m_hat = exp_avg / bias_correction1
    v_hat = exp_avg_sq / bias_correction2
    return m_hat / (v_hat.sqrt() + group['eps'])


class SSAGatedM(torch.optim.Adam):
    """``SSAGatedM`` (variant C): SoftStairs-gated first moment, ungated second.

    With ``D_t = D(W_t)`` evaluated at the pre-update parameters and ``G_t``
    the ungated task gradient (``param.grad`` under ``type='raw_gradient'``):

    - ``m_t = beta1 * m_{t-1} + (1 - beta1) * G_t * D_t``
    - ``v_t = beta2 * v_{t-1} + (1 - beta2) * G_t ** 2``
    - ``W_{t+1} = W_t - lr * m_hat / (sqrt(v_hat) + eps)``

    SoftStairs affects the accumulated update signal but not the
    second-moment normalization signal.
    """

    def __init__(self, params, quantizer, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        _validate_quantizer_type(quantizer)
        super().__init__(params, lr=lr, betas=betas, eps=eps)
        self.quantizer = quantizer
        self._param_names = _refresh_param_names(self)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad, grads, state_steps = _gather_group(self, group)
            for parameter, grad, step_t in zip(params_with_grad, grads, state_steps):
                step_t += 1
                state = self.state[parameter]
                name = _parameter_name(self, parameter)
                derivative = _derivative(self, name)
                gated_grad = grad if derivative is None else grad * derivative
                direction = _normalized_direction(grad, state, group, first_moment_grad=gated_grad)
                parameter.add_(direction, alpha=-group['lr'])
        return loss


class SSAGatedV(torch.optim.Adam):
    """``SSAGatedV`` (variant D): ungated moments, SoftStairs-gated update.

    Both Adam moments use the ungated task gradient ``G_t``; the SoftStairs
    derivative gates the normalized update direction:

    - ``m_t = beta1 * m_{t-1} + (1 - beta1) * G_t``
    - ``v_t = beta2 * v_{t-1} + (1 - beta2) * G_t ** 2``
    - ``W_{t+1} = W_t - lr * D_t * m_hat / (sqrt(v_hat) + eps)``

    with ``D_t = D(W_t)`` evaluated at the pre-update parameters.
    """

    def __init__(self, params, quantizer, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        _validate_quantizer_type(quantizer)
        super().__init__(params, lr=lr, betas=betas, eps=eps)
        self.quantizer = quantizer
        self._param_names = _refresh_param_names(self)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad, grads, state_steps = _gather_group(self, group)
            for parameter, grad, step_t in zip(params_with_grad, grads, state_steps):
                step_t += 1
                state = self.state[parameter]
                name = _parameter_name(self, parameter)
                derivative = _derivative(self, name)
                direction = _normalized_direction(grad, state, group)
                if derivative is not None:
                    direction = direction * derivative
                parameter.add_(direction, alpha=-group['lr'])
        return loss


class SSAG(torch.optim.Adam):
    """``SSAG`` (variant E): ordinary Adam step plus additive SoftStairs correction.

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

    def __init__(self, params, quantizer, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        _validate_quantizer_type(quantizer)
        super().__init__(params, lr=lr, betas=betas, eps=eps)
        self.quantizer = quantizer
        self._param_names = _refresh_param_names(self)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad, grads, state_steps = _gather_group(self, group)
            for parameter, grad, step_t in zip(params_with_grad, grads, state_steps):
                step_t += 1
                state = self.state[parameter]
                name = _parameter_name(self, parameter)
                direction = _normalized_direction(grad, state, group)
                parameter.add_(direction, alpha=-group['lr'])  # stage 1
                derivative = _derivative(self, name, values=parameter)
                if derivative is not None:
                    parameter.add_(direction * derivative, alpha=group['lr'])  # stage 2
        return loss


__all__ = [
    'SSAG',
    'SSAGatedM',
    'SSAGatedV',
]
