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
quantizer (stored scale, zero point, current temperature, and lattice
frequency). The derivative computation never builds an autograd graph and no
division by the derivative is ever performed. Parameters that are not
SoftStairs-quantized (biases, excluded modules) follow standard Adam
behaviour; for ``SSAG`` the additive correction is simply omitted for them.

Code-space updates
------------------

Adam's normalized direction ``U_t`` is dimensionless, and ``param.grad`` under
``type='raw_gradient'`` is the *weight-space* task gradient (the
upscale/downscale factors of the pre-hook cancel). Applying ``-lr * U_t``
directly to the weight-space parameter therefore produces a step of
``lr * U_t * scale * weight_scale`` in normalized quantization coordinates —
inflated by exactly ``scale * weight_scale``, which grows with the bit width
(``scale = q_max / max_abs``). At 8 bits this is a step of many quantization
codes per optimizer step, pushing weights out of the ``[q_min, q_max]`` window
where they saturate on the boundary codes and the usable level structure
collapses.

The optimizers instead compute the SoftStairs-gated update for the *upscaled*
parameter (normalized quantization coordinates, where the SoftStairs
derivative, staircase lattice, and clamp window live) and downscale it back
via ``quantizer.downscale_delta``: with ``s`` the stored upscale factor (the
quantizer's scale, which already includes the constant weight-scale factor),
``z = s * W + zero_point``, the applied weight-space step is

- ``W_{t+1} = W_t - lr / s * U_t`` — i.e. ``z_{t+1} = z_t - lr * U_t`` —
  for the un-gated part, and likewise ``lr / s * D_t * U_t`` for the
  SoftStairs-gated part.

``lr`` is thus expressed in code units (fractions of a quantization level) for
quantized weights and is independent of the bit width and weight range; the
Adam moments need no rescaling because ``U_t`` is invariant to a uniform
rescaling of the gradients. Non-quantized parameters keep the ordinary
weight-space Adam step.

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

import warnings
from typing import Dict, Optional

import torch

_RAW_GRADIENT_TYPE = 'raw_gradient'

_UPDATE_SPACE = 'code'


class _CodeUpdateStateMixin:
    """State-dict marker for the code-space update semantics.

    The SSA optimizers apply SoftStairs-gated updates in code units (an
    update of ``lr`` code levels, downscaled through
    ``quantizer.downscale_delta``). Optimizer state saved by a version of
    these classes without that semantics resumes with a silently different
    effective learning rate for quantized weights. ``state_dict`` stamps
    ``update_space='code'`` and ``load_state_dict`` warns when the marker is
    missing.
    """

    def state_dict(self):
        state = super().state_dict()
        state['update_space'] = _UPDATE_SPACE
        return state

    def load_state_dict(self, state_dict):
        if isinstance(state_dict, dict) and state_dict.get('update_space') != _UPDATE_SPACE:
            warnings.warn(
                f'{type(self).__name__}: optimizer state lacks the '
                f"'update_space={_UPDATE_SPACE}' marker of the code-space update "
                'semantics; if it predates that change, the effective learning '
                'rate for quantized weights differs from the current behaviour.',
                stacklevel=2,
            )
        super().load_state_dict(state_dict)


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


def _weight_step(optimizer, name: Optional[str], direction: torch.Tensor) -> torch.Tensor:
    """Convert a code-space update direction to weight space.

    ``direction`` is the dimensionless Adam update expressed in code units
    (one unit = one quantization level). For quantized parameters it is
    mapped through ``quantizer.downscale_delta`` (divided by the stored
    upscale factor, which already includes the constant weight scale) so
    that an update computed for the upscaled parameter lands correctly on
    the weight-space parameter; for other parameters it is returned
    unchanged (the ordinary Adam step).
    """
    if name is None or not optimizer.quantizer.is_quantized_parameter(name):
        return direction
    return optimizer.quantizer.downscale_delta(name, direction)


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

    Scalar bias corrections and a single reused denominator temporary keep
    the per-parameter allocations at two tensors per step (moments are
    updated in place, as in torch's single-tensor Adam).
    """
    beta1, beta2 = group['betas']
    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
    step = state['step']
    if first_moment_grad is None:
        first_moment_grad = grad
    exp_avg.mul_(beta1).add_(first_moment_grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    denom = exp_avg_sq.div(1 - beta2 ** step).sqrt_().add_(group['eps'])
    return exp_avg.div(1 - beta1 ** step).div_(denom)


class SSAGatedM(_CodeUpdateStateMixin, torch.optim.Adam):
    """``SSAGatedM`` (variant C): SoftStairs-gated first moment, ungated second.

    With ``D_t = D(W_t)`` evaluated at the pre-update parameters, ``G_t``
    the ungated task gradient (``param.grad`` under ``type='raw_gradient'``),
    and ``s`` the stored upscale factor (scale incl. weight scale):

    - ``m_t = beta1 * m_{t-1} + (1 - beta1) * G_t * D_t``
    - ``v_t = beta2 * v_{t-1} + (1 - beta2) * G_t ** 2``
    - ``W_{t+1} = W_t - lr / s * m_hat / (sqrt(v_hat) + eps)``

    The update is computed on the upscaled parameter (``lr`` is in code
    units) and downscaled back to weight space. SoftStairs affects the
    accumulated update signal but not the second-moment normalization
    signal.
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
                parameter.add_(_weight_step(self, name, direction), alpha=-group['lr'])
        return loss


class SSAGatedV(_CodeUpdateStateMixin, torch.optim.Adam):
    """``SSAGatedV`` (variant D): ungated moments, SoftStairs-gated update.

    Both Adam moments use the ungated task gradient ``G_t``; the SoftStairs
    derivative gates the normalized update direction. With
    ``s`` the stored upscale factor (scale incl. weight scale):

    - ``m_t = beta1 * m_{t-1} + (1 - beta1) * G_t``
    - ``v_t = beta2 * v_{t-1} + (1 - beta2) * G_t ** 2``
    - ``W_{t+1} = W_t - lr / s * D_t * m_hat / (sqrt(v_hat) + eps)``

    with ``D_t = D(W_t)`` evaluated at the pre-update parameters. The gated
    update is computed on the upscaled parameter (``lr`` is in code units)
    and downscaled back to weight space.
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
                    direction = _weight_step(self, name, direction * derivative)
                parameter.add_(direction, alpha=-group['lr'])
        return loss


class SSAG(_CodeUpdateStateMixin, torch.optim.Adam):
    """``SSAG`` (variant E): ordinary Adam step plus additive SoftStairs correction.

    Stage 1 performs the ordinary Adam update with the ungated task gradient
    and computes the normalized direction ``U_t`` exactly once. Stage 2
    evaluates the SoftStairs derivative at the intermediate (post-stage-1)
    parameters and applies an additive correction with the cached ``U_t``.
    With ``s`` the stored upscale factor (scale incl. weight scale), both
    stages are computed on the upscaled parameter (``lr`` is in code units)
    and downscaled back to weight space:

    - ``W_{t+1} = W_t - lr / s * U_t``
    - ``W_{t+1} += lr / s * U_t * D(W_t - lr / s * U_t)``

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
                parameter.add_(_weight_step(self, name, direction), alpha=-group['lr'] * self.quantizer.max_derivative_value)  # stage 1
                derivative = _derivative(self, name, values=parameter)
                if derivative is not None:
                    parameter.add_(
                        _weight_step(self, name, direction * derivative), alpha=group['lr'],
                    )  # stage 2
        return loss


__all__ = [
    'SSAG',
    'SSAGatedM',
    'SSAGatedV',
]
