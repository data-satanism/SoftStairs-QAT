"""Deterministic unit tests for the SoftStairs-aware Adam optimizers C, D, E.

References are hand-computed: for a linear layer ``Y = X @ W.T + b`` with
``loss = (Y * L).sum()`` the ungated task gradient is exactly ``G = L.T @ X``
(independent of the weight values), and the SoftStairs derivative is evaluated
with the canonical ``SoftStairs.derivative`` on the upscaled coordinates.

All quantized-model tests use ``type='raw_gradient'`` so that the backward
pass of the SoftStairs function passes the raw task gradient through and
``weight_orig.grad`` is ``G`` (up to floating-point rounding of the
upscale/downscale round trip).
"""

import copy

import pytest
import torch
import torch.nn as nn

from softstairs_qat.core import SoftStairsQuantizer
from softstairs_qat.core.optimizers import SSAG, SSAGatedM, SSAGatedV
from softstairs_qat.core.soft_stairs import SoftStairs
from softstairs_qat.wrappers import QuantizationConfig

BATCH, IN, OUT = 5, 4, 3
LR = 1e-2
BETAS = (0.9, 0.999)
EPS = 1e-8
T = 0.1


def make_setup(optimizer_cls, lr=LR, seed=7, t=T, normalized=False, type_='raw_gradient'):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(IN, OUT))
    config = QuantizationConfig(
        n_bits=4,
        type=type_,
        t_scheduler_strategy='constant',
        t_start=t,
        t_end=t,
        normalized=normalized,
    )
    quantizer = SoftStairsQuantizer(model, config)
    optimizer = None
    if optimizer_cls is not None:
        optimizer = optimizer_cls(model.parameters(), quantizer, lr=lr, betas=BETAS, eps=EPS)
    return model, quantizer, optimizer


def forward_backward(model, seed):
    gen = torch.Generator().manual_seed(seed)
    X = torch.randn(BATCH, IN, generator=gen)
    L = torch.randn(BATCH, OUT, generator=gen)
    model.train()
    ((model(X) * L).sum()).backward()
    return X, L


def softstairs_derivative_at(quantizer, name, values):
    z = quantizer.upscaled_parameter(name, values=values)
    soft = SoftStairs(
        t=quantizer.t,
        normalized=quantizer.config.normalized,
        async_t_factor=quantizer.config.async_t_factor,
    )
    return soft.derivative(z)


def adam_direction(m, v, step, betas=BETAS, eps=EPS):
    beta1, beta2 = betas
    m_hat = m / (1 - beta1 ** step)
    v_hat = v / (1 - beta2 ** step)
    return m_hat / (v_hat.sqrt() + eps)


def run_weight_reference(variant, quantizer, name, grads, initial, lr=LR):
    """Hand-computed multi-step reference for a quantized weight."""
    W = initial.detach().clone()
    m = torch.zeros_like(W)
    v = torch.zeros_like(W)
    for step, (G,) in enumerate(grads, start=1):
        if variant == 'C':
            D = softstairs_derivative_at(quantizer, name, W)
            m = BETAS[0] * m + (1 - BETAS[0]) * (G * D)
            v = BETAS[1] * v + (1 - BETAS[1]) * G ** 2
            W = W - lr * adam_direction(m, v, step)
        elif variant == 'D':
            m = BETAS[0] * m + (1 - BETAS[0]) * G
            v = BETAS[1] * v + (1 - BETAS[1]) * G ** 2
            D = softstairs_derivative_at(quantizer, name, W)
            W = W - lr * D * adam_direction(m, v, step)
        else:  # E
            m = BETAS[0] * m + (1 - BETAS[0]) * G
            v = BETAS[1] * v + (1 - BETAS[1]) * G ** 2
            U = adam_direction(m, v, step)
            W = W - lr * U
            W = W + lr * U * softstairs_derivative_at(quantizer, name, W)
    return W


def run_plain_adam_reference(initial, grads, lr=LR):
    """Plain Adam reference for a non-quantized parameter."""
    W = initial.detach().clone()
    m = torch.zeros_like(W)
    v = torch.zeros_like(W)
    for step, G in enumerate(grads, start=1):
        m = BETAS[0] * m + (1 - BETAS[0]) * G
        v = BETAS[1] * v + (1 - BETAS[1]) * G ** 2
        W = W - lr * adam_direction(m, v, step)
    return W


@pytest.mark.parametrize('variant_cls,variant', [
    (SSAGatedM, 'C'),
    (SSAGatedV, 'D'),
    (SSAG, 'E'),
])
def test_update_matches_hand_reference_two_steps(variant_cls, variant):
    model, quantizer, optimizer = make_setup(variant_cls)
    weight_initial = model.get_parameter('0.weight_orig').detach().clone()
    bias_initial = model.get_parameter('0.bias').detach().clone()

    grads, bias_grads = [], []
    for seed in (101, 102):
        X, L = forward_backward(model, seed)
        grads.append((L.T @ X,))
        bias_grads.append(model.get_parameter('0.bias').grad.detach().clone())
        optimizer.step()
        optimizer.zero_grad()

    expected_weight = run_weight_reference(
        variant, quantizer, '0.weight_orig', grads, weight_initial,
    )
    torch.testing.assert_close(
        model.get_parameter('0.weight_orig'), expected_weight, rtol=1e-5, atol=1e-7,
    )
    # Non-quantized parameters follow plain Adam in every variant.
    expected_bias = run_plain_adam_reference(bias_initial, bias_grads)
    torch.testing.assert_close(
        model.get_parameter('0.bias'), expected_bias, rtol=1e-5, atol=1e-7,
    )


def test_variant_c_moments_and_update():
    model, quantizer, optimizer = make_setup(SSAGatedM)
    X, L = forward_backward(model, 101)
    W0 = model.get_parameter('0.weight_orig').detach().clone()
    G = L.T @ X
    D = softstairs_derivative_at(quantizer, '0.weight_orig', W0)

    optimizer.step()
    state = optimizer.state[model.get_parameter('0.weight_orig')]
    m_expected = (1 - BETAS[0]) * (G * D)
    v_expected = (1 - BETAS[1]) * G ** 2
    torch.testing.assert_close(state['exp_avg'], m_expected, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(state['exp_avg_sq'], v_expected, rtol=1e-5, atol=1e-8)
    U = adam_direction(m_expected, v_expected, 1)
    torch.testing.assert_close(
        model.get_parameter('0.weight_orig'), W0 - LR * U, rtol=1e-5, atol=1e-7,
    )
    optimizer.zero_grad()


def test_variant_d_moments_and_update():
    model, quantizer, optimizer = make_setup(SSAGatedV)
    X, L = forward_backward(model, 101)
    W0 = model.get_parameter('0.weight_orig').detach().clone()
    G = L.T @ X
    D = softstairs_derivative_at(quantizer, '0.weight_orig', W0)

    optimizer.step()
    state = optimizer.state[model.get_parameter('0.weight_orig')]
    torch.testing.assert_close(state['exp_avg'], (1 - BETAS[0]) * G, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(state['exp_avg_sq'], (1 - BETAS[1]) * G ** 2, rtol=1e-5, atol=1e-8)
    U = adam_direction((1 - BETAS[0]) * G, (1 - BETAS[1]) * G ** 2, 1)
    torch.testing.assert_close(
        model.get_parameter('0.weight_orig'), W0 - LR * D * U, rtol=1e-5, atol=1e-7,
    )
    optimizer.zero_grad()


def test_variant_e_two_stage_update():
    model, quantizer, optimizer = make_setup(SSAG)
    X, L = forward_backward(model, 101)
    W0 = model.get_parameter('0.weight_orig').detach().clone()
    G = L.T @ X

    optimizer.step()
    state = optimizer.state[model.get_parameter('0.weight_orig')]
    assert float(state['step']) == 1.0
    torch.testing.assert_close(state['exp_avg'], (1 - BETAS[0]) * G, rtol=1e-5, atol=1e-8)
    torch.testing.assert_close(state['exp_avg_sq'], (1 - BETAS[1]) * G ** 2, rtol=1e-5, atol=1e-8)
    U = adam_direction((1 - BETAS[0]) * G, (1 - BETAS[1]) * G ** 2, 1)
    W_intermediate = W0 - LR * U
    D_intermediate = softstairs_derivative_at(quantizer, '0.weight_orig', W_intermediate)
    torch.testing.assert_close(
        model.get_parameter('0.weight_orig'),
        W_intermediate + LR * U * D_intermediate,
        rtol=1e-5, atol=1e-7,
    )
    optimizer.zero_grad()


class _DerivativeSpy:
    """Wraps a quantizer's derivative evaluation, recording calls."""

    def __init__(self, quantizer):
        self.quantizer = quantizer
        self.calls = []
        self._original = quantizer.softstairs_derivative
        quantizer.softstairs_derivative = self._spy

    def _spy(self, name, values=None):
        if values is None:
            values = self.quantizer.model.get_parameter(name).detach().clone()
        else:
            values = values.detach().clone()
        self.calls.append((name, values))
        return self._original(name, values=values)

    def values_for(self, name):
        return [values for call_name, values in self.calls if call_name == name]


def test_variant_e_derivative_evaluated_once_at_intermediate_parameters():
    model, quantizer, optimizer = make_setup(SSAG)
    X, L = forward_backward(model, 101)
    W0 = model.get_parameter('0.weight_orig').detach().clone()
    G = L.T @ X
    U = adam_direction((1 - BETAS[0]) * G, (1 - BETAS[1]) * G ** 2, 1)

    spy = _DerivativeSpy(quantizer)
    optimizer.step()
    values = spy.values_for('0.weight_orig')
    assert len(values) == 1  # derivative computed exactly once per step
    torch.testing.assert_close(values[0], W0 - LR * U, rtol=1e-5, atol=1e-7)

    # No additional forward/backward pass happened: p.grad still holds exactly
    # the single ungated gradient from the one backward pass we ran.
    torch.testing.assert_close(
        model.get_parameter('0.weight_orig').grad, G, rtol=1e-5, atol=1e-7,
    )
    optimizer.zero_grad()


def test_variant_c_d_derivative_evaluated_at_pre_update_parameters():
    for cls in (SSAGatedM, SSAGatedV):
        model, quantizer, optimizer = make_setup(cls)
        forward_backward(model, 101)
        W0 = model.get_parameter('0.weight_orig').detach().clone()
        spy = _DerivativeSpy(quantizer)
        optimizer.step()
        values = spy.values_for('0.weight_orig')
        assert len(values) == 1
        torch.testing.assert_close(values[0], W0, rtol=1e-6, atol=1e-8)
        optimizer.zero_grad()


class _StubQuantizer:
    """Minimal quantizer stand-in with a prescribed derivative value.

    Used on a plain (unquantized) model, so ``param.grad`` itself is the
    ungated task gradient and the derivative value is fully controlled.
    """

    def __init__(self, model, derivative_value):
        self.model = model
        self.derivative_value = derivative_value

    def is_quantized_parameter(self, name):
        return name.endswith('weight')

    def softstairs_derivative(self, name, values=None):
        return torch.full_like(self.model.get_parameter(name), self.derivative_value)


def _plain_model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(IN, OUT))


def test_zero_derivative_variants_c_and_d():
    for cls in (SSAGatedM, SSAGatedV):
        model = _plain_model(11)
        stub = _StubQuantizer(model, 0.0)
        optimizer = cls(model.parameters(), stub, lr=LR, betas=BETAS, eps=EPS)
        forward_backward(model, 202)
        W0 = model[0].weight.detach().clone()
        optimizer.step()
        assert torch.isfinite(model[0].weight).all()
        torch.testing.assert_close(model[0].weight, W0)  # zero gate nullifies the update
        optimizer.zero_grad()


def test_near_zero_derivative_matches_reference():
    model = _plain_model(12)
    deriv_value = 1e-30
    stub = _StubQuantizer(model, deriv_value)
    optimizer = SSAGatedM(model.parameters(), stub, lr=LR, betas=BETAS, eps=EPS)
    X, L = forward_backward(model, 203)
    W0 = model[0].weight.detach().clone()
    G = L.T @ X
    D = torch.full_like(W0, deriv_value)

    optimizer.step()
    assert torch.isfinite(model[0].weight).all()
    U = adam_direction((1 - BETAS[0]) * (G * D), (1 - BETAS[1]) * G ** 2, 1)
    torch.testing.assert_close(model[0].weight, W0 - LR * U, rtol=1e-5, atol=1e-8)
    optimizer.zero_grad()


def test_zero_derivative_variant_e_reduces_to_stage_one():
    model = _plain_model(13)
    stub = _StubQuantizer(model, 0.0)
    optimizer = SSAG(model.parameters(), stub, lr=LR, betas=BETAS, eps=EPS)
    X, L = forward_backward(model, 204)
    W0 = model[0].weight.detach().clone()
    G = L.T @ X

    optimizer.step()
    U = adam_direction((1 - BETAS[0]) * G, (1 - BETAS[1]) * G ** 2, 1)
    torch.testing.assert_close(model[0].weight, W0 - LR * U, rtol=1e-5, atol=1e-8)
    optimizer.zero_grad()


def test_missing_gradients_are_skipped():
    model, quantizer, optimizer = make_setup(SSAGatedM)
    unused = nn.Linear(IN, OUT)
    optimizer.add_param_group({'params': unused.parameters(), 'lr': LR})
    forward_backward(model, 101)
    optimizer.step()
    for parameter in unused.parameters():
        assert parameter.grad is None
        assert optimizer.state[parameter] == {}
    optimizer.zero_grad()


def test_multiple_param_groups():
    model, quantizer, _ = make_setup(SSAGatedM)
    lr1, lr2 = 1e-2, 3e-3
    optimizer = SSAGatedM(
        [
            {'params': [model[0].weight_orig], 'lr': lr1},
            {'params': [model[0].bias], 'lr': lr2},
        ],
        quantizer,
        betas=BETAS,
        eps=EPS,
    )
    X, L = forward_backward(model, 101)
    W0 = model.get_parameter('0.weight_orig').detach().clone()
    b0 = model.get_parameter('0.bias').detach().clone()
    G = L.T @ X
    D = softstairs_derivative_at(quantizer, '0.weight_orig', W0)
    Gb = L.sum(0)
    optimizer.step()
    U = adam_direction((1 - BETAS[0]) * G * D, (1 - BETAS[1]) * G ** 2, 1)
    torch.testing.assert_close(
        model.get_parameter('0.weight_orig'), W0 - lr1 * U, rtol=1e-5, atol=1e-7,
    )
    Ub = adam_direction((1 - BETAS[0]) * Gb, (1 - BETAS[1]) * Gb ** 2, 1)
    torch.testing.assert_close(model.get_parameter('0.bias'), b0 - lr2 * Ub, rtol=1e-5, atol=1e-7)
    optimizer.zero_grad()


def test_state_dict_round_trip():
    model_a, quantizer_a, optimizer_a = make_setup(SSAGatedV)
    forward_backward(model_a, 101)
    optimizer_a.step()
    optimizer_a.zero_grad()
    forward_backward(model_a, 102)
    optimizer_a.step()
    optimizer_a.zero_grad()

    model_b, quantizer_b, optimizer_b = make_setup(SSAGatedV)
    # Place model_b at the same point in the trajectory as model_a before
    # restoring the optimizer state (load_state_dict restores state, not
    # parameter values). Deep-copy the state dict so the two optimizers do
    # not share state tensors, exactly like a save-to-disk round trip.
    with torch.no_grad():
        model_b.get_parameter('0.weight_orig').copy_(model_a.get_parameter('0.weight_orig'))
        model_b.get_parameter('0.bias').copy_(model_a.get_parameter('0.bias'))
    optimizer_b.load_state_dict(copy.deepcopy(optimizer_a.state_dict()))

    for seed in (103, 104):
        forward_backward(model_a, seed)
        optimizer_a.step()
        optimizer_a.zero_grad()
        forward_backward(model_b, seed)
        optimizer_b.step()
        optimizer_b.zero_grad()
    torch.testing.assert_close(
        model_b.get_parameter('0.weight_orig'), model_a.get_parameter('0.weight_orig'),
        rtol=1e-5, atol=1e-7,
    )
    torch.testing.assert_close(
        optimizer_b.state[model_b.get_parameter('0.weight_orig')]['exp_avg'],
        optimizer_a.state[model_a.get_parameter('0.weight_orig')]['exp_avg'],
        rtol=1e-5, atol=1e-8,
    )


def test_gradient_accumulation_and_zero_grad():
    model, quantizer, optimizer = make_setup(SSAGatedM)
    W0 = model.get_parameter('0.weight_orig').detach().clone()

    X1, L1 = forward_backward(model, 301)
    X2, L2 = forward_backward(model, 302)
    G = L1.T @ X1 + L2.T @ X2
    torch.testing.assert_close(
        model.get_parameter('0.weight_orig').grad, G, rtol=1e-5, atol=1e-7,
    )

    D = softstairs_derivative_at(quantizer, '0.weight_orig', W0)
    optimizer.step()
    U = adam_direction((1 - BETAS[0]) * G * D, (1 - BETAS[1]) * G ** 2, 1)
    torch.testing.assert_close(
        model.get_parameter('0.weight_orig'), W0 - LR * U, rtol=1e-5, atol=1e-7,
    )

    optimizer.zero_grad()
    assert model.get_parameter('0.weight_orig').grad is None

    X3, L3 = forward_backward(model, 303)
    torch.testing.assert_close(
        model.get_parameter('0.weight_orig').grad, L3.T @ X3, rtol=1e-5, atol=1e-7,
    )
    optimizer.zero_grad()


def test_raw_gradient_type_passes_ungated_gradient_through():
    model, quantizer, _ = make_setup(None)
    X, L = forward_backward(model, 101)
    # The raw-gradient backward passes the task gradient through unchanged
    # (upscale/downscale scale factors cancel), so p.grad is the ungated G.
    torch.testing.assert_close(
        model.get_parameter('0.weight_orig').grad, L.T @ X, rtol=1e-5, atol=1e-7,
    )


def test_standard_type_still_gates_gradient():
    model, quantizer, _ = make_setup(None, type_='standard')
    X, L = forward_backward(model, 101)
    G = L.T @ X
    D = softstairs_derivative_at(quantizer, '0.weight_orig', model.get_parameter('0.weight_orig'))
    torch.testing.assert_close(
        model.get_parameter('0.weight_orig').grad, G * D, rtol=1e-5, atol=1e-7,
    )


def test_optimizers_reject_non_raw_gradient_types():
    model, quantizer, _ = make_setup(None, type_='standard')
    with pytest.raises(ValueError, match='raw_gradient'):
        SSAG(model.parameters(), quantizer)


def test_derivative_consistency_with_softstairs_implementation():
    model, quantizer, optimizer = make_setup(SSAGatedM, normalized=True)
    forward_backward(model, 101)
    name = '0.weight_orig'
    W = model.get_parameter(name).detach().clone()
    soft = SoftStairs(
        t=quantizer.t,
        normalized=quantizer.config.normalized,
        async_t_factor=quantizer.config.async_t_factor,
    )
    torch.testing.assert_close(quantizer.softstairs_derivative(name), soft.derivative(quantizer.upscaled_parameter(name, values=W)))
    W_shifted = W + 0.05
    torch.testing.assert_close(
        quantizer.softstairs_derivative(name, values=W_shifted),
        soft.derivative(quantizer.upscaled_parameter(name, values=W_shifted)),
    )
    optimizer.zero_grad()
