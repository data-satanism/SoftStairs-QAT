import torch
import torch.nn as nn
import warnings
from typing import Optional, Dict, List, Set
from peft import PeftModel

from softstairs_qat.core.soft_stairs import SoftStairs, softstairs_naive, SoftStairsShifted, ScaledSoftStairs
from softstairs_qat.core.variance_controller import VarianceController
from softstairs_qat.core.quantization_params import QuantizationParamsCalculator
from softstairs_qat.wrappers.config import QuantizationConfig
from softstairs_qat.utils.t_scheduler import TScheduler, AdaptiveScheduler, FrequencyScheduler


EPSILON = 1e-6
R_CHANGE_THRESHOLD = 1e-12
VARIANCE_CONSTRAINT_INTERVAL = 10

from functools import partial 


class SoftStairsQuantizeFunction(torch.autograd.Function):
    """
    Unified quantization function with optional low-rank adapters.
    Accepts:
        weight:          full weight matrix W
        t:               SoftStairs sharpness parameter
        normalized:      use normalized backward surrogate?
        async_t_factor:  multiplier on t in the backward derivative only
        frequency:       SoftStairs lattice scale (1 = original SoftStairs),
                         realized as ScaledSoftStairs(scale=frequency)
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, t: float, normalized: bool = True, async_t_factor: float = 1., frequency: float = 1.) -> torch.Tensor:
        soft = ScaledSoftStairs(t=t, normalized=normalized, async_t_factor=async_t_factor, scale=frequency)
        x_soft = soft.forward(x)
        ctx.save_for_backward(x)
        ctx.t = t
        ctx.modified = normalized
        ctx.async_t_factor = async_t_factor
        ctx.frequency = frequency
        return x_soft
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        soft = ScaledSoftStairs(t=ctx.t, normalized=ctx.modified, async_t_factor=ctx.async_t_factor, scale=ctx.frequency)
        grad = grad_output * soft.derivative(x)
        return grad, None, None, None, None


class SoftStairsRawGradientQuantizeFunction(torch.autograd.Function):
    """
    SoftStairs forward whose backward passes the raw gradient through unchanged.

    The forward output is identical to ``SoftStairsQuantizeFunction``. The
    backward returns ``grad_output`` as-is, so the gradient accumulated in
    ``weight_orig.grad`` is the ordinary (ungated) task gradient ``G_t``: the
    upscale/downscale scale factors of the pre-hook cancel, and the SoftStairs
    derivative gate is not applied. SoftStairs-aware optimizers (variants C,
    D, E) re-introduce the derivative when forming Adam's moments and updates,
    keeping the backward pass, the gradient, and the gating concerns separate.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, t: float, normalized: bool = True, async_t_factor: float = 1., frequency: float = 1.) -> torch.Tensor:
        soft = ScaledSoftStairs(t=t, normalized=normalized, async_t_factor=async_t_factor, scale=frequency)
        return soft.forward(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None, None, None, None
    

class SoftStairsShiftedQuantizeFunction(torch.autograd.Function):
    """
    Unified quantization function with optional low-rank adapters.
    Accepts:
        weight:          full weight matrix W
        t:               SoftStairs sharpness parameter
        normalized:      use normalized backward surrogate?
        async_t_factor:  multiplier on t in the backward derivative only
        frequency:       SoftStairs lattice scale (1 = original SoftStairs);
                         composed manually as shifted(f x) / f since the scale
                         logic lives only in ScaledSoftStairs
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, t: float, normalized: bool = True, async_t_factor: float = 1., frequency: float = 1.) -> torch.Tensor:
        soft = SoftStairsShifted(t=t, normalized=normalized)
        x_soft = soft.forward(x * frequency) / frequency
        ctx.save_for_backward(x)
        ctx.t = t
        ctx.modified = normalized
        ctx.async_t_factor = async_t_factor
        ctx.frequency = frequency
        return x_soft
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        soft = SoftStairsShifted(t=ctx.t, normalized=ctx.modified, async_t_factor=ctx.async_t_factor)
        grad = grad_output * soft.derivative(x * ctx.frequency)
        return grad, None, None, None, None
    

# def quantize_soft_stairs(
#     x: torch.Tensor,
#     t: float,
#     normalized: bool = False,
#     scale: Optional[torch.Tensor] = None,
# ) -> torch.Tensor:
#     return SoftStairsQuantizeFunction.apply(x, t, normalized, scale)


class SoftStairsQuantizer:
    """
    Manages quantization via SoftStairs.
    Supports:
      - is_lora=False: all model weights are trained; SoftStairs is applied to them during each forward pass.
      - is_lora=True: model weights are frozen; only adapters (LoRA) are trained.
                        During each forward pass, `quantized = soft_stairs(W_orig + A@B)` is calculated.
    Controls adapter variance via VarianceController.

    Constant weight scaling: the weights entering quantization are scaled by
    the fixed factor ``weight_scale`` (``QuantizationConfig.weight_scale``,
    0.9 by default), i.e. ``W' = weight_scale * W``. The factor is folded
    into the stored scale (``scale_stored = weight_scale * scale_raw``), so
    the standard affine dequantization ``W = (code - zero_point) /
    scale_stored`` is the single exact inverse map for training, evaluation,
    deployment, and the SoftStairs-aware optimizers — no consumer needs to
    know about the scaling separately.
    """
    _check_field = '_ssquant_'
    _orig_suffix = '_orig'

    def __init__(
        self,
        model: nn.Module,
        config: QuantizationConfig,
        excluded_modules: Optional[Set[str]] = None,
        verbose=False,
        per_row: bool = False,
    ):
        if getattr(model, self._check_field, False):
            raise RuntimeError('Attempting to double wrap model with SSQuant')
        
        self._is_active = True 
        self.model = model
        self.config = config
        self.verbose=verbose
        self.per_row = per_row
        # self.total_steps = config.t_step
        self.current_step = 0

        self.excluded_modules = excluded_modules or set()

        self.scheduler: Optional[TScheduler | AdaptiveScheduler] = None
        if config.t_scheduler_strategy == "constant":
            current_t = config.t_start
        elif config.t_scheduler_strategy == "adaptive":
            self.scheduler = AdaptiveScheduler.from_config(config)
            current_t = config.t_start
        else:
            self.scheduler = TScheduler.from_config(config, total_steps=config.n_steps)
            current_t = config.t_start

        self._t = current_t 

        self.frequency_scheduler: Optional[FrequencyScheduler] = FrequencyScheduler.from_config(config)
        self._f = self.frequency_scheduler.get_f(self._t) if self.frequency_scheduler is not None else 1.

        self._ws = getattr(config, 'weight_scale', 1.0)

        self._scales: Dict[str, torch.Tensor] = {}
        self._zero_points: Dict[str, torch.Tensor] = {}
        self._q_min: Dict[str, int] = {}
        self._q_max: Dict[str, int] = {}

        # self._is_lora = config.is_lora
        # self._adapter_name = getattr(config, "adapter_name", "default")
        # self._rank = config.rank

        # self._variance_controller = VarianceController(safety_factor=config.safety_factor) if self._is_lora else None
        # self._sigma_A: Dict[str, float] = {}
        # self._sigma_B: Dict[str, float] = {}
        self._infer_hook_type()
        self._hook_handles = []

        self._calc = QuantizationParamsCalculator()
        self._init_quantization()

        self._register_hooks()
        setattr(self.model, self._check_field, True)

    def activate_hooks(self):
        self._is_active = True 

    def deactivate_hooks(self):
        self._is_active = False

    @property
    def t(self):
        return self._t

    @property
    def f(self):
        """Current SoftStairs staircase frequency (1 when homotopy is off)."""
        return self._f

    @property
    def weight_scale(self):
        """Constant weight-scale factor applied to the weights entering
        quantization (``QuantizationConfig.weight_scale``, 0.9 by default).

        The factor is folded into the stored scales (see ``_compute_scale``),
        so nothing else needs to multiply or divide by it: the stored affine
        map is already the exact upscale/downscale of the scaled weights.
        """
        return self._ws

    def _should_quantize_module(self, module_name: str) -> bool:
        return module_name not in self.excluded_modules

    def _get_q_min(self) -> int:
        """Minimum quantized integer code for the configured bit width."""
        return -2 ** (self.config.n_bits - 1)

    def _get_q_max(self) -> int:
        """Maximum quantized integer code for the configured bit width."""
        return 2 ** (self.config.n_bits - 1) - 1

    def _compute_scale(self, parameter: torch.Tensor) -> tuple:
        """Scale and zero-point for ``parameter``.

        Per-row parameters (when ``per_row`` is set and the parameter is at
        least 2-D) get a row-wise scale/zero-point; everything else is
        per-tensor. Override to plug in different logic.

        The constant weight-scale factor ``self._ws`` is folded into the
        returned scale: ``scale_stored = _ws * scale_raw`` computed from the
        raw parameter statistics. The stored affine map ``z = W *
        scale_stored + zero_point`` therefore maps the *scaled* weights
        ``W' = _ws * W`` onto the code window, and the standard dequantization
        ``(code - zero_point) / scale_stored`` is the single exact inverse
        map for training, evaluation, and deployment alike.
        """
        if self.per_row and parameter.dim() >= 2:
            scales, zps = [], []
            for i in range(parameter.shape[0]):
                p = self._calc.compute(
                    parameter[i],
                    n_bits=self.config.n_bits,
                    symmetric=self.config.symmetric,
                )
                scales.append(p.scale.reshape(()))
                zps.append(p.zero_point.reshape(()))

            reshape = (parameter.shape[0],) + (1,) * (parameter.dim() - 1)
            scale      = torch.stack(scales).reshape(reshape)
            zero_point = torch.stack(zps).reshape(reshape)
        else:
            p = self._calc.compute(
                parameter,
                n_bits=self.config.n_bits,
                symmetric=self.config.symmetric,
            )
            scale, zero_point = p.scale, p.zero_point
        return scale * self._ws, zero_point

    def _init_quantization(self):
        """
                Initializes quantization structure.
                For standard mode: weight -> weight_orig in code space (scale once + SoftStairs).
                For LoRA mode: freeze base weight in code space; init existing PEFT adapters
                via VarianceController (do not register lora_A/lora_B yourself).
                """
        with torch.no_grad():            
            for name_m, module in self.model.named_modules():
                if not self._should_quantize_module(name_m):
                    continue
                            
                for name_p, parameter in list(module.named_parameters(recurse=False)):
                    if name_p.endswith('bias'):
                        continue

                    scale, zero_point = self._compute_scale(parameter)
                    q_min = self._get_q_min()
                    q_max = self._get_q_max()

                    full_name = f'{name_m}.{name_p}'
                    if self.verbose:
                        print(f'@@@ INIT {full_name} scale_shape={tuple(scale.shape)}')

                    self._scales[full_name]      = scale
                    self._zero_points[full_name] = zero_point
                    self._q_min[full_name]       = q_min
                    self._q_max[full_name]       = q_max
                    
                    module.register_parameter(f'{name_p}{self._orig_suffix}', parameter)
                    
                    module.register_parameter(name_p, None)
                    if hasattr(module, name_p):
                        delattr(module, name_p)
                    
                    module.register_buffer(name_p, parameter)

    @torch.no_grad()
    def _recalculate_scales(self) -> None:
        """Recompute scale and zero-point for every quantized parameter.

        Recalculates the affine map from the current values of the
        ``*_orig`` parameters, so the code window tracks the weights as they
        drift during training. Called on every training step when
        ``config.recalc_scale_each_step`` is enabled.
        """
        for name_p, parameter in self.model.named_parameters(recurse=True):
            if not name_p.endswith(self._orig_suffix):
                continue
            base_name = self._base_parameter_name(name_p)
            scale, zero_point = self._compute_scale(parameter)
            self._scales[base_name] = scale
            self._zero_points[base_name] = zero_point

    def _register_hooks(self):
        """Registers two separate forward pre-hooks per quantized layer:
        1) SoftStairs (strategy via _make_pre_hook / is_lora) — no scaling
        2) input scale (init-time scale) — no SoftStairs

        The constant weight scaling is not applied here: it lives inside
        ``upscaled_parameter`` (the scaled weights are what SoftStairs sees).
        """
        for name_m, module in self.model.named_modules():
            if not self._should_quantize_module(name_m):
                continue

            self._hook_handles.append(module.register_forward_pre_hook(self._make_pre_hook(name_m)))


    def _make_pre_hook(self, layer_name: str):
        """
        Creates SoftStairs pre-hook; selects LoRA vs standard by is_lora.
        No input/weight scaling here.
        """
        # if self._is_lora:
        #     return self._make_lora_ss_hook(layer_name)
        layer = self.model.get_submodule(layer_name)
        hook = partial(self.hook, layer=layer, layer_name=layer_name)
        return hook
    

    def _infer_hook_type(self):           
        tp = self.config.type
        if tp == 'naive':
            hook_fn = softstairs_naive
        elif tp == 'standard':
            hook_fn = SoftStairsQuantizeFunction.apply
        elif tp == 'raw_gradient':
            hook_fn = SoftStairsRawGradientQuantizeFunction.apply
        elif tp == 'shifted':
            hook_fn = SoftStairsShiftedQuantizeFunction.apply
        else:
            raise ValueError('Unkown hook func type')
        self._hook_fn = hook_fn
        

    def hook(self, module, inputs, layer, layer_name):
        if not self._is_active:
            return
        for name_p, param in list(layer.named_parameters(recurse=False)):
            if not name_p.endswith(self._orig_suffix):
                continue
            full_name = layer_name + '.' + name_p
            if self.verbose:
                    print('@@@ FIRED', full_name)
            W_soft = self._hook_fn(
                        self.upscaled_parameter(full_name),
                        self.t,
                        self.config.normalized,
                        self.config.async_t_factor,
                        self._f,
            )
            if self.config.clamp_codes:
                base_name = self._base_parameter_name(full_name)
                integer_luft = 0.5
                W_soft = W_soft.clamp(self._q_min[base_name] - integer_luft, self._q_max[base_name] + integer_luft)
            W_soft = self.downscale_parameter(W_soft, full_name)
                
            setattr(module, name_p[:-len(self._orig_suffix)], W_soft)

    @torch.no_grad()
    def fake_quantize(self, model=None):
        """Sets all buffers to the quantized float version.

        Values are clamped to ``[q_min, q_max]`` before rounding, matching
        ``finalize``: anything beyond the limits lands on the boundary code.
        """
        model = model or self.model
        for name_p, param in list(model.named_parameters(recurse=True)):
                if not name_p.endswith(self._orig_suffix):
                    continue
                if self.verbose:
                        print('@@@ FIRED', name_p)
                
                W_int = self._hook_fn(
                        self.upscaled_parameter(name_p),
                        self.t,
                        self.config.normalized,
                        self.config.async_t_factor,
                        self._f,
                ).clamp(self._q_min[self._base_parameter_name(name_p)], self._q_max[self._base_parameter_name(name_p)]).round()
                W_soft = self.downscale_parameter(W_int, name_p)

                name_orig = name_p[:-len(self._orig_suffix)]
                module_path, attr_name = name_orig.rsplit('.', 1)
                module = model.get_submodule(module_path)
                setattr(module, attr_name, W_soft)
        self.deactivate_hooks()


    def upscaled_parameter(self, parameter_name, values=None):
        """Map weight values to normalized quantization coordinates.

        Args:
            parameter_name: Full name of a trainable ``*_orig`` parameter (or
                its base name without the suffix).
            values: Optional tensor of weight values to upscale. When omitted,
                the current values of the model parameter are used.

        Returns:
            ``values * scale + zero_point`` with the stored scale (which
            already includes the constant weight-scale factor, see
            :meth:`_compute_scale`): the scaled weights ``W'`` are what
            SoftStairs (and the deployed integer codes) see, and the stored
            affine map is their exact upscale. Every consumer of this method
            (training hook, ``fake_quantize``, error estimates, derivative
            evaluation for the SSA optimizers, level-transition tracking)
            therefore stays consistent with the scaling.
        """
        if values is None:
            param = self.model.get_parameter(parameter_name)
        else:
            param = values
        parameter_name = self._base_parameter_name(parameter_name)
        scale = self._scales[parameter_name]
        zero_point = self._zero_points[parameter_name]
        return param * scale + zero_point


    def downscale_parameter(self, param, parameter_name):
        parameter_name = self._base_parameter_name(parameter_name)
        scale = self._scales[parameter_name]
        zero_point = self._zero_points[parameter_name]
        return (1 / scale) * (param - zero_point)

    def downscale_delta(self, parameter_name: str, delta: torch.Tensor) -> torch.Tensor:
        """Map a code-space increment to weight space.

        Companion of :meth:`downscale_parameter` for increments: a step
        ``dz`` taken in normalized quantization coordinates corresponds to
        ``dz / scale`` in weight space with the stored scale (the zero point
        cancels for increments, and the constant weight-scale factor is
        already folded into the stored scale). The SoftStairs-aware
        optimizers use this to compute their updates on the upscaled
        parameter and downscale them back, so that one learning-rate unit is
        one quantization code regardless of the bit width or the weight
        range.
        """
        parameter_name = self._base_parameter_name(parameter_name)
        return delta / self._scales[parameter_name]

    def _base_parameter_name(self, parameter_name: str) -> str:
        if parameter_name.endswith(self._orig_suffix):
            return parameter_name[:-len(self._orig_suffix)]
        return parameter_name

    def is_quantized_parameter(self, parameter_name: str) -> bool:
        """Whether ``parameter_name`` refers to a SoftStairs-quantized weight."""
        return self._base_parameter_name(parameter_name) in self._scales

    @torch.no_grad()
    def softstairs_derivative(self, parameter_name: str, values: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Evaluate the SoftStairs derivative for a quantized weight.

        Uses the exact conventions of the backward pass: the weight values are
        mapped to normalized quantization coordinates via the stored scale and
        zero point, and the derivative is evaluated with the quantizer's
        current temperature and lattice-frequency settings.

        Args:
            parameter_name: Full name of a trainable ``*_orig`` parameter.
            values: Optional weight values at which to evaluate the derivative
                (e.g. intermediate parameters of a two-stage optimizer update).
                Defaults to the current parameter values.

        Returns:
            Element-wise SoftStairs derivative with the parameter's shape.
        """
        z = self.upscaled_parameter(parameter_name, values=values)
        soft = ScaledSoftStairs(
            t=self._t,
            normalized=self.config.normalized,
            async_t_factor=self.config.async_t_factor,
            scale=self._f,
        )
        return soft.derivative(z)

    @torch.no_grad()
    def estimate_quantization_errors(self) -> Dict[str, float]:
        """Split the mean per-element quantization error into rounding and clamping parts.

        The deployed integer code is ``round(clamp(W, q_min, q_max))`` in code
        space, matching ``fake_quantize``/``finalize``. The error ``W - code``
        decomposes exactly into two disjoint parts:

        * rounding error: ``clamp(W, q_min, q_max) - round(clamp(W, q_min, q_max))``,
          nonzero for in-range weights that have not settled on a code yet;
        * clamping error: ``W - clamp(W, q_min, q_max)``, nonzero only for
          weights beyond ``q_min``/``q_max`` that deployment saturates to a
          boundary code.

        Returns:
            ``rounding`` and ``clamping``: mean absolute errors (in code units);
            ``total``: their sum, the mean absolute distance to the deployed
            code; ``legacy``: the pre-split metric ``mean |W - round(W)|``
            (unclamped), kept under its own key so logs stay comparable across
            the split; ``directed``: signed mean of ``W - code``;
            ``out_of_range``: fraction of weights outside ``[q_min, q_max]``.
        """
        rounding = 0.
        clamping = 0.
        legacy = 0.
        directed = 0.
        out_of_range = 0.
        total_params = 0.
        for name_p, param in list(self.model.named_parameters(recurse=True)):
                if not name_p.endswith(self._orig_suffix):
                    continue

                W = self._hook_fn(
                    self.upscaled_parameter(name_p),
                    self.t,
                    self.config.normalized,
                    self.config.async_t_factor,
                    self._f,
                )
                base_name = self._base_parameter_name(name_p)
                q_min = self._q_min[base_name]
                q_max = self._q_max[base_name]

                W_clamped = W.clamp(q_min, q_max)
                code = W_clamped.round()
                clamp_part = W - W_clamped
                round_part = W_clamped - code

                rounding += round_part.abs().sum()
                clamping += clamp_part.abs().sum()
                legacy += (W - W.round()).abs().sum()
                directed += (clamp_part + round_part).sum()
                out_of_range += ((W < q_min) | (W > q_max)).sum()
                total_params += W.numel()

        denom = max(total_params, 1.)
        return {
            'rounding': float(rounding / denom),
            'clamping': float(clamping / denom),
            'total': float((rounding + clamping) / denom),
            'legacy': float(legacy / denom),
            'directed': float(directed / denom),
            'out_of_range': float(out_of_range / denom),
        }

    def estimate_current_quant_error(self, directed: bool = False):
        """Pre-split quantization error, for backward compatibility.

        With ``directed=False`` returns the legacy ``mean |W - round(W)|``
        (unclamped), i.e. exactly the metric logged before the
        rounding/clamping split; with ``directed=True`` the signed mean error
        ``W - round(clamp(W, q_min, q_max))``. Prefer
        :meth:`estimate_quantization_errors` for the split.
        """
        errors = self.estimate_quantization_errors()
        return errors['legacy'] if not directed else errors['directed']
    
    @property
    def current_backward_t(self):
        return min(1, self.get_current_t() * self.config.async_t_factor)


    def step(self, metric: Optional[float] = None) -> None:
        self.current_step += 1
        if self.config.recalc_scale_each_step:
            self._recalculate_scales()
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, AdaptiveScheduler):
            if metric is None:
                return  
            new_t = self.scheduler.observe(metric)
        else:
            new_t = self.scheduler.get_t(self.current_step)
        if abs(new_t - self._t) > R_CHANGE_THRESHOLD:
            self._t = new_t
        if self.frequency_scheduler is not None:
            self._f = self.frequency_scheduler.get_f(self._t)

    def get_current_t(self) -> float:
        return self._t

    @property
    def max_derivative_value(self) -> float:
        """Peak of the current SoftStairs derivative (1 for the normalized
        surrogate, ``(2 - t) / t`` otherwise), evaluated with the quantizer's
        current temperature, async-t factor, and lattice frequency. The SSA
        optimizers use it to scale the free Adam stage so it matches the
        strongest gated step."""
        soft = ScaledSoftStairs(
            t=self._t,
            normalized=self.config.normalized,
            async_t_factor=self.config.async_t_factor,
            scale=self._f,
        )
        return soft.max_derivative_value()

    def get_current_weight_scale(self) -> float:
        """Constant weight-scale factor (``QuantizationConfig.weight_scale``)."""
        return self._ws

    def get_t_schedule(self) -> Optional[List[float]]:
        if self.scheduler is None:
            return None
        return self.scheduler.get_all_t()
    
    def get_f_schedule(self) -> Optional[List[float]]:
        """Frequencies along the planned t schedule (None without a t scheduler)."""
        if self.frequency_scheduler is None:
            return None
        ts = self.get_t_schedule()
        if ts is None:
            return [self.frequency_scheduler.get_f(self._t)]
        return self.frequency_scheduler.get_all_f(ts)
    
    @torch.no_grad()
    def finalize(self) -> nn.Module:
        """
        Converts model weights to integer codes with clamping and rounding.

        Codes are recomputed from the current ``*_orig`` parameters with the
        exact ``fake_quantize`` math — ``round(clamp(S_{t,f}(z)))`` in code
        space at the final temperature/frequency — so ``finalize`` is
        order-independent w.r.t. ``fake_quantize`` and never depends on the
        stale hook buffers. The codes are strictly ``n_bits``-wide (range
        ``[q_min, q_max]``); when this torch build can materialize
        ``int{n_bits}`` tensors they are cast to that dtype, otherwise they stay
        in float storage with a warning (same codes, FakeQuantize-style
        simulation). The registered float parameters hold the dequantized
        values ``(code - zero_point) / scale``, identical to the
        ``fake_quantize`` buffers.

        For LoRA mode:
        - Applies SoftStairs to frozen weights with FINAL r (from config)
        - Uses already quantized adapters (quantized each forward during training)
        - Combines them and converts to int
        Removes auxiliary buffers properly using register_parameter(name, None).
        """
        n_bits = self.config.n_bits
        dtype = getattr(torch, f'int{n_bits}')

        if self.config.t_scheduler_strategy != "constant":
            final_t = self.config.t_end
        else:
            final_t = self.config.t_start
        final_f = self.frequency_scheduler.get_f(final_t) if self.frequency_scheduler is not None else 1.

        is_lora = getattr(self, '_is_lora', False)

        for name, module in self.model.named_modules():
            orig_params = []
            for name_p, param in list(module.named_parameters(recurse=False)):
                full_name = f'{name}.{name_p}'
                if name_p.endswith(self._orig_suffix) and self._base_parameter_name(full_name) in self._scales:
                    orig_params.append((name_p, param, full_name))

            if not orig_params:
                continue

            deployed = []
            for name_p, param, full_name in orig_params:
                base_name = self._base_parameter_name(full_name)
                orig_name = name_p[:-5]

                W_code = self._hook_fn(
                    self.upscaled_parameter(full_name, values=param),
                    final_t,
                    self.config.normalized,
                    self.config.async_t_factor,
                    final_f,
                )

                if is_lora and hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    adapter_name = getattr(self, '_adapter_name', 'default')
                    a = module.lora_A[adapter_name].weight
                    b = module.lora_B[adapter_name].weight
                    W_code = W_code + torch.matmul(b, a)

                W_int = W_code.clamp(self._q_min[base_name], self._q_max[base_name]).round()
                try:
                    # Strict int{n_bits} storage where ATen supports the cast
                    # (2-bit/4-bit int dtypes have no copy_ kernel in some builds).
                    W_int = W_int.to(dtype)
                except (NotImplementedError, RuntimeError):
                    warnings.warn(
                        f'finalize: torch cannot materialize {dtype} tensors in this '
                        f'build; keeping the exact {n_bits}-bit codes in float storage '
                        f'(same values, range [{self._q_min[base_name]}, '
                        f'{self._q_max[base_name]}]).',
                        RuntimeWarning,
                        stacklevel=2,
                    )
                # The deployed float parameter holds dequantized values (the exact
                # inverse affine map of the stored integer codes), matching the
                # fake_quantize buffers so both paths produce the same model.
                W_deploy = self.downscale_parameter(W_int.float(), base_name)
                deployed.append((orig_name, W_deploy))

            # Strip the aux parameters and hook buffers BEFORE registering the
            # deployed parameters: ``nn.Module.__delattr__`` resolves parameter
            # names first, so removing the 'weight' buffer after registering the
            # 'weight' parameter would delete the parameter itself.
            for name_p in list(module._parameters.keys()):
                if name_p.endswith(self._orig_suffix):
                    module.register_parameter(name_p, None)
                    if hasattr(module, name_p):
                        delattr(module, name_p)

            for buffer_name in list(module._buffers.keys()):
                module.register_buffer(buffer_name, None)
                if hasattr(module, buffer_name):
                    delattr(module, buffer_name)

            for orig_name, W_deploy in deployed:
                module.register_parameter(orig_name, nn.Parameter(W_deploy.float()))

            for attr_name in ['_ss_a_param', '_ss_b_param', 'lora_A_quant', 'lora_B_quant']:
                if hasattr(module, attr_name):
                    delattr(module, attr_name)

            if hasattr(module, '_forward_pre_hooks'):
                module._forward_pre_hooks.clear()

        return self.model