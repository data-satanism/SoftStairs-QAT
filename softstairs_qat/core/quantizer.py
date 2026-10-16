import torch
import torch.nn as nn
from typing import Optional, Dict, List, Set
from peft import PeftModel

from softstairs_qat.core.soft_stairs import SoftStairs, softstairs_naive, SoftStairsShifted
from softstairs_qat.core.variance_controller import VarianceController
from softstairs_qat.core.quantization_params import QuantizationParamsCalculator
from softstairs_qat.wrappers.config import QuantizationConfig
from softstairs_qat.utils.t_scheduler import TScheduler, AdaptiveScheduler


EPSILON = 1e-6
R_CHANGE_THRESHOLD = 1e-12
VARIANCE_CONSTRAINT_INTERVAL = 10

from functools import partial 


class SoftStairsQuantizeFunction(torch.autograd.Function):
    """
    Unified quantization function with optional low-rank adapters.
    Accepts:
        weight:          full weight matrix W
        r:               SoftStairs sharpness parameter
        modified:        use modified SoftStairs?
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, t: float, normalized: bool = True, async_t_factor: float = 1.) -> torch.Tensor:
        soft = SoftStairs(t=t, normalized=normalized, async_t_factor=async_t_factor)
        x_soft = soft.forward(x)
        ctx.save_for_backward(x)
        ctx.t = t
        ctx.modified = normalized
        ctx.async_t_factor = async_t_factor
        return x_soft
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        soft = SoftStairs(t=ctx.t, normalized=ctx.modified, async_t_factor=ctx.async_t_factor)
        grad = grad_output * soft.derivative(x)
        return grad, None, None, None


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
    def forward(ctx, x: torch.Tensor, t: float, normalized: bool = True, async_t_factor: float = 1.) -> torch.Tensor:
        soft = SoftStairs(t=t, normalized=normalized, async_t_factor=async_t_factor)
        return soft.forward(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None, None, None
    

class SoftStairsShiftedQuantizeFunction(torch.autograd.Function):
    """
    Unified quantization function with optional low-rank adapters.
    Accepts:
        weight:          full weight matrix W
        r:               SoftStairs sharpness parameter
        modified:        use modified SoftStairs?
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, t: float, normalized: bool = True, scale: torch.Tensor = None) -> torch.Tensor:
        soft = SoftStairsShifted(t=t, normalized=normalized)
        x_soft = soft.forward(x)
        ctx.save_for_backward(x)
        ctx.t = t
        ctx.modified = normalized
        ctx.scale = scale
        return x_soft
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        soft = SoftStairs(t=ctx.t, normalized=ctx.modified)
        grad = grad_output * soft.derivative(x)
        # alpha = ctx.scale ** 2
        # grad = grad / alpha
        return grad, None, None, None
    

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

    def _should_quantize_module(self, module_name: str) -> bool:
        return module_name not in self.excluded_modules
    

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

                    if self.per_row and parameter.dim() >= 2:
                        scales, zps = [], []
                        q_min = q_max = None
                        for i in range(parameter.shape[0]):
                            p = self._calc.compute(
                                parameter[i],
                                n_bits=self.config.n_bits,
                                symmetric=self.config.symmetric,
                            )
                            scales.append(p.scale.reshape(()))
                            zps.append(p.zero_point.reshape(()))
                            q_min, q_max = p.q_min, p.q_max

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
                        q_min, q_max = p.q_min, p.q_max

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
                

    def _register_hooks(self):
        """Registers two separate forward pre-hooks per quantized layer:
        1) SoftStairs (strategy via _make_pre_hook / is_lora) — no scaling
        2) input scale (init-time scale) — no SoftStairs
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
            )
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
            ``values * scale + zero_point + half_shift / 2``.
        """
        additional_shift = 0.5 * self.config.half_shift 
        if values is None:
            param = self.model.get_parameter(parameter_name)
        else:
            param = values
        if parameter_name.endswith(self._orig_suffix):
            parameter_name = parameter_name[:-len(self._orig_suffix)]
        # if self.config.adaprive_scaling:
        #     scale = 
        scale = self._scales[parameter_name]
        zero_point = self._zero_points[parameter_name]
        return param * scale + zero_point + additional_shift
    

    def downscale_parameter(self, param, parameter_name):
        additional_shift = 0.5 * self.config.half_shift 
        if parameter_name.endswith(self._orig_suffix):
            parameter_name = parameter_name[:-len(self._orig_suffix)]
        scale = self._scales[parameter_name]
        zero_point = self._zero_points[parameter_name]
        return (1 / scale) * (param - zero_point - additional_shift)

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
        mapped to normalized quantization coordinates via the stored scale,
        zero point, and half shift, and ``SoftStairs.derivative`` is evaluated
        with the quantizer's current temperature settings.

        Args:
            parameter_name: Full name of a trainable ``*_orig`` parameter.
            values: Optional weight values at which to evaluate the derivative
                (e.g. intermediate parameters of a two-stage optimizer update).
                Defaults to the current parameter values.

        Returns:
            Element-wise SoftStairs derivative with the parameter's shape.
        """
        z = self.upscaled_parameter(parameter_name, values=values)
        soft = SoftStairs(
            t=self._t,
            normalized=self.config.normalized,
            async_t_factor=self.config.async_t_factor,
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

    def get_current_t(self) -> float:
        return self._t

    def get_t_schedule(self) -> Optional[List[float]]:
        if self.scheduler is None:
            return None
        return self.scheduler.get_all_t()
    
    def finalize(self) -> nn.Module:
        """
        Converts model weights to integer type with clamping.
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
        
        for name, module in self.model.named_modules():
            if name not in self._scales:
                continue
            
            orig_params = []
            for name_p, param in list(module.named_parameters(recurse=False)):
                if name_p.endswith(self._orig_suffix):
                    orig_params.append((name_p, param))
            
            if not orig_params:
                continue
            
            for name_p, param in orig_params:
                orig_name = name_p[:-5] 
                
                if self._is_lora:
                    weight_soft = self._hook_fn(
                        param,
                        final_t,
                        self.config.normalized,
                    )
                    W_code = weight_soft
                    
                    if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                        adapter_name = self._adapter_name
                        a = module.lora_A[adapter_name].weight
                        b = module.lora_B[adapter_name].weight
                        W_code = W_code + torch.matmul(b, a)
                    
                    W_int = W_code.clamp(self._q_min[name], self._q_max[name])
                else:
                    if hasattr(module, orig_name):
                        W_current = getattr(module, orig_name)
                    else:
                        continue
                    
                    W_int = W_current.clamp(self._q_min[name], self._q_max[name])
                
                W_int = W_int.to(dtype)
                
                if hasattr(module, orig_name):
                    delattr(module, orig_name)
                
                module.register_parameter(orig_name, nn.Parameter(W_int.float()))
            
            for name_p in list(module._parameters.keys()):
                if name_p.endswith(self._orig_suffix):
                    module.register_parameter(name_p, None)
                    if hasattr(module, name_p):
                        delattr(module, name_p)
            
            for buffer_name in list(module._buffers.keys()):
                module.register_buffer(buffer_name, None)
                if hasattr(module, buffer_name):
                    delattr(module, buffer_name)
            
            for attr_name in ['_ss_a_param', '_ss_b_param', 'lora_A_quant', 'lora_B_quant']:
                if hasattr(module, attr_name):
                    delattr(module, attr_name)
            
            if hasattr(module, '_forward_pre_hooks'):
                module._forward_pre_hooks.clear()
        
        return self.model