import torch
import torch.nn as nn
from typing import Optional, Dict, List, Set
from peft import PeftModel

from softstairs_qat.core.soft_stairs import SoftStairs
from softstairs_qat.core.variance_controller import VarianceController
from softstairs_qat.core.quantization_params import QuantizationParamsCalculator
from softstairs_qat.wrappers.config import QuantizationConfig
from softstairs_qat.utils.r_scheduler import TScheduler
from softstairs_qat.core.soft_stairs import softstairs_naive

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
    def forward(ctx, x: torch.Tensor, t: float, normalized: bool = True, scale: torch.Tensor = None) -> torch.Tensor:
        soft = SoftStairs(t=t, normalized=normalized)
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
    

def quantize_soft_stairs(
    x: torch.Tensor,
    t: float,
    normalized: bool = False,
    scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return SoftStairsQuantizeFunction.apply(x, t, normalized, scale)


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
    ):
        if getattr(model, self._check_field, False):
            raise RuntimeError('Attempting to double wrap model with SSQuant')
        
        self._is_active = True 
        self.model = model
        self.config = config
        self.verbose=verbose
        # self.total_steps = config.t_step
        self.current_step = 0

        self.excluded_modules = excluded_modules or set()

        self.scheduler: Optional[TScheduler] = None
        if config.t_scheduler_strategy != "constant":
            self.scheduler = TScheduler.from_config(config, config.t_step)
            current_t = config.t_start
        else:
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
        self._hook_handles = []

        self._calc = QuantizationParamsCalculator()
        self._init_quantization()

        self._register_hooks()
        setattr(self.model, self._check_field, True)

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
                    params = self._calc.compute(
                        parameter,
                        self.config.n_bits,
                        symmetric=self.config.symmetric,
                    )
                    full_name = f'{name_m}.{name_p}'
                    if self.verbose:
                        print('@@@ INIT', full_name)
                    self._scales[full_name] = params.scale
                    self._zero_points[full_name] = params.zero_point
                    self._q_min[full_name] = params.q_min
                    self._q_max[full_name] = params.q_max
                    
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
        return self._make_standard_ss_hook(layer_name)
    

    def _make_standard_ss_hook(self, layer_name: str):           
        layer = self.model.get_submodule(layer_name)
        if self.config.naive:
            hook = partial(self.naive_hook, layer=layer, layer_name=layer_name)
        else:
            hook = partial(self.hook, layer=layer, layer_name=layer_name)
    
        return hook
    
    def naive_hook(self, module, inputs, layer, layer_name):
        for name_p, param in list(layer.named_parameters(recurse=False)):
                if not name_p.endswith(self._orig_suffix):
                    continue
                full_name = layer_name + '.' + name_p
                if self.verbose:
                        print('@@@ FIRED', full_name)
                W_soft = softstairs_naive(
                        self.upscaled_parameter(full_name),
                        self.t,
                )
                W_soft = self.downscale_parameter(W_soft, full_name)
                
                setattr(module, name_p[:-len(self._orig_suffix)], W_soft)
    
    def hook(self, module, inputs, layer, layer_name):
        for name_p, param in list(layer.named_parameters(recurse=False)):
            if not name_p.endswith(self._orig_suffix):
                continue
            if self.verbose:
                    print('@@@ FIRED', full_name)
            full_name = layer_name + '.' + name_p
            W_soft = quantize_soft_stairs(
                        self.upscaled_parameter(full_name),
                        self.t,
                        normalized=self.config.normalized,
            )
            W_soft = self.downscale_parameter(W_soft, full_name)
                
            setattr(module, name_p[:-len(self._orig_suffix)], W_soft)


    def fake_quantize(self, model=None):
        model = model or self.model
        for name_p, param in list(model.named_parameters(recurse=True)):
                if not name_p.endswith(self._orig_suffix):
                    continue
                if self.verbose:
                        print('@@@ FIRED', name_p)
                
                W_int = quantize_soft_stairs(
                        self.upscaled_parameter(name_p),
                        self.t,
                        normalized=self.config.normalized,
                ).round()
                W_soft = self.downscale_parameter(W_int, name_p)

                attrs = name_p[:-len(self._orig_suffix)].split('.')
                module = model
                for i in attrs[:-1]:
                    module = getattr(module, i)
                setattr(module, attrs[-1], W_soft)

    def upscaled_parameter(self, parameter_name):
            param = self.model.get_parameter(parameter_name)
            if parameter_name.endswith(self._orig_suffix):
                parameter_name = parameter_name[:-len(self._orig_suffix)]
            scale = self._scales[parameter_name]
            zero_point = self._zero_points[parameter_name]
            return (param - zero_point) * scale
    
    def downscale_parameter(self, param, parameter_name):
        if parameter_name.endswith(self._orig_suffix):
            parameter_name = parameter_name[:-len(self._orig_suffix)]
        scale = self._scales[parameter_name]
        zero_point = self._zero_points[parameter_name]
        return (1 / scale) * param + zero_point

    @torch.no_grad()
    def estimate_current_quant_error(self):
        error = 0.
        total_params = 0. 
        for name_p, param in list(self.model.named_parameters(recurse=True)):
                if not name_p.endswith(self._orig_suffix):
                    continue

                W = (softstairs_naive(
                        self.upscaled_parameter(name_p),
                        self.t,
                ))
                error += torch.abs(W - torch.round(W)).sum()
                total_params += W.numel() 
        return error / total_params


    def step(self):
        """Called after every `optimizer.step()` to update `r` and manage the adapters."""
        self.current_step += 1


        if self.scheduler is not None:
            new_t = self.scheduler.get_t(self.current_step)
            if abs(new_t - self._t) > R_CHANGE_THRESHOLD:
                self._t = new_t

        # if self._is_lora and (self.current_step % VARIANCE_CONSTRAINT_INTERVAL == 0):
        #     for name in self._scales.keys():
        #         self._apply_variance_constraint(name)

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
                    weight_soft = quantize_soft_stairs(
                        param,
                        final_t,
                        normalized=self.config.normalized,
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