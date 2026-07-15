import torch
import torch.nn as nn
from typing import Optional, Dict, List, Set
from peft import PeftModel

from softstairs_qat.core.soft_stairs import SoftStairs
from softstairs_qat.core.variance_controller import VarianceController
from softstairs_qat.core.quantization_params import QuantizationParamsCalculator
from softstairs_qat.wrappers.config import QuantizationConfig
from softstairs_qat.utils.r_scheduler import RScheduler

EPSILON = 1e-6
R_CHANGE_THRESHOLD = 1e-10
VARIANCE_CONSTRAINT_INTERVAL = 10


class SoftStairsQuantizeFunction(torch.autograd.Function):
    """
    Unified quantization function with optional low-rank adapters.
    Accepts:
        weight:          full weight matrix W
        r:               SoftStairs sharpness parameter
        modified:        use modified SoftStairs?
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, r: float, modified: bool = False) -> torch.Tensor:
        soft = SoftStairs(r=r, modified=modified)
        x_soft = soft.forward(x)
        ctx.save_for_backward(x)
        ctx.r = r
        ctx.modified = modified
        return x_soft
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        soft = SoftStairs(r=ctx.r, modified=ctx.modified)
        return grad_output * soft.derivative(x), None, None
    

def quantize_soft_stairs(
    x: torch.Tensor,
    r: float,
    modified: bool = False,
) -> torch.Tensor:
    return SoftStairsQuantizeFunction.apply(x, r, modified)


class SoftStairsQuantizer:
    """
    Manages quantization via SoftStairs.
    Supports:
      - is_lora=False: all model weights are trained; SoftStairs is applied to them during each forward pass.
      - is_lora=True: model weights are frozen; only adapters (LoRA) are trained.
                       During each forward pass, `quantized = soft_stairs(W_orig + A@B)` is calculated.
    Controls adapter variance via VarianceController.
    """
    def __init__(
        self,
        model: nn.Module,
        config: QuantizationConfig,
        total_steps: Optional[int] = None,
        excluded_modules: Optional[Set[str]] = None,
    ):
        self.model = model
        self.config = config
        self.total_steps = total_steps or 0
        self.current_step = 0

        self.excluded_modules = excluded_modules or set()

        self.scheduler: Optional[RScheduler] = None
        if config.r_scheduler_strategy != "constant":
            self.scheduler = RScheduler.from_config(config, total_steps)
            current_r = config.r_start
        else:
            current_r = config.r

        self._r = current_r 

        self._scales: Dict[str, torch.Tensor] = {}
        self._zero_points: Dict[str, torch.Tensor] = {}
        self._q_min: Dict[str, int] = {}
        self._q_max: Dict[str, int] = {}

        self._is_lora = config.is_lora
        self._adapter_name = getattr(config, "adapter_name", "default")
        self._rank = config.rank

        self._variance_controller = VarianceController(safety_factor=config.safety_factor) if self._is_lora else None
        self._sigma_A: Dict[str, float] = {}
        self._sigma_B: Dict[str, float] = {}
        self._hook_handles = []

        self._calc = QuantizationParamsCalculator()
        self._init_quantization()

        self._register_hooks()

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
                if self._is_lora:
                    if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                        continue

                    state = self._variance_controller.initialize_adapters(
                        module.weight.data,
                        self._rank,
                        self.config.n_bits,
                        symmetric=self.config.symmetric,
                    )
                    params = state.params

                    self._scales[name_m] = params.scale
                    self._zero_points[name_m] = params.zero_point
                    self._q_min[name_m] = params.q_min
                    self._q_max[name_m] = params.q_max

                    weight_norm = module.weight.data * params.scale + params.zero_point

                    module.register_parameter("weight_orig", nn.Parameter(weight_norm))
                    module.weight_orig.requires_grad_(False)
                    module.register_parameter("weight", None) 
                    if hasattr(module, "weight"):
                        delattr(module, "weight")

                    module.register_buffer("weight", weight_norm)

                    a = module.lora_A[self._adapter_name].weight
                    b = module.lora_B[self._adapter_name].weight
                    a.data = state.adapter_a.to(device=a.device, dtype=a.dtype)
                    b.data = state.adapter_b.to(device=b.device, dtype=b.dtype)
                    module._ss_a_param = a
                    module._ss_b_param = b
                    a_mod = module.lora_A[self._adapter_name]
                    b_mod = module.lora_B[self._adapter_name]
                    a_mod.register_parameter("weight", None)
                    if hasattr(a_mod, "weight"):
                        delattr(a_mod, "weight")
                    a_mod.register_buffer("weight", a) 
                    b_mod.register_parameter("weight", None)
                    if hasattr(b_mod, "weight"):
                        delattr(b_mod, "weight")
                    b_mod.register_buffer("weight", b)
                    

                else:
                    weight = getattr(module, "weight", None)
                    if not isinstance(weight, nn.Parameter) or weight.ndim < 2:
                        continue

                    params = self._calc.compute(
                        weight.data,
                        self.config.n_bits,
                        symmetric=self.config.symmetric,
                    )

                    self._scales[name_m] = params.scale
                    self._zero_points[name_m] = params.zero_point
                    self._q_min[name_m] = params.q_min
                    self._q_max[name_m] = params.q_max
                    
                    weight_norm = weight.data / params.scale + params.zero_point
                    module.register_parameter("weight_orig", nn.Parameter(weight_norm))
                    module.register_parameter("weight", None)  
                    if hasattr(module, "weight"):
                        delattr(module, "weight")

                    module.register_buffer("weight", weight_norm)

                

    def _apply_variance_constraint(self, name: str):
        """
        Constrains the variance of layer adapters to permissible values.
        Uses permanent PEFT Parameter refs when present.
        """
        module = self._get_module_by_name(name)
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            a = getattr(module, "_ss_a_param", module.lora_A[self._adapter_name].weight)
            b = getattr(module, "_ss_b_param", module.lora_B[self._adapter_name].weight)
            self._variance_controller.constrain_adapters(
                a.data,
                b.data,
                self._sigma_A.get(name, EPSILON),
                self._sigma_B.get(name, EPSILON),
            )
                

    def _get_module_by_name(self, name: str) -> nn.Module:
        return self.model.get_submodule(name)

    def _register_hooks(self):
        """Registers two separate forward pre-hooks per quantized layer:
        1) SoftStairs (strategy via _make_pre_hook / is_lora) — no scaling
        2) input scale (init-time scale) — no SoftStairs
        """
        for name, module in self.model.named_modules():
            if name not in self._scales:
                continue

            self._hook_handles.append(module.register_forward_pre_hook(self._make_input_scale_hook(name)))

            self._hook_handles.append(module.register_forward_pre_hook(self._make_pre_hook(name)))


    def _make_pre_hook(self, layer_name: str):
        """
        Creates SoftStairs pre-hook; selects LoRA vs standard by is_lora.
        No input/weight scaling here.
        """
        if self._is_lora:
            return self._make_lora_ss_hook(layer_name)
        return self._make_standard_ss_hook(layer_name)
    

    def _make_standard_ss_hook(self, layer_name: str):
        def hook(module, inputs):
            module.weight = quantize_soft_stairs(
                module.weight_orig,
                self._r,
                modified=self.config.modified,
            ) 
            return inputs
        return hook
    

    def _make_lora_ss_hook(self, layer_name: str):
        """
        SoftStairs on PEFT adapters.
        Always reads trainable Parameters from _ss_a_param / _ss_b_param
        (saved at init), then assigns soft tensors into the same fields
        PeftModel.forward already uses — no restore post-hook.
        """
        adapter_name = self._adapter_name
        def hook(module, inputs):
            a_mod = module.lora_A[adapter_name]
            b_mod = module.lora_B[adapter_name]
            a_mod.weight = quantize_soft_stairs(
                module._ss_a_param, self._r, modified=self.config.modified
            )
            b_mod.weight = quantize_soft_stairs(
                module._ss_b_param, self._r, modified=self.config.modified
            )
            return inputs
        return hook
    

    def _make_input_scale_hook(self, layer_name: str):
        scale = self._scales[layer_name]
        zero_point = self._zero_points[layer_name]
        def hook(module, inputs):
            x = inputs[0]
            x_scaled = x * scale + zero_point 
            return (x_scaled,)
        return hook

    def step(self):
        """Called after every `optimizer.step()` to update `r` and manage the adapters."""
        self.current_step += 1

        if self.scheduler is not None:
            new_r = self.scheduler.get_r(self.current_step)
            if abs(new_r - self._r) > R_CHANGE_THRESHOLD:
                self._r = new_r

        if self._is_lora and (self.current_step % VARIANCE_CONSTRAINT_INTERVAL == 0):
            for name in self._scales.keys():
                self._apply_variance_constraint(name)

    def get_current_r(self) -> float:
        return self._r

    def get_r_schedule(self) -> Optional[List[float]]:
        if self.scheduler is None:
            return None
        return self.scheduler.get_all_r()
    
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
        
        if self.config.r_scheduler_strategy != "constant":
            final_r = self.config.r_end
        else:
            final_r = self.config.r
        
        for name, module in self.model.named_modules():
            if name not in self._scales:
                continue
            
            if not hasattr(module, 'weight_orig'):
                continue
            
            if self._is_lora:
                weight_soft = quantize_soft_stairs(
                    module.weight_orig,
                    final_r,
                    modified=self.config.modified,
                )
                W_code = weight_soft
                
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    a = getattr(module, "_ss_a_param", module.lora_A[self._adapter_name].weight)
                    b = getattr(module, "_ss_b_param", module.lora_B[self._adapter_name].weight)
                    W_code = W_code + torch.matmul(b, a)
                
                W_int = W_code.clamp(self._q_min[name], self._q_max[name])
            else:
                if hasattr(module, 'weight'):
                    W_current = module.weight
                else:
                    continue
                
                W_int = W_current.clamp(self._q_min[name], self._q_max[name])
            
            W_int = W_int.to(dtype)
            
            for param_name in list(module._parameters.keys()):
                module.register_parameter(param_name, None)
            
            for buffer_name in list(module._buffers.keys()):
                module.register_buffer(buffer_name, None)

            if hasattr(module, 'weight'):
                delattr(module, 'weight')
            
            module.register_parameter('weight', nn.Parameter(W_int.float()))
            
            for attr_name in ['weight_orig', '_ss_a_param', '_ss_b_param', 'lora_A_quant', 'lora_B_quant']:
                if hasattr(module, attr_name):
                    delattr(module, attr_name)
            
            if hasattr(module, '_forward_pre_hooks'):
                module._forward_pre_hooks.clear()
        
        return self.model