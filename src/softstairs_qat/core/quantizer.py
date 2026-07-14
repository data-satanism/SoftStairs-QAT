import torch
import torch.nn as nn
from typing import Optional, Dict, List
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
    ):
        self.model = model
        self.config = config
        self.total_steps = total_steps or 0
        self.current_step = 0

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
        self._rank = config.rank
        self._target_modules = config.target_modules or (nn.Linear,)

        self._variance_controller = VarianceController(safety_factor=config.safety_factor) if self._is_lora else None
        self._sigma_A: Dict[str, float] = {}
        self._sigma_B: Dict[str, float] = {}
        self._layer_scales: Dict[str, float] = {}

        calc = QuantizationParamsCalculator()
        self._init_quantization()

        self._register_hooks()

    def _init_quantization(self):
        """
        Initializes quantization structure.
        For standard mode: renames weight to weight_orig and creates new weight parameter.
        For LoRA mode: freezes weights, initializes adapters via VarianceController.
        """
        with torch.no_grad():
            for name_m, module in self.model.named_modules():
                if not isinstance(module, self._target_modules):
                    continue
                
                for name_p, parameter in list(module.named_parameters()):
                    if name_p == 'weight':
                        module.register_parameter('weight_orig', parameter)
                        module.register_parameter('weight', None)
                        break
                
                if self._is_lora:
                    state = self._variance_controller.initialize_adapters(
                        module.weight_orig,
                        self._rank,
                        self.config.n_bits,
                        symmetric=self.config.symmetric,
                    )
                    
                    module.register_parameter('lora_A', nn.Parameter(state.adapter_a))
                    module.register_parameter('lora_B', nn.Parameter(state.adapter_b))
                    
                    self._sigma_A[name_m] = state.sigma
                    self._sigma_B[name_m] = state.sigma_b_max
                    self._layer_scales[name_m] = 1.0
                    self._params_cache[name_m] = state.params
                    
                    self._apply_variance_constraint(name_m)
                    
                    module.weight_orig.requires_grad_(False)
                    
                    self._scales[name_m] = state.params.scale
                    self._zero_points[name_m] = state.params.zero_point
                    self._q_min[name_m] = state.params.q_min
                    self._q_max[name_m] = state.params.q_max
                    
                elif hasattr(module, 'weight_orig'):
                    module.weight = nn.Parameter(module.weight_orig.data)

    def _apply_variance_constraint(self, name: str):
        """
        Constrains the variance of layer adapters to permissible values.
        Updates scaling factor if adapters were rescaled.
        """
        module = self._get_module_by_name(name)
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            adapter_a, adapter_b, scaling = self._variance_controller.constrain_adapters(
                module.lora_A.data,
                module.lora_B.data,
                self._sigma_A.get(name, EPSILON),
                self._sigma_B.get(name, EPSILON),
            )
            
            if scaling != 1.0:
                module.lora_A.data.copy_(adapter_a)
                module.lora_B.data.copy_(adapter_b)
                self._layer_scales[name] *= scaling
                

    def _get_module_by_name(self, name: str) -> nn.Module:
        return self.model.get_submodule(name)

    def _register_hooks(self):
        """Registers forward pre-hook for each quantized layer."""
        for name, module in self.model.named_modules():
            if name in self._scales or (self._is_lora and hasattr(module, 'lora_A')):
                module.register_forward_pre_hook(self._make_pre_hook(name))

    def _make_pre_hook(self, layer_name: str):
        """
        Creates a pre-hook that selects the appropriate quantization logic
        based on is_lora mode.
        """
        if self._is_lora:
            return self._make_lora_hook(layer_name)
        else:
            return self._make_standard_hook(layer_name)

    def _make_lora_hook(self, layer_name: str):
        """
        Creates a pre-hook for LoRA mode.
        Each forward pass:
          1. Applies SoftStairs to lora_A and lora_B separately
          2. Computes quantized adapter contribution
          3. Adds to quantized base weight for the forward pass
          (Note: weight is recreated each forward pass, not stored)
        """
        def hook(module, input):
            lora_A_soft = quantize_soft_stairs(
                module.lora_A,
                self._r,
                modified=self.config.modified,
            )
            
            lora_B_soft = quantize_soft_stairs(
                module.lora_B,
                self._r,
                modified=self.config.modified,
            )
            
            lora_A_soft = lora_A_soft * self._layer_scales.get(layer_name, 1.0)
            
            module.lora_A_quant = lora_A_soft
            module.lora_B_quant = lora_B_soft
            
        return hook
    
    def _make_standard_hook(self, layer_name: str):
        """
        Creates a pre-hook for standard mode (no LoRA).
        Each forward pass:
          1. Computes quantization parameters
          2. Applies SoftStairs to the entire weight
        """
        def hook(module, input):
            weight = module.weight_orig
            
            params = self._calc.compute(
                weight, 
                self.config.n_bits, 
                symmetric=self.config.symmetric
            )
            
            self._scales[layer_name] = params.scale
            self._zero_points[layer_name] = params.zero_point
            self._q_min[layer_name] = params.q_min
            self._q_max[layer_name] = params.q_max
            
            weight_norm = (weight / params.scale) + params.zero_point
            
            weight_soft = quantize_soft_stairs(
                weight_norm,
                self._r,
                modified=self.config.modified,
            )
            
            weight_quant = (weight_soft - params.zero_point) * params.scale
            
            module.weight.data.copy_(weight_quant)
            
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
                weight_orig = module.weight_orig
                
                weight_norm = (weight_orig / self._scales[name]) + self._zero_points[name]
                weight_soft = quantize_soft_stairs(
                    weight_norm,
                    final_r,
                    modified=self.config.modified,
                )
                weight_base_quant = (weight_soft - self._zero_points[name]) * self._scales[name]
                
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    adapters = torch.matmul(module.lora_A, module.lora_B)
                    adapters = adapters * self._layer_scales.get(name, 1.0)
                    W_final = weight_base_quant + adapters
                else:
                    W_final = weight_base_quant
            else:
                W_final = module.weight_orig
            
            W_scaled = W_final / self._scales[name] + self._zero_points[name]
            W_scaled.clamp_(self._q_min[name], self._q_max[name])
            W_int = W_scaled.to(dtype)
            
            module.register_parameter('weight', None)
            module.register_parameter('weight', nn.Parameter(W_int))
            
            param_names_to_unregister = []
            for param_name in list(module._parameters.keys()):
                if param_name != 'weight':
                    param_names_to_unregister.append(param_name)
            
            for param_name in param_names_to_unregister:
                module.register_parameter(param_name, None)
            
            if hasattr(module, 'lora_A_quant'):
                delattr(module, 'lora_A_quant')
            if hasattr(module, 'lora_B_quant'):
                delattr(module, 'lora_B_quant')
        
        return self.model