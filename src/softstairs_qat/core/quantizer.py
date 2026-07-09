import torch
import torch.nn as nn
from typing import Optional, Dict, List

from softstairs_qat.core.soft_stairs import SoftStairs
from softstairs_qat.core.variance_controller import VarianceController
from softstairs_qat.core.quantization_params import QuantizationParamsCalculator
from softstairs_qat.wrappers.config import QuantizationConfig
from softstairs_qat.utils.r_scheduler import RScheduler


class SoftStairsQuantizeFunction(torch.autograd.Function):
    """
    Unified quantization function with optional low-rank adapters.
    Accepts:
        weight:          full weight matrix W
        A, B:            low-rank matrices (can be None)
        r:               SoftStairs sharpness parameter
        scale, zero_point, q_min, q_max: precomputed quantization parameters
        sigma_B_max:     maximum allowed sigma_B (for variance control)
        modified:        use modified SoftStairs?
    """
    @staticmethod
    def forward(ctx, weight, A, B, r, scale, zero_point, q_min, q_max, sigma_B_max, modified=False, symmetric=False):
        if A is not None and B is not None:
            sigma_B = torch.std(B).item()
            if sigma_B > sigma_B_max:
                B.mul_(sigma_B_max / sigma_B)
            S = weight + torch.matmul(A, B)
        else:
            S = weight.clone()

        S.div_(scale)
        S.add_(zero_point)
        S_norm = S

        soft = SoftStairs(r=r, modified=modified)
        S_soft = soft.forward(S_norm)
        S_soft.clamp_(q_min, q_max)

        S_soft.sub_(zero_point)
        S_soft.mul_(scale)
        S_quant = S_soft

        ctx.save_for_backward(S_norm, scale, zero_point)
        ctx.r = r
        ctx.modified = modified
        ctx.A = A
        ctx.B = B

        del S, S_norm, S_soft
        return S_quant

    @staticmethod
    def backward(ctx, grad_output):
        S_norm, scale, zero_point = ctx.saved_tensors
        r = ctx.r
        modified = ctx.modified
        A = ctx.A
        B = ctx.B

        soft = SoftStairs(r=r, modified=modified)
        deriv = soft.derivative(S_norm)
        grad_S = grad_output * deriv

        grad_weight = grad_S

        if A is not None and B is not None:
            grad_A = torch.matmul(grad_S, B.T)
            grad_B = torch.matmul(A.T, grad_S)
        else:
            grad_A = None
            grad_B = None

        return grad_weight, grad_A, grad_B, None, None, None, None, None, None, None, None


def quantize_soft_stairs(weight, A, B, r, scale, zero_point, q_min, q_max, sigma_B_max, modified=False, symmetric=False):
    return SoftStairsQuantizeFunction.apply(
        weight, A, B, r, scale, zero_point, q_min, q_max, sigma_B_max, modified, symmetric
    )


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

        self._init_quantization()

        self._register_hooks()

    def _init_quantization(self):
        """
        Reserves space for parameters:
          - Saves the original weights as a buffer (W_orig)
          - Calculates quantization parameters
          - If is_lora=True, freezes the weights and creates adapters.
        """
        for name, module in self.model.named_modules():
            if not isinstance(module, self._target_modules):
                continue
            if not hasattr(module, 'weight') or module.weight is None:
                continue

            module.register_buffer('weight_orig', module.weight.data.clone().detach())

            calc = QuantizationParamsCalculator()
            params = calc.compute(module.weight_orig, self.config.n_bits, symmetric=self.config.symmetric)
            self._scales[name] = params.scale
            self._zero_points[name] = params.zero_point
            self._q_min[name] = params.q_min
            self._q_max[name] = params.q_max

            if self._is_lora:
                module.weight.requires_grad_(False)  

                in_features = module.weight.shape[1]
                out_features = module.weight.shape[0]

                lora_A = nn.Parameter(torch.randn(self._rank, in_features) * 0.01)
                lora_B = nn.Parameter(torch.randn(out_features, self._rank) * 0.01)

                module.register_parameter('lora_A', lora_A)
                module.register_parameter('lora_B', lora_B)

                state = self._variance_controller.initialize_adapters(
                    module.weight_orig,
                    self._rank,
                    self.config.n_bits,
                    self.config.symmetric,
                )
                self._sigma_A[name] = state.sigma
                self._sigma_B[name] = state.sigma_b_max

                self._apply_variance_constraint(name)

    def _apply_variance_constraint(self, name: str):
        """Constrains the variance of layer adapters to permissible values."""
        module = self._get_module_by_name(name)
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            with torch.no_grad():
                std_A = module.lora_A.std().item()
                if std_A > self._sigma_A.get(name, 1e-6):
                    module.lora_A.data.mul_(self._sigma_A[name] / std_A)
                std_B = module.lora_B.std().item()
                if std_B > self._sigma_B.get(name, 1e-6):
                    module.lora_B.data.mul_(self._sigma_B[name] / std_B)

    def _get_module_by_name(self, name: str) -> nn.Module:
        parts = name.split('.')
        current = self.model
        for p in parts:
            current = getattr(current, p)
        return current

    def _register_hooks(self):
        """Registers a forward pre-hook for each quantized layer."""
        for name, module in self.model.named_modules():
            if name in self._scales:
                module.register_forward_pre_hook(self._make_pre_hook(name))

    def _make_pre_hook(self, layer_name: str):
        """Creates a pre-hook for a specific layer."""
        def hook(module, input):
            if self._is_lora:
                weight = module.weight_orig
                A = module.lora_A if hasattr(module, 'lora_A') else None
                B = module.lora_B if hasattr(module, 'lora_B') else None
                if A is None or B is None:
                    A = B = None
            else:
                weight = module.weight.data  
                A = None
                B = None

            scale = self._scales[layer_name]
            zero_point = self._zero_points[layer_name]
            q_min = self._q_min[layer_name]
            q_max = self._q_max[layer_name]
            sigma_B_max = self._sigma_B.get(layer_name, 1.0) if self._is_lora else 1.0

            quantized = quantize_soft_stairs(
                weight,
                A,
                B,
                self._r,                    
                scale,
                zero_point,
                q_min,
                q_max,
                sigma_B_max,
                modified=self.config.modified,
                symmetric=self.config.symmetric,
            )

            module.weight.data.copy_(quantized)

        return hook

    def step(self):
        """Called after every `optimizer.step()` to update `r` and manage the adapters."""
        self.current_step += 1

        if self.scheduler is not None:
            new_r = self.scheduler.get_r(self.current_step)
            if abs(new_r - self._r) > 1e-10:
                self._r = new_r

        if self._is_lora and (self.current_step % 10 == 0):
            for name in self._scales.keys():
                self._apply_variance_constraint(name)

    def get_current_r(self) -> float:
        return self._r

    def get_r_schedule(self) -> Optional[List[float]]:
        if self.scheduler is None:
            return None
        return self.scheduler.get_all_r()
    


def finalize_model(model: nn.Module, quantizer: SoftStairsQuantizer) -> nn.Module:
    """
    Converts model weights to the integer type,
    since SoftStairs has already brought them to values ​​close to integers.
    Removes auxiliary buffers (weight_orig, lora_A, lora_B).
    """
    n_bits = quantizer.config.n_bits
    dtype = getattr(torch, f'int{n_bits}')

    for name, module in model.named_modules():
        if name not in quantizer._scales:
            continue

        if quantizer._is_lora and hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            W_final = module.weight_orig + torch.matmul(module.lora_B, module.lora_A)
        else:
            W_final = module.weight_orig

        W_scaled = W_final / quantizer._scales[name] + quantizer._zero_points[name]

        W_int = W_scaled.to(dtype)

        module.weight.data = W_int

        del module.weight_orig
        if hasattr(module, 'lora_A'):
            del module.lora_A
        if hasattr(module, 'lora_B'):
            del module.lora_B

    return model