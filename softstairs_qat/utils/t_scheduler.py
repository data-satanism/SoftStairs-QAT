# softstairs_qat/utils/r_scheduler.py

from __future__ import annotations

import math
import torch
from enum import Enum
from typing import List, Optional
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback as LightningCallback
from pytorch_lightning.core import LightningModule

from softstairs_qat.wrappers.config import QuantizationConfig

class TSchedulerType(Enum):
    LINEAR = "linear"
    EXP = "exp"
    STEP = "step"
    COS = "cos"
    CONSTANT = "constant"
    ADAPTIVE = "adaptive"
    CYCLIC = "cyclic"


class TScheduler:
    def __init__(
        self,
        strategy: TSchedulerType | str = TSchedulerType.LINEAR,
        start_t: float = 0.5,
        end_t: float = 0.01,
        total_steps: int = 1000,
        tau: float = 8.0,
        step_size: int = 100,
        early_power: float = 0.7,
        min_majorant: float = 0.01,   
    ):
        self.strategy = strategy if isinstance(strategy, TSchedulerType) else TSchedulerType(strategy)
        self.start_t = start_t
        self.end_t = end_t
        self.total_steps = total_steps
        self.tau = tau
        self.step_size = step_size
        self.early_power = early_power
        self.min_majorant = min_majorant

        self._diff = end_t - start_t
        self._inv_total = 1.0 / (total_steps - 1) if total_steps > 1 else 1.0

        self._strategies = {
            TSchedulerType.LINEAR: self._linear,
            TSchedulerType.EXP: self._exp,
            TSchedulerType.STEP: self._step,
            TSchedulerType.COS: self._cos,
            TSchedulerType.CONSTANT: self._constant,
            TSchedulerType.CYCLIC: self._cyclic,
        }

        self._precomputed: Optional[List[float]] = None
        if total_steps > 0:
            self._precomputed = [self._compute_t(i) for i in range(total_steps)]

        self._cache: dict[int, float] = {}

    def _linear(self, step: int) -> float:
        if self.total_steps <= 1:
            return self.end_t
        return self.start_t + self._diff * (step * self._inv_total)

    def _exp(self, step: int) -> float:
        if self.total_steps <= 1:
            return self.end_t
        progress = step * self._inv_total
        accelerated_progress = progress ** self.early_power
        exp_factor = 1 - math.exp(-accelerated_progress * self.tau)
        return self.start_t + self._diff * exp_factor

    def _step(self, step: int) -> float:
        if self.total_steps <= self.step_size:
            return self.end_t
        num_steps = max(1, self.total_steps // self.step_size)
        current_step = min(step // self.step_size, num_steps)
        return self.start_t + self._diff * (current_step / num_steps)

    def _cos(self, step: int) -> float:
        if self.total_steps <= 1:
            return self.end_t
        progress = step * self._inv_total
        cos_factor = (1 - math.cos(math.pi * progress)) / 2
        return self.start_t + self._diff * cos_factor

    def _constant(self, step: int) -> float:
        return self.start_t

    def _cyclic(self, step: int) -> float:
        if self.total_steps <= 1:
            return self.end_t

        self.min_majorant = self.end_t if self.min_majorant==0 else self.min_majorant
        
        progress = step * self._inv_total  
        
        exp_factor = 1 - math.exp(-progress * self.tau)  
        
        minorant = self.start_t + (self.end_t - self.start_t) * exp_factor
        majorant = 1.0 + (self.min_majorant - 1.0) * exp_factor

        if step == self.total_steps - 1:
            return minorant
    
        base = (majorant + minorant) / 2
        amplitude = (majorant - minorant) * 0.5

        sign = 1 - 2 * (step % 2)  
        oscillation = amplitude * sign
        
        t = base + oscillation
         
        t = max(minorant, min(majorant, t))
        
        return t

    def _compute_t(self, step: int) -> float:
        step = max(0, min(step, self.total_steps - 1))
        strategy_func = self._strategies.get(self.strategy, self._linear)
        r = strategy_func(step)
        return max(0.0, min(1.0, r))

    def get_t(self, step: int) -> float:
        if self._precomputed is not None:
            return self._precomputed[max(0, min(step, self.total_steps - 1))]
        if step in self._cache:
            return self._cache[step]
        r = self._compute_t(step)
        self._cache[step] = r
        return r

    def get_all_t(self) -> List[float]:
        if self._precomputed is not None:
            return self._precomputed.copy()
        return [self.get_t(step) for step in range(self.total_steps)]

    def reset(self) -> None:
        self._cache.clear()

    def __repr__(self) -> str:
        return (
            f"TScheduler(strategy={self.strategy.value}, "
            f"start_r={self.start_t}, end_r={self.end_t}, "
            f"total_steps={self.total_steps})"
        )

    @classmethod
    def from_config(cls, config: QuantizationConfig, total_steps: int) -> Optional[TScheduler]:
        if config.t_scheduler_strategy == "constant":
            return None
        return cls(
            strategy=config.t_scheduler_strategy,
            start_t=config.t_start,
            end_t=config.t_end,
            total_steps=total_steps,
            tau=config.t_tau,
            step_size=config.n_steps,
            early_power=config.early_power,
            min_majorant=config.min_majorant,
        )

class FrequencyScheduler:
    """Frequency schedule of the SoftStairs frequency homotopy, coupled to ``t``.

    The staircase frequency ``f`` is a linear function of the current
    temperature ``t``, interpolating between ``(t_start, f_start)`` and
    ``(t_end, f_end)``::

        f(t) = f_end + (f_start - f_end) * (t - t_end) / (t_start - t_end)

    Because the schedule is defined on ``t`` (not on the step index), it
    composes with every ``TScheduler`` strategy (linear, exp, cos, ...) and
    with the adaptive scheduler: whenever the quantizer's ``t`` moves, ``f``
    follows. The canonical homotopy starts at ``f_start = 2`` (doubled
    staircase frequency, a twice denser soft-quantization lattice) and anneals
    to ``f_end = 1``, recovering the original SoftStairs exactly.
    """

    def __init__(
        self,
        f_start: float = 2.0,
        f_end: float = 1.0,
        t_start: float = 0.9,
        t_end: float = 0.01,
    ):
        if f_start <= 0 or f_end <= 0:
            raise ValueError(f"frequencies must be positive, got f_start={f_start}, f_end={f_end}")
        self.f_start = f_start
        self.f_end = f_end
        self.t_start = t_start
        self.t_end = t_end

    def get_f(self, t: float) -> float:
        """Frequency at temperature ``t``, linear in ``t`` and clamped.

        Args:
            t: Current SoftStairs temperature.

        Returns:
            Frequency clamped to ``[min(f_start, f_end), max(f_start, f_end)]``.
            When ``t_start == t_end`` the frequency is constant at ``f_start``.
        """
        if self.t_start == self.t_end:
            return float(self.f_start)
        progress = (t - self.t_end) / (self.t_start - self.t_end)
        progress = max(0.0, min(1.0, progress))
        f = self.f_end + (self.f_start - self.f_end) * progress
        lo, hi = min(self.f_start, self.f_end), max(self.f_start, self.f_end)
        return float(max(lo, min(hi, f)))

    def get_all_f(self, ts: List[float]) -> List[float]:
        """Frequencies along a sequence of temperatures (e.g. a t schedule)."""
        return [self.get_f(t) for t in ts]

    def __repr__(self) -> str:
        return (
            f"FrequencyScheduler(f_start={self.f_start}, f_end={self.f_end}, "
            f"t_start={self.t_start}, t_end={self.t_end})"
        )

    @classmethod
    def from_config(cls, config: QuantizationConfig) -> Optional[FrequencyScheduler]:
        if not getattr(config, "frequency_homotopy", False):
            return None
        return cls(
            f_start=config.f_start,
            f_end=config.f_end,
            t_start=config.t_start,
            t_end=config.t_end,
        )


class AdaptiveScheduler:
    """Metric-driven t schedule: t = tau^n after each stagnation."""
    def __init__(
        self,
        start_t: float = 0.2,
        tau: float = 0.9,
        metric_delta: float = 1e-3,
    ):
        if not (0.0 < tau < 1.0):
            raise ValueError(f"tau must be in (0, 1), got {tau}")
        if metric_delta < 0.0:
            raise ValueError(f"metric_delta must be >= 0, got {metric_delta}")
        self.start_t = start_t
        self.tau = tau
        self.metric_delta = metric_delta
        self._n = 0
        self._t = start_t
        self._prev_metric: Optional[float] = None
        self._history: List[float] = [start_t]


    def get_t(self, step: int | None = None) -> float:
        return self._t

    
    def observe(self, metric: float) -> float:
        if self._prev_metric is not None:
            if abs(metric - self._prev_metric) < self.metric_delta:
                self._n += 1
                # self._t = max(0.0, min(1.0, 1.0 - self.tau ** self._n))
                k = 6.0
                progress = 1.0 - self.tau ** self._n
                self._t = self.start_t * (math.exp(k) - math.exp(k * progress)) / (math.exp(k) - 1.0)
                self._t = max(0.0, min(1.0, self._t))
                self._history.append(self._t)
        self._prev_metric = metric
        return self._t

    
    def get_all_t(self) -> List[float]:
        return self._history.copy()

    
    def reset(self) -> None:
        self._n = 0
        self._t = self.start_t
        self._prev_metric = None
        self._history = [self.start_t]


    def __repr__(self) -> str:
        return (
            f"AdaptiveScheduler(start_t={self.start_t}, tau={self.tau}, "
            f"metric_delta={self.metric_delta}, n={self._n}, t={self._t})"
        )

    
    @classmethod
    def from_config(cls, config: QuantizationConfig) -> AdaptiveScheduler:
        return cls(
            start_t=config.t_start,
            tau=config.t_tau,
            metric_delta=config.t_metric_delta,
        )

    
class AdaptiveTCallback(LightningCallback):
    """Lightning adapter: reads an epoch metric and updates AdaptiveScheduler."""
    def __init__(self, monitor: str = "val_loss"):
        self.monitor = monitor

        
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        quantizer = getattr(pl_module, "quantizer", None)

        if quantizer is None:
            return
        
        scheduler = getattr(quantizer, "scheduler", None)

        if not isinstance(scheduler, AdaptiveScheduler):
            return

        raw = trainer.callback_metrics.get(self.monitor)
        if raw is None:
            return
        
        metric = float(raw.detach().cpu()) if hasattr(raw, "detach") else float(raw)
        new_t = scheduler.observe(metric)
        quantizer._t = new_t
        pl_module.log("quantizer_t", quantizer.t, on_epoch=True, on_step=False)