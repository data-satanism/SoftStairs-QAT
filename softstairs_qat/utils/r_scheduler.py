# softstairs_qat/utils/r_scheduler.py

from __future__ import annotations

import math
from enum import Enum
from typing import List, Optional

from softstairs_qat.wrappers.config import QuantizationConfig


class TSchedulerType(Enum):
    LINEAR = "linear"
    EXP = "exp"
    STEP = "step"
    COS = "cos"
    CONSTANT = "constant"


class TScheduler:
    def __init__(
        self,
        strategy: TSchedulerType | str = TSchedulerType.LINEAR,
        start_t: float = 0.5,
        end_t: float = 0.01,
        total_steps: int = 1000,
        tau: float = 8.0,
        step_size: int = 100,
    ):
        self.strategy = strategy if isinstance(strategy, TSchedulerType) else TSchedulerType(strategy)
        self.start_t = start_t
        self.end_t = end_t
        self.total_steps = total_steps
        self.tau = tau
        self.step_size = step_size

        self._diff = end_t - start_t
        self._inv_total = 1.0 / (total_steps - 1) if total_steps > 1 else 1.0

        self._strategies = {
            TSchedulerType.LINEAR: self._linear,
            TSchedulerType.EXP: self._exp,
            TSchedulerType.STEP: self._step,
            TSchedulerType.COS: self._cos,
            TSchedulerType.CONSTANT: self._constant,
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
        k = 6.0
        exp_factor = (1.0 - math.exp(k * progress)) / (1.0 - math.exp(k))
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
            step_size=config.t_step,
        )