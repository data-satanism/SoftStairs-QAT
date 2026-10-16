# softstairs_qat/analysis/weight_types.py

"""On-the-fly weight-type tracker: current state and lifetime activation behaviour.

This module is a strictly *observational* training-harness component.  It never
touches gradients, optimizer state, parameter values or the SoftStairs
forward/backward computation; it only consumes tensors that the QAT step already
produced.

Two separate concepts are tracked, and they are deliberately kept apart:

CURRENT STATE (per tracking step, mutually exclusive)
    ``DORMANT`` / ``RESERVOIR`` / ``ACTIVE`` -- the same three labels used by
    :mod:`softstairs_qat.analysis.tracking` (:class:`softstairs_qat.analysis.tracking.Label`).

LIFETIME BEHAVIOUR (derived from the full history, never reset)
    ``NEVER_ACTIVE`` / ``RARE_ACTIVE`` / ``CHURNING`` / ``PERSISTENT_ACTIVE``.
    A ``PERSISTENT_ACTIVE`` weight may temporarily sit in ``RESERVOIR``; the two
    categoricals are never collapsed into one variable.

Activity score
--------------
The classification score is the quantization-sensitivity-weighted optimization
pressure ``|H| * D`` where ``H`` is the upstream (loss) gradient before the
SoftStairs derivative and ``D`` is
:meth:`softstairs_qat.core.soft_stairs.SoftStairs.derivative`.  Because the QAT
gradient arriving at the trainable leaf is ``g = H * D`` with ``D >= 0``, the
identity ``|H| * D == |g|`` holds, so the tracker accepts whichever quantity the
training loop already has:

* ``gradient=g`` -- the stored QAT gradient; the score is ``|g|``;
* ``pressure=|g|`` -- a precomputed magnitude;
* ``upstream_gradient=H`` together with ``D`` (explicit ``sensitivity``, or
  ``parameter`` + ``t`` so ``dSS`` can be evaluated); the literal ``|H| * D`` is
  formed and ``EMA(|H|)`` / ``EMA(D)`` are tracked separately;
* ``activity_score=...`` -- a score computed elsewhere (for example the
  ``active_score`` exposed by an existing
  :class:`~softstairs_qat.analysis.tracking.ReservoirActiveTracker`, or a
  quantity derived from optimizer momenta), reused verbatim.

Scores are optionally mapped to ``[0, 1]`` with the same adaptive quantile
normalization used by :func:`softstairs_qat.analysis.scores.normalize_activity`,
which keeps the default thresholds ``active_threshold`` / ``reservoir_threshold``
meaningful across models and gradient scales.  No second, competing definition
of the score is introduced.

Classification is threshold-based::

    score >= active_threshold      -> ACTIVE
    score >= reservoir_threshold   -> RESERVOIR
    otherwise                      -> DORMANT

This complements the ranking-based (TopK) classifier of
:mod:`softstairs_qat.analysis.tracking`; both consume the same score quantity.

State is stored as flat ``[N]`` / ``[6, N]`` tensors over one logical parameter
vector (:class:`~softstairs_qat.analysis.flattening.ParameterLayout` keeps the
mapping back to individual parameter tensors).  There are no Python objects per
weight, no per-step snapshots of weights/gradients/masks, and the only Python
loop is over the (few) parameter tensors.

The scientific purpose is to decide whether QAT contains a persistent
quantization-critical parameter core and how that core interacts with a
dynamically churning reservoir.  The tracker only exposes the masks and
statistics; using them for optimization is out of scope here.

Live-training usage::

    tracker = build_weight_type_tracker(model, active_threshold=0.8,
                                        reservoir_threshold=0.3,
                                        total_steps=len(train_loader) * epochs)
    for step, batch in enumerate(train_loader):
        loss.backward()
        tracker.update(step=step, gradients={n: p.grad for n, p in model.named_parameters()},
                       parameters=dict(model.named_parameters()), t=quantizer.t)
        optimizer.step(); optimizer.zero_grad()
        if step % log_interval == 0:
            log(tracker.get_metrics())   # compact scalars + quantiles
    torch.save(tracker.state_dict(), ckpt)          # resume keeps lifetime stats
    tracker.export_weight_statistics(path="weight_stats.pt")   # end of training
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch import Tensor

from softstairs_qat.analysis.flattening import ParameterLayout, flatten_tensors, select_parameters
from softstairs_qat.analysis.scores import normalize_activity, quantile_last_dim
from softstairs_qat.analysis.tracking import Label, jaccard
from softstairs_qat.core.soft_stairs import SoftStairs

__all__ = [
    "WeightType",
    "WEIGHT_TYPE_TRANSITIONS",
    "WeightTypeConfig",
    "classify_weight_states",
    "WeightTypeTracker",
    "ModelWeightTypeTracker",
    "build_weight_type_tracker",
]

_LOW_PRECISION = (torch.float16, torch.bfloat16)

_SCORE_NORMALIZATIONS = ("quantile", "none")


class WeightType(IntEnum):
    """Long-term weight type derived from lifetime activation behaviour.

    This is intentionally a different categorical from
    :class:`softstairs_qat.analysis.tracking.Label`: the current state
    (dormant / active / reservoir) fluctuates every step, while the weight type
    summarizes the whole history so far.
    """

    NEVER_ACTIVE = 0
    RARE_ACTIVE = 1
    CHURNING = 2
    PERSISTENT_ACTIVE = 3

    @property
    def key(self) -> str:
        """Lower-case type name, used as a mask dictionary key."""
        return self.name.lower()


#: The six off-diagonal state transitions, in the fixed order of the counter rows.
WEIGHT_TYPE_TRANSITIONS: tuple[tuple[str, int, int], ...] = (
    ("dormant_to_active", int(Label.DORMANT), int(Label.ACTIVE)),
    ("dormant_to_reservoir", int(Label.DORMANT), int(Label.RESERVOIR)),
    ("reservoir_to_active", int(Label.RESERVOIR), int(Label.ACTIVE)),
    ("reservoir_to_dormant", int(Label.RESERVOIR), int(Label.DORMANT)),
    ("active_to_reservoir", int(Label.ACTIVE), int(Label.RESERVOIR)),
    ("active_to_dormant", int(Label.ACTIVE), int(Label.DORMANT)),
)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _work_dtype(dtype: torch.dtype) -> torch.dtype:
    """Promote half precision dtypes used for accumulation.

    Args:
        dtype: Configured dtype.

    Returns:
        ``torch.float32`` for half/bfloat16 inputs, otherwise ``dtype``.
    """
    return torch.float32 if dtype in _LOW_PRECISION else dtype


def _ema_step(state: Tensor | None, observation: Tensor, decay: float) -> Tensor:
    """One EMA recursion step, seeded from the first observation.

    Matches :func:`softstairs_qat.analysis.ema.EmaState.update` with
    ``init="first"``.

    Args:
        state: Previous EMA value, or ``None`` before the first update.
        observation: Current observation.
        decay: Decay factor in ``[0, 1]``.

    Returns:
        The updated EMA value.
    """
    if state is None:
        return observation.clone()
    return float(decay) * state + (1.0 - float(decay)) * observation


def _quantile_scalar(values: Tensor, q: float) -> float:
    """Quantile of a flat vector as a Python float.

    Args:
        values: Arbitrary-shape tensor; flattened first.
        q: Quantile in ``[0, 1]``.

    Returns:
        The quantile value.
    """
    return float(quantile_last_dim(values.reshape(1, -1).float(), float(q)).reshape(()))


def _fraction(numerator: float, denominator: float) -> float:
    """Ratio guarded against an empty denominator.

    Args:
        numerator: Numerator.
        denominator: Denominator.

    Returns:
        ``numerator / denominator``, or ``0.0`` when the denominator is ``<= 0``.
    """
    return 0.0 if denominator <= 0 else float(numerator) / float(denominator)


# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class WeightTypeConfig:
    """Configuration of the on-the-fly weight-type tracker.

    All state-dependent thresholds are configurable; none are hard-coded in the
    state machine.

    Attributes:
        active_threshold: ``score >= active_threshold`` classifies a weight as
            ``ACTIVE``.  Interpreted in ``[0, 1]`` when ``score_normalization``
            is ``"quantile"`` (the default), in raw score units otherwise.
        reservoir_threshold: ``reservoir_threshold <= score < active_threshold``
            classifies a weight as ``RESERVOIR``; below it the weight is
            ``DORMANT``.
        score_normalization: ``"quantile"`` maps the score to ``[0, 1]`` per step
            with :func:`softstairs_qat.analysis.scores.normalize_activity`;
            ``"none"`` compares the raw score against the thresholds.
        normalization_quantile: Upper quantile bound of the normalization.
        normalization_eps: Stabilizer of the normalization denominator.
        decay_activity: Decay of ``EMA(|H|)`` and ``EMA(|H| * D)``.
        decay_sensitivity: Decay of ``EMA(D)``.
        decay_velocity: Decay of ``EMA(|p_t - p_{t-1}|)``.
        persistent_active_threshold: ``time_active >=`` this value marks a weight
            persistent (both for the long-term type and the persistent core).
        churn_min_time_active: Minimum ``time_active`` for ``CHURNING``.
        churn_min_spells: Minimum number of activation spells for ``CHURNING``.
        discovery_fraction: Optional length of the discovery phase as a fraction
            of ``total_steps``.  Requires ``total_steps`` unless
            ``discovery_steps`` is given.
        discovery_steps: Optional explicit discovery length in tracking steps;
            takes precedence over ``discovery_fraction``.
        total_steps: Expected total number of tracking steps, used only to
            resolve ``discovery_fraction``.
        track_step_jaccard: Whether to accumulate the per-step active/reservoir
            Jaccard means reported as ``*_jaccard_step_mean``.  Disable to shave
            a few reductions per step.
        report_quantiles: Interior quantiles reported for ``time_active`` and the
            spell counts.
        use_sensitivity: Whether ``dSS`` may be evaluated from ``parameter`` and
            ``t`` when no explicit sensitivity is supplied.
        normalized: Forwarded to ``SoftStairs.normalized``.
        async_t_factor: Forwarded to ``SoftStairs.async_t_factor``; the factor the
            backward pass applies, see ``SoftStairsQuantizer.current_backward_t``.
        dtype: Accumulation dtype of the float state tensors.
        device: Device of the tracker tensors; resolved from the first update
            input when ``None``.
    """

    active_threshold: float = 0.8
    reservoir_threshold: float = 0.3
    score_normalization: str = "quantile"
    normalization_quantile: float = 0.9
    normalization_eps: float = 1e-12
    decay_activity: float = 0.9
    decay_sensitivity: float = 0.9
    decay_velocity: float = 0.9
    persistent_active_threshold: float = 0.90
    churn_min_time_active: float = 0.10
    churn_min_spells: int = 2
    discovery_fraction: float | None = 0.1
    discovery_steps: int | None = None
    total_steps: int | None = None
    track_step_jaccard: bool = True
    report_quantiles: tuple[float, ...] = (0.25, 0.5, 0.75)
    use_sensitivity: bool = True
    normalized: bool = False
    async_t_factor: float = 1.0
    dtype: torch.dtype = torch.float32
    device: str | torch.device | None = None

    def __post_init__(self) -> None:
        """Validate the configured ranges.

        Raises:
            ValueError: If a threshold, decay, fraction or quantile is out of
                range or a mode name is unknown.
        """
        if self.score_normalization not in _SCORE_NORMALIZATIONS:
            raise ValueError(
                f"score_normalization must be one of {_SCORE_NORMALIZATIONS}, got {self.score_normalization!r}"
            )
        if not 0.0 <= float(self.reservoir_threshold) <= float(self.active_threshold):
            raise ValueError(
                "reservoir_threshold must lie in [0, active_threshold], got "
                f"{self.reservoir_threshold} with active_threshold={self.active_threshold}"
            )
        for name in ("normalization_quantile",):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value}")
        if not 0.0 < float(self.persistent_active_threshold) <= 1.0:
            raise ValueError(f"persistent_active_threshold must lie in (0, 1], got {self.persistent_active_threshold}")
        if not 0.0 <= float(self.churn_min_time_active) < float(self.persistent_active_threshold):
            raise ValueError(
                "churn_min_time_active must lie in [0, persistent_active_threshold), got "
                f"{self.churn_min_time_active}"
            )
        if int(self.churn_min_spells) < 1:
            raise ValueError(f"churn_min_spells must be positive, got {self.churn_min_spells}")
        for name in ("decay_activity", "decay_sensitivity", "decay_velocity"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value}")
        if self.discovery_fraction is not None and not 0.0 < float(self.discovery_fraction) <= 1.0:
            raise ValueError(f"discovery_fraction must lie in (0, 1], got {self.discovery_fraction}")
        if self.discovery_steps is not None and int(self.discovery_steps) < 0:
            raise ValueError(f"discovery_steps must be non-negative, got {self.discovery_steps}")
        if self.total_steps is not None and int(self.total_steps) < 0:
            raise ValueError(f"total_steps must be non-negative, got {self.total_steps}")
        for value in self.report_quantiles:
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"report_quantiles entries must lie in [0, 1], got {value}")

    def replace(self, **changes: Any) -> "WeightTypeConfig":
        """Return a copy of this config with ``changes`` applied.

        Args:
            **changes: Field overrides.

        Returns:
            The updated :class:`WeightTypeConfig`.
        """
        return replace(self, **changes)


# ----------------------------------------------------------------------
# classification
# ----------------------------------------------------------------------
@torch.no_grad()
def prepare_scores(scores: Tensor, config: WeightTypeConfig, *, dtype: torch.dtype) -> Tensor:
    """Flatten a score tensor and apply the configured normalization.

    Args:
        scores: Raw activity score of any shape; flattened to ``[N]``.
        config: Tracker configuration providing the normalization knobs.
        dtype: Work dtype of the tracker.

    Returns:
        Flat ``[N]`` tensor ready for threshold classification.

    Raises:
        ValueError: If the tensor is empty.
    """
    flat = scores.detach().reshape(-1).to(dtype=dtype)
    if flat.numel() == 0:
        raise ValueError("cannot classify an empty score tensor")
    if config.score_normalization == "quantile":
        normalized = normalize_activity(
            flat.reshape(1, -1),
            quantile=config.normalization_quantile,
            eps=config.normalization_eps,
        )
        return normalized.reshape(-1)
    return flat


def classify_weight_states(scores: Tensor, config: WeightTypeConfig, *, dtype: torch.dtype | None = None) -> Tensor:
    """Threshold-classify a score tensor into the three :class:`Label` states.

    Args:
        scores: Activity score ``|H| * D`` of any shape.
        config: Tracker configuration providing the thresholds.
        dtype: Work dtype; defaults to the promoted ``config.dtype``.

    Returns:
        Long tensor of shape ``[N]`` holding ``Label`` values.
    """
    work = _work_dtype(config.dtype if dtype is None else dtype)
    value = prepare_scores(scores, config, dtype=work)
    state = torch.full_like(value, int(Label.DORMANT), dtype=torch.long)
    state = torch.where(
        value >= float(config.reservoir_threshold),
        torch.full_like(state, int(Label.RESERVOIR)),
        state,
    )
    return torch.where(
        value >= float(config.active_threshold),
        torch.full_like(state, int(Label.ACTIVE)),
        state,
    )


# ----------------------------------------------------------------------
# streaming tracker over one flat weight vector
# ----------------------------------------------------------------------
class WeightTypeTracker:
    """Streaming weight-type tracker over one flat ``[N]`` weight vector.

    Every :meth:`update` is one *tracking step* (typically one optimizer step).
    The tracker classifies the flat vector into the three current states, folds
    the step into per-weight lifetime counters, advances four EMA signals and
    keeps only ``O(N)`` tensor state -- no per-weight Python objects and no
    history snapshots.

    Per-weight state (all tensors):

    =======================  ==========================  =====================
    tensor                    shape                        dtype
    =======================  ==========================  =====================
    ``current_state``         ``[N]``                      long
    ``previous_state``        ``[N]``                      long
    ``active_steps``          ``[N]``                      long
    ``spell_count``           ``[N]``                      long
    ``current_spell_length``  ``[N]``                      long
    ``longest_spell``         ``[N]``                      long
    ``transition_counts``     ``[6, N]``                   long
    EMA signals               ``[N]``                      float
    =======================  ==========================  =====================

    ``transition_counts`` rows follow :data:`WEIGHT_TYPE_TRANSITIONS`.
    Aggregate ``[3]`` state-population totals provide the denominators of the
    source-normalized transition rates ``P(dst | src)``.
    """

    def __init__(self, num_parameters: int, *, config: WeightTypeConfig | None = None) -> None:
        """Initialize the tracker state.

        Args:
            num_parameters: Total number of tracked scalar weights ``N``.
            config: Tracker configuration; dataclass defaults are used when ``None``.

        Raises:
            ValueError: If ``num_parameters`` is not positive or the config is
                invalid.
        """
        config = config or WeightTypeConfig()
        if int(num_parameters) <= 0:
            raise ValueError(f"num_parameters must be positive, got {num_parameters}")
        self.num_parameters = int(num_parameters)
        self.config = config
        self._dtype = _work_dtype(config.dtype)
        self._device: torch.device | None = torch.device(config.device) if config.device is not None else None
        self._steps = 0
        self._last_step = -1

        shape = (self.num_parameters,)
        self._current_state = torch.full(shape, int(Label.DORMANT), dtype=torch.long, device=self._device)
        self._previous_state = torch.full(shape, int(Label.DORMANT), dtype=torch.long, device=self._device)
        self._active_steps = torch.zeros(shape, dtype=torch.long, device=self._device)
        self._spell_count = torch.zeros(shape, dtype=torch.long, device=self._device)
        self._current_spell_length = torch.zeros(shape, dtype=torch.long, device=self._device)
        self._longest_spell = torch.zeros(shape, dtype=torch.long, device=self._device)
        self._transition_counts = torch.zeros(
            (len(WEIGHT_TYPE_TRANSITIONS), self.num_parameters), dtype=torch.long, device=self._device
        )
        self._state_totals = torch.zeros(3, dtype=torch.long, device=self._device)

        self._ema_activity: Tensor | None = None
        self._ema_pressure: Tensor | None = None
        self._ema_sensitivity: Tensor | None = None
        self._ema_velocity: Tensor | None = None
        self._previous_parameter: Tensor | None = None

        self._step_jac_active_sum: Tensor | None = None
        self._step_jac_reservoir_sum: Tensor | None = None
        self._step_jac_n = 0

        self._discovery_core: Tensor | None = None
        self._discovery_end_step: int | None = None

        self._metrics_active: Tensor | None = None
        self._metrics_reservoir: Tensor | None = None
        self._metrics_persistent: Tensor | None = None
        self._metrics_transitions: Tensor | None = None
        self._metrics_state_totals: Tensor | None = None

    # ------------------------------------------------------------------
    # state views
    # ------------------------------------------------------------------
    @property
    def seen(self) -> int:
        """Number of tracking steps folded into the state."""
        return self._steps

    @property
    def step(self) -> int:
        """Step index of the last :meth:`update` (``-1`` before it)."""
        return self._last_step

    @property
    def device(self) -> torch.device | None:
        """Device of the tracker tensors; ``None`` before the first update."""
        return self._device

    @property
    def work_dtype(self) -> torch.dtype:
        """Accumulation dtype of the float state tensors."""
        return self._dtype

    @property
    def current_state(self) -> Tensor:
        """Flat long tensor of :class:`Label` values at the last tracking step."""
        return self._current_state

    @property
    def previous_state(self) -> Tensor:
        """Flat long tensor of :class:`Label` values at the previous step."""
        return self._previous_state

    @property
    def active_steps(self) -> Tensor:
        """Per-weight count of tracking steps spent in the ``ACTIVE`` state."""
        return self._active_steps

    @property
    def spell_count(self) -> Tensor:
        """Per-weight number of activation spells (contiguous ``ACTIVE`` episodes)."""
        return self._spell_count

    @property
    def current_spell_length(self) -> Tensor:
        """Length of the ongoing active spell per weight (``0`` outside a spell)."""
        return self._current_spell_length

    @property
    def longest_spell(self) -> Tensor:
        """Longest active spell per weight so far."""
        return self._longest_spell

    @property
    def transition_counts(self) -> Tensor:
        """``[6, N]`` long tensor of per-weight transition counters.

        Rows follow :data:`WEIGHT_TYPE_TRANSITIONS`.
        """
        return self._transition_counts

    @property
    def ema_activity(self) -> Tensor | None:
        """``EMA(|H| * D)`` per weight, or ``None`` before the first update."""
        return self._ema_activity

    @property
    def ema_pressure(self) -> Tensor | None:
        """``EMA(|H|)`` per weight, or ``None`` until an upstream gradient was seen."""
        return self._ema_pressure

    @property
    def ema_sensitivity(self) -> Tensor | None:
        """``EMA(D)`` per weight, or ``None`` until a sensitivity was seen."""
        return self._ema_sensitivity

    @property
    def ema_velocity(self) -> Tensor | None:
        """``EMA(|p_t - p_{t-1}|)`` per weight, or ``None`` before a snapshot was seen."""
        return self._ema_velocity

    @property
    def active_mask(self) -> Tensor:
        """Flat boolean mask of the currently active weights."""
        return self._current_state == int(Label.ACTIVE)

    @property
    def reservoir_mask(self) -> Tensor:
        """Flat boolean mask of the current reservoir weights."""
        return self._current_state == int(Label.RESERVOIR)

    @property
    def dormant_mask(self) -> Tensor:
        """Flat boolean mask of the currently dormant weights."""
        return self._current_state == int(Label.DORMANT)

    @property
    def activation_count(self) -> Tensor:
        """Number of activation spells per weight; alias of :attr:`spell_count`."""
        return self._spell_count

    @property
    def spells(self) -> Tensor:
        """Number of activation spells per weight; alias of :attr:`spell_count`."""
        return self._spell_count

    @property
    def current_active_episode_length(self) -> Tensor:
        """Length of the ongoing active spell per weight; alias of :attr:`current_spell_length`."""
        return self._current_spell_length

    @property
    def longest_active_episode(self) -> Tensor:
        """Longest completed active spell per weight; alias of :attr:`longest_spell`."""
        return self._longest_spell

    def time_active(self) -> Tensor:
        """Fraction of tracking steps each weight has been active so far.

        Returns:
            Float tensor of shape ``[N]`` in ``[0, 1]``; zero before the first update.
        """
        if self._steps == 0:
            return torch.zeros(self.num_parameters, dtype=self._dtype, device=self._device_or_default())
        return self._active_steps.to(self._dtype) / float(self._steps)

    def activation_rate(self) -> Tensor:
        """Number of activation spells per tracking step per weight.

        Returns:
            Float tensor of shape ``[N]``.
        """
        if self._steps == 0:
            return torch.zeros(self.num_parameters, dtype=self._dtype, device=self._device_or_default())
        return self._spell_count.to(self._dtype) / float(self._steps)

    def persistent_mask(self) -> Tensor:
        """Running persistent-active core: ``time_active >= persistent_active_threshold``."""
        return self.time_active() >= float(self.config.persistent_active_threshold)

    @property
    def discovery_core_mask(self) -> Tensor | None:
        """Core frozen at the end of the discovery phase, or ``None`` before it."""
        return self._discovery_core

    @property
    def discovery_active(self) -> bool:
        """Whether a configured discovery phase is still running."""
        return self._discovery_limit() is not None and self._discovery_core is None

    def state_counts(self) -> dict[str, int]:
        """Current population per state.

        Returns:
            Mapping with the ``"active"`` / ``"reservoir"`` / ``"dormant"`` counts.
        """
        return {
            "active": int(self._current_state.eq(int(Label.ACTIVE)).sum()),
            "reservoir": int(self._current_state.eq(int(Label.RESERVOIR)).sum()),
            "dormant": int(self._current_state.eq(int(Label.DORMANT)).sum()),
        }

    def transition_totals(self) -> dict[str, int]:
        """Cumulative transition counts aggregated over the whole model.

        Returns:
            Mapping from :data:`WEIGHT_TYPE_TRANSITIONS` name to total count.
        """
        totals = self._transition_counts.sum(dim=1)
        return {name: int(totals[index]) for index, (name, _, _) in enumerate(WEIGHT_TYPE_TRANSITIONS)}

    def long_term_types(self) -> Tensor:
        """Classify every weight into its long-term :class:`WeightType`.

        The categories partition the weights, evaluated in the order
        ``NEVER_ACTIVE`` -> ``RARE_ACTIVE`` -> ``CHURNING`` -> ``PERSISTENT_ACTIVE``:

        * ``NEVER_ACTIVE``: ``active_steps == 0``;
        * ``PERSISTENT_ACTIVE``: ``time_active >= persistent_active_threshold``;
        * ``CHURNING``: ``time_active >= churn_min_time_active`` *and*
          ``spell_count >= churn_min_spells``;
        * ``RARE_ACTIVE``: anything else that has been active at least once.

        Returns:
            Long tensor of shape ``[N]`` holding :class:`WeightType` values.
        """
        config = self.config
        time_active = self.time_active()
        kinds = torch.full_like(self._active_steps, int(WeightType.NEVER_ACTIVE))
        kinds = torch.where(
            self._active_steps > 0,
            torch.full_like(kinds, int(WeightType.RARE_ACTIVE)),
            kinds,
        )
        churning = (time_active >= float(config.churn_min_time_active)) & (
            self._spell_count >= int(config.churn_min_spells)
        )
        kinds = torch.where(churning, torch.full_like(kinds, int(WeightType.CHURNING)), kinds)
        return torch.where(
            time_active >= float(config.persistent_active_threshold),
            torch.full_like(kinds, int(WeightType.PERSISTENT_ACTIVE)),
            kinds,
        )

    def long_term_type_counts(self) -> dict[str, int]:
        """Population per long-term :class:`WeightType`.

        Returns:
            Mapping from lower-case type name to count.
        """
        kinds = self.long_term_types()
        return {kind.key: int(kinds.eq(int(kind)).sum()) for kind in WeightType}

    def _device_or_default(self) -> torch.device:
        """Resolve the current tracking device.

        Returns:
            The tracker device, or CPU before the first update.
        """
        return self._device if self._device is not None else torch.device("cpu")

    def to(self, device: str | torch.device) -> "WeightTypeTracker":
        """Move every tracker tensor to ``device``.

        Args:
            device: Target device.

        Returns:
            This tracker, for chaining.
        """
        self._device = torch.device(device)
        for name in (
            "_current_state",
            "_previous_state",
            "_active_steps",
            "_spell_count",
            "_current_spell_length",
            "_longest_spell",
            "_transition_counts",
            "_state_totals",
            "_ema_activity",
            "_ema_pressure",
            "_ema_sensitivity",
            "_ema_velocity",
            "_previous_parameter",
            "_step_jac_active_sum",
            "_step_jac_reservoir_sum",
            "_discovery_core",
            "_metrics_active",
            "_metrics_reservoir",
            "_metrics_persistent",
            "_metrics_transitions",
            "_metrics_state_totals",
        ):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.to(self._device))
        return self

    def reset(self) -> None:
        """Drop every accumulated statistic and return to the initial state."""
        shape = (self.num_parameters,)
        self._steps = 0
        self._last_step = -1
        self._current_state.fill_(int(Label.DORMANT))
        self._previous_state.fill_(int(Label.DORMANT))
        self._active_steps.zero_()
        self._spell_count.zero_()
        self._current_spell_length.zero_()
        self._longest_spell.zero_()
        self._transition_counts.zero_()
        self._state_totals.zero_()
        self._ema_activity = None
        self._ema_pressure = None
        self._ema_sensitivity = None
        self._ema_velocity = None
        self._previous_parameter = None
        self._step_jac_active_sum = None
        self._step_jac_reservoir_sum = None
        self._step_jac_n = 0
        self._discovery_core = None
        self._discovery_end_step = None
        self._metrics_active = None
        self._metrics_reservoir = None
        self._metrics_persistent = None
        self._metrics_transitions = None
        self._metrics_state_totals = None

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(
        self,
        *,
        gradient: Tensor | None = None,
        upstream_gradient: Tensor | None = None,
        sensitivity: Tensor | None = None,
        pressure: Tensor | None = None,
        activity_score: Tensor | None = None,
        parameter: Tensor | None = None,
        velocity: Tensor | None = None,
        t: float | None = None,
        step: int | None = None,
    ) -> None:
        """Fold one tracking step into the state machine.

        Exactly one of ``gradient`` / ``upstream_gradient`` / ``pressure`` /
        ``activity_score`` supplies the activity score (see the module docstring
        for how ``|H| * D`` is recovered in each case).  ``parameter`` enables
        the velocity EMA, ``t`` (with ``parameter``) lets ``dSS`` be evaluated,
        and ``sensitivity`` accepts a precomputed ``D``.

        Args:
            gradient: Current QAT gradient ``g = H * D``.
            upstream_gradient: Upstream gradient ``H`` before the SoftStairs
                derivative; needs ``D`` from ``sensitivity`` or ``parameter``+``t``.
            sensitivity: Precomputed SoftStairs derivative ``D``.
            pressure: Precomputed instantaneous pressure ``|g|``.
            activity_score: Score computed elsewhere, reused verbatim.
            parameter: Current weight snapshot; enables the velocity EMA.
            velocity: Explicit weight velocity ``|p_t - p_{t-1}|``; overrides the
                displacement computed from successive ``parameter`` inputs.
            t: Live SoftStairs temperature used to evaluate ``dSS``.
            step: Training-step index; defaults to an internal counter.

        Raises:
            ValueError: If the score source is missing or ambiguous, the shapes
                disagree with ``num_parameters``, or ``upstream_gradient`` has no
                way to obtain ``D``.
        """
        sources = [
            name
            for name, value in (
                ("gradient", gradient),
                ("upstream_gradient", upstream_gradient),
                ("pressure", pressure),
                ("activity_score", activity_score),
            )
            if value is not None
        ]
        if len(sources) > 1:
            raise ValueError(f"pass only one of gradient/upstream_gradient/pressure/activity_score, got {sources}")
        if not sources:
            raise ValueError("update requires gradient, pressure, upstream_gradient, or activity_score")
        if upstream_gradient is not None and sensitivity is None and (parameter is None or t is None):
            raise ValueError(
                "upstream_gradient needs dSS: pass parameter together with t, or pass sensitivity explicitly"
            )
        source = gradient if gradient is not None else (
            upstream_gradient if upstream_gradient is not None else pressure if pressure is not None else activity_score
        )
        if source.numel() != self.num_parameters:
            raise ValueError(f"expected {self.num_parameters} weights, got {source.numel()}")

        if self._device is None:
            self._device = source.device
            self.to(self._device)

        upstream = None
        if upstream_gradient is not None:
            upstream = upstream_gradient.detach().reshape(-1).to(device=self._device, dtype=self._dtype)
        dss = self._resolve_sensitivity(parameter, sensitivity, t)

        if activity_score is not None:
            observation = activity_score.detach().reshape(-1).to(device=self._device, dtype=self._dtype)
        elif upstream is not None and dss is not None:
            observation = upstream.abs() * dss
        else:
            observation = source.detach().reshape(-1).to(device=self._device, dtype=self._dtype).abs()

        self._ema_activity = _ema_step(self._ema_activity, observation, self.config.decay_activity)
        if upstream is not None:
            self._ema_pressure = _ema_step(self._ema_pressure, upstream.abs(), self.config.decay_activity)
        if dss is not None:
            self._ema_sensitivity = _ema_step(self._ema_sensitivity, dss.abs(), self.config.decay_sensitivity)
        self._advance_velocity(parameter, velocity)

        new_state = classify_weight_states(observation, self.config, dtype=self._dtype)
        self._advance_state_machine(new_state)
        self._steps += 1
        self._last_step = self._steps - 1 if step is None else int(step)
        self._freeze_discovery_if_due()

    def _resolve_sensitivity(
        self,
        parameter: Tensor | None,
        sensitivity: Tensor | None,
        t: float | None,
    ) -> Tensor | None:
        """Resolve the SoftStairs derivative ``D`` for this step.

        Args:
            parameter: Weight snapshot in code space, or ``None``.
            sensitivity: Precomputed derivative, or ``None``.
            t: Live temperature, or ``None``.

        Returns:
            Flat ``[N]`` derivative tensor, or ``None`` when unavailable.

        Raises:
            ValueError: If a supplied tensor has the wrong number of elements.
        """
        if sensitivity is not None:
            if sensitivity.numel() != self.num_parameters:
                raise ValueError(f"expected {self.num_parameters} sensitivities, got {sensitivity.numel()}")
            return sensitivity.detach().reshape(-1).to(device=self._device, dtype=self._dtype)
        if parameter is None or t is None or not self.config.use_sensitivity:
            return None
        if parameter.numel() != self.num_parameters:
            raise ValueError(f"expected {self.num_parameters} weights, got {parameter.numel()}")
        weights = parameter.detach().reshape(-1).to(device=self._device, dtype=self._dtype)
        staircase = SoftStairs(
            t=float(t),
            normalized=self.config.normalized,
            async_t_factor=self.config.async_t_factor,
        )
        return staircase.derivative(weights)

    def _advance_velocity(self, parameter: Tensor | None, velocity: Tensor | None) -> None:
        """Advance the ``EMA(|weight velocity|)`` by one step.

        Args:
            parameter: Current weight snapshot, or ``None`` to freeze the state.
            velocity: Explicit velocity observation taking precedence.
        """
        if velocity is not None:
            displacement = velocity.detach().reshape(-1).to(device=self._device, dtype=self._dtype).abs()
        elif parameter is not None:
            current = parameter.detach().reshape(-1).to(device=self._device, dtype=self._dtype)
            if self._previous_parameter is None:
                displacement = torch.zeros_like(current)
            else:
                displacement = (current - self._previous_parameter).abs()
            self._previous_parameter = current
        else:
            return
        self._ema_velocity = _ema_step(self._ema_velocity, displacement, self.config.decay_velocity)

    def _advance_state_machine(self, new_state: Tensor) -> None:
        """Fold one classification into counters, spells and transitions.

        The first classification seeds the spells but records no transition and
        no source population, because there is no previous state to leave.

        Args:
            new_state: Newly classified ``[N]`` label tensor.
        """
        previous = self._current_state
        active_now = new_state.eq(int(Label.ACTIVE))
        if self._steps > 0:
            self._state_totals += torch.bincount(previous, minlength=3)
            for index, (_, src, dst) in enumerate(WEIGHT_TYPE_TRANSITIONS):
                self._transition_counts[index] += previous.eq(src) & new_state.eq(dst)
            if self.config.track_step_jaccard:
                self._accumulate_step_jaccard(
                    new_state.eq(int(Label.ACTIVE)), previous.eq(int(Label.ACTIVE)), Label.ACTIVE
                )
                self._accumulate_step_jaccard(
                    new_state.eq(int(Label.RESERVOIR)), previous.eq(int(Label.RESERVOIR)), Label.RESERVOIR
                )
                self._step_jac_n += 1

        was_active = previous.eq(int(Label.ACTIVE))
        self._active_steps += active_now
        self._spell_count += active_now & ~was_active
        self._current_spell_length = torch.where(
            active_now, self._current_spell_length + 1, torch.zeros_like(self._current_spell_length)
        )
        torch.maximum(self._longest_spell, self._current_spell_length, out=self._longest_spell)

        self._previous_state.copy_(previous)
        self._current_state.copy_(new_state)

    def _accumulate_step_jaccard(self, current: Tensor, previous: Tensor, label: Label) -> None:
        """Add one per-step Jaccard observation to the interval accumulators.

        Args:
            current: Boolean mask at the current step.
            previous: Boolean mask at the previous step.
            label: Which set the masks describe.
        """
        value = jaccard(current, previous).to(torch.float32)
        if label is Label.ACTIVE:
            self._step_jac_active_sum = (
                value.clone() if self._step_jac_active_sum is None else self._step_jac_active_sum + value
            )
        else:
            self._step_jac_reservoir_sum = (
                value.clone() if self._step_jac_reservoir_sum is None else self._step_jac_reservoir_sum + value
            )

    def _discovery_limit(self) -> int | None:
        """Resolve the discovery-phase length in tracking steps.

        Returns:
            The limit, or ``None`` when no discovery phase is configured.
        """
        config = self.config
        if config.discovery_steps is not None:
            return int(config.discovery_steps)
        if config.discovery_fraction is not None and config.total_steps is not None:
            limit = int(round(float(config.discovery_fraction) * int(config.total_steps)))
            return limit if limit > 0 else None
        return None

    def _freeze_discovery_if_due(self) -> None:
        """Freeze the persistent core at the end of the discovery phase."""
        limit = self._discovery_limit()
        if limit is None or self._discovery_core is not None or self._steps < limit:
            return
        self._discovery_core = self.persistent_mask().clone()
        self._discovery_end_step = self._steps

    # ------------------------------------------------------------------
    # metrics
    # ------------------------------------------------------------------
    def get_metrics(self) -> dict[str, float | int | bool]:
        """Compact scalars and quantiles for logging.

        Interval quantities (``*_jaccard``, per-transition counts and rates) are
        taken since the previous :meth:`get_metrics` call, ``*_total`` quantities
        accumulate over the whole run.  Source-normalized rates divide by the
        number of (weight, step) pairs whose *source* state matched, i.e.
        ``P(dst -> | src)``, which is not the population-level flow fraction.

        Calling this method only advances the interval bookkeeping; it never
        touches the lifetime state.

        Returns:
            Flat mapping of metric name to scalar.  Empty before the first
            update; keys whose value is undefined (for example the first
            interval Jaccard) are omitted.
        """
        if self._steps == 0:
            return {}
        config = self.config
        total = self.num_parameters
        active = self._current_state.eq(int(Label.ACTIVE))
        reservoir = self._current_state.eq(int(Label.RESERVOIR))
        dormant = self._current_state.eq(int(Label.DORMANT))
        time_active = self.time_active()

        metrics: dict[str, float | int | bool] = {
            "step": self._last_step,
            "tracking_steps": self._steps,
            "active_fraction": float(active.sum()) / total,
            "reservoir_fraction": float(reservoir.sum()) / total,
            "dormant_fraction": float(dormant.sum()) / total,
        }

        if self._metrics_active is not None:
            metrics["active_jaccard"] = float(jaccard(active, self._metrics_active))
            metrics["reservoir_jaccard"] = float(jaccard(reservoir, self._metrics_reservoir))
        if self._step_jac_n > 0:
            metrics["active_jaccard_step_mean"] = float(self._step_jac_active_sum / self._step_jac_n)
            metrics["reservoir_jaccard_step_mean"] = float(self._step_jac_reservoir_sum / self._step_jac_n)

        totals = self._transition_counts.sum(dim=1)
        state_totals = self._state_totals
        for index, (name, src, dst) in enumerate(WEIGHT_TYPE_TRANSITIONS):
            cumulative = int(totals[index])
            source_total = int(state_totals[src])
            previous_totals = 0 if self._metrics_transitions is None else int(self._metrics_transitions[index])
            interval = cumulative - previous_totals
            interval_source = (
                source_total
                if self._metrics_state_totals is None
                else source_total - int(self._metrics_state_totals[src])
            )
            metrics[name] = interval
            metrics[f"{name}_rate"] = _fraction(interval, interval_source)
            metrics[f"{name}_total"] = cumulative
            metrics[f"{name}_total_rate"] = _fraction(cumulative, source_total)

        persistent = time_active >= float(config.persistent_active_threshold)
        metrics["persistent_active_fraction"] = float(persistent.sum()) / total
        if self._metrics_persistent is not None:
            metrics["persistent_core_jaccard"] = float(jaccard(persistent, self._metrics_persistent))

        core = self._discovery_core if self._discovery_core is not None else persistent
        if self._discovery_core is not None:
            metrics["discovery_active"] = False
            metrics["discovery_end_step"] = self._discovery_end_step
        else:
            metrics["discovery_active"] = self.discovery_active
        intersection = int((active & core).sum())
        outside = int((active & ~core).sum())
        missed = int((core & ~active).sum())
        core_size = int(core.sum())
        active_size = int(active.sum())
        metrics["core_intersection"] = intersection
        metrics["core_intersection_fraction"] = _fraction(intersection, core_size)
        metrics["active_outside_core"] = outside
        metrics["active_outside_core_fraction"] = _fraction(outside, active_size)
        metrics["core_outside_active"] = missed
        metrics["core_outside_active_fraction"] = _fraction(missed, core_size)

        for q in config.report_quantiles:
            tag = f"q{int(round(float(q) * 100)):02d}"
            metrics[f"time_active_{tag}"] = _quantile_scalar(time_active, q)
            metrics[f"spells_{tag}"] = _quantile_scalar(self._spell_count, q)

        self._metrics_active = active.clone()
        self._metrics_reservoir = reservoir.clone()
        self._metrics_persistent = persistent.clone()
        self._metrics_transitions = totals.clone()
        self._metrics_state_totals = state_totals.clone()
        if self._step_jac_active_sum is not None:
            self._step_jac_active_sum.zero_()
        if self._step_jac_reservoir_sum is not None:
            self._step_jac_reservoir_sum.zero_()
        self._step_jac_n = 0
        return metrics

    def summary(self) -> dict[str, float | int | bool]:
        """Alias of :meth:`get_metrics` for harness logging code.

        Returns:
            The same compact mapping as :meth:`get_metrics`.
        """
        return self.get_metrics()

    # ------------------------------------------------------------------
    # checkpointing
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Snapshot every lifetime statistic for checkpointing.

        Returns:
            Mapping of state name to tensor (or plain scalar).  Optional tensors
            that were never built (``ema_pressure`` before an upstream update,
            the frozen discovery core before the phase ends) are stored as
            ``None``.
        """
        state: dict[str, Any] = {
            "num_parameters": self.num_parameters,
            "steps": self._steps,
            "last_step": self._last_step,
            "current_state": self._current_state.clone(),
            "previous_state": self._previous_state.clone(),
            "active_steps": self._active_steps.clone(),
            "spell_count": self._spell_count.clone(),
            "current_spell_length": self._current_spell_length.clone(),
            "longest_spell": self._longest_spell.clone(),
            "transition_counts": self._transition_counts.clone(),
            "state_totals": self._state_totals.clone(),
            "discovery_end_step": self._discovery_end_step,
        }
        for key in (
            "ema_activity",
            "ema_pressure",
            "ema_sensitivity",
            "ema_velocity",
            "previous_parameter",
            "discovery_core",
        ):
            attribute = getattr(self, f"_{key}")
            state[key] = None if attribute is None else attribute.clone()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore lifetime statistics saved by :meth:`state_dict`.

        Interval bookkeeping (the previous :meth:`get_metrics` snapshot) is reset
        so a resumed run starts a fresh logging interval without corrupting the
        lifetime statistics.

        Args:
            state: Mapping produced by :meth:`state_dict`.

        Raises:
            ValueError: If ``num_parameters`` disagrees with this tracker or a
                restored tensor has the wrong shape.
        """
        if int(state["num_parameters"]) != self.num_parameters:
            raise ValueError(
                f"state_dict holds {int(state['num_parameters'])} parameters, tracker has {self.num_parameters}"
            )
        first = next(value for value in state.values() if torch.is_tensor(value))
        if self._device is None:
            self._device = first.device
            self.to(self._device)

        self._steps = int(state["steps"])
        self._last_step = int(state["last_step"])
        for name, attribute in (
            ("current_state", "_current_state"),
            ("previous_state", "_previous_state"),
            ("active_steps", "_active_steps"),
            ("spell_count", "_spell_count"),
            ("current_spell_length", "_current_spell_length"),
            ("longest_spell", "_longest_spell"),
            ("transition_counts", "_transition_counts"),
            ("state_totals", "_state_totals"),
        ):
            value = state[name].to(device=self._device)
            if tuple(value.shape) != tuple(getattr(self, attribute).shape):
                raise ValueError(f"state_dict entry {name!r} has shape {tuple(value.shape)}")
            getattr(self, attribute).copy_(value)
        for name, attribute in (
            ("ema_activity", "_ema_activity"),
            ("ema_pressure", "_ema_pressure"),
            ("ema_sensitivity", "_ema_sensitivity"),
            ("ema_velocity", "_ema_velocity"),
            ("previous_parameter", "_previous_parameter"),
            ("discovery_core", "_discovery_core"),
        ):
            value = state.get(name)
            setattr(self, attribute, None if value is None else value.to(device=self._device).clone())
        self._discovery_end_step = state["discovery_end_step"]

        self._metrics_active = None
        self._metrics_reservoir = None
        self._metrics_persistent = None
        self._metrics_transitions = None
        self._metrics_state_totals = None
        self._step_jac_active_sum = None
        self._step_jac_reservoir_sum = None
        self._step_jac_n = 0

    # ------------------------------------------------------------------
    # export
    # ------------------------------------------------------------------
    def export_weight_statistics(
        self,
        *,
        layout: ParameterLayout | None = None,
        path: str | None = None,
    ) -> dict[str, Any]:
        """Per-weight statistics for offline analysis at the end of training.

        Nothing is written during training; exporting is explicit.  With a
        :class:`~softstairs_qat.analysis.flattening.ParameterLayout` the flat
        rows are mapped back to ``(parameter_id, weight_index)`` pairs so each
        row identifies one scalar weight inside one named parameter tensor.

        Args:
            layout: Optional layout of the tracked parameters.
            path: Optional ``torch.save`` destination; the mapping is also
                returned.

        Returns:
            Mapping with ``parameter_id``, ``weight_index``, ``current_state``,
            ``long_term_type``, ``time_active``, ``activation_rate``,
            ``spell_count``, ``current_spell_length``, ``longest_active_spell``,
            ``active_steps``, the available EMA signals and (with a layout) the
            ``parameter_names`` tuple.  EMA entries never updated are omitted.
        """
        total = self.num_parameters
        if layout is not None and layout.num_parameters != total:
            raise ValueError(f"layout holds {layout.num_parameters} parameters, tracker has {total}")
        if layout is not None:
            parameter_id = torch.zeros(total, dtype=torch.long)
            weight_index = torch.empty(total, dtype=torch.long)
            for index, (offset, numel) in enumerate(zip(layout.offsets, layout.numels)):
                parameter_id[offset : offset + numel] = index
                weight_index[offset : offset + numel] = torch.arange(numel)
        else:
            parameter_id = torch.zeros(total, dtype=torch.long)
            weight_index = torch.arange(total, dtype=torch.long)

        export: dict[str, Any] = {
            "parameter_id": parameter_id,
            "weight_index": weight_index,
            "current_state": self._current_state.detach().cpu().clone(),
            "long_term_type": self.long_term_types().cpu(),
            "time_active": self.time_active().detach().cpu(),
            "activation_rate": self.activation_rate().detach().cpu(),
            "spell_count": self._spell_count.detach().cpu().clone(),
            "current_spell_length": self._current_spell_length.detach().cpu().clone(),
            "longest_active_spell": self._longest_spell.detach().cpu().clone(),
            "active_steps": self._active_steps.detach().cpu().clone(),
        }
        if self._ema_pressure is not None:
            export["ema_abs_h"] = self._ema_pressure.detach().cpu().clone()
        if self._ema_sensitivity is not None:
            export["ema_d"] = self._ema_sensitivity.detach().cpu().clone()
        if self._ema_activity is not None:
            export["ema_abs_h_times_d"] = self._ema_activity.detach().cpu().clone()
        if self._ema_velocity is not None:
            export["ema_velocity"] = self._ema_velocity.detach().cpu().clone()
        if layout is not None:
            export["parameter_names"] = tuple(layout.names)
        if path is not None:
            torch.save(export, path)
        return export

    def __repr__(self) -> str:
        if self._discovery_core is not None:
            discovery = "done"
        else:
            discovery = "active" if self.discovery_active else "off"
        return (
            f"WeightTypeTracker(num_parameters={self.num_parameters}, steps={self._steps}, "
            f"discovery={discovery}, device={self._device})"
        )


# ----------------------------------------------------------------------
# model-level harness
# ----------------------------------------------------------------------
class ModelWeightTypeTracker:
    """Weight-type tracker bound to a model or a mapping of parameter tensors.

    Heterogeneous parameter tensors are flattened into one logical weight vector
    (:class:`~softstairs_qat.analysis.flattening.ParameterLayout`), tracked by a
    single :class:`WeightTypeTracker`, and every flat result can be mapped back
    onto the original tensor shapes.  This is the entry point for a live QAT
    training loop; the observational guarantee is inherited from the core
    tracker.
    """

    def __init__(
        self,
        parameters: torch.nn.Module | Mapping[str, Tensor] | Sequence[Tensor],
        *,
        config: WeightTypeConfig | None = None,
        layout: ParameterLayout | None = None,
        trainable_only: bool = True,
        exclude_suffix: str | None = None,
    ) -> None:
        """Bind the tracker to a model or an explicit parameter container.

        Args:
            parameters: A model to select trainable parameters from, or an
                explicit mapping/sequence of named tensors.
            config: Tracker configuration.
            layout: Optional precomputed layout; required when ``parameters`` is
                a one-shot iterable.
            trainable_only: For a model, skip parameters with ``requires_grad=False``.
            exclude_suffix: For a model, skip names ending with this suffix
                (``"_orig"`` follows the raw leaves instead of the quantized
                buffers).

        Raises:
            ValueError: If the parameter set is empty or a generator was passed
                without a ``layout``.
        """
        config = config or WeightTypeConfig()
        if isinstance(parameters, torch.nn.Module):
            selected = select_parameters(parameters, trainable_only=trainable_only, exclude_suffix=exclude_suffix)
            if not selected:
                raise ValueError("no trainable parameters matched; adjust trainable_only/exclude_suffix")
            source: Any = selected
        else:
            source = parameters
        if layout is None:
            if isinstance(source, Iterator):
                raise ValueError("pass an explicit layout when parameters is a one-shot iterable")
            layout = ParameterLayout.from_tensors(source)
        self.layout = layout
        self.config = config
        self._tracker = WeightTypeTracker(layout.num_parameters, config=config)

    @property
    def tracker(self) -> WeightTypeTracker:
        """Underlying flat-vector tracker."""
        return self._tracker

    @property
    def num_parameters(self) -> int:
        """Total number of tracked scalar weights."""
        return self.layout.num_parameters

    def _flatten(self, mapping: Mapping[str, Tensor] | None) -> Tensor | None:
        """Flatten a per-parameter mapping onto the tracker's flat vector.

        Args:
            mapping: Mapping from parameter name to tensor, or ``None``.

        Returns:
            Flat tensor, or ``None`` when ``mapping`` is ``None``.
        """
        if mapping is None:
            return None
        return flatten_tensors(mapping, layout=self.layout, dtype=self._tracker.work_dtype)

    def update(
        self,
        *,
        step: int | None = None,
        gradients: Mapping[str, Tensor] | None = None,
        upstream_gradients: Mapping[str, Tensor] | None = None,
        sensitivities: Mapping[str, Tensor] | None = None,
        pressures: Mapping[str, Tensor] | None = None,
        activity_scores: Mapping[str, Tensor] | None = None,
        parameters: Mapping[str, Tensor] | None = None,
        velocities: Mapping[str, Tensor] | None = None,
        t: float | None = None,
    ) -> None:
        """Run one tracking step over every tracked parameter tensor.

        Args:
            step: Training-step index; defaults to an internal counter.
            gradients: Mapping from parameter name to the QAT gradient ``g``.
            upstream_gradients: Mapping to the upstream gradient ``H``; switches
                the score to the literal ``|H| * D`` and enables ``EMA(|H|)``.
            sensitivities: Mapping to a precomputed ``dSS`` per parameter.
            pressures: Mapping to precomputed ``|g|`` per parameter.
            activity_scores: Mapping to scores computed elsewhere, reused
                verbatim (for example from optimizer momenta).
            parameters: Mapping to the current weight snapshots; enables the
                velocity EMA and (with ``t``) ``dSS`` evaluation.
            velocities: Mapping to explicit per-parameter velocities.
            t: Live SoftStairs temperature.

        Raises:
            ValueError: If no score source is supplied or a mapping is missing a
                layout entry.
        """
        self._tracker.update(
            step=step,
            gradient=self._flatten(gradients),
            upstream_gradient=self._flatten(upstream_gradients),
            sensitivity=self._flatten(sensitivities),
            pressure=self._flatten(pressures),
            activity_score=self._flatten(activity_scores),
            parameter=self._flatten(parameters),
            velocity=self._flatten(velocities),
            t=t,
        )

    def state_masks(self) -> dict[str, dict[str, Tensor]]:
        """Current-state masks mapped back onto the parameter shapes.

        Returns:
            ``{'active': {name: tensor}, 'reservoir': {...}, 'dormant': {...}}``.
        """
        return {
            "active": self.layout.unflatten_masks(self._tracker.active_mask),
            "reservoir": self.layout.unflatten_masks(self._tracker.reservoir_mask),
            "dormant": self.layout.unflatten_masks(self._tracker.dormant_mask),
        }

    def long_term_masks(self) -> dict[str, dict[str, Tensor]]:
        """Long-term type masks mapped back onto the parameter shapes.

        Returns:
            ``{'persistent_active': {...}, 'churning': {...}, 'rare_active':
            {...}, 'never_active': {...}}``.
        """
        kinds = self._tracker.long_term_types()
        return {
            kind.key: self.layout.unflatten_masks(kinds.eq(int(kind)))
            for kind in WeightType
        }

    def per_tensor(self, flat: Tensor) -> dict[str, Tensor]:
        """Split any flat per-weight tensor back into the parameter shapes.

        Args:
            flat: Flat tensor whose last axis holds ``num_parameters`` entries.

        Returns:
            Mapping from parameter name to a view with the original shape.
        """
        return self.layout.unflatten(flat)

    def time_active(self) -> Tensor:
        """See :meth:`WeightTypeTracker.time_active`.

        Returns:
            Flat ``[N]`` fraction of tracking steps spent active.
        """
        return self._tracker.time_active()

    def activation_rate(self) -> Tensor:
        """See :meth:`WeightTypeTracker.activation_rate`.

        Returns:
            Flat ``[N]`` spells-per-step rate.
        """
        return self._tracker.activation_rate()

    def long_term_types(self) -> Tensor:
        """See :meth:`WeightTypeTracker.long_term_types`.

        Returns:
            Flat ``[N]`` :class:`WeightType` tensor.
        """
        return self._tracker.long_term_types()

    def persistent_mask(self) -> Tensor:
        """See :meth:`WeightTypeTracker.persistent_mask`.

        Returns:
            Flat boolean mask of the running persistent-active core.
        """
        return self._tracker.persistent_mask()

    def get_metrics(self) -> dict[str, float | int | bool]:
        """See :meth:`WeightTypeTracker.get_metrics`.

        Returns:
            The compact metric mapping of the underlying tracker.
        """
        return self._tracker.get_metrics()

    def summary(self) -> dict[str, float | int | bool]:
        """Alias of :meth:`get_metrics`.

        Returns:
            The compact metric mapping of the underlying tracker.
        """
        return self._tracker.summary()

    def state_dict(self) -> dict[str, Any]:
        """See :meth:`WeightTypeTracker.state_dict`.

        Returns:
            The checkpoint mapping of the underlying tracker.
        """
        return self._tracker.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """See :meth:`WeightTypeTracker.load_state_dict`.

        Args:
            state: Mapping produced by :meth:`state_dict`.
        """
        self._tracker.load_state_dict(state)

    def export_weight_statistics(self, path: str | None = None) -> dict[str, Any]:
        """Export per-weight statistics with the parameter mapping resolved.

        Args:
            path: Optional ``torch.save`` destination.

        Returns:
            See :meth:`WeightTypeTracker.export_weight_statistics`; the export
            additionally carries the ``parameter_names`` tuple.
        """
        return self._tracker.export_weight_statistics(layout=self.layout, path=path)


def build_weight_type_tracker(
    model: torch.nn.Module,
    *,
    active_threshold: float = 0.8,
    reservoir_threshold: float = 0.3,
    total_steps: int | None = None,
    discovery_fraction: float | None = 0.1,
    config: WeightTypeConfig | None = None,
    trainable_only: bool = True,
    exclude_suffix: str | None = None,
) -> ModelWeightTypeTracker:
    """Convenience constructor for a live QAT model.

    Args:
        model: Model whose trainable parameters are tracked.
        active_threshold: Active threshold on the (normalized) activity score.
        reservoir_threshold: Reservoir threshold on the same score.
        total_steps: Expected number of tracking steps, used to resolve the
            discovery phase.
        discovery_fraction: Fraction of ``total_steps`` spent collecting
            statistics before the persistent core is frozen; ``None`` disables.
        config: Full configuration; overrides the convenience arguments.
        trainable_only: Skip parameters with ``requires_grad=False``.
        exclude_suffix: Skip parameter names ending with this suffix.

    Returns:
        The configured :class:`ModelWeightTypeTracker`.
    """
    if config is None:
        config = WeightTypeConfig(
            active_threshold=active_threshold,
            reservoir_threshold=reservoir_threshold,
            total_steps=total_steps,
            discovery_fraction=discovery_fraction,
        )
    return ModelWeightTypeTracker(
        model,
        config=config,
        trainable_only=trainable_only,
        exclude_suffix=exclude_suffix,
    )
