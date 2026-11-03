# softstairs_qat/analysis/tracking.py

"""Active / reservoir / dormant set classification and set-change tracking.

Three mutually exclusive sets are produced at every support update:

``active``
    weights whose persistent pressure *and* current SoftStairs sensitivity mark
    them as relevant to the ongoing quantization transition;
``reservoir``
    weights with persistent directional pressure that is currently *not*
    sensitive enough, i.e. candidates for a later promotion;
``dormant``
    everything else, ``dormant = ~(active | reservoir)``.

Classification is a ranking, never an absolute threshold: the active set is
``TopK(S_active, K_active)`` and the reservoir set is
``TopK(S_reservoir among the weights that are neither active nor below the
pressure floor, K_reservoir)``.  Because the reservoir candidates are masked with
``~active_mask`` before the second top-k, disjointness ``active ∩ reservoir == ∅``
holds by construction rather than by a post-hoc intersection.  ``K_reservoir`` is
additionally clamped to ``N - K_active`` so the two budgets can never exceed the
parameter count.

Bounded budgets
---------------
Both budgets accept an absolute count (``active_k``) or a relative one
(``active_fraction``); the count wins when both are given.  ``K = clamp(round(
fraction * N), 0, N)``, with a minimum of one element for a non-zero fraction so
a small fraction on a small layer still yields an observable set.

Pressure floor
--------------
A weight with negligible pressure must not become a reservoir member merely
because it is inactive, so the reservoir ranking is restricted to weights with
``pressure >= Quantile_q(pressure)``.  The default ``q = 0`` is a no-op; raising
it (for example to ``0.3``) prunes the negligible-pressure tail structurally,
before any ranking happens.

Set-change semantics
--------------------
All transitions between two consecutive support updates are read from stacked
``[T - 1, 3, 3]`` confusion matrices over the labels
``(dormant=0, active=1, reservoir=2)``.  Each matrix is accumulated with a single
``torch.bincount`` whose indices carry a ``9 * step`` offset, so the whole
transition history is obtained without a Python loop over time or weights.  For
every transition the record reports the count, the fraction of the *source* set
and the fraction of the *destination* set:

=============================  ==================  ==================
transition                     source set           destination set
=============================  ==================  ==================
``active_added``               ``~active_{t-1}``    ``active_t``
``active_removed``             ``active_{t-1}``     ``~active_t``
``active_persistent``          ``active_{t-1}``     ``active_t``
``reservoir_added``            ``~reservoir_{t-1}`` ``reservoir_t``
``reservoir_removed``          ``reservoir_{t-1}``  ``~reservoir_t``
``reservoir_persistent``       ``reservoir_{t-1}``  ``reservoir_t``
``reservoir_to_active``        ``reservoir_{t-1}``  ``active_t``
``active_to_reservoir``        ``active_{t-1}``     ``reservoir_t``
``reservoir_to_dormant``       ``reservoir_{t-1}``  ``dormant_t``
``dormant_to_reservoir``       ``dormant_{t-1}``    ``reservoir_t``
``dormant_to_active``          ``dormant_{t-1}``    ``active_t``
``active_to_dormant``          ``active_{t-1}``     ``dormant_t``
=============================  ==================  ==================

The three ``added`` / ``removed`` / ``persistent`` rows describe the symmetric
difference of the set between two consecutive updates:
``active_added = reservoir_to_active + dormant_to_active`` and
``active_removed = active_to_reservoir + active_to_dormant``, while
``|active_t| = active_persistent + active_added``.  The ``added`` rows take the
*complement* of the previous set as their source, and the ``removed`` rows the
complement of the current set as their destination -- "what share of the weights
that are *not* active now used to be active".  A source or destination set of size
zero yields a fraction of ``0.0``.

Stability
---------
``Jaccard(A_t, A_{t-1}) = |A_t ∩ A_{t-1}| / |A_t ∪ A_{t-1}|`` with the convention
that two empty sets score ``1.0``.  Turnover is ``1 - Jaccard``.  Both are
reported for the active and the reservoir set.  A falling active Jaccard together
with a high ``reservoir_to_active`` rate is exactly the
``reservoir -> active -> reservoir`` churn this tracker is built to expose.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Any, Mapping

import torch
from torch import Tensor

from softstairs_qat.analysis.scores import (
    ActivityScores,
    ScoreConfig,
    ScoreSeries,
    compute_score_series,
    normalize_activity,
    quantile_last_dim,
)

__all__ = [
    "Label",
    "TRANSITION_NAMES",
    "SetConfig",
    "TrackerConfig",
    "TransitionEntry",
    "TrackerRecord",
    "resolve_budgets",
    "classify_series",
    "labels_from_masks",
    "transition_matrices",
    "summarize_transitions",
    "jaccard",
    "ReservoirActiveTracker",
    "track_sequence",
    "records_to_history",
]

_LOW_PRECISION = (torch.float16, torch.bfloat16)

_NEG_INF = float("-inf")


class Label(IntEnum):
    """Label of a weight at one support update."""

    DORMANT = 0
    ACTIVE = 1
    RESERVOIR = 2

    @property
    def key(self) -> str:
        """Lower-case label name, used as a mask dictionary key."""
        return self.name.lower()


TRANSITION_NAMES: tuple[str, ...] = (
    "active_added",
    "active_removed",
    "active_persistent",
    "reservoir_added",
    "reservoir_removed",
    "reservoir_persistent",
    "reservoir_to_active",
    "active_to_reservoir",
    "reservoir_to_dormant",
    "dormant_to_reservoir",
    "dormant_to_active",
    "active_to_dormant",
)


# ----------------------------------------------------------------------
# private helpers
# ----------------------------------------------------------------------
def _work_dtype(dtype: torch.dtype) -> torch.dtype:
    """Promote half precision dtypes used for accumulation.

    Args:
        dtype: Configured dtype.

    Returns:
        ``torch.float32`` for half/bfloat16 inputs, otherwise ``dtype``.
    """
    return torch.float32 if dtype in _LOW_PRECISION else dtype


def _flat_quantile(values: Tensor, q: float) -> Tensor:
    """Quantile of a flat vector.

    Args:
        values: Flat tensor.
        q: Quantile in ``[0, 1]``.

    Returns:
        Zero-dimensional tensor holding the quantile.
    """
    return quantile_last_dim(values.reshape(1, -1), float(q)).reshape(())


def _flat_normalize(scores: Tensor, config: ScoreConfig) -> Tensor:
    """Normalized activity of a flat score vector.

    Args:
        scores: Flat score vector.
        config: Score configuration.

    Returns:
        Tensor in ``[0, 1]`` shaped like ``scores``.
    """
    normalized = normalize_activity(scores.reshape(1, -1), quantile=config.activity_quantile, eps=config.activity_eps)
    return normalized.reshape(scores.shape)


def _fraction(count: int, size: int) -> float:
    """Ratio guarded against an empty set.

    Args:
        count: Numerator.
        size: Denominator.

    Returns:
        ``count / size``, or ``0.0`` when ``size <= 0``.
    """
    return 0.0 if size <= 0 else float(count) / float(size)


def labels_from_masks(active: Tensor, reservoir: Tensor, dormant: Tensor) -> Tensor:
    """Encode three masks into one integer label tensor.

    Args:
        active: Active mask.
        reservoir: Reservoir mask.
        dormant: Dormant mask (unused for encoding, kept for symmetry).

    Returns:
        Long tensor holding :class:`Label` values.
    """
    labels = torch.full_like(active, int(Label.DORMANT), dtype=torch.long)
    labels = torch.where(reservoir, torch.full_like(labels, int(Label.RESERVOIR)), labels)
    return torch.where(active, torch.full_like(labels, int(Label.ACTIVE)), labels)


def _topk_mask(scores: Tensor, k: int) -> Tensor:
    """Select the ``k`` largest scores along the last dimension.

    Args:
        scores: Score tensor of shape ``[..., N]``.
        k: Requested count, clamped to ``N``.

    Returns:
        Boolean tensor with the shape of ``scores`` and ``min(k, N)`` set entries
        per row.
    """
    total = scores.shape[-1]
    k = min(max(int(k), 0), total)
    if k == 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    indices = torch.topk(scores, k, dim=-1, largest=True, sorted=False).indices
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, indices, True)


def _ema_step(state: Tensor | None, observation: Tensor, decay: float) -> Tensor:
    """One EMA recursion step, seeded from the first observation.

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


def _single_sensitivity(parameter: Tensor, config: ScoreConfig, t: float) -> Tensor | None:
    """Evaluate the SoftStairs derivative for one weight vector at a known temperature.

    ``dSS`` needs the code-space weight and the live temperature, so this returns
    ``None`` whenever no usable temperature is available; the caller then scores
    from ``|g|``, which is the same estimator.

    Args:
        parameter: Weight snapshot in quantization code space.
        config: Score configuration.
        t: Live SoftStairs temperature.

    Returns:
        The derivative, or ``None`` when it cannot be evaluated.
    """
    try:
        from softstairs_qat.analysis.sensitivity import softstairs_sensitivity

        return softstairs_sensitivity(
            parameter.reshape(1, -1),
            float(t),
            normalized=config.normalized,
            async_t_factor=config.async_t_factor,
        ).reshape(-1)
    except (ValueError, TypeError, RuntimeError):
        return None


# ----------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SetConfig:
    """Budgets and ranking knobs of the three-set classifier.

    Attributes:
        active_fraction: Fraction of weights eligible for the active set.
        active_k: Absolute active budget; takes precedence over ``active_fraction``.
        reservoir_fraction: Fraction of weights eligible for the reservoir set.
        reservoir_k: Absolute reservoir budget; takes precedence over
            ``reservoir_fraction``.
        active_hysteresis: Optional stickiness bonus in ``[0, 1]``.  The active score
            of a weight that was already active at the previous support update is
            multiplied by ``1 + active_hysteresis``.  ``0.0`` (default) keeps the
            classification purely score-driven and lets the EMA alone provide the
            temporal smoothing.
    """

    active_fraction: float | None = None
    active_k: int | None = None
    reservoir_fraction: float | None = None
    reservoir_k: int | None = None
    active_hysteresis: float = 0.0

    def __post_init__(self) -> None:
        """Validate the configured ranges.

        Raises:
            ValueError: If a fraction, count or the hysteresis bonus is out of range.
        """
        for name in ("active_fraction", "reservoir_fraction"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value}")
        for name in ("active_k", "reservoir_k"):
            value = getattr(self, name)
            if value is not None and int(value) < 0:
                raise ValueError(f"{name} must be non-negative, got {value}")
        if not 0.0 <= float(self.active_hysteresis) <= 1.0:
            raise ValueError(f"active_hysteresis must lie in [0, 1], got {self.active_hysteresis}")

    def replace(self, **changes: Any) -> "SetConfig":
        """Return a copy of this config with ``changes`` applied.

        Args:
            **changes: Field overrides.

        Returns:
            The updated :class:`SetConfig`.
        """
        return replace(self, **changes)


@dataclass(frozen=True)
class TrackerConfig:
    """Full configuration of the reservoir-vs-active tracker.

    Attributes:
        score: Temporal score configuration.
        sets: Budget and hysteresis configuration.
        record_distributions: Whether each record carries compact score summaries.
        distribution_quantiles: Interior quantiles of those summaries.
        chunk_size: Number of time steps processed per block when classifying a full
            series.  Bounds peak memory of the ``[T, N]`` label tensor; ``None``
            processes the whole series at once.
    """

    score: ScoreConfig = field(default_factory=ScoreConfig)
    sets: SetConfig = field(default_factory=SetConfig)
    record_distributions: bool = False
    distribution_quantiles: tuple[float, ...] = (0.05, 0.5, 0.95)
    chunk_size: int | None = None

    def replace(self, **changes: Any) -> "TrackerConfig":
        """Return a copy of this config with ``changes`` applied.

        Args:
            **changes: Field overrides.

        Returns:
            The updated :class:`TrackerConfig`.
        """
        return replace(self, **changes)


# ----------------------------------------------------------------------
# records
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class TransitionEntry:
    """One tracked set change between two consecutive support updates.

    Attributes:
        count: Number of weights exhibiting the transition.
        source_fraction: ``count / size(source set)``; ``0.0`` for an empty source.
        destination_fraction: ``count / size(destination set)``; ``0.0`` for an empty
            destination.
    """

    count: int
    source_fraction: float
    destination_fraction: float

    def to_dict(self) -> dict[str, float | int]:
        """Return the entry as a flat dictionary.

        Returns:
            Mapping with ``count``, ``source_fraction`` and ``destination_fraction``.
        """
        return {
            "count": self.count,
            "source_fraction": self.source_fraction,
            "destination_fraction": self.destination_fraction,
        }


@dataclass(frozen=True)
class TrackerRecord:
    """Compact per-support-update statistics.

    The record deliberately stores counts, fractions and order statistics only,
    never a ``[T, num_parameters]`` history.

    Attributes:
        step: Support-update index.
        num_active: Size of the active set.
        num_reservoir: Size of the reservoir set.
        num_dormant: Size of the dormant set.
        num_parameters: Total number of tracked weights ``N``.
        active_jaccard: Jaccard similarity of the active set with the previous update;
            ``None`` for the first update.
        reservoir_jaccard: Jaccard similarity of the reservoir set with the previous
            update; ``None`` for the first update.
        active_turnover: ``1 - active_jaccard``.
        reservoir_turnover: ``1 - reservoir_jaccard``.
        transitions: Per-transition counts and source/destination fractions.
        distributions: Compact score summaries; empty unless enabled in the config.
    """

    step: int
    num_active: int
    num_reservoir: int
    num_dormant: int
    num_parameters: int
    active_jaccard: float | None = None
    reservoir_jaccard: float | None = None
    active_turnover: float | None = None
    reservoir_turnover: float | None = None
    transitions: dict[str, TransitionEntry] = field(default_factory=dict)
    distributions: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def active_fraction(self) -> float:
        """Share of tracked weights that are active."""
        return self.num_active / self.num_parameters if self.num_parameters else 0.0

    @property
    def reservoir_fraction(self) -> float:
        """Share of tracked weights that are in the reservoir."""
        return self.num_reservoir / self.num_parameters if self.num_parameters else 0.0

    @property
    def dormant_fraction(self) -> float:
        """Share of tracked weights that are dormant."""
        return self.num_dormant / self.num_parameters if self.num_parameters else 0.0

    def transition(self, name: str) -> TransitionEntry:
        """Look up one transition.

        Args:
            name: One of :data:`TRANSITION_NAMES`.

        Returns:
            The matching entry, or a zeroed entry when it was not recorded.

        Raises:
            KeyError: If ``name`` is not a known transition.
        """
        if name not in TRANSITION_NAMES:
            raise KeyError(f"unknown transition {name!r}")
        return self.transitions.get(name, TransitionEntry(0, 0.0, 0.0))

    def to_dict(self, *, flat_transitions: bool = True) -> dict[str, Any]:
        """Return the record as a plain dictionary, ready for tabular logging.

        Args:
            flat_transitions: Emit ``"<name>_count"``, ``"<name>_source_fraction"``
                and ``"<name>_destination_fraction"`` columns instead of nested
                dictionaries.

        Returns:
            Mapping of column name to value.
        """
        row: dict[str, Any] = {
            "step": self.step,
            "num_active": self.num_active,
            "num_reservoir": self.num_reservoir,
            "num_dormant": self.num_dormant,
            "num_parameters": self.num_parameters,
            "active_fraction": self.active_fraction,
            "reservoir_fraction": self.reservoir_fraction,
            "dormant_fraction": self.dormant_fraction,
            "active_jaccard": self.active_jaccard,
            "reservoir_jaccard": self.reservoir_jaccard,
            "active_turnover": self.active_turnover,
            "reservoir_turnover": self.reservoir_turnover,
        }
        for name in TRANSITION_NAMES:
            entry = self.transition(name)
            if flat_transitions:
                row[f"{name}_count"] = entry.count
                row[f"{name}_source_fraction"] = entry.source_fraction
                row[f"{name}_destination_fraction"] = entry.destination_fraction
            else:
                row[name] = entry.to_dict()
        for signal, stats in self.distributions.items():
            for key, value in stats.items():
                row[f"{signal}_{key}"] = value
        return row

    def __repr__(self) -> str:
        return (
            f"TrackerRecord(step={self.step}, num_active={self.num_active}, "
            f"num_reservoir={self.num_reservoir}, num_dormant={self.num_dormant}, "
            f"active_jaccard={self.active_jaccard}, reservoir_jaccard={self.reservoir_jaccard})"
        )


def records_to_history(records: list[TrackerRecord] | tuple[TrackerRecord, ...]) -> dict[str, list[Any]]:
    """Transpose a record list into a column-oriented history.

    The result maps a column name to a per-update list, which is what the
    diagnostics of the experimental analysis plot (``active-set size vs step``,
    Jaccard curves, transition rates, score distributions).

    Args:
        records: Records produced by :func:`track_sequence` or
            :meth:`ReservoirActiveTracker.update`.

    Returns:
        Mapping of column name to a list with one entry per record.
    """
    rows = [record.to_dict() for record in records]
    if not rows:
        return {}
    return {key: [row.get(key) for row in rows] for key in rows[0]}


# ----------------------------------------------------------------------
# budgets and classification
# ----------------------------------------------------------------------
def _resolve_budget(count: int | None, fraction: float | None, total: int, label: str, *, required: bool) -> int:
    """Turn a count/fraction budget pair into an absolute ``K``.

    Args:
        count: Absolute budget, or ``None``.
        fraction: Relative budget in ``[0, 1]``, or ``None``.
        total: Total number of tracked weights.
        label: Budget name used in error messages.
        required: When ``False`` an unset budget resolves to ``0`` instead of raising.

    Returns:
        The resolved budget clamped to ``[0, total]``.

    Raises:
        ValueError: If neither ``count`` nor ``fraction`` is given for a required budget.
    """
    if count is not None:
        return min(max(int(count), 0), total)
    if fraction is None:
        if not required:
            return 0
        raise ValueError(f"either {label}_k or {label}_fraction must be provided")
    value = min(max(int(round(float(fraction) * total)), 0), total)
    return 1 if fraction > 0.0 and value == 0 else value


def resolve_budgets(config: SetConfig, num_parameters: int) -> tuple[int, int]:
    """Resolve the active and reservoir budgets for ``N`` weights.

    The active budget is mandatory: without it the classification has no meaning.
    The reservoir budget is optional and defaults to ``0``, which tracks the active
    and dormant sets only.

    Args:
        config: Set configuration.
        num_parameters: Total number of tracked weights ``N``.

    Returns:
        Tuple ``(k_active, k_reservoir)``.  The reservoir budget is clamped to
        ``N - k_active``, which is what keeps the two budgets disjoint.

    Raises:
        ValueError: If ``num_parameters`` is negative or the active budget is
            unspecified.
    """
    if int(num_parameters) < 0:
        raise ValueError(f"num_parameters must be non-negative, got {num_parameters}")
    k_active = _resolve_budget(config.active_k, config.active_fraction, int(num_parameters), "active", required=True)
    k_reservoir = _resolve_budget(
        config.reservoir_k, config.reservoir_fraction, int(num_parameters), "reservoir", required=False
    )
    return k_active, min(k_reservoir, int(num_parameters) - k_active)


@torch.no_grad()
def classify_series(
    series: ScoreSeries,
    sets: SetConfig,
    *,
    previous_active: Tensor | None = None,
    num_parameters: int | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Split a score series into active / reservoir / dormant mask tensors.

    Every signal is first collapsed to ``[T, N]`` so that the ranking sees the whole
    logical weight vector -- not each trailing axis of the snapshot separately.
    Ranking is then a batched ``topk`` plus ``scatter`` per frame, so there is no
    Python loop over time and no loop over individual weights.  The returned masks are
    reshaped back to the shape of the score series.

    Args:
        series: Score series shaped ``[T, ...]``.
        sets: Budget configuration.
        previous_active: Optional active mask with the shape of ``series.active_score``
            used for the hysteresis bonus.
        num_parameters: Optional explicit ``N``; inferred from the series otherwise.

    Returns:
        Tuple of masks shaped like ``series.active_score``.

    Raises:
        ValueError: If ``previous_active`` is not aligned with the series.
    """
    steps = series.num_snapshots
    weight_shape = tuple(series.active_score.shape[1:])
    active_score = series.active_score.reshape(steps, -1)
    pressure = series.pressure.reshape(steps, -1)
    reservoir_score = series.reservoir_score.reshape(steps, -1)
    total = int(active_score.shape[-1]) if num_parameters is None else int(num_parameters)
    if total != active_score.shape[-1]:
        raise ValueError(f"num_parameters={total} disagrees with the {active_score.shape[-1]} weights of the series")
    k_active, k_reservoir = resolve_budgets(sets, total)

    ranking_score = active_score
    if sets.active_hysteresis > 0.0 and previous_active is not None:
        if previous_active.shape != series.active_score.shape:
            raise ValueError(
                f"previous_active with shape {tuple(previous_active.shape)} is not aligned with a "
                f"{tuple(series.active_score.shape)} score series"
            )
        bonus = previous_active.reshape(steps, -1).to(device=active_score.device, dtype=active_score.dtype)
        ranking_score = active_score * (1.0 + float(sets.active_hysteresis) * bonus)

    active = _topk_mask(ranking_score, k_active)

    floor = series.pressure_floor.reshape(steps, 1) if series.pressure_floor.ndim else series.pressure_floor
    eligible = (pressure >= floor) & ~active

    reservoir = _topk_mask(reservoir_score.masked_fill(~eligible, _NEG_INF), k_reservoir) & eligible
    dormant = ~(active | reservoir)
    return (
        active.reshape((steps,) + weight_shape),
        reservoir.reshape((steps,) + weight_shape),
        dormant.reshape((steps,) + weight_shape),
    )


# ----------------------------------------------------------------------
# transitions
# ----------------------------------------------------------------------
@torch.no_grad()
def jaccard(current: Tensor, previous: Tensor) -> Tensor:
    """Jaccard similarity along the last dimension.

    Args:
        current: Boolean mask of shape ``[..., N]``.
        previous: Boolean mask with the same shape.

    Returns:
        Tensor of shape ``current.shape[:-1]``.  Two empty sets score ``1.0``.

    Raises:
        ValueError: If the shapes differ.
    """
    if current.shape != previous.shape:
        raise ValueError(f"mask shapes differ: {tuple(current.shape)} vs {tuple(previous.shape)}")
    left = current.to(torch.bool)
    right = previous.to(torch.bool)
    intersection = (left & right).sum(-1).to(torch.float64)
    union = (left | right).sum(-1).to(torch.float64)
    return torch.where(union > 0, intersection / union.clamp_min(1.0), torch.ones_like(union))


@torch.no_grad()
def transition_matrices(labels: Tensor, *, chunk_size: int | None = None) -> Tensor:
    """Stack one ``[3, 3]`` label confusion matrix per consecutive pair.

    Each matrix is accumulated with a single ``torch.bincount`` whose flat indices
    carry a ``9 * step`` offset, so the whole history is computed without a Python
    loop over time.

    Args:
        labels: ``[T, N]`` integer labels, typically from :func:`labels_from_masks`.
        chunk_size: Number of time steps per block; bounds peak memory of the index
            tensor.  ``None`` processes the whole series at once.

    Returns:
        Long tensor of shape ``[T - 1, 3, 3]`` where entry ``[s, i, j]`` counts the
        weights that moved from label ``i`` to label ``j`` between frames ``s`` and
        ``s + 1``.

    Raises:
        ValueError: If fewer than two frames are supplied or labels are out of range.
    """
    if labels.ndim != 2:
        raise ValueError(f"labels must be [T, N], got {tuple(labels.shape)}")
    steps = labels.shape[0]
    if steps < 2:
        raise ValueError(f"need at least two frames to compute transitions, got {steps}")
    flat = labels.to(torch.long)
    if int(flat.min()) < 0 or int(flat.max()) > 2:
        raise ValueError("labels must be in {0, 1, 2}")

    width = flat.shape[1]
    pairs = steps - 1
    block = pairs if chunk_size is None else max(int(chunk_size), 1)
    counts = torch.zeros(9 * pairs, dtype=torch.long, device=flat.device)
    for start in range(0, pairs, block):
        stop = min(start + block, pairs)
        length = stop - start
        previous = flat[start:stop]
        current = flat[start + 1 : stop + 1]
        # Each step owns a disjoint slot of 9 bins, so the blocks never collide.
        offset = ((start + torch.arange(length, device=flat.device, dtype=torch.long)) * 9).unsqueeze(1)
        index = (previous * 3 + current + offset).reshape(-1)
        counts += torch.bincount(index, minlength=9 * pairs)
    return counts.reshape(pairs, 3, 3)


@torch.no_grad()
def _pair_matrix(current_labels: Tensor, previous_labels: Tensor) -> Tensor:
    """Confusion matrix of a single pair of label frames.

    Args:
        current_labels: ``[1, N]`` labels of the new support update.
        previous_labels: ``[1, N]`` labels of the previous support update.

    Returns:
        ``[3, 3]`` matrix where entry ``[i, j]`` counts the weights that moved from
        label ``i`` to label ``j``.
    """
    previous = previous_labels.reshape(-1).to(torch.long)
    current = current_labels.reshape(-1).to(torch.long)
    return torch.bincount(previous * 3 + current, minlength=9).reshape(3, 3)


def summarize_transitions(
    counts: Tensor,
    previous_sizes: Mapping[str, int],
    current_sizes: Mapping[str, int],
) -> dict[str, TransitionEntry]:
    """Turn one label confusion matrix into per-transition entries.

    The source and destination sets of every name are documented in the module
    docstring.

    Args:
        counts: ``[3, 3]`` matrix from :func:`transition_matrices`.
        previous_sizes: Set sizes at ``t - 1`` keyed by ``"active"``/``"reservoir"``/``"dormant"``.
        current_sizes: Set sizes at ``t``.

    Returns:
        Mapping from transition name to :class:`TransitionEntry`.
    """
    active_prev = int(previous_sizes.get("active", 0))
    active_now = int(current_sizes.get("active", 0))
    reservoir_prev = int(previous_sizes.get("reservoir", 0))
    reservoir_now = int(current_sizes.get("reservoir", 0))
    dormant_prev = int(previous_sizes.get("dormant", 0))
    dormant_now = int(current_sizes.get("dormant", 0))
    total = active_now + reservoir_now + dormant_now

    entries = {
        # |active_t \ active_{t-1}|, and its symmetric-difference partner.
        "active_added": (int(counts[0, 1]) + int(counts[2, 1]), total - active_prev, active_now),
        "active_removed": (int(counts[1, 0]) + int(counts[1, 2]), active_prev, total - active_now),
        "active_persistent": (int(counts[1, 1]), active_prev, active_now),
        "reservoir_added": (int(counts[0, 2]) + int(counts[1, 2]), total - reservoir_prev, reservoir_now),
        "reservoir_removed": (int(counts[2, 0]) + int(counts[2, 1]), reservoir_prev, total - reservoir_now),
        "reservoir_persistent": (int(counts[2, 2]), reservoir_prev, reservoir_now),
        "reservoir_to_active": (int(counts[2, 1]), reservoir_prev, active_now),
        "active_to_reservoir": (int(counts[1, 2]), active_prev, reservoir_now),
        "reservoir_to_dormant": (int(counts[2, 0]), reservoir_prev, dormant_now),
        "dormant_to_reservoir": (int(counts[0, 2]), dormant_prev, reservoir_now),
        "dormant_to_active": (int(counts[0, 1]), dormant_prev, active_now),
        "active_to_dormant": (int(counts[1, 0]), active_prev, dormant_now),
    }
    return {
        name: TransitionEntry(int(count), _fraction(int(count), source), _fraction(int(count), destination))
        for name, (count, source, destination) in entries.items()
    }


def _build_record(
    step: int,
    sizes: Mapping[str, int],
    previous_sizes: Mapping[str, int] | None,
    counts: Tensor | None,
    active_jaccard: float | None,
    reservoir_jaccard: float | None,
    width: int,
    distributions: dict[str, dict[str, float]],
) -> TrackerRecord:
    """Assemble one :class:`TrackerRecord`.

    Args:
        step: Support-update index.
        sizes: Set sizes at ``step``.
        previous_sizes: Set sizes at ``step - 1``, or ``None`` for the first update.
        counts: ``[3, 3]`` matrix for the transition, or ``None`` for the first update.
        active_jaccard: Active-set similarity with the previous update.
        reservoir_jaccard: Reservoir-set similarity with the previous update.
        width: Total number of tracked weights.
        distributions: Compact score summaries.

    Returns:
        The assembled record.
    """
    transitions = summarize_transitions(counts, previous_sizes, sizes) if counts is not None else {}
    return TrackerRecord(
        step=step,
        num_active=int(sizes["active"]),
        num_reservoir=int(sizes["reservoir"]),
        num_dormant=int(sizes["dormant"]),
        num_parameters=width,
        active_jaccard=active_jaccard,
        reservoir_jaccard=reservoir_jaccard,
        active_turnover=None if active_jaccard is None else 1.0 - active_jaccard,
        reservoir_turnover=None if reservoir_jaccard is None else 1.0 - reservoir_jaccard,
        transitions=transitions,
        distributions=distributions,
    )


@torch.no_grad()
def _records_from_masks(
    active: Tensor,
    reservoir: Tensor,
    dormant: Tensor,
    series: ScoreSeries,
    config: TrackerConfig,
) -> tuple[list[TrackerRecord], ScoreSeries]:
    """Build one record per frame from precomputed ``[T, ...]`` masks.

    Set sizes, transition matrices and Jaccard values are computed vectorized; the
    only Python loop iterates over the (short) record list.

    Args:
        active: ``[T, ...]`` active mask.
        reservoir: ``[T, ...]`` reservoir mask.
        dormant: ``[T, ...]`` dormant mask.
        series: Score series used for the optional distribution summaries.
        config: Tracker configuration.

    Returns:
        Tuple of the record list and the score series.

    Raises:
        ValueError: If the temporal length is zero.
    """
    steps = active.shape[0]
    if steps == 0:
        raise ValueError("cannot build records for an empty series")
    width = int(active.reshape(steps, -1).shape[1])

    flat_active = active.reshape(steps, width)
    flat_reservoir = reservoir.reshape(steps, width)
    flat_dormant = dormant.reshape(steps, width)

    size_table = {
        "active": flat_active.sum(-1),
        "reservoir": flat_reservoir.sum(-1),
        "dormant": flat_dormant.sum(-1),
    }

    if steps > 1:
        labels = labels_from_masks(flat_active, flat_reservoir, flat_dormant)
        matrices = transition_matrices(labels, chunk_size=config.chunk_size)
        active_jaccard_series = jaccard(flat_active[1:], flat_active[:-1])
        reservoir_jaccard_series = jaccard(flat_reservoir[1:], flat_reservoir[:-1])
    else:
        matrices = None
        active_jaccard_series = torch.zeros(0, dtype=torch.float64)
        reservoir_jaccard_series = torch.zeros(0, dtype=torch.float64)

    records: list[TrackerRecord] = []
    for step in range(steps):
        sizes = {name: int(value[step]) for name, value in size_table.items()}
        distributions = (
            _frame_distributions(series, step, config.distribution_quantiles) if config.record_distributions else {}
        )
        if step == 0 or matrices is None:
            records.append(_build_record(step, sizes, None, None, None, None, width, distributions))
            continue
        records.append(
            _build_record(
                step,
                sizes,
                {name: int(value[step - 1]) for name, value in size_table.items()},
                matrices[step - 1],
                float(active_jaccard_series[step - 1]),
                float(reservoir_jaccard_series[step - 1]),
                width,
                distributions,
            )
        )
    return records, series


def _frame_distributions(series: ScoreSeries, step: int, quantiles: tuple[float, ...]) -> dict[str, dict[str, float]]:
    """Distribution summary of one frame of a score series.

    Args:
        series: Score series.
        step: Frame index.
        quantiles: Interior quantiles to report.

    Returns:
        Compact statistics of every signal at ``step``.
    """
    scores = ActivityScores(
        pressure=series.pressure[step],
        signed=series.signed[step],
        consistency=series.consistency[step],
        motion=series.motion[step],
        active_score=series.active_score[step],
        reservoir_score=series.reservoir_score[step],
        activity=series.activity[step],
        pressure_floor=series.pressure_floor[step],
        sensitivity=None if series.sensitivity is None else series.sensitivity[step],
        active_score_source=series.active_score_source,
        step=step,
    )
    return scores.distributions(quantiles)


# ----------------------------------------------------------------------
# offline driver
# ----------------------------------------------------------------------
@torch.no_grad()
def track_sequence(
    ps: Tensor,
    gs: Tensor,
    *,
    config: TrackerConfig | None = None,
    **score_kwargs: Any,
) -> tuple[list[TrackerRecord], ScoreSeries]:
    """Score and classify a whole snapshot sequence offline.

    The sequence is scored once with vectorized EMAs, then classified with batched
    ``topk``.  Transitions and Jaccard values for every consecutive pair come from a
    single ``bincount`` and a pair of vectorized set operations, so there is no
    Python loop over time or over weights.

    Args:
        ps: Weight snapshots shaped ``[T, ...]``.
        gs: QAT gradients shaped ``[T, ...]``.
        config: Tracker configuration.
        **score_kwargs: Forwarded to
            :func:`softstairs_qat.analysis.scores.compute_score_series`, for example
            ``t_values``, ``snapshots_per_epoch``, ``schedule_offset`` or ``owner``.

    Returns:
        Tuple of the per-update :class:`TrackerRecord` list and the underlying
        :class:`ScoreSeries`.

    Raises:
        ValueError: If the snapshot counts disagree or a budget cannot be resolved.
    """
    config = config or TrackerConfig()
    series = compute_score_series(ps, gs, config=config.score, **score_kwargs)
    steps = series.num_snapshots
    width = int(series.pressure.reshape(steps, -1).shape[1])
    resolve_budgets(config.sets, width)

    block = steps if config.chunk_size is None else max(int(config.chunk_size), 1)
    active_frames: list[Tensor] = []
    reservoir_frames: list[Tensor] = []
    dormant_frames: list[Tensor] = []
    for start in range(0, steps, block):
        stop = min(start + block, steps)
        chunk = _slice_series(series, start, stop)
        active, reservoir, dormant = classify_series(chunk, config.sets, num_parameters=width)
        active_frames.append(active)
        reservoir_frames.append(reservoir)
        dormant_frames.append(dormant)

    active = torch.cat(active_frames, dim=0)
    reservoir = torch.cat(reservoir_frames, dim=0)
    dormant = torch.cat(dormant_frames, dim=0)
    return _records_from_masks(active, reservoir, dormant, series, config)


def _slice_series(series: ScoreSeries, start: int, stop: int) -> ScoreSeries:
    """Slice every tensor of a score series along the temporal axis.

    Args:
        series: Source series.
        start: First frame.
        stop: Exclusive last frame.

    Returns:
        The sliced :class:`ScoreSeries`.
    """
    return ScoreSeries(
        pressure=series.pressure[start:stop],
        signed=series.signed[start:stop],
        consistency=series.consistency[start:stop],
        motion=series.motion[start:stop],
        active_score=series.active_score[start:stop],
        reservoir_score=series.reservoir_score[start:stop],
        activity=series.activity[start:stop],
        pressure_floor=series.pressure_floor[start:stop],
        sensitivity=None if series.sensitivity is None else series.sensitivity[start:stop],
        active_score_source=series.active_score_source,
    )


# ----------------------------------------------------------------------
# streaming driver
# ----------------------------------------------------------------------
class ReservoirActiveTracker:
    """Streaming reservoir-vs-active tracker over one flat weight vector.

    Each :meth:`update` is one *support update*: the current gradient enters the EMA
    recursion, the persistent pressure estimate is ranked, and the three sets are
    updated.  Only the EMA state and the previous masks are retained, so a long run
    never builds a ``[T, N]`` history.

    The temporal behaviour matches the offline path exactly: the pressure EMA is
    ``e_k = decay * e_{k-1} + (1 - decay) |g_k|`` seeded with ``|g_0|``, which is the
    same recursion :func:`softstairs_qat.analysis.ema.ema_over_time` evaluates in
    closed form.  A single gradient spike therefore cannot redefine the reservoir
    unless it is strong relative to the accumulated history, and a persistent
    gradient accumulates until the weight is promoted into the active set.
    """

    def __init__(self, num_parameters: int, *, config: TrackerConfig | None = None) -> None:
        """Initialize the tracker.

        Args:
            num_parameters: Total number of tracked weights ``N``.
            config: Tracker configuration; dataclass defaults are used when ``None``.

        Raises:
            ValueError: If ``num_parameters`` is negative or the budgets cannot be
                resolved for ``N``.
        """
        config = config or TrackerConfig()
        if int(num_parameters) < 0:
            raise ValueError(f"num_parameters must be non-negative, got {num_parameters}")
        self.num_parameters = int(num_parameters)
        self.config = config
        self.k_active, self.k_reservoir = resolve_budgets(config.sets, self.num_parameters)
        self._dtype = _work_dtype(config.score.dtype)
        self._step = -1
        self._pressure: Tensor | None = None
        self._signed: Tensor | None = None
        self._gated = False
        self._motion: Tensor | None = None
        self._sensitivity: Tensor | None = None
        self._previous_parameter: Tensor | None = None
        self._previous_masks: dict[str, Tensor] | None = None
        self._records: list[TrackerRecord] = []

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------
    @property
    def seen(self) -> int:
        """Number of support updates folded into the EMA state."""
        return self._step + 1

    @property
    def records(self) -> list[TrackerRecord]:
        """Compact per-update statistics collected so far."""
        return list(self._records)

    @property
    def scores(self) -> ActivityScores:
        """Current score snapshot.

        Raises:
            RuntimeError: If called before the first :meth:`update`.
        """
        return self._build_scores()

    @property
    def active_mask(self) -> Tensor:
        """Flat boolean mask of the active set at the last update."""
        return self._mask(Label.ACTIVE)

    @property
    def reservoir_mask(self) -> Tensor:
        """Flat boolean mask of the reservoir set at the last update."""
        return self._mask(Label.RESERVOIR)

    @property
    def dormant_mask(self) -> Tensor:
        """Flat boolean mask of the dormant set at the last update."""
        return self._mask(Label.DORMANT)

    def mask(self, label: Label | str) -> Tensor:
        """Flat mask of one set at the last update.

        Args:
            label: A :class:`Label` or its lower-case name.

        Returns:
            Boolean tensor of shape ``[N]``.

        Raises:
            RuntimeError: If no update has been recorded yet.
            ValueError: If ``label`` is unknown.
        """
        key = label.key if isinstance(label, Label) else str(label).lower()
        try:
            resolved = Label[key.upper()]
        except KeyError as error:
            raise ValueError(f"unknown set {label!r}; expected one of {[item.key for item in Label]}") from error
        return self._mask(resolved)

    def _mask(self, label: Label) -> Tensor:
        """Return the cached flat mask of ``label``.

        Args:
            label: Target label.

        Returns:
            Boolean tensor of shape ``[N]``.

        Raises:
            RuntimeError: If no update has been recorded yet.
        """
        if self._previous_masks is None:
            raise RuntimeError(f"{label.key}_mask is unavailable before the first update")
        return self._previous_masks[label.key]

    def reset(self) -> None:
        """Drop the EMA state, the previous sets and the collected records."""
        self._step = -1
        self._pressure = None
        self._signed = None
        self._gated = False
        self._motion = None
        self._sensitivity = None
        self._previous_parameter = None
        self._previous_masks = None
        self._records.clear()

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(
        self,
        *,
        gradient: Tensor | None = None,
        parameter: Tensor | None = None,
        upstream_gradient: Tensor | None = None,
        sensitivity: Tensor | None = None,
        pressure: Tensor | None = None,
        t: float | None = None,
        step: int | None = None,
    ) -> TrackerRecord:
        """Fold one support update into the EMA state and reclassify the sets.

        Args:
            gradient: Current QAT gradient ``g`` (no temporal axis), shaped ``[N]``.
                Ignored when ``pressure`` or ``upstream_gradient`` is given.
            parameter: Current weight snapshot ``p``, shaped ``[N]``.  Enables the
                motion EMA and, together with ``t``, the SoftStairs sensitivity.
            upstream_gradient: Optional upstream gradient ``H``.  Combined with
                ``sensitivity`` (or with ``parameter`` *and* ``t``) the active score
                becomes the literal ``EMA(|H| * D)``.
            sensitivity: Optional precomputed SoftStairs derivative ``D``, shaped
                ``[N]``.
            pressure: Optional precomputed instantaneous QAT pressure ``|g|``.
            t: Live SoftStairs temperature, used to evaluate ``dSS`` from ``parameter``.
                Required when the active score should be the literal ``EMA(|H| * D)``
                and no ``sensitivity`` is supplied.
            step: Support-update index; defaults to an internal counter.

        Returns:
            The :class:`TrackerRecord` describing this update.

        Raises:
            ValueError: If no pressure source is given, several are given, an input has
                the wrong shape, or the upstream gradient has no way to obtain ``D``.
        """
        config = self.config.score
        sources = [
            name
            for name, value in (
                ("gradient", gradient),
                ("upstream_gradient", upstream_gradient),
                ("pressure", pressure),
            )
            if value is not None
        ]
        if len(sources) > 1:
            raise ValueError(f"pass only one of gradient/upstream_gradient/pressure, got {sources}")
        if sensitivity is not None and parameter is None:
            raise ValueError("sensitivity requires the weight snapshot it was evaluated on")
        if upstream_gradient is not None and sensitivity is None and (parameter is None or t is None):
            raise ValueError(
                "upstream_gradient needs dSS: pass parameter together with t, or pass sensitivity explicitly"
            )

        source = gradient if gradient is not None else upstream_gradient if upstream_gradient is not None else pressure
        if source is None:
            raise ValueError("update requires gradient, pressure, or upstream_gradient")
        if self.num_parameters and source.numel() != self.num_parameters:
            raise ValueError(f"expected {self.num_parameters} weights, got {source.numel()}")

        raw = source.detach().reshape(-1).to(dtype=self._dtype)

        self._step = self._step + 1 if step is None else int(step)
        current_step = self._step

        if sensitivity is not None:
            self._sensitivity = sensitivity.detach().reshape(-1).to(dtype=self._dtype)
        elif parameter is not None and config.use_sensitivity and t is not None:
            self._sensitivity = _single_sensitivity(parameter.detach().reshape(-1).to(dtype=self._dtype), config, t)
        else:
            self._sensitivity = None

        # Rebuild the QAT gradient from its two factors when the upstream gradient is
        # available: g = H * D.  Because D >= 0, |g| == |H| * D, so the pressure EMA of
        # the gated magnitude is exactly the EMA(|H| * D) of the design; when the caller
        # already passes the gated gradient, ``raw`` *is* g and the two routes agree.
        gated = upstream_gradient is not None and self._sensitivity is not None
        self._gated = gated
        observation = raw * self._sensitivity if gated else raw
        instantaneous = observation.abs()

        self._signed = _ema_step(self._signed, observation, config.decay_consistency)
        self._pressure = _ema_step(self._pressure, instantaneous, config.decay_pressure)
        self._motion = self._motion_step(parameter, config)
        return self._classify_and_record(self._build_scores(), current_step)

    def _motion_step(self, parameter: Tensor | None, config: ScoreConfig) -> Tensor | None:
        """Advance the weight-motion EMA by one step.

        Args:
            parameter: Current weight snapshot, or ``None`` to leave the state frozen.
            config: Score configuration providing the decay and dtype.

        Returns:
            The updated motion EMA, or ``None`` when no snapshot was ever supplied.
        """
        if parameter is None:
            return self._motion
        current = parameter.detach().reshape(-1).to(dtype=self._dtype)
        if self._previous_parameter is None:
            displacement = torch.zeros_like(current)
        else:
            displacement = (current - self._previous_parameter).abs()
        self._previous_parameter = current
        self._motion = (
            displacement
            if self._motion is None
            else float(config.decay_motion) * self._motion + (1.0 - float(config.decay_motion)) * displacement
        )
        return self._motion

    def _build_scores(self) -> ActivityScores:
        """Assemble the current :class:`ActivityScores` from the EMA state.

        Returns:
            The current score snapshot.

        Raises:
            RuntimeError: If called before the first :meth:`update`.
        """
        config = self.config.score
        pressure = self._pressure
        signed = self._signed
        if pressure is None or signed is None:
            raise RuntimeError("scores are unavailable before the first update")
        # S_active = EMA(|H| * D).  With D >= 0 that equals EMA(|g|), which is the
        # pressure EMA in both routes: here it is EMA of the gated magnitude when the
        # upstream gradient was supplied, and EMA(|g|) of the stored gradient otherwise.
        active = pressure

        consistency = (signed.abs() / (pressure + config.eps)).clamp(0.0, 1.0)
        activity = _flat_normalize(active, config)
        return ActivityScores(
            pressure=pressure,
            signed=signed,
            consistency=consistency,
            motion=self._motion if self._motion is not None else torch.zeros_like(pressure),
            active_score=active,
            reservoir_score=pressure * consistency * (1.0 - activity),
            activity=activity,
            pressure_floor=_flat_quantile(pressure, config.pressure_quantile),
            sensitivity=self._sensitivity,
            active_score_source="upstream" if self._gated else "gradient",
            step=self._step,
        )

    def _classify_and_record(self, scores: ActivityScores, step: int) -> TrackerRecord:
        """Rank the scores into three disjoint sets and build the record.

        Args:
            scores: Current score snapshot.
            step: Support-update index.

        Returns:
            The assembled :class:`TrackerRecord`.
        """
        sets = self.config.sets
        ranking = scores.active_score
        if float(sets.active_hysteresis) > 0.0 and self._previous_masks is not None:
            ranking = ranking * (
                1.0 + float(sets.active_hysteresis) * self._previous_masks[Label.ACTIVE.key].to(ranking.dtype)
            )

        active = _topk_mask(ranking, self.k_active)
        eligible = (scores.pressure >= scores.pressure_floor) & ~active
        reservoir = _topk_mask(scores.reservoir_score.masked_fill(~eligible, _NEG_INF), self.k_reservoir) & eligible
        dormant = ~(active | reservoir)

        active_flat = active.reshape(-1)
        reservoir_flat = reservoir.reshape(-1)
        dormant_flat = dormant.reshape(-1)
        sizes = {
            "active": int(active_flat.sum()),
            "reservoir": int(reservoir_flat.sum()),
            "dormant": int(dormant_flat.sum()),
        }

        previous_sizes: dict[str, int] | None = None
        counts: Tensor | None = None
        active_jaccard: float | None = None
        reservoir_jaccard: float | None = None
        if self._previous_masks is not None:
            previous = self._previous_masks
            previous_sizes = {name: int(mask.sum()) for name, mask in previous.items()}
            current_labels = labels_from_masks(active_flat, reservoir_flat, dormant_flat)[None]
            previous_labels = labels_from_masks(
                previous[Label.ACTIVE.key], previous[Label.RESERVOIR.key], previous[Label.DORMANT.key]
            )[None]
            counts = _pair_matrix(current_labels, previous_labels)
            active_jaccard = float(jaccard(active_flat, previous[Label.ACTIVE.key]))
            reservoir_jaccard = float(jaccard(reservoir_flat, previous[Label.RESERVOIR.key]))

        record = _build_record(
            step,
            sizes,
            previous_sizes,
            counts,
            active_jaccard,
            reservoir_jaccard,
            self.num_parameters,
            scores.distributions(self.config.distribution_quantiles) if self.config.record_distributions else {},
        )
        self._previous_masks = {
            Label.ACTIVE.key: active_flat,
            Label.RESERVOIR.key: reservoir_flat,
            Label.DORMANT.key: dormant_flat,
        }
        self._records.append(record)
        return record