# softstairs_qat/analysis/scores.py

"""Temporal (EMA) pressure, consistency, motion and QAT activity scores.

The tracker must not classify weights from a single gradient.  This module
builds, for a stack of snapshots laid out as ``ps.shape = [T, ...]`` and
``gs.shape = [T, ...]``, the exponentially weighted signals that separate an
*active* weight from a *reservoir* weight:

===================  ===============================================  ==========================================
symbol              definition                                      interpretation
===================  ===============================================  ==========================================
``P`` (pressure)     ``EMA_t(|g|)``                                   persistent optimization pressure
``M`` (signed)       ``EMA_t(g)``                                     persistent signed push
``C`` (consistency)  ``|M| / (P + eps)`` in ``[0, 1]``                is the push directional or cancelling?
``V`` (motion)       ``EMA_t(|p_t - p_{t-1}|)``                       how actively the weight moves
``S_active``         ``EMA_t(|H| * D)``, see below                    persistent pressure at the quantizer
``S_reservoir``      ``P * C * (1 - activity)``                       persistent pressure, not sensitive now
===================  ===============================================  ==========================================

Relation to the SoftStairs QAT gradient
---------------------------------------
During QAT the weight gradient reaching the trainable leaf is the gated
gradient ``g = H * D``, where ``H`` is the upstream gradient and ``D`` is the
SoftStairs derivative (see :meth:`softstairs_qat.core.soft_stairs.SoftStairs.derivative`,
the very tensor used by ``SoftStairsQuantizeFunction.backward``).  The
``GradSaver`` callback in ``experiments/CV/inception_stl10.ipynb`` stores that
already-gated gradient, i.e. ``gs`` *is* ``H * D``.

Consequently ``|H| * D = |H * D| = |g|`` because ``D >= 0``, so the two
expressions the design allows are algebraically identical:

.. code-block:: text

    EMA(|H| * D) == EMA(|g|)

:func:`compute_score_series` therefore defaults to ``EMA(|g|)`` and never
re-derives a second, competing approximation of ``dSS``.  When a live training
loop can hand over the *upstream* gradient ``H`` (for example a forward hook
that reads the incoming gradient before the SoftStairs autograd function
applies ``D``), pass it as ``upstream_gradients`` and the literal
``EMA(|H| * D)`` product is formed instead.  ``D`` itself is obtained only via
:func:`softstairs_qat.analysis.sensitivity.resolve_sensitivity`, which prefers a
sensitivity published by a live quantizer and otherwise calls
``SoftStairs.derivative`` on the code-space weight snapshot with the correct
per-snapshot temperature.

A weight reaches a high active score only when *both* factors of ``g = H * D``
are large: sustained upstream pressure **and** current quantization
sensitivity.  That is exactly the definition of an active weight, and it is why
the active score is sensitive to the quantization transition rather than to
gradient magnitude alone.

Reservoir score
---------------
``S_reservoir = P * C * (1 - activity)``.  Note ``P * C == |M|``, so the score is
the persistent *signed* momentum magnitude discounted by how strongly the
weight currently sits at the quantization transition.  Two properties follow
directly and are relied upon by the classifier:

* a weight with negligible pressure has ``M ~ 0`` and therefore scores ``~0``
  -- it can never be ranked into the reservoir just for being inactive;
* a weight that is oscillating has ``|M| << P``, i.e. ``C ~ 0``, and is
  suppressed even when its instantaneous ``|g|`` is large.

``activity`` is the normalized, scale-free version of ``S_active`` used to
suppress currently sensitive weights.  Its normalization is deliberately
adaptive rather than an absolute threshold, so the score stays meaningful as
the gradient scale decays over training: with ``activity_quantile = q``,

.. code-block:: text

    lo_t = min_i S_active[t, i]
    hi_t = Quantile_q( S_active[t, ...] )
    activity[t, i] = clip( (S_active[t, i] - lo_t) / max(hi_t - lo_t, activity_eps), 0, 1 )

The denominator is clamped, so the mapping is safe when every weight has the
same score.  Bounds are taken per time step from that step's own distribution,
which keeps the offline sequence causal: snapshot ``t`` never depends on
snapshots after ``t``.

Temporal behaviour
------------------
Every EMA is evaluated in closed form over ``dim=0`` (see
:func:`softstairs_qat.analysis.ema.ema_over_time`), so there is no Python loop
over time and no ``[T, num_parameters]`` Python container.  Weight displacement
is zero-padded at ``t = 0`` so that every signal shares the ``[T, ...]`` shape
of the snapshots and can be ranked frame by frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Sequence

import torch
from torch import Tensor

from softstairs_qat.analysis.ema import ema_over_time
from softstairs_qat.analysis.sensitivity import resolve_sensitivity

__all__ = [
    "ScoreConfig",
    "ScoreSeries",
    "ActivityScores",
    "normalize_activity",
    "quantile_last_dim",
    "compute_score_series",
    "finalize_scores",
    "compute_scores",
]

#: ``torch.quantile`` refuses very large inputs, so the sorted fallback is used above this size.
_QUANTILE_ELEMENT_LIMIT = 1 << 24

_ACTIVE_SCORE_GRADIENT = "gradient"
_ACTIVE_SCORE_UPSTREAM = "upstream"
_ACTIVE_SCORE_SENSITIVITY = "sensitivity"

_LOW_PRECISION = (torch.float16, torch.bfloat16)


def _work_dtype(dtype: torch.dtype | None) -> torch.dtype:
    """Return the accumulation dtype for score inputs.

    Args:
        dtype: Requested dtype, or ``None``.

    Returns:
        ``torch.float32`` for half/bfloat16 inputs and for ``None``, otherwise ``dtype``.
    """
    if dtype is None or dtype in _LOW_PRECISION:
        return torch.float32
    return dtype


def quantile_last_dim(values: Tensor, q: float) -> Tensor:
    """Compute one quantile per row along the last dimension.

    ``torch.quantile`` has a hard element limit; this wrapper falls back to a
    full sort so the tracker keeps working on large layers.

    Args:
        values: Tensor to reduce.
        q: Quantile in ``[0, 1]``.

    Returns:
        Tensor with the same shape as ``values`` except the last dimension, which is kept.

    Raises:
        ValueError: If ``q`` is outside ``[0, 1]`` or the last dimension is empty.
    """
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError(f"q must lie in [0, 1], got {q}")
    width = values.shape[-1]
    if width == 0:
        raise ValueError("cannot take a quantile of an empty axis")
    index = min(max(math.ceil(float(q) * width) - 1, 0), width - 1)
    if values.numel() <= _QUANTILE_ELEMENT_LIMIT:
        return torch.quantile(values, float(q), dim=-1, keepdim=True)
    ordered, _ = torch.sort(values, dim=-1)
    return ordered[..., index : index + 1]


@dataclass(frozen=True)
class ScoreConfig:
    """Knobs of the temporal score computation.

    Attributes:
        decay_pressure: Decay of ``EMA(|g|)``.
        decay_consistency: Decay of the signed ``EMA(g)``.
        decay_motion: Decay of ``EMA(|p_t - p_{t-1}|)``.
        eps: Stabilizer of the consistency ratio ``|M| / (P + eps)``.
        activity_quantile: Quantile of ``S_active`` used as the upper bound of the
            normalized activity, in ``[0, 1]``.
        activity_eps: Stabilizer of the activity normalization denominator.
        pressure_quantile: Quantile of ``P`` used as the eligibility floor for the
            reservoir set.  ``0.0`` (default) admits every weight; raise it to
            ``0.3`` to prune the negligible-pressure tail before ranking.
        use_sensitivity: Whether to consult the SoftStairs derivative at all.  Set
            to ``False`` to score purely from ``|g|``.
        normalized: Forward for ``SoftStairs.normalized``.
        async_t_factor: Forward for ``SoftStairs.async_t_factor``; this is the factor
            applied in the backward pass, see ``SoftStairsQuantizer.current_backward_t``.
        dtype: Accumulation dtype of every score.
    """

    decay_pressure: float = 0.9
    decay_consistency: float = 0.9
    decay_motion: float = 0.9
    eps: float = 1e-12
    activity_quantile: float = 0.9
    activity_eps: float = 1e-12
    pressure_quantile: float = 0.0
    use_sensitivity: bool = True
    normalized: bool = False
    async_t_factor: float = 1.0
    dtype: torch.dtype = torch.float32

    def replace(self, **changes: Any) -> "ScoreConfig":
        """Return a copy of this config with ``changes`` applied.

        Args:
            **changes: Field overrides.

        Returns:
            The updated :class:`ScoreConfig`.
        """
        return replace(self, **changes)


@dataclass(frozen=True)
class ScoreSeries:
    """Per-snapshot score series, all shaped ``[T, ...]``.

    Attributes:
        pressure: ``EMA(|g|)``.
        signed: ``EMA(g)``.
        consistency: ``|signed| / (pressure + eps)`` clipped to ``[0, 1]``.
        motion: ``EMA(|p_t - p_{t-1}|)``.
        active_score: ``EMA(|H| * D)``; equal to ``EMA(|g|)`` unless the upstream
            gradient is supplied explicitly.
        reservoir_score: ``pressure * consistency * (1 - activity)``.
        activity: Normalized active score in ``[0, 1]``.
        pressure_floor: Per-snapshot scalar eligibility floor of the reservoir set.
        sensitivity: SoftStairs derivative ``D`` per snapshot, or ``None`` when it was
            neither exposed by the quantizer nor recomputed.
        active_score_source: Which expression produced :attr:`active_score`, one of
            ``"gradient"``, ``"upstream"`` or ``"sensitivity"``.
    """

    pressure: Tensor
    signed: Tensor
    consistency: Tensor
    motion: Tensor
    active_score: Tensor
    reservoir_score: Tensor
    activity: Tensor
    pressure_floor: Tensor
    sensitivity: Tensor | None = None
    active_score_source: str = _ACTIVE_SCORE_GRADIENT

    @property
    def num_snapshots(self) -> int:
        """Temporal length ``T`` of the series."""
        return int(self.pressure.shape[0])

    def final(self) -> "ActivityScores":
        """Extract the last snapshot of every series.

        Returns:
            The :class:`ActivityScores` describing the current support update.
        """
        last = self.num_snapshots - 1
        return ActivityScores(
            pressure=self.pressure[last],
            signed=self.signed[last],
            consistency=self.consistency[last],
            motion=self.motion[last],
            active_score=self.active_score[last],
            reservoir_score=self.reservoir_score[last],
            activity=self.activity[last],
            pressure_floor=self.pressure_floor[last],
            sensitivity=None if self.sensitivity is None else self.sensitivity[last],
            active_score_source=self.active_score_source,
            step=last,
        )

    def __repr__(self) -> str:
        return (
            f"ScoreSeries(num_snapshots={self.num_snapshots}, weight_shape={tuple(self.pressure.shape[1:])}, "
            f"active_score_source={self.active_score_source!r})"
        )


@dataclass(frozen=True)
class ActivityScores:
    """Current-snapshot view of :class:`ScoreSeries`, shaped like one weight tensor.

    Attributes:
        pressure: ``EMA(|g|)``.
        signed: ``EMA(g)``.
        consistency: ``|signed| / (pressure + eps)``.
        motion: ``EMA(|p_t - p_{t-1}|)``.
        active_score: Ranking key of the active set.
        reservoir_score: Ranking key of the reservoir set.
        activity: Normalized active score in ``[0, 1]``.
        pressure_floor: Eligibility floor of the reservoir set.
        sensitivity: SoftStairs derivative ``D``, or ``None``.
        active_score_source: Expression used for :attr:`active_score`.
        step: Index of the snapshot inside the series.
    """

    pressure: Tensor
    signed: Tensor
    consistency: Tensor
    motion: Tensor
    active_score: Tensor
    reservoir_score: Tensor
    activity: Tensor
    pressure_floor: Tensor
    sensitivity: Tensor | None = None
    active_score_source: str = _ACTIVE_SCORE_GRADIENT
    step: int = -1

    def distributions(self, quantiles: Sequence[float] = (0.05, 0.5, 0.95)) -> dict[str, dict[str, float]]:
        """Summarize every scalar signal with a compact set of statistics.

        Only order statistics are reported, so logging a support update never
        retains a ``[num_parameters]`` history.

        Args:
            quantiles: Interior quantiles to report, in ``[0, 1]``.

        Returns:
            Mapping from signal name to ``{"mean", "q00", ..., "q100", "max", "min"}``.
        """
        summary: dict[str, dict[str, float]] = {}
        signals = {
            "pressure": self.pressure,
            "consistency": self.consistency,
            "motion": self.motion,
            "active_score": self.active_score,
            "reservoir_score": self.reservoir_score,
            "activity": self.activity,
        }
        if self.sensitivity is not None:
            signals["sensitivity"] = self.sensitivity
        for name, signal in signals.items():
            flat = signal.detach().reshape(-1).float()
            stats = {
                "mean": float(flat.mean()) if flat.numel() else 0.0,
                "min": float(flat.min()) if flat.numel() else 0.0,
                "max": float(flat.max()) if flat.numel() else 0.0,
}
            for q in quantiles:
                value = quantile_last_dim(flat.reshape(1, -1), float(q)).reshape(())
                stats[f"q{int(round(float(q) * 100)):02d}"] = float(value)
            summary[name] = stats
        return summary


def normalize_activity(scores: Tensor, *, quantile: float = 0.9, eps: float = 1e-12) -> Tensor:
    """Map a raw score to ``[0, 1]`` using adaptive per-row quantile bounds.

    A degenerate row -- every weight carrying the same score -- has ``lo == hi``, so
    the clamped denominator makes the result ``0.0`` for every entry.  That is the
    consistent extension of a min-max normalization: without any spread there is no
    weight that stands out as *more* active than the rest, and the reservoir score
    keeps its full ``(1 - activity) == 1`` factor instead of being arbitrarily
    suppressed.

    Args:
        scores: Score tensor of any shape; the last dimension is the weight axis.
        quantile: Upper bound of the normalization, taken as this quantile of the row.
        eps: Stabilizer of the denominator.

    Returns:
        Tensor with the shape of ``scores`` and values in ``[0, 1]``.
    """
    low = scores.amin(dim=-1, keepdim=True)
    high = quantile_last_dim(scores, quantile)
    span = (high - low).clamp_min(eps)
    return ((scores - low) / span).clamp(0.0, 1.0)


def _zero_padded_displacement(ps: Tensor) -> Tensor:
    """Build ``|p_t - p_{t-1}|`` with a zero first frame.

    Args:
        ps: Weight snapshots shaped ``[T, ...]``.

    Returns:
        Tensor with the shape of ``ps``.

    Raises:
        ValueError: If the temporal axis is empty.
    """
    if ps.shape[0] == 0:
        raise ValueError("ps must contain at least one snapshot")
    displacement = torch.zeros_like(ps)
    if ps.shape[0] > 1:
        displacement[1:] = (ps[1:] - ps[:-1]).abs()
    return displacement


def _resolve_series_sensitivity(
    ps: Tensor,
    config: ScoreConfig,
    t_values: Sequence[float] | float | Tensor | None,
    snapshots_per_epoch: int | None,
    schedule_offset: int,
    upstream_gradients: Tensor | None,
    owner: Any,
    name: str | None,
) -> tuple[Tensor | None, str]:
    """Determine ``D`` and the expression used for the active score.

    Args:
        ps: Weight snapshots in quantization code space, shaped ``[T, ...]``.
        config: Score configuration.
        t_values: Temperature per snapshot, or a full ``t`` schedule.
        snapshots_per_epoch: Snapshots recorded per epoch ``P``.
        schedule_offset: Index of the schedule entry used by snapshot ``0``.
        upstream_gradients: Optional upstream gradient ``H``, shaped ``[T, ...]``.
        owner: Optional quantizer or model probed for an already exposed ``dSS``.
        name: Optional parameter name for a named sensitivity mapping.

    Returns:
        Tuple of the sensitivity series (``None`` when unavailable) and the
        active-score expression tag.
    """
    if upstream_gradients is not None:
        if not config.use_sensitivity:
            raise ValueError("upstream_gradients requires config.use_sensitivity=True")
        dss = resolve_sensitivity(
            ps,
            owner=owner,
            t_values=t_values,
            name=name,
            use_sensitivity=True,
            normalized=config.normalized,
            async_t_factor=config.async_t_factor,
            snapshots_per_epoch=snapshots_per_epoch,
            schedule_offset=schedule_offset,
        )
        return dss, _ACTIVE_SCORE_UPSTREAM

    if config.use_sensitivity:
        dss = resolve_sensitivity(
            ps,
            owner=owner,
            t_values=t_values,
            name=name,
            use_sensitivity=True,
            normalized=config.normalized,
            async_t_factor=config.async_t_factor,
            snapshots_per_epoch=snapshots_per_epoch,
            schedule_offset=schedule_offset,
        )
        return dss, _ACTIVE_SCORE_SENSITIVITY

    return None, _ACTIVE_SCORE_GRADIENT


@torch.no_grad()
def compute_score_series(
    ps: Tensor,
    gs: Tensor,
    *,
    config: ScoreConfig | None = None,
    t_values: Sequence[float] | float | Tensor | None = None,
    snapshots_per_epoch: int | None = None,
    schedule_offset: int = 0,
    upstream_gradients: Tensor | None = None,
    sensitivity: Tensor | None = None,
    owner: Any = None,
    name: str | None = None,
    pressure_override: Tensor | None = None,
) -> ScoreSeries:
    """Build every temporal score for a stack of snapshots.

    All EMAs are closed-form and vectorized over ``dim=0``; only the per-frame
    reductions (``amin``, single quantile, ``topk`` in the classifier) touch time.

    Args:
        ps: Weight snapshots, shaped ``[T, ...]``.
        gs: QAT gradients ``g = H * D``, shaped ``[T, ...]``.
        config: Score configuration; the dataclass defaults are used when ``None``.
        t_values: Temperature per snapshot, a scalar, a full ``t`` schedule, or ``None``.
        snapshots_per_epoch: Snapshots recorded per epoch ``P``.  Required when
            ``t_values`` is a full schedule, because ``t`` is stepped once per epoch.
        schedule_offset: Index of the schedule entry used by snapshot ``0``.
        upstream_gradients: Optional upstream gradient ``H`` used to build the literal
            ``EMA(|H| * D)``.
        sensitivity: Optional precomputed ``D`` series.  Bypasses quantizer probing.
        owner: Optional quantizer or model probed for an exposed ``dSS``.
        name: Optional parameter name for a named sensitivity mapping.
        pressure_override: Optional precomputed instantaneous QAT pressure series that
            replaces ``|g|`` in the pressure EMA.  Used when the caller already knows
            the gated magnitude.

    Returns:
        The :class:`ScoreSeries`, all elements shaped ``[T, ...]``.

    Raises:
        ValueError: If ``ps`` and ``gs`` disagree on ``T`` or an input is empty.
    """
    config = config or ScoreConfig()
    if not torch.is_tensor(ps) or not torch.is_tensor(gs):
        raise TypeError("ps and gs must be tensors")
    if ps.shape[0] != gs.shape[0]:
        raise ValueError(f"ps has {ps.shape[0]} snapshots but gs has {gs.shape[0]}")
    if ps.shape[0] == 0:
        raise ValueError("ps and gs must contain at least one snapshot")

    dtype = _work_dtype(config.dtype)
    weights = ps.detach().to(dtype=dtype)
    grads = gs.detach().to(dtype=dtype)

    instantaneous = grads.abs() if pressure_override is None else pressure_override.detach().to(dtype=dtype)
    pressure = ema_over_time(instantaneous, config.decay_pressure)
    signed = ema_over_time(grads, config.decay_consistency)
    consistency = (signed.abs() / (pressure + config.eps)).clamp(0.0, 1.0)
    motion = ema_over_time(_zero_padded_displacement(weights), config.decay_motion)

    if upstream_gradients is not None:
        upstream = upstream_gradients.detach().to(dtype=dtype)
        if upstream.shape != weights.shape:
            raise ValueError(f"upstream_gradients has shape {tuple(upstream.shape)}, expected {tuple(weights.shape)}")

    dss: Tensor | None
    if sensitivity is not None:
        dss = sensitivity.detach().to(dtype=dtype)
        if dss.shape != weights.shape:
            raise ValueError(f"sensitivity has shape {tuple(dss.shape)}, expected {tuple(weights.shape)}")
        source = _ACTIVE_SCORE_UPSTREAM if upstream_gradients is not None else _ACTIVE_SCORE_SENSITIVITY
    elif upstream_gradients is not None or config.use_sensitivity:
        dss, source = _resolve_series_sensitivity(
            weights,
            config,
            t_values,
            snapshots_per_epoch,
            schedule_offset,
            upstream_gradients,
            owner,
            name,
        )
    else:
        dss, source = None, _ACTIVE_SCORE_GRADIENT

    if upstream_gradients is not None and dss is not None:
        # Literal EMA(|H| * D): the gated gradient rebuilt from its two factors.
        gated = upstream.abs() * dss
    else:
        # |H| * D == |H * D| == |g| for D >= 0, so EMA(|g|) is the same estimator.
        gated = instantaneous

    active_score = ema_over_time(gated, config.decay_pressure)
    activity = normalize_activity(active_score, quantile=config.activity_quantile, eps=config.activity_eps)
    reservoir_score = pressure * consistency * (1.0 - activity)
    pressure_floor = quantile_last_dim(pressure.reshape(pressure.shape[0], -1), config.pressure_quantile).squeeze(-1)

    return ScoreSeries(
        pressure=pressure,
        signed=signed,
        consistency=consistency,
        motion=motion,
        active_score=active_score,
        reservoir_score=reservoir_score,
        activity=activity,
        pressure_floor=pressure_floor,
        sensitivity=dss,
        active_score_source=source,
    )


def finalize_scores(series: ScoreSeries) -> ActivityScores:
    """Reduce a :class:`ScoreSeries` to its last snapshot.

    Args:
        series: Series produced by :func:`compute_score_series`.

    Returns:
        The current-snapshot :class:`ActivityScores`.
    """
    return series.final()


def compute_scores(
    ps: Tensor,
    gs: Tensor,
    *,
    config: ScoreConfig | None = None,
    **kwargs: Any,
) -> ActivityScores:
    """Convenience wrapper scoring the latest snapshot only.

    Args:
        ps: Weight snapshots, shaped ``[T, ...]``.
        gs: QAT gradients, shaped ``[T, ...]``.
        config: Score configuration.
        **kwargs: Forwarded to :func:`compute_score_series`.

    Returns:
        The :class:`ActivityScores` of the final snapshot.
    """
    return compute_score_series(ps, gs, config=config, **kwargs).final()