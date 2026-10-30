# softstairs_qat/analysis/paramset.py

"""Multi-tensor support sets for a whole model.

:class:`ParameterSetTracker` flattens an arbitrary set of trainable parameter
tensors into one logical weight vector, scores and ranks that vector in a single
:class:`softstairs_qat.analysis.tracking.ReservoirActiveTracker`, and maps the
resulting flat masks back onto the original tensors with
:meth:`softstairs_qat.analysis.flattening.ParameterLayout.unflatten_masks`.

The flat index bookkeeping is stored once as contiguous offset ranges, so the
mapping ``flat index -> parameter tensor -> local index`` costs no allocation at
run time and the per-parameter masks are views rather than copies.

Usage from a live QAT run
-------------------------
A quantized model exposes the trainable leaves as ``<name>_orig`` while ``<name>``
becomes a buffer, so the natural selection is

.. code-block:: python

    tracker = ParameterSetTracker(model, config=TrackerConfig(
        score=ScoreConfig(decay_pressure=0.9),
        sets=SetConfig(active_fraction=0.01, reservoir_fraction=0.25),
    ))
    # one support update
    record = tracker.update(
        step=global_step,
        gradients={name: param.grad for name, param in model.named_parameters()},
        parameters={name: param for name, param in model.named_parameters()},
    )
    masks = tracker.masks()  # {'active': {name: bool tensor}, 'reservoir': ..., 'dormant': ...}

Usage from saved snapshots
--------------------------
A run that saved ``[T, ...]`` snapshots per parameter is handled by
:meth:`update_sequence`, which feeds the frames one by one through the same
streaming tracker.  This keeps peak memory at ``O(N)`` instead of the
``O(T * N)`` a stacked offline pass would need, and produces exactly the records
:func:`softstairs_qat.analysis.tracking.track_sequence` produces for a single
tensor.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from softstairs_qat.analysis.flattening import ParameterLayout, flatten_tensors, select_parameters
from softstairs_qat.analysis.scores import ScoreConfig
from softstairs_qat.analysis.tracking import (
    Label,
    ReservoirActiveTracker,
    SetConfig,
    TrackerConfig,
    TrackerRecord,
)

__all__ = ["ParameterSetTracker", "build_tracker", "summarize"]


class ParameterSetTracker:
    """Reservoir / active / dormant sets spanning several parameter tensors.

    Attributes are read-only views of the underlying flat state; use
    :meth:`masks` for per-tensor masks and :meth:`counts` for per-tensor counts.

    Example:
        >>> model = torch.nn.Sequential(torch.nn.Linear(4, 3))  # doctest: +SKIP
        >>> tracker = ParameterSetTracker(model, config=TrackerConfig(
        ...     sets=SetConfig(active_k=2, reservoir_k=5)))  # doctest: +SKIP
        >>> tracker.layout.num_parameters  # doctest: +SKIP
        15
    """

    def __init__(
        self,
        parameters: torch.nn.Module | Mapping[str, Tensor] | Sequence[Tensor] | Iterable[Tensor],
        *,
        config: TrackerConfig | None = None,
        layout: ParameterLayout | None = None,
        trainable_only: bool = True,
        exclude_suffix: str | None = None,
    ) -> None:
        """Bind the tracker to a set of parameters or a whole model.

        Args:
            parameters: A model to select trainable parameters from, or an explicit
                mapping/sequence of named tensors.
            config: Tracker configuration.
            layout: Optional precomputed layout; required when ``parameters`` is a
                generator, otherwise derived from ``parameters``.
            trainable_only: For a model, skip parameters with ``requires_grad=False``.
            exclude_suffix: For a model, skip parameter names ending with this suffix.

        Raises:
            ValueError: If the parameter set is empty or a generator was passed
                without a ``layout``.
        """
        config = config or TrackerConfig()
        if isinstance(parameters, torch.nn.Module):
            selected = select_parameters(
                parameters,
                trainable_only=trainable_only,
                exclude_suffix=exclude_suffix,
            )
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
        self._tracker = ReservoirActiveTracker(layout.num_parameters, config=config)
        self._reference: dict[str, Tensor] | None = None
        if isinstance(source, Mapping):
            self._reference = dict(source)

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------
    @property
    def tracker(self) -> ReservoirActiveTracker:
        """Underlying single-vector streaming tracker."""
        return self._tracker

    @property
    def num_parameters(self) -> int:
        """Total number of tracked scalar weights ``N``."""
        return self.layout.num_parameters

    @property
    def records(self) -> list[TrackerRecord]:
        """Compact per-update statistics collected so far."""
        return self._tracker.records

    @property
    def scores(self) -> Any:
        """Current flat score snapshot.

        Raises:
            RuntimeError: If called before the first update.
        """
        return self._tracker.scores

    def flat_masks(self) -> dict[str, Tensor]:
        """Flat boolean masks of the three sets at the last update.

        Returns:
            Mapping with the keys ``"active"``, ``"reservoir"`` and ``"dormant"``,
            each of shape ``[N]``.

        Raises:
            RuntimeError: If called before the first update.
        """
        return {
            "active": self._tracker.active_mask,
            "reservoir": self._tracker.reservoir_mask,
            "dormant": self._tracker.dormant_mask,
        }

    def masks(self, label: Label | str | None = None) -> dict[str, dict[str, Tensor]]:
        """Map the flat masks back onto the original parameter shapes.

        Args:
            label: Restrict the result to one set; ``None`` returns all three.

        Returns:
            ``{'active': {name: tensor}, 'reservoir': {...}, 'dormant': {...}}`` when
            ``label`` is ``None``, otherwise ``{name: tensor}`` for that set alone.

        Raises:
            RuntimeError: If called before the first update.
            ValueError: If ``label`` is unknown.
        """
        flat = self.flat_masks()
        if label is None:
            return {name: self.layout.unflatten_masks(mask) for name, mask in flat.items()}
        key = label.key if isinstance(label, Label) else str(label).lower()
        if key not in flat:
            raise ValueError(f"unknown set {label!r}")
        return self.layout.unflatten_masks(flat[key])

    def counts(self) -> dict[str, dict[str, int]]:
        """Per-parameter set sizes at the last update.

        Returns:
            Mapping from set name to a mapping from parameter name to count, plus the
            totals under the key ``"__total__"``.
        """
        result: dict[str, dict[str, int]] = {}
        for name, flat in self.flat_masks().items():
            per_tensor = self.layout.per_parameter_counts(flat)
            per_tensor["__total__"] = int(flat.sum())
            result[name] = per_tensor
        return result

    def history(self, *, flat_transitions: bool = True) -> dict[str, list[Any]]:
        """Column-oriented history of every collected record.

        Args:
            flat_transitions: Forwarded to
                :meth:`softstairs_qat.analysis.tracking.TrackerRecord.to_dict`.

        Returns:
            Mapping from column name to a per-update list, ready for plotting or
            export to a dataframe.
        """
        rows = [record.to_dict(flat_transitions=flat_transitions) for record in self.records]
        if not rows:
            return {}
        return {key: [row.get(key) for row in rows] for key in rows[0]}

    def reset(self) -> None:
        """Drop the EMA state, the previous sets and the collected records."""
        self._tracker.reset()

    # ------------------------------------------------------------------
    # updates
    # ------------------------------------------------------------------
    def update(
        self,
        *,
        step: int | None = None,
        gradients: Mapping[str, Tensor] | None = None,
        parameters: Mapping[str, Tensor] | None = None,
        upstream_gradients: Mapping[str, Tensor] | None = None,
        sensitivities: Mapping[str, Tensor] | None = None,
        pressures: Mapping[str, Tensor] | None = None,
        t: float | None = None,
    ) -> TrackerRecord:
        """Run one support update over every tracked parameter tensor.

        Args:
            step: Support-update index; defaults to an internal counter.
            gradients: Mapping from parameter name to the current QAT gradient ``g``.
            parameters: Mapping from parameter name to the current weight snapshot
                ``p`` (required to enable motion and ``dSS``).
            upstream_gradients: Mapping from parameter name to the upstream gradient
                ``H``; switches the active score to the literal ``EMA(|H| * D)``.
            sensitivities: Mapping from parameter name to a precomputed ``dSS``.
            pressures: Mapping from parameter name to a precomputed ``|g|``; an
                alternative to ``gradients``.
            t: Live SoftStairs temperature used to evaluate ``dSS`` from ``parameters``.

        Returns:
            The :class:`softstairs_qat.analysis.tracking.TrackerRecord` of this update.

        Raises:
            ValueError: If no pressure source is supplied or a mapping is missing a
                layout entry.
        """
        if gradients is None and pressures is None and upstream_gradients is None:
            raise ValueError("update requires gradients, pressures, or upstream_gradients")
        source = gradients if gradients is not None else pressures if pressures is not None else upstream_gradients
        flat = flatten_tensors(source, layout=self.layout, dtype=self._tracker._dtype)
        flat_parameter = (
            flatten_tensors(parameters, layout=self.layout, dtype=self._tracker._dtype)
            if parameters is not None
            else None
        )
        flat_sensitivity = (
            flatten_tensors(sensitivities, layout=self.layout, dtype=self._tracker._dtype)
            if sensitivities is not None
            else None
        )
        return self._tracker.update(
            step=step,
            gradient=flat,
            parameter=flat_parameter,
            sensitivity=flat_sensitivity,
            upstream_gradient=flat if upstream_gradients is not None else None,
            t=t,
        )

    def update_sequence(
        self,
        ps: Mapping[str, Tensor] | Sequence[Tensor],
        gs: Mapping[str, Tensor] | Sequence[Tensor],
        *,
        sensitivities: Mapping[str, Tensor] | None = None,
        upstream_gradients: Mapping[str, Tensor] | None = None,
        t_values: Any = None,
    ) -> list[TrackerRecord]:
        """Replay ``[T, ...]`` snapshots per tensor as consecutive support updates.

        Frames are fed one at a time through the streaming tracker, so peak memory
        stays at ``O(N)``.  The resulting records match
        :func:`softstairs_qat.analysis.tracking.track_sequence` on the concatenated
        flat vector.

        Args:
            ps: Mapping (or sequence) of weight snapshot stacks, each ``[T, ...]``.
            gs: Matching gradient snapshot stacks, each ``[T, ...]``.
            sensitivities: Optional per-tensor ``dSS`` stacks, each ``[T, ...]``.
            upstream_gradients: Optional per-tensor upstream gradient stacks ``[T, ...]``.
            t_values: Temperature per snapshot: a ``[T]`` tensor, a scalar, or ``None``
                to score purely from ``|g|``.

        Returns:
            One :class:`softstairs_qat.analysis.tracking.TrackerRecord` per snapshot.

        Raises:
            ValueError: If the two mappings disagree on the snapshot count.
        """
        names = list(self.layout.names)
        stacked_p = _stack_per_tensor(ps, names, self.layout, "ps")
        stacked_g = _stack_per_tensor(gs, names, self.layout, "gs")
        steps = stacked_g.shape[0]
        if stacked_p.shape[0] != steps:
            raise ValueError(f"ps has {stacked_p.shape[0]} snapshots but gs has {steps}")

        stacked_sensitivity = (
            _stack_per_tensor(sensitivities, names, self.layout, "sensitivities")
            if sensitivities is not None
            else None
        )
        stacked_upstream = (
            _stack_per_tensor(upstream_gradients, names, self.layout, "upstream_gradients")
            if upstream_gradients is not None
            else None
        )
        temperatures = _as_temperature_series(t_values, steps)

        records: list[TrackerRecord] = []
        for step in range(steps):
            records.append(
                self._tracker.update(
                    step=step,
                    gradient=stacked_g[step],
                    parameter=stacked_p[step],
                    sensitivity=None if stacked_sensitivity is None else stacked_sensitivity[step],
                    upstream_gradient=None if stacked_upstream is None else stacked_upstream[step],
                    t=None if temperatures is None else float(temperatures[step]),
                )
            )
        return records


def _as_temperature_series(t_values: Any, steps: int) -> Tensor | None:
    """Normalize a temperature specification into a ``[T]`` float tensor.

    Args:
        t_values: Scalar, ``[T]`` tensor or sequence, or ``None``.
        steps: Number of snapshots.

    Returns:
        Tensor of shape ``[steps]``, or ``None`` when ``t_values`` is ``None``.

    Raises:
        ValueError: If the length of ``t_values`` disagrees with ``steps``.
    """
    if t_values is None:
        return None
    values = torch.as_tensor(t_values, dtype=torch.float32).detach().flatten()
    if values.numel() == 1:
        return values.expand(steps).clone()
    if values.numel() != steps:
        raise ValueError(f"t_values has {values.numel()} entries but the sequence has {steps} snapshots")
    return values


def _stack_per_tensor(
    stacks: Mapping[str, Tensor] | Sequence[Tensor],
    names: Sequence[str],
    layout: ParameterLayout,
    label: str,
) -> Tensor:
    """Flatten per-tensor snapshot stacks into one ``[T, N]`` tensor.

    Args:
        stacks: Mapping from parameter name to a ``[T, ...]`` stack, or a sequence in
            layout order.
        names: Layout parameter names, used to normalize a mapping.
        layout: Layout describing the packing.
        label: Argument name used in error messages.

    Returns:
        Tensor of shape ``[T, num_parameters]``.

    Raises:
        ValueError: If the temporal lengths disagree or the input is malformed.
    """
    if isinstance(stacks, Mapping):
        missing = [name for name in names if name not in stacks]
        if missing:
            raise KeyError(f"{label} is missing the layout entries {missing}")
        ordered = {name: stacks[name] for name in names}
    else:
        ordered = {name: tensor for name, tensor in zip(names, stacks)}

    steps = None
    for name in names:
        tensor = ordered[name]
        if tensor.ndim < 1:
            raise ValueError(f"{label}[{name!r}] must carry a temporal axis")
        if steps is None:
            steps = tensor.shape[0]
        elif tensor.shape[0] != steps:
            raise ValueError(f"{label}[{name!r}] has {tensor.shape[0]} snapshots, expected {steps}")

    frames = []
    for step in range(int(steps or 0)):
        frames.append(flatten_tensors({name: ordered[name][step] for name in names}, layout=layout))
    if not frames:
        raise ValueError(f"{label} contains no snapshots")
    return torch.stack(frames)


def build_tracker(
    model: torch.nn.Module,
    *,
    active_fraction: float | None = 0.01,
    reservoir_fraction: float | None = 0.25,
    decay_pressure: float = 0.9,
    score: ScoreConfig | None = None,
    sets: SetConfig | None = None,
    trainable_only: bool = True,
    exclude_suffix: str | None = None,
) -> ParameterSetTracker:
    """Convenience constructor for a live QAT model.

    Args:
        model: Model whose trainable parameters are tracked.
        active_fraction: Default active budget as a fraction of ``N``.
        reservoir_fraction: Default reservoir budget as a fraction of ``N``.
        decay_pressure: Decay of the pressure EMA, used when ``score`` is ``None``.
        score: Full score configuration; overrides ``decay_pressure``.
        sets: Full set configuration; overrides the two fractions.
        trainable_only: Skip parameters with ``requires_grad=False``.
        exclude_suffix: Skip parameter names ending with this suffix.

    Returns:
        The configured :class:`ParameterSetTracker`.
    """
    score_config = score or ScoreConfig(decay_pressure=decay_pressure)
    set_config = sets or SetConfig(active_fraction=active_fraction, reservoir_fraction=reservoir_fraction)
    return ParameterSetTracker(
        model,
        config=TrackerConfig(score=score_config, sets=set_config),
        trainable_only=trainable_only,
        exclude_suffix=exclude_suffix,
    )


def summarize(records: Sequence[TrackerRecord]) -> dict[str, float]:
    """Aggregate the repeated ``reservoir -> active -> reservoir`` churn.

    The hypothesis under test is that during late QAT training most weights become
    dormant or reservoir while a small subset repeatedly enters the active state.
    The two transition rates below are the direct measurements, averaged over the
    support updates that follow the first one.

    Args:
        records: Per-update records from either tracker.

    Returns:
        Mapping with the mean active/reservoir sizes and the transition rates,
        expressed as a fraction of the source set.
    """
    if not records:
        return {}
    tails = records[1:]
    reservoir_to_active = [record.transition("reservoir_to_active").source_fraction for record in tails]
    active_to_reservoir = [record.transition("active_to_reservoir").source_fraction for record in tails]
    active_jaccard = [record.active_jaccard for record in tails if record.active_jaccard is not None]
    reservoir_jaccard = [record.reservoir_jaccard for record in tails if record.reservoir_jaccard is not None]

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "updates": float(len(records)),
        "mean_num_active": _mean([float(record.num_active) for record in records]),
        "mean_num_reservoir": _mean([float(record.num_reservoir) for record in records]),
        "mean_dormant_fraction": _mean([record.dormant_fraction for record in records]),
        "mean_reservoir_to_active_rate": _mean(reservoir_to_active),
        "mean_active_to_reservoir_rate": _mean(active_to_reservoir),
        "mean_active_jaccard": _mean(active_jaccard),
        "mean_reservoir_jaccard": _mean(reservoir_jaccard),
        "churn_excess": _mean(reservoir_to_active) - _mean(active_to_reservoir),
    }