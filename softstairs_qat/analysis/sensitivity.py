# softstairs_qat/analysis/sensitivity.py

"""Quantization sensitivity (``dSS``) plumbing for active-set scoring.

The active-set score must reflect the QAT gradient ``g = H * D`` where ``D`` is
the SoftStairs derivative.  This module never re-derives that derivative: it
calls :meth:`softstairs_qat.core.soft_stairs.SoftStairs.derivative`, the same
entry point used by ``SoftStairsQuantizeFunction.backward``.

Two sources are supported, in priority order:

1. A sensitivity already exposed by a live quantizer or model (any attribute
   named ``dSS``), used verbatim via :func:`resolve_sensitivity`.
2. Recomputation from the upscaled (code-space) weight snapshot and the current
   ``t`` via :func:`softstairs_sensitivity`.

Because ``t`` is stepped once per epoch while snapshots are taken ``P`` times
per epoch, the per-snapshot temperature has to be built by index arithmetic
(see :func:`t_schedule_for_snapshots`) rather than by zipping the schedule.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from softstairs_qat.core.soft_stairs import SoftStairs

__all__ = [
    "softstairs_sensitivity",
    "t_schedule_for_snapshots",
    "collect_t_schedule",
    "find_exposed_sensitivity",
    "resolve_sensitivity",
]

_DSS_ATTRIBUTE_CANDIDATES = ("dSS", "dss", "sensitivity", "sensibility")


def collect_t_schedule(source: Any) -> list[float]:
    """Normalize anything schedule-like into a list of floats.

    Args:
        source: ``TScheduler``/``AdaptiveScheduler`` instance exposing
            ``get_all_t()``, a callable returning such a list, or an iterable
            of numbers.

    Returns:
        The schedule as a list of floats.

    Raises:
        TypeError: If ``source`` cannot be interpreted as a t schedule.
    """
    if source is None:
        raise TypeError("t schedule source is None")
    if hasattr(source, "get_all_t"):
        values: Iterable[Any] = source.get_all_t()
    elif callable(source):
        values = source()
    elif isinstance(source, (int, float)):
        values = [source]
    else:
        values = source
    return [float(v) for v in values]


def t_schedule_for_snapshots(
    t_schedule: Sequence[float] | float | None,
    num_snapshots: int,
    *,
    snapshots_per_epoch: int | None = None,
    schedule_offset: int = 0,
) -> Tensor:
    """Map snapshot indices to the temperature that was live at that time.

    Training runs as ``start of epoch -> P optimizer steps with snapshots ->
    training -> one t-scheduler step -> start of epoch -> ...``.  ``t`` is
    therefore constant inside an epoch and advances once per epoch, so snapshot
    ``i`` must be scored with ``t_schedule[(i + schedule_offset) //
    snapshots_per_epoch]``.  Passing ``snapshots_per_epoch=None`` assumes one
    temperature per snapshot, which is only correct when the scheduler is
    stepped on every snapshot.

    Args:
        t_schedule: Schedule values, a single float, or ``None`` to use
            ``schedule_offset`` as a constant temperature.
        num_snapshots: Number of snapshots ``T``.
        snapshots_per_epoch: Snapshots recorded per epoch (``P``).  ``None``
            disables the epoch grouping.
        schedule_offset: Index into the schedule that snapshot ``0`` maps to.

    Returns:
        Float tensor of shape ``[T]`` with the temperature per snapshot.

    Raises:
        ValueError: If ``snapshots_per_epoch`` is not positive, or the schedule
            is too short for the requested number of snapshots.
    """
    if num_snapshots < 0:
        raise ValueError(f"num_snapshots must be non-negative, got {num_snapshots}")
    if snapshots_per_epoch is not None and snapshots_per_epoch <= 0:
        raise ValueError(f"snapshots_per_epoch must be positive, got {snapshots_per_epoch}")

    indices = torch.arange(num_snapshots, dtype=torch.long)
    if snapshots_per_epoch is not None:
        indices = torch.div(indices, snapshots_per_epoch, rounding_mode="floor")
    indices = indices + schedule_offset

    if t_schedule is None:
        return torch.full((num_snapshots,), float(schedule_offset), dtype=torch.float32)
    if isinstance(t_schedule, (int, float)):
        return torch.full((num_snapshots,), float(t_schedule), dtype=torch.float32)

    values = torch.as_tensor(list(t_schedule), dtype=torch.float32)
    if values.numel() == 0:
        raise ValueError("t_schedule is empty")
    if values.numel() == 1:
        return values.expand(num_snapshots).clone()
    last = int(indices[-1].item()) if num_snapshots else int(indices.max().item()) if indices.numel() else 0
    if last >= values.numel():
        raise ValueError(
            f"t_schedule has {values.numel()} entries but snapshot {last} needs index {last}; "
            "pass a longer schedule or adjust schedule_offset"
        )
    return values[indices.clamp_max(values.numel() - 1)]


def softstairs_sensitivity(
    ps: Tensor,
    t_values: Sequence[float] | float | Tensor | None = None,
    *,
    normalized: bool = False,
    async_t_factor: float = 1.0,
    snapshots_per_epoch: int | None = None,
    schedule_offset: int = 0,
    out: Tensor | None = None,
) -> Tensor:
    """Evaluate the SoftStairs derivative ``dSS`` for a stack of snapshots.

    ``SoftStairs.derivative`` accepts a single temperature, so snapshots are
    grouped by temperature and each group is evaluated in one broadcast call.
    The Python loop is therefore over *distinct* ``t`` values (one per epoch),
    never over weights, and never over the temporal axis.

    Args:
        ps: Weight snapshots in quantization code space, shaped ``[T, ...]``.
        t_values: Temperature per snapshot (``[T]``), a scalar temperature, a
            full schedule (see ``snapshots_per_epoch``), or ``None``.
        normalized: Forward for ``SoftStairs.normalized``.
        async_t_factor: Forward for ``SoftStairs.async_t_factor``.  This is the
            factor used in the backward pass, see ``current_backward_t``.
        snapshots_per_epoch: Snapshots per epoch ``P``; required when
            ``t_values`` is a full schedule rather than a per-snapshot vector.
        schedule_offset: Index of the schedule entry that snapshot ``0`` uses.
        out: Optional destination with the shape of ``ps``.

    Returns:
        Tensor with the shape and dtype of ``ps`` holding ``dSS`` values.

    Raises:
        ValueError: If ``t_values`` is a scalar-less schedule shorter than the
            requested snapshot range, or if ``out`` has the wrong shape.
    """
    if ps.ndim == 0:
        raise ValueError("ps must have at least one dimension")
    steps = ps.shape[0]
    work = ps if ps.dtype not in (torch.float16, torch.bfloat16) else ps.float()

    if out is not None and tuple(out.shape) != tuple(ps.shape):
        raise ValueError(f"out must have shape {tuple(ps.shape)}, got {tuple(out.shape)}")

    if t_values is None:
        schedule = t_schedule_for_snapshots(
            None, steps, snapshots_per_epoch=snapshots_per_epoch, schedule_offset=schedule_offset
        )
    elif isinstance(t_values, (int, float)):
        schedule = torch.full((steps,), float(t_values), dtype=torch.float32)
    elif torch.is_tensor(t_values) and t_values.ndim > 0 and t_values.numel() == steps:
        # One temperature per snapshot.
        schedule = t_values.detach().flatten().to(torch.float32)
    else:
        # A per-epoch schedule that has to be expanded to one temperature per
        # snapshot, or any other iterable of temperatures.
        schedule = t_schedule_for_snapshots(
            t_values, steps, snapshots_per_epoch=snapshots_per_epoch, schedule_offset=schedule_offset
        )

    result = torch.empty_like(work)
    unique, inverse = torch.unique(schedule, return_inverse=True)
    for group, temperature in enumerate(unique.tolist()):
        selector = inverse == group
        staircase = SoftStairs(t=temperature, normalized=normalized, async_t_factor=async_t_factor)
        result[selector] = staircase.derivative(work[selector])

    if out is not None:
        out.copy_(result)
        return out
    return result.to(ps.dtype)


def find_exposed_sensitivity(owner: Any) -> Tensor | None:
    """Look for a sensitivity tensor already published by ``owner``.

    Checks ``owner`` itself and the objects it holds most commonly: the model,
    a ``quantizer`` attribute, and the quantizer's ``_hook_fn``.

    Args:
        owner: Quantizer, model, or module to probe.

    Returns:
        The first sensitivity tensor found, or ``None``.
    """
    if owner is None:
        return None

    candidates: list[Any] = [owner]
    for attribute in ("model", "quantizer", "_hook_fn", "soft"):
        nested = getattr(owner, attribute, None)
        if nested is not None:
            candidates.append(nested)

    for candidate in candidates:
        for name in _DSS_ATTRIBUTE_CANDIDATES:
            value = getattr(candidate, name, None)
            if torch.is_tensor(value):
                return value
            if isinstance(value, Mapping):
                tensors = [v for v in value.values() if torch.is_tensor(v)]
                if len(tensors) == 1:
                    return tensors[0]
    return None


def resolve_sensitivity(
    ps: Tensor | None,
    owner: Any = None,
    t_values: Sequence[float] | float | Tensor | None = None,
    *,
    name: str | None = None,
    use_sensitivity: bool = True,
    **kwargs: Any,
) -> Tensor:
    """Return ``dSS`` for one snapshot, preferring an already exposed value.

    Args:
        ps: Current weight snapshot in code space (may be ``None`` when the
            owner exposes its own sensitivity).
        owner: Quantizer or model probed by :func:`find_exposed_sensitivity`.
        t_values: Temperature for this snapshot, or a schedule.
        name: Optional parameter name; when the owner publishes a mapping keyed
            by parameter name this entry is preferred.
        use_sensitivity: Set to ``False`` to always recompute from ``ps``.
        **kwargs: Forwarded to :func:`softstairs_sensitivity`.

    Returns:
        The sensitivity tensor.

    Raises:
        ValueError: If neither an exposed sensitivity nor ``ps`` is available.
    """
    if use_sensitivity:
        exposed = _lookup_named_sensitivity(owner, name)
        if exposed is not None:
            return exposed
        exposed = find_exposed_sensitivity(owner)
        if exposed is not None:
            return exposed

    if ps is None:
        raise ValueError("no exposed dSS and no weight snapshot available to recompute it from")
    return softstairs_sensitivity(ps, t_values, **kwargs)


def _lookup_named_sensitivity(owner: Any, name: str | None) -> Tensor | None:
    """Pick ``owner``'s named sensitivity entry when it publishes a mapping.

    Args:
        owner: Quantizer or model to probe.
        name: Parameter name, or ``None`` to skip the lookup.

    Returns:
        The sensitivity tensor for ``name``, or ``None``.
    """
    if owner is None or name is None:
        return None
    for attribute in _DSS_ATTRIBUTE_CANDIDATES:
        value = getattr(owner, attribute, None)
        if isinstance(value, Mapping) and name in value and torch.is_tensor(value[name]):
            return value[name]
    quantizer = getattr(owner, "quantizer", None)
    if quantizer is not None:
        for attribute in _DSS_ATTRIBUTE_CANDIDATES:
            value = getattr(quantizer, attribute, None)
            if isinstance(value, Mapping) and name in value and torch.is_tensor(value[name]):
                return value[name]
    return None
