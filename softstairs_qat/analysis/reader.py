# softstairs_qat/analysis/reader.py

"""Locating and loading saved ``[T, ...]`` weight/gradient snapshots.

``GradSaver`` (a ``pytorch_lightning`` callback in
``experiments/CV/inception_stl10.ipynb``) writes one pair of files per *support
update* into a directory named after the tracked parameter:

.. code-block:: text

    <run>/<version_N>/<param_name>/param_<epoch>.<index>.pth
    <run>/<version_N>/<param_name>/grad_<epoch>.<index>.pth

``index`` is the snapshot taken inside the epoch, so an epoch contributes ``P``
support updates and the epoch contributes a single temperature step.  Older runs
omitted the index and produced one snapshot per epoch; both layouts are
accepted here.

Pointing the analysis at a run
------------------------------
:data:`EXPERIMENT_CHECKPOINTS` maps a human-readable key to a snapshot directory.
It is a plain module-level dictionary on purpose: a notebook (or a test) sets it
before loading, exactly as requested by the analysis workflow.

.. code-block:: python

    from softstairs_qat.analysis import reader

    reader.EXPERIMENT_CHECKPOINTS["my_run/version_2/fc.weight"] = "/path/to/fc.weight"
    name, ps, gs, layout = reader.load_registered_snapshots("my_run/version_2/fc.weight")

The prefilled entry points at the directory documented in the analysis setup, so
``load_registered_snapshots()`` also works without any user edit.

Temperature schedule
--------------------
``load_t_schedule`` rebuilds the run's :class:`softstairs_qat.utils.TScheduler`
from ``hparams.yaml`` and expands it to one temperature *per snapshot* with
:func:`softstairs_qat.analysis.sensitivity.t_schedule_for_snapshots`, which
accounts for the ``P`` snapshots per epoch.  Zipping the schedule with the
snapshots directly is wrong for the ``param_<epoch>.<index>`` layout and raises
``IndexError`` once ``T`` exceeds ``n_steps``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from softstairs_qat.analysis.flattening import ParameterLayout, flatten_tensors
from softstairs_qat.analysis.sensitivity import collect_t_schedule

__all__ = [
    "EXPERIMENT_CHECKPOINTS",
    "SnapshotSet",
    "register_checkpoint",
    "available_checkpoints",
    "resolve_checkpoint",
    "load_snapshots",
    "load_registered_snapshots",
    "load_hparams",
    "load_t_schedule",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Registry mapping an experiment key to a snapshot directory.
#:
#: Populated with the reference run of the analysis setup; add further entries (or
#: overwrite this one) to analyse another run.  Keys are free-form labels, paths may
#: also be passed directly to the loading helpers.
EXPERIMENT_CHECKPOINTS: dict[str, str] = {
    "InceptionNet_STL10_strcyclic_af2_b8_tstandard_majNone_norm/version_2/fc.weight": str(
        _REPO_ROOT
        / "experiments/CV/logs/InceptionNet_STL10_strcyclic_af2_b8_tstandard_majNone_norm/version_2/fc.weight"
    ),
}


@dataclass(frozen=True)
class SnapshotSet:
    """One loaded set of ``[T, ...]`` snapshots.

    Attributes:
        name: Registry key or directory name the snapshots came from.
        path: Directory holding the ``param_*`` / ``grad_*`` files.
        ps: Stacked weight snapshots, shaped ``[T, ...]``; values are in
            quantization code space because ``GradSaver`` saves
            ``quantizer.upscaled_parameter(...)``.
        gs: Stacked QAT gradients, shaped ``[T, ...]``; already gated by ``dSS``.
        snapshots_per_epoch: Number of support updates recorded per epoch (``P``).
        epochs: Epoch index of every snapshot, ``[T]``.
        indices: In-epoch snapshot index of every snapshot, ``[T]``.
    """

    name: str
    path: Path
    ps: Tensor
    gs: Tensor
    snapshots_per_epoch: int
    epochs: Tensor
    indices: Tensor

    @property
    def num_snapshots(self) -> int:
        """Temporal length ``T``."""
        return int(self.ps.shape[0])

    @property
    def weight_shape(self) -> tuple[int, ...]:
        """Shape of a single weight snapshot, without the temporal axis."""
        return tuple(self.ps.shape[1:])

    def flat(self, *, dtype: torch.dtype = torch.float32) -> tuple[Tensor, Tensor]:
        """Return the snapshots flattened to ``[T, N]``.

        Args:
            dtype: Accumulation dtype.

        Returns:
            Tuple ``(ps, gs)`` of shape ``[T, N]``.
        """
        layout = ParameterLayout.from_tensors({"weight": self.ps[0]})
        stacked_p = torch.stack([flatten_tensors({"weight": frame}, layout=layout, dtype=dtype) for frame in self.ps])
        stacked_g = torch.stack([flatten_tensors({"weight": frame}, layout=layout, dtype=dtype) for frame in self.gs])
        return stacked_p, stacked_g

    def __repr__(self) -> str:
        return (
            f"SnapshotSet(name={self.name!r}, num_snapshots={self.num_snapshots}, "
            f"weight_shape={self.weight_shape}, snapshots_per_epoch={self.snapshots_per_epoch})"
        )


def register_checkpoint(key: str, path: str | os.PathLike[str]) -> str:
    """Add or replace an entry in :data:`EXPERIMENT_CHECKPOINTS`.

    Args:
        key: Human-readable experiment key.
        path: Snapshot directory.

    Returns:
        The stored absolute path.
    """
    resolved = str(Path(path).expanduser().resolve())
    EXPERIMENT_CHECKPOINTS[key] = resolved
    return resolved


def available_checkpoints() -> dict[str, str]:
    """Return a copy of the registry.

    Returns:
        Mapping from experiment key to snapshot directory.
    """
    return dict(EXPERIMENT_CHECKPOINTS)


def resolve_checkpoint(key_or_path: str | os.PathLike[str] | None = None) -> tuple[str, Path]:
    """Resolve a registry key or a raw path to a snapshot directory.

    Args:
        key_or_path: Registry key, or a directory path (absolute or relative to the
            current working directory).  ``None`` selects the only entry of
            :data:`EXPERIMENT_CHECKPOINTS`, which is the prefilled reference run.

    Returns:
        Tuple of the registry key (or the directory name) and the resolved path.

    Raises:
        FileNotFoundError: If the resolved directory does not exist.
        ValueError: If ``key_or_path`` is ``None`` and the registry is empty or holds
            several entries, so an explicit choice is required.
    """
    if key_or_path is None:
        if len(EXPERIMENT_CHECKPOINTS) == 1:
            key_or_path = next(iter(EXPERIMENT_CHECKPOINTS))
        elif not EXPERIMENT_CHECKPOINTS:
            raise ValueError("EXPERIMENT_CHECKPOINTS is empty; register a checkpoint first")
        else:
            raise ValueError(
                "EXPERIMENT_CHECKPOINTS holds several entries, pass one explicitly: "
                f"{sorted(EXPERIMENT_CHECKPOINTS)}"
            )
    text = str(key_or_path)
    if text in EXPERIMENT_CHECKPOINTS:
        path = Path(EXPERIMENT_CHECKPOINTS[text]).expanduser()
        name = text
    else:
        path = Path(text).expanduser()
        name = path.name
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"snapshot directory does not exist: {path}")
    return name, path


def _parse_tag(stem: str, kind: str) -> tuple[int, int]:
    """Parse ``<kind>_<epoch>[.<index>]`` into an ``(epoch, index)`` pair.

    Args:
        stem: File stem without the extension.
        kind: Expected file prefix (``"param"`` or ``"grad"``).

    Returns:
        Tuple ``(epoch, index)``; ``index`` is ``0`` for the older single-snapshot
        layout.

    Raises:
        ValueError: If the stem does not follow the convention.
    """
    prefix = f"{kind}_"
    if not stem.startswith(prefix):
        raise ValueError(f"expected a {prefix} file, got {stem!r}")
    body = stem[len(prefix) :]
    parts = body.split(".")
    try:
        epoch = int(parts[0])
        index = int(parts[1]) if len(parts) > 1 and parts[1] != "" else 0
    except ValueError as error:
        raise ValueError(f"cannot parse snapshot tag {stem!r}") from error
    return epoch, index


def _collect(directory: Path, kind: str, device: torch.device | str | None) -> tuple[Tensor, Tensor, Tensor]:
    """Load and order every file of one snapshot kind.

    Args:
        directory: Snapshot directory.
        kind: ``"param"`` or ``"grad"``.
        device: Optional device for the stacked tensors.

    Returns:
        Tuple ``(stacked, epochs, indices)`` where ``stacked`` has the leading temporal
        axis, ordered by ``(epoch, index)``.

    Raises:
        FileNotFoundError: If the directory holds no file of that kind.
    """
    entries: list[tuple[tuple[int, int], Path]] = []
    for name in os.listdir(directory):
        if not name.endswith(".pth"):
            continue
        stem = name[: -len(".pth")]
        if not stem.startswith(f"{kind}_"):
            continue
        entries.append((_parse_tag(stem, kind), directory / name))
    if not entries:
        raise FileNotFoundError(f"no {kind}_*.pth snapshots in {directory}")

    entries.sort(key=lambda item: item[0])
    tensors = []
    for _, path in entries:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if not torch.is_tensor(loaded):
            raise TypeError(f"{path} does not contain a tensor")
        tensors.append(loaded.detach().float())
    shapes = {tuple(tensor.shape) for tensor in tensors}
    if len(shapes) != 1:
        raise ValueError(f"{kind} snapshots in {directory} disagree on shape: {sorted(shapes)}")
    stacked = torch.stack(tensors).to(device=device)
    tags = torch.tensor([tag for tag, _ in entries], dtype=torch.long)
    return stacked, tags[:, 0], tags[:, 1]


def load_snapshots(
    path: str | os.PathLike[str],
    *,
    name: str | None = None,
    device: torch.device | str | None = None,
) -> SnapshotSet:
    """Load ``ps`` and ``gs`` stacks from a snapshot directory.

    The two kinds are ordered independently by their own ``(epoch, index)`` tags, so
    a directory with a missing file on one side is still reported rather than
    silently misaligned.

    Args:
        path: Snapshot directory.
        name: Label stored in the result; defaults to the directory name.
        device: Optional device for the stacked tensors.

    Returns:
        The loaded :class:`SnapshotSet`.

    Raises:
        FileNotFoundError: If the directory does not exist or holds no snapshots.
        ValueError: If ``ps`` and ``gs`` disagree on the snapshot count.
    """
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"snapshot directory does not exist: {directory}")

    ps, param_epochs, param_indices = _collect(directory, "param", device)
    gs, grad_epochs, grad_indices = _collect(directory, "grad", device)
    if ps.shape[0] != gs.shape[0]:
        raise ValueError(f"{directory} holds {ps.shape[0]} param and {gs.shape[0]} grad snapshots")

    indices = torch.cat([param_indices, grad_indices])
    per_epoch = int(torch.unique(param_indices).numel())
    snapshots_per_epoch = max(per_epoch, int(param_indices.max()) + 1, int(grad_indices.max()) + 1)
    del indices

    return SnapshotSet(
        name=name or directory.name,
        path=directory,
        ps=ps,
        gs=gs,
        snapshots_per_epoch=snapshots_per_epoch,
        epochs=torch.cat([param_epochs, grad_epochs])[: ps.shape[0]],
        indices=torch.cat([param_indices, grad_indices])[: ps.shape[0]],
    )


def load_registered_snapshots(
    key_or_path: str | os.PathLike[str] | None = None,
    *,
    device: torch.device | str | None = None,
) -> SnapshotSet:
    """Load the snapshots of a registry entry or a raw directory.

    Args:
        key_or_path: Registry key or directory path.  ``None`` selects the only entry
            of :data:`EXPERIMENT_CHECKPOINTS`.
        device: Optional device for the stacked tensors.

    Returns:
        The loaded :class:`SnapshotSet`.
    """
    name, path = resolve_checkpoint(key_or_path)
    return load_snapshots(path, name=name, device=device)


def load_hparams(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a run's ``hparams.yaml``.

    Args:
        path: The run directory (the ``version_N`` folder) or the file itself.

    Returns:
        The parsed mapping; empty when no YAML parser is installed.

    Raises:
        FileNotFoundError: If neither the directory nor the file exists.
    """
    target = Path(path).expanduser()
    if target.is_dir():
        target = target / "hparams.yaml"
    if not target.is_file():
        raise FileNotFoundError(f"hparams file does not exist: {target}")
    try:
        import yaml
    except ImportError:  # pragma: no cover - PyYAML is an experiment dependency
        return {}
    with target.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_t_schedule(
    run_dir: str | os.PathLike[str],
    *,
    key: str = "quantization_hparams",
    **overrides: Any,
) -> list[float]:
    """Rebuild the run's ``t`` schedule from ``hparams.yaml``.

    Args:
        run_dir: The ``version_N`` run directory, or any ancestor of it.
        key: Sub-mapping holding the scheduler hyper-parameters.
        **overrides: Field overrides applied on top of the file values.

    Returns:
        One temperature per scheduler step.

    Raises:
        FileNotFoundError: If no ``hparams.yaml`` can be located.
        ValueError: If the scheduler cannot be constructed.
    """
    from softstairs_qat.utils import TScheduler

    directory = Path(run_dir).expanduser()
    for candidate in (directory, *directory.parents):
        if (candidate / "hparams.yaml").is_file():
            directory = candidate
            break
    else:
        raise FileNotFoundError(f"no hparams.yaml above {run_dir}")

    raw: Mapping[str, Any] = load_hparams(directory)
    section = dict(raw.get(key) or {})
    section.update(overrides)
    strategy = section.pop("t_scheduler_strategy", "linear")
    if strategy == "constant":
        return [float(section.get("t_start", 0.2))]
    return collect_t_schedule(
        TScheduler(
            strategy=strategy,
            start_t=section.get("t_start", 0.5),
            end_t=section.get("t_end", 0.01),
            total_steps=int(section.get("n_steps", 1000)),
            early_power=section.get("early_power", 0.7),
            min_majorant=section.get("min_majorant", 0.01),
        )
    )