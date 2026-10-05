# softstairs_qat/analysis/flattening.py

"""Flatten heterogeneous parameter tensors into one logical weight vector.

Scores and masks are computed on a single flat ``[N]`` vector so that ranking,
top-k selection and set algebra are one vectorized operation instead of a loop
over layers.  :class:`ParameterLayout` stores the flat index bookkeeping

    flat index -> parameter name -> local index

as contiguous offset ranges, which keeps the mapping allocation free at run
time and makes mask scattering a ``view`` rather than a copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import torch
from torch import Tensor

__all__ = [
    "ParameterLayout",
    "flatten_tensors",
    "unflatten_tensor",
    "unflatten_masks",
    "select_parameters",
]


def _as_named_sequence(
    parameters: Mapping[str, Tensor] | Sequence[Tensor] | Iterable[Tensor],
) -> list[tuple[str, Tensor]]:
    """Normalize the accepted parameter containers into ``(name, tensor)`` pairs.

    Args:
        parameters: Mapping of names to tensors, or a sequence of tensors.

    Returns:
        List of ``(name, tensor)`` pairs in iteration order.

    Raises:
        TypeError: If ``parameters`` is not one of the accepted containers.
    """
    if isinstance(parameters, Mapping):
        return [(str(name), tensor) for name, tensor in parameters.items()]
    if torch.is_tensor(parameters):
        return [("flat", parameters)]
    if isinstance(parameters, Sequence):
        return [(f"param_{index}", tensor) for index, tensor in enumerate(parameters)]
    raise TypeError(f"unsupported parameter container: {type(parameters)!r}")


@dataclass(frozen=True)
class ParameterLayout:
    """Offset table describing how named tensors pack into a flat vector.

    Attributes:
        names: Parameter names in flat order.
        shapes: Per-parameter shapes, matching :attr:`names`.
        numels: Per-parameter element counts.
        offsets: Start offset of each parameter inside the flat vector.
    """

    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    numels: tuple[int, ...]
    offsets: tuple[int, ...]

    @classmethod
    def from_tensors(
        cls,
        parameters: Mapping[str, Tensor] | Sequence[Tensor] | Iterable[Tensor],
    ) -> "ParameterLayout":
        """Build a layout from a container of tensors.

        Args:
            parameters: Mapping of names to tensors, or a sequence of tensors.

        Returns:
            The corresponding :class:`ParameterLayout`.

        Raises:
            ValueError: If the container is empty or a tensor is not 0-d/1-d
                compatible (every tensor is flattened, so any shape is fine) and
                a name repeats.
        """
        pairs = _as_named_sequence(parameters)
        if not pairs:
            raise ValueError("cannot build a ParameterLayout from an empty parameter set")
        seen: set[str] = set()
        for name, _ in pairs:
            if name in seen:
                raise ValueError(f"duplicate parameter name {name!r}")
            seen.add(name)

        shapes = tuple(tuple(tensor.shape) for _, tensor in pairs)
        numels = tuple(tensor.numel() for _, tensor in pairs)
        offsets: list[int] = []
        cursor = 0
        for numel in numels:
            offsets.append(cursor)
            cursor += numel
        return cls(
            names=tuple(name for name, _ in pairs),
            shapes=shapes,
            numels=numels,
            offsets=tuple(offsets),
        )

    @property
    def num_parameters(self) -> int:
        """Total number of scalar weights ``N``."""
        return sum(self.numels)

    def index_of(self, name: str) -> int:
        """Return the flat offset of a named parameter.

        Args:
            name: Parameter name.

        Returns:
            The start offset inside the flat vector.

        Raises:
            KeyError: If the name is not part of the layout.
        """
        try:
            return self.offsets[self.names.index(name)]
        except ValueError as error:
            raise KeyError(f"unknown parameter name {name!r}") from error

    def ranges(self) -> dict[str, tuple[int, int]]:
        """Return the ``[start, stop)`` flat range of every parameter.

        Returns:
            Mapping from parameter name to a half-open flat range.
        """
        return {
            name: (start, start + numel)
            for name, start, numel in zip(self.names, self.offsets, self.numels)
        }

    def flatten(self, parameters: Mapping[str, Tensor]) -> Tensor:
        """Concatenate the given tensors into the flat vector.

        Args:
            parameters: Mapping of names to tensors; every name in the layout
                must be present.

        Returns:
            Flat tensor of shape ``[num_parameters]``.

        Raises:
            KeyError: If a layout parameter is missing from ``parameters``.
        """
        return flatten_tensors(parameters, layout=self)

    def unflatten(self, flat: Tensor) -> dict[str, Tensor]:
        """Split a flat vector back into per-parameter views.

        Args:
            flat: Tensor whose last dimension is the flat parameter axis.

        Returns:
            Mapping from parameter name to a ``[*flat.shape[:-1], *shape]`` view.
        """
        return unflatten_tensor(flat, layout=self)

    def unflatten_masks(self, flat_mask: Tensor) -> dict[str, Tensor]:
        """Map a flat boolean mask back onto the original parameter shapes.

        Args:
            flat_mask: Boolean tensor whose last dimension is the flat axis.

        Returns:
            Mapping from parameter name to a boolean view of the original shape.
        """
        return unflatten_masks(flat_mask, layout=self)

    def per_parameter_counts(self, flat_mask: Tensor) -> dict[str, int]:
        """Count set entries of a flat mask per parameter.

        Args:
            flat_mask: Boolean tensor whose last axis is the flat parameter axis, so a
                bare ``[N]`` mask or a ``[T, N]`` history.  The last axis is split
                along the layout offsets; no reshaping of the weight axis happens.

        Returns:
            Mapping from parameter name to the number of set entries.  A mask with
            leading axes yields a list of counts per frame instead of a single int.

        Raises:
            ValueError: If the last axis of ``flat_mask`` is not the flat parameter
                axis of this layout.
        """
        if flat_mask.shape[-1] != self.num_parameters:
            raise ValueError(
                f"flat_mask has {flat_mask.shape[-1]} entries on its last axis, "
                f"layout expects {self.num_parameters}"
            )
        counts = {
            name: flat_mask[..., start:start + numel].sum(-1)
            for name, start, numel in zip(self.names, self.offsets, self.numels)
        }
        if flat_mask.ndim == 1:
            return {name: int(value) for name, value in counts.items()}
        return {name: [int(item) for item in value] for name, value in counts.items()}

    def __repr__(self) -> str:
        return (
            f"ParameterLayout(n_tensors={len(self.names)}, num_parameters={self.num_parameters}, "
            f"names={list(self.names[:4])}{'...' if len(self.names) > 4 else ''})"
        )


def flatten_tensors(
    parameters: Mapping[str, Tensor] | Sequence[Tensor] | Iterable[Tensor],
    *,
    layout: ParameterLayout | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """Flatten one or more tensors into a single 1-D vector.

    Args:
        parameters: Mapping of names to tensors, or a sequence of tensors.
        layout: Optional precomputed layout; must describe ``parameters``.
        dtype: Optional dtype for the result, defaulting to the promoted input
            dtype.
        device: Optional device for the result.

    Returns:
        Tensor of shape ``[num_parameters]``.

    Raises:
        KeyError: If ``parameters`` is a mapping missing a layout entry.
        ValueError: If the layout does not match ``parameters``.
    """
    if isinstance(parameters, Mapping) and layout is not None:
        pairs = [(name, parameters[name]) for name in layout.names]
    else:
        pairs = _as_named_sequence(parameters)
        layout = layout or ParameterLayout.from_tensors(parameters)

    tensors = []
    for name, tensor in pairs:
        if not torch.is_tensor(tensor):
            raise TypeError(f"parameter {name!r} is not a tensor: {type(tensor)!r}")
        tensors.append(tensor.detach().reshape(-1))
    if not tensors:
        raise ValueError("nothing to flatten")

    flat = torch.cat(tensors)
    if dtype is not None:
        flat = flat.to(dtype=dtype)
    if device is not None:
        flat = flat.to(device=device)
    return flat


def unflatten_tensor(flat: Tensor, *, layout: ParameterLayout) -> dict[str, Tensor]:
    """Split a flat tensor back into per-parameter views.

    Args:
        flat: Tensor whose last dimension is the flat parameter axis.
        layout: Layout describing the packing.

    Returns:
        Mapping from parameter name to a view with the original parameter shape
        appended after the leading dimensions of ``flat``.
    """
    if flat.shape[-1] != layout.num_parameters:
        raise ValueError(f"flat has {flat.shape[-1]} entries, layout expects {layout.num_parameters}")
    return {
        name: flat[..., start:start + numel].reshape(flat.shape[:-1] + shape)
        for name, start, numel, shape in zip(layout.names, layout.offsets, layout.numels, layout.shapes)
    }


def unflatten_masks(flat_mask: Tensor, *, layout: ParameterLayout) -> dict[str, Tensor]:
    """Map a flat boolean mask back onto the original parameter shapes.

    The returned tensors are views into ``flat_mask``, so no memory is copied.

    Args:
        flat_mask: Boolean tensor whose last dimension is the flat axis.
        layout: Layout describing the packing.

    Returns:
        Mapping from parameter name to a boolean view of the original shape.
    """
    return unflatten_tensor(flat_mask, layout=layout)


def select_parameters(
    model: torch.nn.Module,
    *,
    include: Callable[[str, Tensor], bool] | Sequence[str] | None = None,
    exclude: Callable[[str, Tensor], bool] | Sequence[str] | None = None,
    trainable_only: bool = True,
    exclude_suffix: str | None = None,
) -> dict[str, Tensor]:
    """Pick the parameters of a model that should be tracked.

    Args:
        model: Model to inspect.
        include: Optional predicate or name collection; a predicate receives
            ``(name, parameter)``.
        exclude: Optional predicate or name collection applied after ``include``.
        trainable_only: Skip parameters with ``requires_grad=False``.
        exclude_suffix: Skip names ending with this suffix.  Quantized models
            expose the trainable leaves as ``<name>_orig`` while ``<name>`` is a
            buffer, so tracking ``weight`` is the natural default here; pass
            ``"_orig"`` to follow the raw leaves instead.

    Returns:
        Mapping from parameter name to the live parameter tensor, in
        ``named_parameters`` order.
    """

    def _matches(spec: Callable[[str, Tensor], bool] | Sequence[str] | None, name: str, tensor: Tensor) -> bool:
        """Whether ``name`` satisfies ``spec``; an absent spec never matches."""
        if spec is None:
            return False
        if callable(spec):
            return bool(spec(name, tensor))
        wanted = set(spec)
        return name in wanted or any(name.startswith(f"{item}.") or name.endswith(item) for item in wanted)

    selected: dict[str, Tensor] = {}
    for name, parameter in model.named_parameters():
        if not torch.is_tensor(parameter):
            continue
        if trainable_only and not parameter.requires_grad:
            continue
        if exclude_suffix is not None and name.endswith(exclude_suffix):
            continue
        if include is not None and not _matches(include, name, parameter):
            continue
        if _matches(exclude, name, parameter):
            continue
        selected[name] = parameter
    return selected
