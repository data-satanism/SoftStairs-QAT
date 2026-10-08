# softstairs_qat/analysis/ema.py

"""Exponential moving averages over the temporal (snapshot) dimension.

The snapshot convention used across this repository is

    ps.shape = [T, ...]
    gs.shape = [T, ...]

where ``T`` indexes support updates taken inside training (see
``experiments/CV/gradient_analysis.ipynb`` for the offline reader and
``experiments/CV/inception_stl10.ipynb`` for the ``GradSaver`` callback).

Every routine here is vectorized over that leading dimension: the recursion
``e_k = decay * e_{k-1} + (1 - decay) * x_k`` is evaluated in closed form by a
logarithmic-depth inclusive scan (:func:`_scan`), so there is no Python loop over
time and no ``[T, num_parameters]`` Python list.  Chunking along ``dim=0`` keeps
peak memory at one block instead of the whole series while producing (up to float
associativity) the same numbers as the unchunked computation.  The batch and
streaming entry points (:func:`ema_over_time` and :class:`EmaState`) share the
same recursion and the same ``init`` convention, so a score computed frame by
frame equals the same score computed on the whole series.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = [
    "ema_over_time",
    "ema_final",
    "ema_weights",
    "EmaState",
]

_VALID_INITS = ("first", "zero")
_LOW_PRECISION = (torch.float16, torch.bfloat16)


def _check_decay(decay: float) -> float:
    """Validate an EMA decay factor.

    Args:
        decay: Decay factor in ``[0, 1]``.

    Returns:
        The decay as a ``float``.

    Raises:
        ValueError: If ``decay`` is outside ``[0, 1]``.
    """
    if not 0.0 <= decay <= 1.0:
        raise ValueError(f"decay must lie in [0, 1], got {decay}")
    return float(decay)


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Promote low-precision dtypes so cumulative sums stay accurate.

    Args:
        dtype: Dtype of the input tensor.

    Returns:
        ``torch.float32`` for half/bfloat16 inputs, otherwise ``dtype``.
    """
    return torch.float32 if dtype in _LOW_PRECISION else dtype


def _as_series(x: Tensor, dim: int) -> Tensor:
    """Move ``dim`` to the front and flatten every trailing dimension.

    Args:
        x: Input tensor of any shape.
        dim: Temporal dimension to average over.

    Returns:
        A ``[T, N]`` tensor (possibly a copy when ``x`` is not contiguous).
    """
    if x.ndim == 0:
        return x.reshape(1, 1)
    if dim != 0:
        x = x.movedim(dim, 0)
    return x.reshape(x.shape[0], -1)


def ema_weights(length: int, decay: float, device=None, dtype=None) -> Tensor:
    """Build normalized geometric weights for a closed-form, bias-free EMA.

    Args:
        length: Number of steps ``T``.
        decay: Decay factor in ``[0, 1]``.
        device: Optional device for the returned tensor.
        dtype: Optional dtype for the returned tensor.

    Returns:
        Tensor of shape ``[T]`` holding ``(1 - decay) * decay ** (T - 1 - k)``
        divided by ``1 - decay ** T``.  The weights sum to one, which removes
        the warm-up bias of a zero-initialized EMA.
    """
    decay = _check_decay(decay)
    dtype = _compute_dtype(dtype or torch.float32)
    powers = torch.arange(length - 1, -1, -1, device=device, dtype=dtype)
    weights = (1.0 - decay) * torch.pow(decay, powers)
    return weights / weights.sum().clamp_min(torch.finfo(dtype).tiny)


def _scan(series: Tensor, decay: float) -> Tensor:
    """Evaluate ``y_k = decay * y_{k-1} + x_k`` in closed form, loop-free over time.

    A first-order linear recurrence with a constant coefficient is solved by a
    Hillis-Steele inclusive scan: doubling the stride each round covers exponentially
    more history, so ``ceil(log2(T))`` vectorized rounds replace ``T`` sequential
    ones.  Every multiplier is ``decay ** stride <= 1``, so the scan is numerically
    stable -- unlike the closed form ``decay ** -k * cumsum(x * decay ** k)``, whose
    reciprocal powers overflow for small decay.

    Args:
        series: Observations shaped ``[L, N]``.
        decay: Decay factor in ``[0, 1]``.

    Returns:
        Tensor with the shape of ``series`` holding the recurrence result.
    """
    length = series.shape[0]
    result = series.clone()
    stride = 1
    while stride < length:
        factor = decay**stride
        result[stride:] = factor * result[: length - stride].clone() + result[stride:]
        stride *= 2
    return result


def ema_over_time(
    x: Tensor,
    decay: float = 0.9,
    *,
    dim: int = 0,
    init: str = "first",
    chunk_size: int | None = None,
    out: Tensor | None = None,
) -> Tensor:
    """Compute the EMA series of ``x`` along ``dim`` without a time loop.

    Two initialization conventions are supported:

    ``init="first"``
        ``e_0 = x_0`` and ``e_k = decay * e_{k-1} + (1 - decay) * x_k`` for
        ``k >= 1``.  This mirrors how optimizer momentum buffers are seeded
        from the first observed gradient, and it is the convention used by
        :class:`EmaState`, so streaming and batch scoring agree.
    ``init="zero"``
        ``e_{-1} = 0``, i.e. the geometric kernel ``(1 - decay) * decay ** (k - j)``.
        The warm-up bias is removed by construction because the kernel is
        normalized inside the routine.

    Args:
        x: Input tensor, typically ``[T, ...]``.
        decay: Decay factor in ``[0, 1]``.
        dim: Temporal dimension.  Non-zero values are moved to the front.
        init: One of ``"first"`` or ``"zero"``.
        chunk_size: Number of leading slices handled per block.  ``None`` processes
            the whole tensor at once; a positive integer caps peak memory at
            ``chunk_size * prod(x.shape[1:])`` elements.  Results agree with the
            unchunked computation up to float associativity.
        out: Optional destination tensor with the shape of ``x``, useful to reuse
            buffers across repeated calls.

    Returns:
        Tensor with the same shape as ``x`` holding the EMA series.
    """
    if init not in _VALID_INITS:
        raise ValueError(f"init must be one of {_VALID_INITS}, got {init!r}")
    decay = _check_decay(decay)
    if chunk_size is not None and int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    original_shape = tuple(x.shape)
    series = _as_series(x, dim)
    steps, width = series.shape

    if out is not None and tuple(out.shape) != original_shape:
        raise ValueError(f"out must have shape {original_shape}, got {tuple(out.shape)}")

    result_dtype = series.dtype
    work = series if result_dtype not in _LOW_PRECISION else series.float()
    result = torch.empty((steps, width), dtype=work.dtype, device=work.device)

    if steps == 0:
        if out is not None:
            out.copy_(result.reshape(original_shape))
            return out
        return result.to(result_dtype).reshape(original_shape)

    # Seed exactly like EmaState: the first observation initializes the recursion.
    state = work[0].clone() if init == "first" else torch.zeros(width, dtype=work.dtype, device=work.device)
    offset = 1 if init == "first" else 0
    if offset:
        result[0] = state

    remaining = steps - offset
    if remaining > 0:
        block = remaining if chunk_size is None else min(int(chunk_size), remaining)
        for start in range(offset, steps, block):
            stop = min(start + block, steps)
            length = stop - start
            chunk = work[start:stop]

            # Recursion inside the block, carried in by ``state``:
            #   e_{start + m} = decay ** (m + 1) * state + (1 - decay) * sum_j decay ** (m - j) * x_{start + j}
            recent = _scan((1.0 - decay) * chunk, decay)
            exponents = torch.arange(1, length + 1, dtype=work.dtype, device=work.device)
            carried = state.unsqueeze(0) * torch.pow(decay, exponents).unsqueeze(1)
            result[start:stop] = recent + carried
            state = result[stop - 1].clone()

    if out is not None:
        out.copy_(result.reshape(original_shape))
        return out
    return result.to(result_dtype).reshape(original_shape)


def ema_final(x: Tensor, decay: float = 0.9, *, dim: int = 0, init: str = "first") -> Tensor:
    """Return only the last element of :func:`ema_over_time`.

    Useful when the temporal history is not needed but the final persistent
    estimate is, for example when scoring a single support update.

    Args:
        x: Input tensor, typically ``[T, ...]``.
        decay: Decay factor in ``[0, 1]``.
        dim: Temporal dimension.
        init: One of ``"first"`` or ``"zero"``.

    Returns:
        Tensor shaped like ``x`` with the temporal dimension removed.
    """
    if x.ndim == 0 or x.shape[dim] == 0:
        raise ValueError("cannot take the final EMA of an empty temporal axis")
    series = ema_over_time(x, decay, dim=dim, init=init)
    return series.select(dim, series.shape[dim] - 1)


class EmaState:
    """Streaming EMA state for one signal.

    Holds only the current value, so a long run never accumulates a
    ``[T, num_parameters]`` history.  Each :meth:`update` is a single
    vectorized recursion step over the parameter axis.
    """

    def __init__(
        self,
        decay: float = 0.9,
        *,
        init: str = "first",
        shape: tuple[int, ...] | None = None,
        device=None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize an empty EMA state.

        Args:
            decay: Decay factor in ``[0, 1]``.
            init: One of ``"first"`` or ``"zero"``.
            shape: Optional shape of the signal; lets :attr:`value` be read
                before the first :meth:`update`.
            device: Device for the state buffer.
            dtype: Dtype for the state buffer.
        """
        if init not in _VALID_INITS:
            raise ValueError(f"init must be one of {_VALID_INITS}, got {init!r}")
        self.decay = _check_decay(decay)
        self.init = init
        self.dtype = _compute_dtype(dtype)
        self._value: Tensor | None = torch.zeros(shape, dtype=self.dtype, device=device) if shape is not None else None
        self._seen = 0

    @property
    def seen(self) -> int:
        """Number of observations folded into the state."""
        return self._seen

    @property
    def value(self) -> Tensor:
        """Current EMA estimate.

        Raises:
            RuntimeError: If the state has never been updated.
        """
        if self._value is None:
            raise RuntimeError("EmaState.value is unavailable before the first update")
        return self._value

    def update(self, x: Tensor) -> Tensor:
        """Fold a new observation into the state.

        Args:
            x: Observation shaped like the configured ``shape``.

        Returns:
            The updated EMA estimate.
        """
        if not torch.is_tensor(x):
            raise TypeError(f"expected a tensor observation, got {type(x)!r}")
        x = x.detach().to(dtype=self.dtype)
        if self._value is None:
            self._value = torch.zeros_like(x)
        elif x.shape != self._value.shape:
            raise ValueError(f"observation shape {tuple(x.shape)} != state shape {tuple(self._value.shape)}")

        if self._seen == 0 and self.init == "first":
            self._value = x.clone()
        else:
            self._value = self.decay * self._value + (1.0 - self.decay) * x
        self._seen += 1
        return self._value

    def reset(self) -> None:
        """Drop the accumulated state, keeping the allocated buffer."""
        if self._value is not None:
            self._value.zero_()
        self._seen = 0

    def effective_window(self) -> float:
        """Approximate number of observations shaping the current estimate.

        Returns:
            ``1 / (1 - decay)``, or ``inf`` for a non-decaying average.
        """
        return math.inf if self.decay >= 1.0 else 1.0 / (1.0 - self.decay)

    def __repr__(self) -> str:
        shape = None if self._value is None else tuple(self._value.shape)
        return f"EmaState(decay={self.decay}, init={self.init!r}, seen={self._seen}, shape={shape})"
