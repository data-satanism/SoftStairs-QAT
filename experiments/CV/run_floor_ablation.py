#!/usr/bin/env python3
"""Derivative-floor ablation for SoftStairs QAT (Inception v3 / STL-10).

Ablation: the SoftStairs backward gate ``D(x) = (2-t)t / (t^2 + 4(1-t)cos^2(pi x))``
(normalized variant: ``t^2 / ...``) collapses to ~``t/2`` (normalized: ~``t^2/4``)
inside the quantization wells, so late in training almost no gradient flows
through settled weights. This script patches the SoftStairs derivative ON THE
FLY (monkeypatch, nothing in ``softstairs_qat`` is modified) so the gate never
drops below a floor::

    gate(x) = max(D(x), floor)

and sweeps several floors x ``async_t_factor in {1, 10, 100}``.

Fixed protocol (as requested): exp t-schedule, no weight decay, 30 epochs,
2-bit asymmetric, type='standard'. Floor ``0`` is the unpatched baseline.

Run with bash::

    python experiments/CV/run_floor_ablation.py                     # full 4x3 grid
    python experiments/CV/run_floor_ablation.py --self-test         # verify the patch
    python experiments/CV/run_floor_ablation.py --floors 0,0.01 --async-t-factors 1,10
    python experiments/CV/run_floor_ablation.py --summary-only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CV_DIR = Path(__file__).resolve().parent
if str(CV_DIR) not in sys.path:
    sys.path.insert(0, str(CV_DIR))

import torch  # noqa: E402

import inception_stl10 as base  # noqa: E402

from softstairs_qat.core.soft_stairs import ScaledSoftStairs  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults (ablation protocol)
# ---------------------------------------------------------------------------
N_BITS = 2
N_EPOCHS = 30
WEIGHT_DECAY = 0.0
STRATEGY = "exp"
BETAS_DEFAULT = base.parse_betas("0.9:0.999")

# Gate floors to sweep, in gate units (peak of the normalized gate is 1;
# the natural well floor is ~t^2/4, i.e. ~6e-8 at t_end=5e-4). 0 = baseline.
FLOORS_DEFAULT = (0.0, 1e-3, 1e-2, 1e-1)
ASYNC_T_FACTORS_DEFAULT = (1.0, 10.0, 100.0)


# ---------------------------------------------------------------------------
# On-the-fly patched SoftStairs: floored backward gate
# ---------------------------------------------------------------------------
class _GateFloorState:
    """Mutable holder for the current gate floor (read at every backward)."""

    floor: float = 0.0


_orig_scaled_derivative = ScaledSoftStairs.derivative


def _floored_scaled_derivative(self: ScaledSoftStairs, x: torch.Tensor) -> torch.Tensor:
    """``ScaledSoftStairs.derivative`` clamped from below at the current floor."""
    deriv = _orig_scaled_derivative(self, x)
    floor = _GateFloorState.floor
    if floor > 0.0:
        deriv = torch.clamp(deriv, min=floor)
    return deriv


def apply_gate_floor_patch() -> None:
    """Monkeypatch the derivative once; only the backward gate is affected.

    The forward map is untouched. Every consumer that builds
    ``ScaledSoftStairs`` (``SoftStairsQuantizeFunction.backward``, the
    quantizer's derivative probes) picks up the floored gate.
    """
    if ScaledSoftStairs.derivative is not _floored_scaled_derivative:
        ScaledSoftStairs.derivative = _floored_scaled_derivative


def set_gate_floor(floor: float) -> None:
    _GateFloorState.floor = float(floor)


def self_test() -> None:
    """Verify the patched gate never drops below the floor."""
    apply_gate_floor_patch()
    soft = ScaledSoftStairs(t=0.01, normalized=True, async_t_factor=1.0, scale=1.0)
    x = torch.linspace(-2.0, 2.0, 200001)
    for floor in (0.0, 1e-3, 1e-2, 1e-1):
        set_gate_floor(floor)
        gate = soft.derivative(x)
        assert gate.min() >= floor, (floor, gate.min().item())
        assert torch.allclose(gate, torch.clamp(_orig_scaled_derivative(soft, x), min=floor))
    # Forward is unpatched.
    set_gate_floor(0.0)
    fwd = soft.forward(x)
    assert fwd.shape == x.shape
    print("self-test OK: floored gate >= floor everywhere; forward unpatched")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_floats(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(v) for v in value.split(",") if v.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float list {value!r}: {exc}") from exc


def build_parser():
    parser = argparse.ArgumentParser(
        description="SoftStairs derivative-floor ablation (exp schedule, no weight decay, "
                    "30 epochs, 2-bit asymmetric)",
        parents=[base.build_parser()],
        conflict_handler="resolve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(
        n_bits=N_BITS,
        epochs=N_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        strategy=STRATEGY,
        betas=BETAS_DEFAULT,
    )

    abl = parser.add_argument_group("floor ablation")
    abl.add_argument("--floors", type=parse_floats, default=FLOORS_DEFAULT,
                     metavar="F[,F...]", help="gate floor values (0 = baseline)")
    abl.add_argument("--async-t-factors", type=parse_floats, default=ASYNC_T_FACTORS_DEFAULT,
                     metavar="A[,A...]", help="async_t_factor values to sweep")
    abl.add_argument("--self-test", action="store_true",
                     help="verify the patched derivative and exit")
    return parser


# ---------------------------------------------------------------------------
# Run naming
# ---------------------------------------------------------------------------
def run_save_name(args, floor: float, beta1: float, beta2: float) -> str:
    """``beta_save_name`` plus the floor tag (async factor comes from args)."""
    name = base.beta_save_name(args, beta1, beta2)
    if floor > 0.0:
        name += f"_gfloor{floor:g}"
    else:
        name += "_gfloor0"
    return name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = build_parser().parse_args()

    if args.self_test:
        self_test()
        return

    apply_gate_floor_patch()

    if args.summary_only:
        names = []
        for floor in args.floors:
            for async_t_factor in args.async_t_factors:
                args.async_t_factor = async_t_factor
                for beta1, beta2 in args.betas:
                    names.append(run_save_name(args, floor, beta1, beta2))
        base.summarize(names, logs_dir=args.logs_dir, scale="epoch")
        return

    train_loader, val_loader, _ = base.build_data_loaders(args)
    base.reset_specific_gpu(0)

    runs = []
    for floor in args.floors:
        for async_t_factor in args.async_t_factors:
            args.async_t_factor = async_t_factor
            set_gate_floor(floor)
            for beta1, beta2 in args.betas:
                save_name = run_save_name(args, floor, beta1, beta2)
                print(f"\n=== floor={floor:g}, async_t_factor={async_t_factor:g}, "
                      f"beta1={beta1}, beta2={beta2}")
                print(f"=== save_name: {save_name}")

                base.train_model(args, save_name, beta1, beta2, train_loader, val_loader)
                runs.append(save_name)

    base.summarize(runs, logs_dir=args.logs_dir, scale="epoch")


if __name__ == "__main__":
    main()
