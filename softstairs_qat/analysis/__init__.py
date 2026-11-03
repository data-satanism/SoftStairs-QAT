# softstairs_qat/analysis/__init__.py

"""Offline analysis utilities for SoftStairs quantization-aware training.

The package is strictly read-only: nothing here mutates the quantizer, the model
or its parameters, so the SoftStairs core stays untouched.

Modules
-------
:mod:`softstairs_qat.analysis.ema`
    Closed-form EMAs over the temporal snapshot axis and a streaming state object.
:mod:`softstairs_qat.analysis.sensitivity`
    SoftStairs derivative (``dSS``) resolution and per-snapshot temperature mapping.
:mod:`softstairs_qat.analysis.flattening`
    Flat ``[N]`` parameter vector with ``flat index -> parameter tensor -> local index``
    bookkeeping.
:mod:`softstairs_qat.analysis.scores`
    EMA pressure, directional consistency, weight motion and the active / reservoir
    scores.
:mod:`softstairs_qat.analysis.tracking`
    Ranked three-set classification, transition tracking and per-update records.
:mod:`softstairs_qat.analysis.paramset`
    Whole-model support sets across heterogeneous parameter tensors.
:mod:`softstairs_qat.analysis.weight_types`
    On-the-fly weight-type tracker: threshold-based current state, lifetime
    activation behaviour, persistent core and discovery phase.
:mod:`softstairs_qat.analysis.reader`
    Locating and loading the saved ``[T, ...]`` snapshots.

Quick start
-----------
.. code-block:: python

    from softstairs_qat.analysis import (
        ScoreConfig, SetConfig, TrackerConfig, ParameterSetTracker, track_sequence,
    )

    records, series = track_sequence(ps, gs, config=TrackerConfig(
        score=ScoreConfig(decay_pressure=0.9),
        sets=SetConfig(active_fraction=0.01, reservoir_fraction=0.25),
    ))
    print(records[-1].transition("reservoir_to_active"))
"""

from softstairs_qat.analysis.ema import EmaState, ema_final, ema_over_time, ema_weights
from softstairs_qat.analysis.flattening import (
    ParameterLayout,
    flatten_tensors,
    select_parameters,
    unflatten_masks,
    unflatten_tensor,
)
from softstairs_qat.analysis.paramset import ParameterSetTracker, build_tracker, summarize
from softstairs_qat.analysis.reader import (
    EXPERIMENT_CHECKPOINTS,
    SnapshotSet,
    load_registered_snapshots,
    load_snapshots,
    register_checkpoint,
)
from softstairs_qat.analysis.scores import (
    ActivityScores,
    ScoreConfig,
    ScoreSeries,
    compute_scores,
    compute_score_series,
    normalize_activity,
)
from softstairs_qat.analysis.sensitivity import (
    resolve_sensitivity,
    softstairs_sensitivity,
    t_schedule_for_snapshots,
)
from softstairs_qat.analysis.tracking import (
    TRANSITION_NAMES,
    Label,
    ReservoirActiveTracker,
    SetConfig,
    TrackerConfig,
    TrackerRecord,
    TransitionEntry,
    classify_series,
    jaccard,
    labels_from_masks,
    records_to_history,
    resolve_budgets,
    summarize_transitions,
    track_sequence,
    transition_matrices,
)
from softstairs_qat.analysis.weight_types import (
    WEIGHT_TYPE_TRANSITIONS,
    ModelWeightTypeTracker,
    WeightType,
    WeightTypeConfig,
    WeightTypeTracker,
    build_weight_type_tracker,
    classify_weight_states,
)

__all__ = [
    # ema
    "EmaState",
    "ema_over_time",
    "ema_final",
    "ema_weights",
    # sensitivity
    "softstairs_sensitivity",
    "resolve_sensitivity",
    "t_schedule_for_snapshots",
    # flattening
    "ParameterLayout",
    "flatten_tensors",
    "unflatten_tensor",
    "unflatten_masks",
    "select_parameters",
    # scores
    "ScoreConfig",
    "ScoreSeries",
    "ActivityScores",
    "compute_score_series",
    "compute_scores",
    "normalize_activity",
    # tracking
    "Label",
    "TRANSITION_NAMES",
    "SetConfig",
    "TrackerConfig",
    "TrackerRecord",
    "TransitionEntry",
    "ReservoirActiveTracker",
    "track_sequence",
    "classify_series",
    "jaccard",
    "labels_from_masks",
    "transition_matrices",
    "summarize_transitions",
    "resolve_budgets",
    "records_to_history",
    # parameter sets
    "ParameterSetTracker",
    "build_tracker",
    "summarize",
    # weight types
    "WeightType",
    "WEIGHT_TYPE_TRANSITIONS",
    "WeightTypeConfig",
    "WeightTypeTracker",
    "ModelWeightTypeTracker",
    "build_weight_type_tracker",
    "classify_weight_states",
    # reader
    "EXPERIMENT_CHECKPOINTS",
    "SnapshotSet",
    "load_snapshots",
    "load_registered_snapshots",
    "register_checkpoint",
]