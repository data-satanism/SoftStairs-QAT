import pytest
import torch

from softstairs_qat.analysis import (
    Label,
    ParameterSetTracker,
    ReservoirActiveTracker,
    ScoreConfig,
    SetConfig,
    TrackerConfig,
    TRANSITION_NAMES,
    classify_series,
    compute_score_series,
    jaccard,
    labels_from_masks,
    records_to_history,
    resolve_budgets,
    summarize_transitions,
    track_sequence,
)
from softstairs_qat.analysis.flattening import ParameterLayout, flatten_tensors
from softstairs_qat.analysis.paramset import build_tracker, summarize
from softstairs_qat.analysis.tracking import transition_matrices
from softstairs_qat.core.soft_stairs import SoftStairs


def naive_ema(series, decay):
    """Reference recursion used to validate the vectorized EMA."""
    out = [series[0].clone()]
    for value in series[1:]:
        out.append(decay * out[-1] + (1.0 - decay) * value)
    return torch.stack(out)


def test_pressure_consistency_motion_match_reference_recursion():
    torch.manual_seed(0)
    ps = torch.randn(12, 4, 7)
    gs = torch.randn(12, 4, 7)
    decay = 0.8

    series = compute_score_series(
        ps,
        gs,
        config=ScoreConfig(decay_pressure=decay, decay_consistency=decay, decay_motion=decay),
    )

    expected_pressure = naive_ema(gs.abs(), decay)
    expected_signed = naive_ema(gs, decay)
    displacement = torch.zeros_like(ps)
    displacement[1:] = (ps[1:] - ps[:-1]).abs()
    expected_motion = naive_ema(displacement, decay)
    expected_consistency = (expected_signed.abs() / (expected_pressure + 1e-12)).clamp(0, 1)

    assert torch.allclose(series.pressure, expected_pressure, atol=1e-6)
    assert torch.allclose(series.signed, expected_signed, atol=1e-6)
    assert torch.allclose(series.motion, expected_motion, atol=1e-6)
    assert torch.allclose(series.consistency, expected_consistency, atol=1e-5)


def test_consistency_is_the_signed_ema_ratio():
    """``P * C == |M|``: the reservoir score is the persistent signed magnitude."""
    torch.manual_seed(1)
    series = compute_score_series(torch.randn(20, 6, 5), torch.randn(20, 6, 5))
    scores = series.final()

    assert torch.allclose(scores.pressure * scores.consistency, scores.signed.abs(), atol=1e-9)
    assert float(scores.consistency.min()) >= 0.0
    assert float(scores.consistency.max()) <= 1.0


def test_consistency_is_low_for_oscillating_gradients():
    torch.manual_seed(2)
    gs = torch.stack([torch.full((32,), 1.0 if step % 2 == 0 else -1.0) for step in range(30)])
    scores = compute_score_series(torch.randn(30, 32), gs).final()

    assert float(scores.consistency.max()) < 0.2
    assert float(scores.pressure.mean()) > 0.5


def test_active_score_is_ema_of_the_qat_gradient():
    torch.manual_seed(3)
    gs = torch.randn(10, 8)
    scores = compute_score_series(torch.randn(10, 8), gs, config=ScoreConfig(decay_pressure=0.75)).final()

    assert scores.active_score_source == "sensitivity"
    assert torch.allclose(scores.active_score, naive_ema(gs.abs(), 0.75)[-1], atol=1e-7)


def test_upstream_gradient_reproduces_the_same_active_score():
    """EMA(|H| * D) == EMA(|g|) because g = H * D and D >= 0."""
    torch.manual_seed(4)
    ps = torch.randn(9, 16)
    upstream = torch.randn(9, 16)
    temperatures = torch.linspace(0.9, 0.01, 9)
    dss = torch.stack(
        [
            SoftStairs(t=float(temperature), normalized=False, async_t_factor=1.0).derivative(frame)
            for frame, temperature in zip(ps, temperatures)
        ]
    )
    gated = upstream * dss

    default = compute_score_series(ps, gated, config=ScoreConfig(), t_values=temperatures)
    literal = compute_score_series(
        ps,
        upstream,
        config=ScoreConfig(),
        t_values=temperatures,
        upstream_gradients=upstream,
    )

    assert literal.active_score_source == "upstream"
    assert default.active_score_source == "sensitivity"
    assert torch.allclose(default.active_score, literal.active_score, atol=1e-6)


def test_sensitivity_uses_the_softstairs_derivative():
    """The tracker never re-derives dSS; it delegates to SoftStairs.derivative."""
    torch.manual_seed(5)
    ps = torch.randn(5, 12)
    temperature = 0.37
    series = compute_score_series(
        ps,
        torch.randn(5, 12),
        config=ScoreConfig(normalized=True, async_t_factor=2.0),
        t_values=temperature,
    )

    expected = SoftStairs(t=temperature, normalized=True, async_t_factor=2.0).derivative(ps)
    assert series.sensitivity is not None
    assert torch.allclose(series.sensitivity, expected, atol=1e-7)


def test_sensitivity_uses_the_per_snapshot_temperature():
    """``t`` is stepped once per epoch while snapshots are taken P times per epoch."""
    torch.manual_seed(6)
    ps = torch.randn(8, 4, 4)
    temperatures = torch.tensor([0.9, 0.5, 0.1])
    series = compute_score_series(
        ps,
        torch.randn(8, 4, 4),
        config=ScoreConfig(),
        t_values=temperatures,
        snapshots_per_epoch=4,
    )

    for step in range(8):
        expected = SoftStairs(t=float(temperatures[min(step // 4, 2)])).derivative(ps[step])
        assert torch.allclose(series.sensitivity[step], expected, atol=1e-7)


def test_reservoir_score_definition():
    torch.manual_seed(7)
    scores = compute_score_series(torch.randn(7, 30), torch.randn(7, 30)).final()

    expected = scores.pressure * scores.consistency * (1.0 - scores.activity)
    assert torch.allclose(scores.reservoir_score, expected, atol=1e-12)
    assert float(scores.activity.min()) >= 0.0
    assert float(scores.activity.max()) <= 1.0


def test_activity_normalization_is_stable_for_constant_scores():
    """A degenerate row has no spread, so nothing is *more* active than the rest."""
    scores = compute_score_series(torch.ones(4, 8), torch.ones(4, 8)).final()

    assert torch.isfinite(scores.activity).all()
    assert torch.allclose(scores.activity, torch.zeros_like(scores.activity))
    assert torch.allclose(scores.reservoir_score, scores.pressure * scores.consistency)


def test_dtypes_are_promoted_for_accumulation():
    torch.manual_seed(8)
    scores = compute_score_series(
        torch.randn(5, 16, dtype=torch.float64),
        torch.randn(5, 16, dtype=torch.bfloat16),
    ).final()

    assert scores.pressure.dtype == torch.float32
    assert scores.active_score.dtype == torch.float32
    assert torch.isfinite(scores.pressure).all()


def test_empty_and_mismatched_inputs_raise():
    with pytest.raises(ValueError):
        compute_score_series(torch.randn(0, 4), torch.randn(0, 4))
    with pytest.raises(ValueError):
        compute_score_series(torch.randn(3, 4), torch.randn(2, 4))


# ----------------------------------------------------------------------
# budgets and classification
# ----------------------------------------------------------------------
def test_budget_resolution_and_disjoint_clamping():
    assert resolve_budgets(SetConfig(active_fraction=0.1, reservoir_fraction=0.2), 1000) == (100, 200)
    assert resolve_budgets(SetConfig(active_k=10, reservoir_k=5), 1000) == (10, 5)
    # an unset reservoir budget is optional and resolves to 0
    assert resolve_budgets(SetConfig(active_k=10), 1000) == (10, 0)
    assert resolve_budgets(SetConfig(active_k=10, active_fraction=0.5), 1000) == (10, 0)
    # the reservoir budget can never exceed what the active budget leaves over
    assert resolve_budgets(SetConfig(active_fraction=0.9, reservoir_fraction=0.9), 100) == (90, 10)
    assert resolve_budgets(SetConfig(active_fraction=0.01, reservoir_fraction=0.01), 10) == (1, 1)

    with pytest.raises(ValueError):
        resolve_budgets(SetConfig(), 100)
    with pytest.raises(ValueError):
        SetConfig(active_fraction=1.5)
    with pytest.raises(ValueError):
        SetConfig(active_hysteresis=-0.1)
    with pytest.raises(ValueError):
        SetConfig(active_k=-3)


def test_ranking_is_global_over_the_whole_weight_vector():
    """The top-k is taken over the flattened tensor, not per trailing axis."""
    torch.manual_seed(9)
    series = compute_score_series(torch.randn(4, 10, 20), torch.randn(4, 10, 20))
    sets = SetConfig(active_fraction=0.05, reservoir_fraction=0.1)
    active, reservoir, dormant = classify_series(series, sets, num_parameters=200)

    assert int(active[0].sum()) == 10
    assert int(reservoir[0].sum()) == 20
    scores = series.active_score[0].reshape(-1)
    assert bool(active[0].reshape(-1)[torch.topk(scores, 10).indices].all())


def test_sets_are_disjoint_and_dormant_is_the_complement():
    torch.manual_seed(10)
    ps = torch.randn(15, 5, 9)
    gs = torch.randn(15, 5, 9)
    config = TrackerConfig(sets=SetConfig(active_fraction=2 / 45, reservoir_fraction=11 / 45))
    records, series = track_sequence(ps, gs, config=config)
    active, reservoir, dormant = classify_series(series, config.sets, num_parameters=45)

    assert not bool((active & reservoir).any())
    assert torch.equal(dormant, ~(active | reservoir))
    for step in range(15):
        assert int(active[step].sum()) == 2
        assert int(reservoir[step].sum()) == 11
        assert int(dormant[step].sum()) == 32
        assert records[step].num_active + records[step].num_reservoir + records[step].num_dormant == 45


def test_active_hysteresis_requires_aligned_previous_mask():
    series = compute_score_series(torch.randn(4, 10), torch.randn(4, 10))
    with pytest.raises(ValueError):
        classify_series(
            series,
            SetConfig(active_fraction=0.2, active_hysteresis=0.5),
            previous_active=torch.zeros(3, 10, dtype=torch.bool),
        )


def test_active_hysteresis_increases_stability():
    torch.manual_seed(11)
    gs = torch.randn(12, 40) * (1.0 + torch.arange(12, dtype=torch.float32).abs().reshape(-1, 1) * 0.4)

    def _run(hysteresis):
        config = TrackerConfig(sets=SetConfig(active_fraction=0.1, active_hysteresis=hysteresis))
        tracker = ReservoirActiveTracker(40, config=config)
        records = [tracker.update(step=step, gradient=gs[step]) for step in range(12)]
        return sum(record.active_jaccard for record in records[1:])

    assert _run(0.9) > _run(0.0)


# ----------------------------------------------------------------------
# transitions
# ----------------------------------------------------------------------
def test_summarize_transitions_uses_source_and_destination_sizes():
    counts = torch.tensor(
        [
            [1, 2, 3],  # from dormant
            [4, 5, 6],  # from active
            [7, 8, 9],  # from reservoir
        ]
    )
    previous = {"dormant": 6, "active": 15, "reservoir": 24}
    current = {"dormant": 12, "active": 15, "reservoir": 18}
    entries = summarize_transitions(counts, previous, current)

    assert entries["reservoir_to_active"].count == 8
    assert entries["reservoir_to_active"].source_fraction == pytest.approx(8 / 24)
    assert entries["reservoir_to_active"].destination_fraction == pytest.approx(8 / 15)

    assert entries["active_to_dormant"].count == 4
    assert entries["active_to_dormant"].source_fraction == pytest.approx(4 / 15)
    assert entries["active_to_dormant"].destination_fraction == pytest.approx(4 / 12)

    assert entries["active_removed"].count == 10
    assert entries["active_removed"].destination_fraction == pytest.approx(10 / 30)

    assert entries["reservoir_removed"].count == 15
    assert entries["reservoir_removed"].destination_fraction == pytest.approx(15 / 27)

    assert entries["active_persistent"].count == 5
    assert entries["reservoir_persistent"].count == 9
    assert entries["dormant_to_active"].source_fraction == pytest.approx(2 / 6)
    assert entries["dormant_to_reservoir"].destination_fraction == pytest.approx(3 / 18)
    assert set(entries) == set(TRANSITION_NAMES)


def test_transition_matrices_counts_moves():
    labels = torch.tensor(
        [
            [0, 1, 2],
            [1, 1, 0],
            [2, 1, 2],
        ]
    )
    matrices = transition_matrices(labels)

    assert matrices.shape == (2, 3, 3)
    assert int(matrices[0].sum()) == 3
    assert int(matrices[1].sum()) == 3
    assert int(matrices[0][0, 1]) == 1  # dormant -> active
    assert int(matrices[0][1, 1]) == 1  # active -> active
    assert int(matrices[0][2, 0]) == 1  # reservoir -> dormant
    assert int(matrices[1][0, 2]) == 1  # dormant -> reservoir
    assert int(matrices[1][1, 2]) == 1  # active -> reservoir
    assert int(matrices[1][1, 1]) == 1  # active -> active
    assert int(matrices[1][2, 2]) == 0  # nothing was in the reservoir at frame 1

    chunked = transition_matrices(labels, chunk_size=1)
    assert torch.equal(matrices, chunked)

    with pytest.raises(ValueError):
        transition_matrices(torch.zeros(1, 3, dtype=torch.long))


def test_labels_from_masks_encodes_all_three_sets():
    active = torch.tensor([True, False, False])
    reservoir = torch.tensor([False, True, False])
    dormant = torch.tensor([False, False, True])
    labels = labels_from_masks(active, reservoir, dormant)

    assert labels.tolist() == [int(Label.ACTIVE), int(Label.RESERVOIR), int(Label.DORMANT)]


def test_every_required_transition_is_reported():
    torch.manual_seed(12)
    records, _ = track_sequence(
        torch.randn(6, 20),
        torch.randn(6, 20),
        config=TrackerConfig(sets=SetConfig(active_fraction=0.2, reservoir_fraction=0.3)),
    )

    required = {
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
    }
    assert required == set(TRANSITION_NAMES)
    assert required.issubset(set(records[1].transitions))
    for name in TRANSITION_NAMES:
        entry = records[-1].transition(name)
        assert 0.0 <= entry.source_fraction <= 1.0
        assert 0.0 <= entry.destination_fraction <= 1.0

    with pytest.raises(KeyError):
        records[-1].transition("nonsense")


def test_transition_accounting_is_consistent():
    torch.manual_seed(13)
    records, _ = track_sequence(
        torch.randn(10, 40),
        torch.randn(10, 40),
        config=TrackerConfig(sets=SetConfig(active_fraction=0.1, reservoir_fraction=0.3)),
    )

    for record in records[1:]:
        removed = record.transition("active_removed").count
        moved_out = record.transition("active_to_reservoir").count + record.transition("active_to_dormant").count
        assert removed == moved_out

        dropped = record.transition("reservoir_removed").count
        left_reservoir = (
            record.transition("reservoir_to_active").count + record.transition("reservoir_to_dormant").count
        )
        assert dropped == left_reservoir

        # a weight can only enter the active set from the reservoir or from dormant
        assert record.transition("active_added").count == (
            record.transition("reservoir_to_active").count + record.transition("dormant_to_active").count
        )
        assert record.transition("reservoir_added").count == (
            record.transition("active_to_reservoir").count + record.transition("dormant_to_reservoir").count
        )
        # the set sizes close: persistent + added == current, persistent + removed == previous
        persistent = record.transition("active_persistent").count
        assert persistent + record.transition("active_added").count == record.num_active
        assert persistent + record.transition("active_removed").count == _previous(record, "active")
        held = record.transition("reservoir_persistent").count
        assert held + record.transition("reservoir_added").count == record.num_reservoir
        assert held + record.transition("reservoir_removed").count == _previous(record, "reservoir")
        assert record.num_active + record.num_reservoir + record.num_dormant == record.num_parameters


def _previous(record, label):
    """Recover the previous size of ``label`` from the recorded set change."""
    if label == "active":
        return record.num_active - record.transition("active_added").count + record.transition("active_removed").count
    added = record.transition("reservoir_added").count
    removed = record.transition("reservoir_removed").count
    return record.num_reservoir - added + removed


def test_jaccard_and_turnover():
    current = torch.tensor([True, True, False, False])
    previous = torch.tensor([True, False, False, False])

    assert float(jaccard(current, previous)) == pytest.approx(0.5)

    empty = torch.zeros(4, dtype=torch.bool)
    assert float(jaccard(empty, empty)) == 1.0
    assert float(jaccard(empty, current)) == 0.0

    with pytest.raises(ValueError):
        jaccard(current, current[:2])


def test_set_stability_and_turnover_are_recorded():
    torch.manual_seed(14)
    records, _ = track_sequence(
        torch.randn(8, 30),
        torch.randn(8, 30).cumsum(0) * 0.1,
        config=TrackerConfig(sets=SetConfig(active_fraction=0.1, reservoir_fraction=0.2)),
    )

    assert records[0].active_jaccard is None
    assert records[0].reservoir_jaccard is None
    for record in records[1:]:
        assert 0.0 <= record.active_jaccard <= 1.0
        assert 0.0 <= record.reservoir_jaccard <= 1.0
        assert record.active_turnover == pytest.approx(1.0 - record.active_jaccard)
        assert record.reservoir_turnover == pytest.approx(1.0 - record.reservoir_jaccard)


def test_record_fractions_and_history_columns():
    torch.manual_seed(15)
    config = TrackerConfig(
        sets=SetConfig(active_fraction=0.1, reservoir_fraction=0.2),
        record_distributions=True,
    )
    records, _ = track_sequence(torch.randn(6, 40), torch.randn(6, 40), config=config)

    assert records[0].active_fraction == pytest.approx(4 / 40)
    assert records[0].reservoir_fraction == pytest.approx(8 / 40)
    assert records[0].dormant_fraction == pytest.approx(28 / 40)

    history = records_to_history(records)
    for column in (
        "step",
        "num_active",
        "num_reservoir",
        "active_fraction",
        "reservoir_fraction",
        "active_jaccard",
        "reservoir_jaccard",
        "active_added_count",
        "active_removed_count",
        "reservoir_added_count",
        "reservoir_removed_count",
        "reservoir_to_active_count",
        "active_to_reservoir_count",
        "reservoir_to_dormant_count",
        "dormant_to_reservoir_count",
        "dormant_to_active_count",
        "active_to_dormant_count",
        "pressure_q50",
        "consistency_q50",
        "motion_q50",
        "active_score_q50",
        "reservoir_score_q50",
    ):
        assert column in history
        assert len(history[column]) == len(records)


def test_chunked_classification_matches_the_unchunked_path():
    torch.manual_seed(16)
    ps = torch.randn(20, 60)
    gs = torch.randn(20, 60)
    sets = SetConfig(active_fraction=0.1, reservoir_fraction=0.3)

    whole, series = track_sequence(ps, gs, config=TrackerConfig(sets=sets))
    chunked, _ = track_sequence(ps, gs, config=TrackerConfig(sets=sets, chunk_size=4))

    assert series.num_snapshots == 20
    for expected, actual in zip(whole, chunked):
        assert expected.num_active == actual.num_active
        assert expected.num_reservoir == actual.num_reservoir


# ----------------------------------------------------------------------
# temporal behaviour
# ----------------------------------------------------------------------
def test_streaming_tracker_matches_the_offline_path():
    torch.manual_seed(17)
    ps = torch.randn(14, 25)
    gs = torch.randn(14, 25)
    config = TrackerConfig(
        score=ScoreConfig(decay_pressure=0.85, decay_consistency=0.7, decay_motion=0.6),
        sets=SetConfig(active_fraction=0.12, reservoir_fraction=0.28),
        record_distributions=True,
    )

    offline, _ = track_sequence(ps, gs, config=config)
    tracker = ReservoirActiveTracker(25, config=config)
    streaming = [tracker.update(step=step, gradient=gs[step], parameter=ps[step]) for step in range(14)]

    assert len(streaming) == len(offline)
    for expected, actual in zip(offline, streaming):
        assert expected.num_active == actual.num_active
        assert expected.num_reservoir == actual.num_reservoir
        assert expected.num_dormant == actual.num_dormant
        assert expected.active_jaccard == pytest.approx(actual.active_jaccard)
        assert expected.reservoir_jaccard == pytest.approx(actual.reservoir_jaccard)
        for name in TRANSITION_NAMES:
            assert expected.transition(name).count == actual.transition(name).count
        expected_stats = expected.distributions["pressure"]
        assert expected_stats["q50"] == pytest.approx(actual.distributions["pressure"]["q50"], rel=1e-5)

    assert int(tracker.active_mask.sum()) == offline[-1].num_active
    assert tracker.active_mask.shape == (25,)
    assert int((tracker.active_mask & tracker.reservoir_mask).sum()) == 0
    assert torch.equal(tracker.dormant_mask, ~(tracker.active_mask | tracker.reservoir_mask))
    assert len(tracker.records) == 14


def _history_and_spike(decay):
    """Build a settled history, then apply one spike or a sustained push to weight 32."""
    torch.manual_seed(26)
    total = 64
    config = TrackerConfig(
        score=ScoreConfig(decay_pressure=decay),
        sets=SetConfig(active_fraction=0.25, reservoir_fraction=0.25),
    )
    tracker = ReservoirActiveTracker(total, config=config)
    base = torch.linspace(1.0, 2.0, total)
    for step in range(30):
        tracker.update(step=step, gradient=base)
    return tracker, base.clone()


def test_single_gradient_spike_does_not_redefine_the_reservoir():
    """One spike must not overturn a persisted EMA history, a sustained push must."""
    target = 32
    tracker, base = _history_and_spike(0.9)
    stable_active = tracker.active_mask.clone()
    stable_reservoir = tracker.reservoir_mask.clone()

    # a single moderate spike on a mid-pressure weight
    spiked = base.clone()
    spiked[target] = 3.0
    record = tracker.update(step=30, gradient=spiked)

    assert not bool(tracker.active_mask[target])
    assert int((stable_active & tracker.active_mask).sum()) == int(stable_active.sum())
    assert int((stable_reservoir & tracker.reservoir_mask).sum()) == int(stable_reservoir.sum())
    assert record.active_jaccard == 1.0
    assert record.reservoir_jaccard == 1.0

    # the same magnitude, sustained, does accumulate into the active set
    for step in range(1, 40):
        tracker.update(step=30 + step, gradient=spiked)
    assert bool(tracker.active_mask[target])
    assert float(tracker.scores.active_score[target]) > float(tracker.scores.active_score[[0, 1, 2]].max())


def test_ema_history_outlives_the_burst_that_created_it():
    """Persistence comes from the EMA, not from the latest gradient."""
    torch.manual_seed(27)
    total = 64

    def _run(decay):
        config = TrackerConfig(
            score=ScoreConfig(decay_pressure=decay),
            sets=SetConfig(active_fraction=0.25, reservoir_fraction=0.25),
        )
        tracker = ReservoirActiveTracker(total, config=config)
        quiet = torch.full((total,), 0.2)
        for step in range(10):
            tracker.update(step=step, gradient=quiet)
        burst = quiet.clone()
        burst[:16] = 5.0
        tracker.update(step=10, gradient=burst)
        for step in range(11, 60):
            tracker.update(step=step, gradient=quiet)
        return int(tracker.active_mask[:16].sum()), float(tracker.scores.pressure[:16].mean())

    with_memory, pressure_with = _run(0.95)
    without_memory, pressure_without = _run(0.0)

    assert pressure_with > pressure_without
    assert with_memory == 16
    assert without_memory == 0


def test_persistent_gradient_accumulates_into_the_active_set():
    torch.manual_seed(19)
    total = 32
    config = TrackerConfig(score=ScoreConfig(decay_pressure=0.8), sets=SetConfig(active_k=4, reservoir_k=8))
    tracker = ReservoirActiveTracker(total, config=config)

    target = slice(0, 4)
    for step in range(40):
        gradient = torch.zeros(total)
        gradient[target] = 0.05
        tracker.update(step=step, gradient=gradient)

    assert int(tracker.active_mask[target].sum()) == 4
    assert float(tracker.scores.pressure[target].max()) > float(tracker.scores.pressure[4:].max())


def test_reservoir_rejects_negligible_pressure_and_oscillation():
    torch.manual_seed(20)
    total = 12
    config = TrackerConfig(
        score=ScoreConfig(decay_pressure=0.5, pressure_quantile=0.0),
        sets=SetConfig(active_k=2, reservoir_k=4),
    )
    tracker = ReservoirActiveTracker(total, config=config)

    gradient = torch.zeros(total)
    gradient[:2] = 10.0
    gradient[2:6] = 1.0
    for step in range(10):
        tracker.update(step=step, gradient=gradient)

    assert float(tracker.scores.pressure[6:].max()) == 0.0
    assert int(tracker.reservoir_mask[6:].sum()) == 0
    assert int(tracker.reservoir_mask[:6].sum()) == 4

    oscillating = ReservoirActiveTracker(total, config=config)
    for step in range(10):
        signal = torch.zeros(total)
        signal[2:6] = 1.0 if step % 2 == 0 else -1.0
        oscillating.update(step=step, gradient=signal)
    assert float(oscillating.scores.consistency[2:6].max()) < 0.5


def test_pressure_floor_prunes_the_negligible_tail():
    torch.manual_seed(21)
    ps = torch.randn(6, 200)
    gs = torch.randn(6, 200)
    gs[:, 150:] = 0.0
    sets = SetConfig(active_fraction=0.1, reservoir_fraction=0.5)

    unfiltered, series = track_sequence(
        ps,
        gs,
        config=TrackerConfig(score=ScoreConfig(pressure_quantile=0.0), sets=sets),
    )
    filtered, _ = track_sequence(ps, gs, config=TrackerConfig(score=ScoreConfig(pressure_quantile=0.5), sets=sets))
    active, reservoir, _ = classify_series(series, sets, num_parameters=200)

    assert unfiltered[-1].num_reservoir == 100
    assert filtered[-1].num_reservoir <= 150
    assert int(reservoir[-1].reshape(-1)[150:].sum()) == 0
    del active


# ----------------------------------------------------------------------
# parameter flattening
# ----------------------------------------------------------------------
def test_arbitrary_shapes_dtypes_and_layout_round_trip():
    torch.manual_seed(22)
    tensors = {
        "layer_a.weight": torch.randn(4, 3),
        "layer_a.bias": torch.randn(4),
        "layer_b.weight": torch.randn(2, 5, 2),
        "scalar": torch.randn(()),
    }
    layout = ParameterLayout.from_tensors(tensors)
    assert layout.num_parameters == 4 * 3 + 4 + 2 * 5 * 2 + 1

    flat = flatten_tensors(tensors, layout=layout)
    for name, tensor in tensors.items():
        start, stop = layout.ranges()[name]
        assert torch.equal(flat[start:stop], tensor.reshape(-1))
        assert torch.equal(layout.unflatten(flat)[name], tensor)

    config = TrackerConfig(sets=SetConfig(active_k=3, reservoir_k=5))
    tracker = ParameterSetTracker(tensors, config=config)
    tracker.update(step=0, gradients={name: torch.randn_like(tensor) for name, tensor in tensors.items()})

    masks = tracker.masks()
    for name, tensor in tensors.items():
        for label in ("active", "reservoir", "dormant"):
            assert masks[label][name].shape == tensor.shape
            assert masks[label][name].dtype == torch.bool
        assert not bool((masks["active"][name] & masks["reservoir"][name]).any())

    counts = tracker.counts()
    assert counts["active"]["__total__"] == 3
    assert counts["reservoir"]["__total__"] == 5
    assert sum(counts["dormant"][name] for name in tensors) == layout.num_parameters - 8
    assert set(counts["dormant"]) == set(tensors) | {"__total__"}

    single = tracker.masks(Label.ACTIVE)
    assert set(single) == set(tensors)


def test_parameter_set_tracker_over_a_model():
    torch.manual_seed(23)
    model = torch.nn.Sequential(torch.nn.Linear(6, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    tracker = build_tracker(model, active_fraction=0.1, reservoir_fraction=0.2)
    assert tracker.num_parameters == 4 * 6 + 4 + 2 * 4 + 2

    parameters = dict(model.named_parameters())
    for step in range(3):
        model(torch.randn(3, 6)).sum().backward()
        tracker.update(
            step=step,
            gradients={name: parameter.grad for name, parameter in parameters.items()},
            parameters=parameters,
            t=0.5,
        )
        model.zero_grad()

    assert int(tracker.tracker.active_mask.sum()) == round(0.1 * tracker.num_parameters)
    assert int((tracker.tracker.active_mask & tracker.tracker.reservoir_mask).sum()) == 0
    assert set(tracker.masks("reservoir")) == set(parameters)
    assert len(tracker.history()) > 0


def test_update_sequence_matches_track_sequence_for_multiple_tensors():
    torch.manual_seed(24)
    shapes = {"a": (3, 4), "b": (2, 5)}
    ps = {name: torch.randn(9, *shape) for name, shape in shapes.items()}
    gs = {name: torch.randn(9, *shape) for name, shape in shapes.items()}
    config = TrackerConfig(
        score=ScoreConfig(decay_pressure=0.8),
        sets=SetConfig(active_fraction=0.1, reservoir_fraction=0.3),
    )

    tracker = ParameterSetTracker({name: ps[name][0] for name in shapes}, config=config)
    records = tracker.update_sequence(ps, gs, t_values=0.4)
    assert len(records) == 9

    layout = ParameterLayout.from_tensors({name: ps[name][0] for name in shapes})
    flat_p = torch.stack(
        [flatten_tensors({name: ps[name][step] for name in shapes}, layout=layout) for step in range(9)]
    )
    flat_g = torch.stack(
        [flatten_tensors({name: gs[name][step] for name in shapes}, layout=layout) for step in range(9)]
    )
    offline, _ = track_sequence(flat_p, flat_g, config=config)

    for expected, actual in zip(offline, records):
        assert expected.num_active == actual.num_active
        assert expected.num_reservoir == actual.num_reservoir
        assert expected.active_jaccard == pytest.approx(actual.active_jaccard)


# ----------------------------------------------------------------------
# tracker input validation and lifecycle
# ----------------------------------------------------------------------
def test_upstream_gradient_path_reproduces_the_gated_gradient_path():
    """Streaming with ``H`` + ``t`` must equal streaming with the gated ``g = H * D``."""
    torch.manual_seed(28)
    total, steps = 40, 12
    ps = torch.randn(steps, total)
    upstream = torch.randn(steps, total)
    temperatures = torch.linspace(0.8, 0.05, steps)
    dss = torch.stack([SoftStairs(t=float(temperatures[step])).derivative(ps[step]) for step in range(steps)])
    gated = upstream * dss
    config = TrackerConfig(
        score=ScoreConfig(decay_pressure=0.85),
        sets=SetConfig(active_fraction=0.1, reservoir_fraction=0.2),
    )

    from_gated = ReservoirActiveTracker(total, config=config)
    from_upstream = ReservoirActiveTracker(total, config=config)
    for step in range(steps):
        from_gated.update(step=step, gradient=gated[step])
        from_upstream.update(step=step, upstream_gradient=upstream[step], parameter=ps[step], t=float(temperatures[step]))

    assert from_upstream.scores.active_score_source == "upstream"
    assert torch.allclose(from_upstream.scores.pressure, from_gated.scores.pressure, atol=1e-5)
    assert torch.allclose(from_upstream.scores.signed, from_gated.scores.signed, atol=1e-5)
    assert torch.equal(from_upstream.active_mask, from_gated.active_mask)
    assert torch.equal(from_upstream.reservoir_mask, from_gated.reservoir_mask)


def test_upstream_gradient_path_requires_a_sensitivity_source():
    tracker = ReservoirActiveTracker(8, config=TrackerConfig(sets=SetConfig(active_k=2)))
    with pytest.raises(ValueError):
        tracker.update()
    with pytest.raises(ValueError):
        tracker.update(upstream_gradient=torch.randn(8))
    with pytest.raises(ValueError):
        tracker.update(upstream_gradient=torch.randn(8), parameter=torch.randn(8))

    record = tracker.update(upstream_gradient=torch.randn(8), parameter=torch.randn(8), t=0.25)
    assert record.num_active == 2
    assert tracker.scores.active_score_source == "upstream"


def test_tracker_rejects_conflicting_sources_and_bad_shapes():
    tracker = ReservoirActiveTracker(8, config=TrackerConfig(sets=SetConfig(active_k=2)))
    with pytest.raises(ValueError):
        tracker.update(gradient=torch.randn(8), pressure=torch.rand(8))
    with pytest.raises(ValueError):
        tracker.update(gradient=torch.randn(9))

    tracker.update(step=0, gradient=torch.randn(8))
    with pytest.raises(ValueError):
        tracker.update(step=1, sensitivity=torch.rand(8))


def test_state_is_unavailable_before_the_first_update():
    tracker = ReservoirActiveTracker(8, config=TrackerConfig(sets=SetConfig(active_k=2)))
    with pytest.raises(RuntimeError, match="scores are unavailable"):
        _ = tracker.scores
    with pytest.raises(RuntimeError):
        _ = tracker.active_mask


def test_mask_helper_and_reset():
    tracker = ReservoirActiveTracker(4, config=TrackerConfig(sets=SetConfig(active_k=1, reservoir_k=1)))
    tracker.update(step=0, gradient=torch.tensor([3.0, 2.0, 1.0, 0.0]))

    assert tracker.mask(Label.ACTIVE).shape == (4,)
    assert tracker.mask("reservoir").shape == (4,)
    assert int(tracker.mask(Label.ACTIVE).sum()) == 1
    with pytest.raises(ValueError):
        tracker.mask("nonsense")

    assert tracker.seen == 1
    tracker.reset()
    assert tracker.seen == 0
    assert tracker.records == []
    with pytest.raises(RuntimeError):
        _ = tracker.active_mask


def test_summarize_reports_the_churn_diagnostic():
    torch.manual_seed(25)
    records, _ = track_sequence(
        torch.randn(10, 60),
        torch.randn(10, 60),
        config=TrackerConfig(sets=SetConfig(active_fraction=0.1, reservoir_fraction=0.2)),
    )
    summary = summarize(records)

    for key in (
        "mean_num_active",
        "mean_num_reservoir",
        "mean_reservoir_to_active_rate",
        "mean_active_to_reservoir_rate",
        "mean_active_jaccard",
        "mean_reservoir_jaccard",
        "churn_excess",
    ):
        assert key in summary
    assert 0.0 <= summary["mean_reservoir_to_active_rate"] <= 1.0
    assert 0.0 <= summary["mean_active_jaccard"] <= 1.0