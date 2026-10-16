import pytest
import torch

from softstairs_qat.analysis import (
    Label,
    ModelWeightTypeTracker,
    WeightType,
    WeightTypeConfig,
    WeightTypeTracker,
    build_weight_type_tracker,
    classify_weight_states,
)
from softstairs_qat.analysis.weight_types import WEIGHT_TYPE_TRANSITIONS
from softstairs_qat.core.soft_stairs import SoftStairs

ACTIVE_SCORE = 1.0
RESERVOIR_SCORE = 0.5
DORMANT_SCORE = 0.0


def synthetic_config(**overrides):
    """Threshold config driven by synthetic scores, independent from SoftStairs."""
    defaults = dict(score_normalization="none", active_threshold=0.8, reservoir_threshold=0.3)
    defaults.update(overrides)
    return WeightTypeConfig(**defaults)


def run_scores(tracker, scores):
    """Feed a ``[T, N]`` synthetic score matrix as one tracking step per row."""
    for index in range(scores.shape[0]):
        tracker.update(step=index, activity_score=scores[index])
    return tracker


def transition_total(tracker, name):
    index = [entry[0] for entry in WEIGHT_TYPE_TRANSITIONS].index(name)
    return int(tracker._transition_counts[index].sum())


# ----------------------------------------------------------------------
# 1. permanently dormant
# ----------------------------------------------------------------------
def test_all_weights_permanently_dormant():
    tracker = run_scores(WeightTypeTracker(5, config=synthetic_config()), torch.zeros(6, 5))

    assert torch.equal(tracker.current_state, torch.full((5,), int(Label.DORMANT), dtype=torch.long))
    assert tracker.state_counts() == {"active": 0, "reservoir": 0, "dormant": 5}
    assert int(tracker.active_steps.sum()) == 0
    assert int(tracker.spell_count.sum()) == 0
    assert torch.equal(tracker.long_term_types(), torch.full((5,), int(WeightType.NEVER_ACTIVE)))
    assert tracker.long_term_type_counts() == {
        "never_active": 5,
        "rare_active": 0,
        "churning": 0,
        "persistent_active": 0,
    }
    for name, _, _ in WEIGHT_TYPE_TRANSITIONS:
        assert transition_total(tracker, name) == 0

    tracker.get_metrics()  # seeds the logging interval
    metrics = tracker.get_metrics()
    assert metrics["active_fraction"] == 0.0
    assert metrics["reservoir_fraction"] == 0.0
    assert metrics["dormant_fraction"] == 1.0
    assert metrics["active_jaccard"] == 1.0  # two empty sets score 1.0
    assert metrics["persistent_active_fraction"] == 0.0
    assert metrics["core_intersection"] == 0


# ----------------------------------------------------------------------
# 2. permanently active
# ----------------------------------------------------------------------
def test_all_weights_permanently_active():
    tracker = run_scores(WeightTypeTracker(4, config=synthetic_config()), torch.ones(5, 4))

    assert torch.equal(tracker.current_state, torch.full((4,), int(Label.ACTIVE), dtype=torch.long))
    assert torch.equal(tracker.spell_count, torch.ones(4, dtype=torch.long))
    assert torch.equal(tracker.longest_spell, torch.full((4,), 5, dtype=torch.long))
    assert torch.equal(tracker.current_spell_length, torch.full((4,), 5, dtype=torch.long))
    assert torch.allclose(tracker.time_active(), torch.ones(4))
    for name, _, _ in WEIGHT_TYPE_TRANSITIONS:
        assert transition_total(tracker, name) == 0

    tracker.get_metrics()  # seeds the logging interval
    metrics = tracker.get_metrics()
    assert metrics["active_fraction"] == 1.0
    assert metrics["dormant_fraction"] == 0.0
    assert metrics["persistent_active_fraction"] == 1.0
    assert metrics["persistent_core_jaccard"] == 1.0
    assert metrics["time_active_q50"] == pytest.approx(1.0)
    assert metrics["spells_q50"] == pytest.approx(1.0)


# ----------------------------------------------------------------------
# 3. one activation spell
# ----------------------------------------------------------------------
def test_single_activation_spell():
    scores = torch.tensor(
        [
            [DORMANT_SCORE],
            [DORMANT_SCORE],
            [ACTIVE_SCORE],
            [ACTIVE_SCORE],
            [ACTIVE_SCORE],
            [DORMANT_SCORE],
            [DORMANT_SCORE],
        ]
    )
    tracker = run_scores(WeightTypeTracker(1, config=synthetic_config()), scores)

    assert int(tracker.spell_count[0]) == 1
    assert int(tracker.longest_spell[0]) == 3
    assert int(tracker.current_spell_length[0]) == 0
    assert int(tracker.active_steps[0]) == 3
    assert float(tracker.time_active()[0]) == pytest.approx(3 / 7)
    assert tracker.current_state[0] == int(Label.DORMANT)


# ----------------------------------------------------------------------
# 4. multiple activation spells
# ----------------------------------------------------------------------
def test_multiple_activation_spells():
    pattern = [
        ACTIVE_SCORE,
        ACTIVE_SCORE,
        DORMANT_SCORE,
        ACTIVE_SCORE,
        ACTIVE_SCORE,
        ACTIVE_SCORE,
        DORMANT_SCORE,
        ACTIVE_SCORE,
    ]
    tracker = run_scores(
        WeightTypeTracker(1, config=synthetic_config()),
        torch.tensor([[value] for value in pattern]),
    )

    assert int(tracker.spell_count[0]) == 3
    assert int(tracker.longest_spell[0]) == 3
    assert int(tracker.active_steps[0]) == 6
    assert float(tracker.activation_rate()[0]) == pytest.approx(3 / 8)
    assert float(tracker.time_active()[0]) == pytest.approx(6 / 8)
    assert tracker.current_state[0] == int(Label.ACTIVE)
    assert int(tracker.current_spell_length[0]) == 1


# ----------------------------------------------------------------------
# 5. D -> A -> R -> D
# ----------------------------------------------------------------------
def test_dormant_to_active_to_reservoir_to_dormant():
    scores = torch.tensor([[value] for value in (DORMANT_SCORE, ACTIVE_SCORE, RESERVOIR_SCORE, DORMANT_SCORE)])
    tracker = run_scores(WeightTypeTracker(1, config=synthetic_config()), scores)

    assert transition_total(tracker, "dormant_to_active") == 1
    assert transition_total(tracker, "active_to_reservoir") == 1
    assert transition_total(tracker, "reservoir_to_dormant") == 1
    for name in ("dormant_to_reservoir", "reservoir_to_active", "active_to_dormant"):
        assert transition_total(tracker, name) == 0

    metrics = tracker.get_metrics()
    assert metrics["dormant_to_active"] == 1
    assert metrics["active_to_reservoir"] == 1
    assert metrics["reservoir_to_dormant"] == 1


# ----------------------------------------------------------------------
# 6. source-normalized transition rates
# ----------------------------------------------------------------------
def test_source_normalized_transition_rates():
    # w0: A A R D D   -> sources per event: A, A, R, D
    # w1: D A A D A   -> sources per event: D, A, A, D
    # totals: D->A 2, A->R 1, A->D 1, R->D 1; source populations D 3, A 4, R 1
    scores = torch.tensor(
        [
            [ACTIVE_SCORE, DORMANT_SCORE],
            [ACTIVE_SCORE, ACTIVE_SCORE],
            [RESERVOIR_SCORE, ACTIVE_SCORE],
            [DORMANT_SCORE, DORMANT_SCORE],
            [DORMANT_SCORE, ACTIVE_SCORE],
        ]
    )
    tracker = run_scores(WeightTypeTracker(2, config=synthetic_config()), scores)

    assert transition_total(tracker, "dormant_to_active") == 2
    assert transition_total(tracker, "active_to_reservoir") == 1
    assert transition_total(tracker, "active_to_dormant") == 1
    assert transition_total(tracker, "reservoir_to_dormant") == 1

    counts = tracker._transition_counts
    assert counts[[entry[0] for entry in WEIGHT_TYPE_TRANSITIONS].index("dormant_to_active")].tolist() == [0, 2]
    assert counts[[entry[0] for entry in WEIGHT_TYPE_TRANSITIONS].index("active_to_reservoir")].tolist() == [1, 0]

    metrics = tracker.get_metrics()
    assert metrics["dormant_to_active_total"] == 2
    assert metrics["dormant_to_active_total_rate"] == pytest.approx(2 / 3)
    assert metrics["active_to_reservoir_total_rate"] == pytest.approx(1 / 4)
    assert metrics["active_to_dormant_total_rate"] == pytest.approx(1 / 4)
    assert metrics["reservoir_to_dormant_total_rate"] == pytest.approx(1.0)
    assert metrics["dormant_to_reservoir_total_rate"] == 0.0
    assert metrics["reservoir_to_active_total_rate"] == 0.0


# ----------------------------------------------------------------------
# 7. active Jaccard
# ----------------------------------------------------------------------
def test_active_jaccard_metric():
    tracker = WeightTypeTracker(4, config=synthetic_config())
    tracker.update(step=0, activity_score=torch.tensor([1.0, 1.0, 1.0, 0.0]))
    first = tracker.get_metrics()
    assert "active_jaccard" not in first  # no previous tracked state yet

    tracker.update(step=1, activity_score=torch.tensor([0.0, 1.0, 1.0, 1.0]))
    metrics = tracker.get_metrics()

    assert metrics["active_jaccard"] == pytest.approx(2 / 4)
    assert metrics["active_jaccard_step_mean"] == pytest.approx(0.5)
    assert 0.0 <= metrics["reservoir_jaccard"] <= 1.0


# ----------------------------------------------------------------------
# 8. time_active
# ----------------------------------------------------------------------
def test_time_active():
    scores = torch.tensor(
        [
            [ACTIVE_SCORE, DORMANT_SCORE, DORMANT_SCORE, DORMANT_SCORE],
            [ACTIVE_SCORE, ACTIVE_SCORE, DORMANT_SCORE, DORMANT_SCORE],
            [ACTIVE_SCORE, ACTIVE_SCORE, DORMANT_SCORE, DORMANT_SCORE],
            [ACTIVE_SCORE, DORMANT_SCORE, ACTIVE_SCORE, DORMANT_SCORE],
        ]
    )
    tracker = run_scores(WeightTypeTracker(4, config=synthetic_config()), scores)

    expected = torch.tensor([1.0, 0.5, 0.25, 0.0])
    assert torch.allclose(tracker.time_active(), expected)
    metrics = tracker.get_metrics()
    # torch.quantile interpolates: q25 of [0.0, 0.25, 0.5, 1.0] and the median.
    assert metrics["time_active_q25"] == pytest.approx(0.1875)
    assert metrics["time_active_q50"] == pytest.approx(0.375)


# ----------------------------------------------------------------------
# 9. spell count
# ----------------------------------------------------------------------
def test_spell_count():
    scores = torch.tensor(
        [
            # two spells        # one spell         # never
            [ACTIVE_SCORE, ACTIVE_SCORE, DORMANT_SCORE],
            [DORMANT_SCORE, ACTIVE_SCORE, DORMANT_SCORE],
            [ACTIVE_SCORE, DORMANT_SCORE, DORMANT_SCORE],
        ]
    )
    tracker = run_scores(WeightTypeTracker(3, config=synthetic_config()), scores)

    assert tracker.spell_count.tolist() == [2, 1, 0]
    assert tracker.activation_count.tolist() == [2, 1, 0]
    assert tracker.longest_spell.tolist() == [1, 2, 0]


# ----------------------------------------------------------------------
# 10. persistent-active classification
# ----------------------------------------------------------------------
def test_persistent_active_classification():
    steps = 100
    scores = torch.zeros(steps, 5)
    scores[:, 0] = ACTIVE_SCORE  # permanently active -> PERSISTENT_ACTIVE
    scores[0:20, 1] = ACTIVE_SCORE  # three spells, time_active 0.55 -> CHURNING
    scores[40:60, 1] = ACTIVE_SCORE
    scores[80:95, 1] = ACTIVE_SCORE
    scores[0:5, 2] = ACTIVE_SCORE  # one short spell, time_active 0.05 -> RARE_ACTIVE
    # w3 never active -> NEVER_ACTIVE
    scores[10:, 4] = ACTIVE_SCORE  # exactly 0.90 -> PERSISTENT_ACTIVE boundary

    tracker = run_scores(WeightTypeTracker(5, config=synthetic_config()), scores)
    kinds = tracker.long_term_types()

    assert kinds.tolist() == [
        int(WeightType.PERSISTENT_ACTIVE),
        int(WeightType.CHURNING),
        int(WeightType.RARE_ACTIVE),
        int(WeightType.NEVER_ACTIVE),
        int(WeightType.PERSISTENT_ACTIVE),
    ]
    assert tracker.long_term_type_counts() == {
        "persistent_active": 2,
        "churning": 1,
        "rare_active": 1,
        "never_active": 1,
    }
    # first classifications seed spells without a transition: w1 re-enters twice,
    # w4 enters once; exits: w1 three times plus w2 once.
    assert transition_total(tracker, "dormant_to_active") == 3
    assert transition_total(tracker, "active_to_dormant") == 4

    metrics = tracker.get_metrics()
    assert metrics["persistent_active_fraction"] == pytest.approx(0.4)
    assert metrics["core_intersection"] == 2


# ----------------------------------------------------------------------
# 11. checkpoint save / load
# ----------------------------------------------------------------------
def test_checkpoint_save_load():
    generator = torch.Generator().manual_seed(7)
    raw = torch.rand(15, 10, generator=generator)
    scores = torch.where(raw > 0.6, ACTIVE_SCORE, torch.where(raw > 0.3, RESERVOIR_SCORE, DORMANT_SCORE))
    config = synthetic_config()

    full = run_scores(WeightTypeTracker(10, config=config), scores)

    head = WeightTypeTracker(10, config=config)
    for step in range(8):
        head.update(step=step, activity_score=scores[step])
    saved = head.state_dict()

    resumed = WeightTypeTracker(10, config=config)
    resumed.load_state_dict(saved)
    for step in range(8, 15):
        resumed.update(step=step, activity_score=scores[step])

    assert full.seen == resumed.seen
    assert torch.equal(full.current_state, resumed.current_state)
    assert torch.equal(full.previous_state, resumed.previous_state)
    assert torch.equal(full.active_steps, resumed.active_steps)
    assert torch.equal(full.spell_count, resumed.spell_count)
    assert torch.equal(full.current_spell_length, resumed.current_spell_length)
    assert torch.equal(full.longest_spell, resumed.longest_spell)
    assert torch.equal(full._transition_counts, resumed._transition_counts)
    assert torch.equal(full._state_totals, resumed._state_totals)
    assert torch.allclose(full.time_active(), resumed.time_active())
    assert torch.equal(full.long_term_types(), resumed.long_term_types())
    assert torch.allclose(full._ema_activity, resumed._ema_activity)
    assert full.transition_totals() == resumed.transition_totals()

    full_metrics = full.get_metrics()
    resumed_metrics = resumed.get_metrics()
    for key in ("active_fraction", "reservoir_fraction", "dormant_fraction", "persistent_active_fraction",
                "dormant_to_active_total", "dormant_to_active_total_rate", "time_active_q50", "spells_q50"):
        assert full_metrics[key] == pytest.approx(resumed_metrics[key])

    with pytest.raises(ValueError):
        WeightTypeTracker(9, config=config).load_state_dict(saved)


# ----------------------------------------------------------------------
# 12. multiple parameter tensors
# ----------------------------------------------------------------------
def test_multiple_parameter_tensors():
    tensors = {"a": torch.ones(2, 3), "b": torch.zeros(4)}
    tracker = ModelWeightTypeTracker(tensors, config=synthetic_config())
    assert tracker.num_parameters == 10

    tracker.update(step=0, activity_scores=tensors, parameters=tensors)

    masks = tracker.state_masks()
    assert masks["active"]["a"].shape == (2, 3)
    assert bool(masks["active"]["a"].all())
    assert bool(masks["dormant"]["b"].all())
    assert not bool((masks["active"]["a"] & masks["reservoir"]["a"]).any())

    types = tracker.long_term_masks()
    assert bool(types["never_active"]["b"].all())

    metrics = tracker.get_metrics()
    assert metrics["active_fraction"] == pytest.approx(0.6)
    assert metrics["dormant_fraction"] == pytest.approx(0.4)

    export = tracker.export_weight_statistics()
    assert export["parameter_id"].tolist() == [0] * 6 + [1] * 4
    assert export["weight_index"].tolist() == [0, 1, 2, 3, 4, 5, 0, 1, 2, 3]
    assert export["parameter_names"] == ("a", "b")
    assert export["current_state"].tolist() == [int(Label.ACTIVE)] * 6 + [int(Label.DORMANT)] * 4
    assert export["time_active"].tolist() == [1.0] * 6 + [0.0] * 4

    per_tensor = tracker.per_tensor(tracker.time_active())
    assert per_tensor["a"].shape == (2, 3)
    assert float(per_tensor["b"].sum()) == 0.0


# ----------------------------------------------------------------------
# 13. GPU execution
# ----------------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_gpu_execution():
    config = synthetic_config(device="cuda")
    tracker = WeightTypeTracker(8, config=config)
    for step in range(3):
        tracker.update(step=step, activity_score=torch.ones(8, device="cuda"))

    assert tracker.current_state.is_cuda
    assert tracker.current_state.dtype == torch.long
    assert tracker.time_active().is_cuda
    assert int(tracker.active_mask.sum()) == 8

    metrics = tracker.get_metrics()
    assert metrics["active_fraction"] == 1.0
    assert isinstance(metrics["active_fraction"], float)

    model_tracker = ModelWeightTypeTracker(
        {"w": torch.ones(2, 2)},
        config=synthetic_config(device="cuda"),
    )
    model_tracker.update(step=0, activity_scores={"w": torch.ones(2, 2)})
    assert model_tracker.tracker.current_state.is_cuda


# ----------------------------------------------------------------------
# discovery phase
# ----------------------------------------------------------------------
def test_discovery_phase_freezes_the_persistent_core():
    config = synthetic_config(discovery_fraction=None, discovery_steps=4)
    scores = torch.tensor(
        [
            [ACTIVE_SCORE, ACTIVE_SCORE, DORMANT_SCORE, DORMANT_SCORE],
            [ACTIVE_SCORE, ACTIVE_SCORE, DORMANT_SCORE, DORMANT_SCORE],
            [ACTIVE_SCORE, ACTIVE_SCORE, ACTIVE_SCORE, DORMANT_SCORE],
            [ACTIVE_SCORE, ACTIVE_SCORE, ACTIVE_SCORE, DORMANT_SCORE],
            [ACTIVE_SCORE, DORMANT_SCORE, ACTIVE_SCORE, DORMANT_SCORE],
            [ACTIVE_SCORE, DORMANT_SCORE, ACTIVE_SCORE, DORMANT_SCORE],
        ]
    )
    tracker = WeightTypeTracker(4, config=config)
    for step in range(2):
        tracker.update(step=step, activity_score=scores[step])
    assert tracker.discovery_active
    assert "discovery_end_step" not in tracker.get_metrics()

    for step in range(2, 4):
        tracker.update(step=step, activity_score=scores[step])
    assert not tracker.discovery_active
    assert tracker.discovery_core_mask is not None
    # time_active at the freeze point: w0 1.0, w1 1.0, w2 0.5, w3 0.0 -> core {w0, w1}
    assert tracker.discovery_core_mask.tolist() == [True, True, False, False]

    for step in range(4, 6):
        tracker.update(step=step, activity_score=scores[step])
    metrics = tracker.get_metrics()

    assert metrics["discovery_active"] is False
    assert metrics["discovery_end_step"] == 4
    # current active {w0, w2} vs frozen core {w0, w1}
    assert metrics["core_intersection"] == 1
    assert metrics["active_outside_core"] == 1
    assert metrics["core_outside_active"] == 1
    assert metrics["core_intersection_fraction"] == pytest.approx(0.5)
    assert metrics["active_outside_core_fraction"] == pytest.approx(0.5)
    assert metrics["core_outside_active_fraction"] == pytest.approx(0.5)


# ----------------------------------------------------------------------
# score sources
# ----------------------------------------------------------------------
def test_activity_score_reuse_path_matches_gradient_path():
    torch.manual_seed(3)
    gradient = torch.rand(6)
    from_score = WeightTypeTracker(6, config=synthetic_config())
    from_gradient = WeightTypeTracker(6, config=synthetic_config())
    from_score.update(step=0, activity_score=gradient.abs())
    from_gradient.update(step=0, gradient=gradient)

    assert torch.equal(from_score.current_state, from_gradient.current_state)
    assert torch.allclose(from_score._ema_activity, from_gradient._ema_activity)


def test_upstream_gradient_path_tracks_the_separate_emas():
    torch.manual_seed(11)
    steps, total = 5, 3
    ps = torch.randn(steps, total)
    upstream = torch.randn(steps, total)
    temperature = 0.5

    tracker = WeightTypeTracker(total, config=synthetic_config())
    for step in range(steps):
        tracker.update(step=step, upstream_gradient=upstream[step], parameter=ps[step], t=temperature)

    dss = [SoftStairs(t=temperature).derivative(ps[step]) for step in range(steps)]

    def naive_ema(series):
        out = series[0].abs().clone()
        for value in series[1:]:
            out = 0.9 * out + 0.1 * value.abs()
        return out

    assert torch.allclose(tracker._ema_pressure, naive_ema(upstream), atol=1e-6)
    assert torch.allclose(tracker._ema_sensitivity, naive_ema(torch.stack(dss)), atol=1e-6)
    gated = torch.stack([upstream[step].abs() * dss[step] for step in range(steps)])
    assert torch.allclose(tracker._ema_activity, naive_ema(gated), atol=1e-6)

    from_gated = WeightTypeTracker(total, config=synthetic_config())
    for step in range(steps):
        from_gated.update(step=step, gradient=upstream[step] * dss[step])
    assert torch.equal(tracker.current_state, from_gated.current_state)


def test_velocity_ema_tracks_weight_motion():
    tracker = WeightTypeTracker(2, config=synthetic_config(decay_velocity=0.9))
    parameters = [torch.zeros(2), torch.ones(2), torch.full((2,), 2.0), torch.full((2,), 3.0)]
    for step, parameter in enumerate(parameters):
        tracker.update(step=step, activity_score=torch.ones(2), parameter=parameter)

    expected = torch.tensor([0.0, 0.1, 0.19, 0.271])
    assert torch.allclose(tracker.ema_velocity, expected[-1:].expand(2), atol=1e-6)


def test_quantile_normalization_is_the_default():
    scores = torch.cat([torch.tensor([10.0]), torch.ones(9)])
    tracker = WeightTypeTracker(10, config=WeightTypeConfig())
    tracker.update(step=0, gradient=scores)

    states = tracker.current_state.tolist()
    assert states[0] == int(Label.ACTIVE)
    assert all(state == int(Label.DORMANT) for state in states[1:])


def test_classify_weight_states_matches_the_tracker():
    config = synthetic_config()
    scores = torch.tensor([1.0, 0.5, 0.31, 0.29, 0.0])
    tracker = WeightTypeTracker(5, config=config)
    tracker.update(step=0, activity_score=scores)

    assert classify_weight_states(scores, config).tolist() == tracker.current_state.tolist()
    assert tracker.current_state.tolist() == [
        int(Label.ACTIVE),
        int(Label.RESERVOIR),
        int(Label.RESERVOIR),
        int(Label.DORMANT),
        int(Label.DORMANT),
    ]


# ----------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------
def test_input_validation():
    tracker = WeightTypeTracker(4, config=synthetic_config())
    with pytest.raises(ValueError):
        tracker.update()
    with pytest.raises(ValueError):
        tracker.update(gradient=torch.randn(4), pressure=torch.rand(4))
    with pytest.raises(ValueError):
        tracker.update(gradient=torch.randn(5))
    with pytest.raises(ValueError):
        tracker.update(upstream_gradient=torch.randn(4))
    with pytest.raises(ValueError):
        tracker.update(activity_score=torch.randn(3))
    with pytest.raises(ValueError):
        WeightTypeTracker(0)

    metrics = tracker.get_metrics()
    assert metrics == {}

    tracker.update(step=0, activity_score=torch.ones(4))
    tracker.reset()
    assert tracker.seen == 0
    assert tracker.get_metrics() == {}
    assert int(tracker.spell_count.sum()) == 0


def test_config_validation():
    with pytest.raises(ValueError):
        WeightTypeConfig(active_threshold=0.2, reservoir_threshold=0.5)
    with pytest.raises(ValueError):
        WeightTypeConfig(score_normalization="bogus")
    with pytest.raises(ValueError):
        WeightTypeConfig(persistent_active_threshold=0.0)
    with pytest.raises(ValueError):
        WeightTypeConfig(churn_min_spells=0)
    with pytest.raises(ValueError):
        WeightTypeConfig(churn_min_time_active=0.95)
    with pytest.raises(ValueError):
        WeightTypeConfig(discovery_fraction=1.5)
    with pytest.raises(ValueError):
        WeightTypeConfig(decay_activity=1.5)
    with pytest.raises(ValueError):
        WeightTypeConfig(report_quantiles=(0.5, 2.0))


def test_build_weight_type_tracker_over_a_model():
    model = torch.nn.Sequential(torch.nn.Linear(6, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    tracker = build_weight_type_tracker(
        model,
        active_threshold=0.8,
        reservoir_threshold=0.3,
        total_steps=10,
        discovery_fraction=0.2,
    )
    assert tracker.num_parameters == 4 * 6 + 4 + 2 * 4 + 2
    assert tracker.tracker._discovery_limit() == 2

    parameters = dict(model.named_parameters())
    for step in range(3):
        tracker.update(
            step=step,
            gradients={name: torch.rand_like(tensor) for name, tensor in parameters.items()},
            parameters=parameters,
            t=0.5,
        )

    metrics = tracker.get_metrics()
    assert metrics["tracking_steps"] == 3
    assert 0.0 <= metrics["active_fraction"] <= 1.0
    assert 0.0 <= metrics["dormant_fraction"] <= 1.0
    export = tracker.export_weight_statistics()
    assert set(export["parameter_names"]) == set(parameters)
