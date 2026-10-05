import pathlib

import pytest
import torch

from softstairs_qat.analysis.reader import (
    EXPERIMENT_CHECKPOINTS,
    available_checkpoints,
    load_hparams,
    load_registered_snapshots,
    load_snapshots,
    load_t_schedule,
    register_checkpoint,
    resolve_checkpoint,
)
from softstairs_qat.analysis.scores import ScoreConfig
from softstairs_qat.analysis.sensitivity import t_schedule_for_snapshots
from softstairs_qat.analysis.tracking import SetConfig, TrackerConfig, track_sequence

REFERENCE_RUN = EXPERIMENT_CHECKPOINTS[
    "InceptionNet_STL10_strcyclic_af2_b8_tstandard_majNone_norm/version_2/fc.weight"
]

requires_reference_run = pytest.mark.skipif(
    not pathlib.Path(REFERENCE_RUN).is_dir(),
    reason="the reference snapshot directory is not present",
)


def write_snapshots(directory, epochs, indices):
    """Write ``param_*`` / ``grad_*`` files with the given epoch and index grid.

    Each snapshot receives a distinct constant value so ordering is verifiable.
    """
    directory.mkdir(parents=True, exist_ok=True)
    shape = (4, 5)
    state = {"value": 0.0}

    def _next():
        current = state["value"]
        state["value"] = current + 1.0
        return current

    for epoch in epochs:
        # An empty ``indices`` collection means the older one-snapshot-per-epoch layout.
        for index in indices or [None]:
            tag = f"{epoch}.{index}" if index is not None else str(epoch)
            value = torch.full(shape, _next())
            torch.save(value, directory / f"param_{tag}.pth")
            torch.save(-value, directory / f"grad_{tag}.pth")
    return shape


def test_load_snapshots_flat_layout(tmp_path):
    """The older layout writes one snapshot per epoch, so P == 1."""
    shape = write_snapshots(tmp_path / "fc.weight", epochs=range(5), indices=[])
    snapshots = load_snapshots(tmp_path / "fc.weight")

    assert snapshots.num_snapshots == 5
    assert snapshots.snapshots_per_epoch == 1
    assert snapshots.weight_shape == shape
    assert snapshots.ps.shape == (5, *shape)
    assert snapshots.gs.shape == (5, *shape)
    assert torch.equal(snapshots.gs, -snapshots.ps)
    assert snapshots.epochs.tolist() == [0, 1, 2, 3, 4]


def test_load_snapshots_dotted_layout(tmp_path):
    """``param_<epoch>.<index>`` carries P support updates per epoch."""
    shape = write_snapshots(tmp_path / "fc.weight", epochs=range(3), indices=range(4))
    snapshots = load_snapshots(tmp_path / "fc.weight")

    assert snapshots.num_snapshots == 12
    assert snapshots.snapshots_per_epoch == 4
    assert snapshots.weight_shape == shape
    assert snapshots.epochs.tolist() == [epoch for epoch in range(3) for _ in range(4)]
    assert snapshots.indices.tolist() == list(range(4)) * 3


def test_load_snapshots_sorts_by_epoch_then_index(tmp_path):
    directory = tmp_path / "fc.weight"
    write_snapshots(directory, epochs=range(3), indices=range(3))
    snapshots = load_snapshots(directory)

    assert snapshots.epochs.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert snapshots.indices.tolist() == [0, 1, 2] * 3
    assert torch.all(snapshots.ps[1:] > snapshots.ps[:-1])


def test_load_snapshots_reports_a_missing_kind(tmp_path):
    write_snapshots(tmp_path / "fc.weight", epochs=range(2), indices=[])
    (tmp_path / "fc.weight" / "grad_0.pth").unlink()
    (tmp_path / "fc.weight" / "grad_1.pth").unlink()

    with pytest.raises(FileNotFoundError):
        load_snapshots(tmp_path / "fc.weight")


def test_load_snapshots_rejects_a_count_mismatch(tmp_path):
    directory = tmp_path / "fc.weight"
    write_snapshots(directory, epochs=range(3), indices=[])
    (directory / "param_2.pth").unlink()

    with pytest.raises(ValueError, match="snapshots"):
        load_snapshots(directory)


def test_load_snapshots_reports_an_empty_directory(tmp_path):
    directory = tmp_path / "fc.weight"
    directory.mkdir()

    with pytest.raises(FileNotFoundError):
        load_snapshots(directory)
    with pytest.raises(FileNotFoundError):
        load_snapshots(tmp_path / "missing")


def test_snapshot_set_flattens_to_a_time_by_weight_matrix(tmp_path):
    write_snapshots(tmp_path / "fc.weight", epochs=range(3), indices=[])
    snapshots = load_snapshots(tmp_path / "fc.weight")
    ps, gs = snapshots.flat()

    assert ps.shape == (3, 20)
    assert gs.shape == (3, 20)
    assert torch.equal(ps[0].reshape(4, 5), snapshots.ps[0])


def test_register_and_resolve_checkpoint(tmp_path):
    directory = tmp_path / "custom" / "fc.weight"
    write_snapshots(directory, epochs=range(2), indices=[])
    key = "custom/version_9/fc.weight"
    try:
        stored = register_checkpoint(key, directory)
        assert stored == str(directory.resolve())
        assert available_checkpoints()[key] == stored

        name, path = resolve_checkpoint(key)
        assert name == key
        assert path == directory.resolve()

        name, path = resolve_checkpoint(directory)
        assert name == "fc.weight"
        assert path == directory.resolve()

        snapshots = load_registered_snapshots(key)
        assert snapshots.name == key
        assert snapshots.num_snapshots == 2
    finally:
        EXPERIMENT_CHECKPOINTS.pop(key, None)
    assert key not in available_checkpoints()


def test_resolve_checkpoint_requires_a_choice_when_ambiguous(tmp_path):
    write_snapshots(tmp_path / "a" / "fc.weight", epochs=range(1), indices=[])
    write_snapshots(tmp_path / "b" / "fc.weight", epochs=range(1), indices=[])
    register_checkpoint("a/version_0/fc.weight", tmp_path / "a" / "fc.weight")
    register_checkpoint("b/version_0/fc.weight", tmp_path / "b" / "fc.weight")
    try:
        with pytest.raises(ValueError, match="several entries"):
            resolve_checkpoint(None)
    finally:
        del EXPERIMENT_CHECKPOINTS["a/version_0/fc.weight"]
        del EXPERIMENT_CHECKPOINTS["b/version_0/fc.weight"]

    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(tmp_path / "does-not-exist")


@requires_reference_run
def test_default_registry_entry_loads_the_user_specified_run():
    snapshots = load_registered_snapshots()

    assert snapshots.path.name == "fc.weight"
    assert snapshots.num_snapshots == 30
    assert snapshots.weight_shape == (10, 2048)
    assert snapshots.ps.shape == snapshots.gs.shape
    assert snapshots.indices.numel() == snapshots.num_snapshots

    explicit = load_registered_snapshots(
        "InceptionNet_STL10_strcyclic_af2_b8_tstandard_majNone_norm/version_2/fc.weight"
    )
    assert torch.equal(explicit.ps, snapshots.ps)


@requires_reference_run
def test_t_schedule_is_rebuilt_from_hparams_and_expanded_per_epoch():
    run_dir = pathlib.Path(REFERENCE_RUN).parent
    hparams = load_hparams(run_dir)
    assert hparams["quantization_hparams"]["type"] == "standard"

    schedule = load_t_schedule(run_dir)
    assert len(schedule) == hparams["quantization_hparams"]["n_steps"]
    assert all(0.0 <= value <= 1.0 for value in schedule)

    snapshots = load_registered_snapshots()
    per_snapshot = t_schedule_for_snapshots(
        schedule, snapshots.num_snapshots, snapshots_per_epoch=snapshots.snapshots_per_epoch
    )
    assert per_snapshot.shape == (snapshots.num_snapshots,)
    for step in range(snapshots.num_snapshots):
        epoch = int(snapshots.epochs[step])
        assert float(per_snapshot[step]) == pytest.approx(schedule[epoch])


@requires_reference_run
def test_end_to_end_on_the_user_specified_run():
    snapshots = load_registered_snapshots()
    run_dir = snapshots.path.parent
    config = TrackerConfig(
        score=ScoreConfig(normalized=True, async_t_factor=2.0),
        sets=SetConfig(active_fraction=0.01, reservoir_fraction=0.10),
    )
    temperatures = t_schedule_for_snapshots(
        load_t_schedule(run_dir), snapshots.num_snapshots, snapshots_per_epoch=snapshots.snapshots_per_epoch
    )

    records, series = track_sequence(snapshots.ps, snapshots.gs, config=config, t_values=temperatures)
    total = snapshots.weight_shape[0] * snapshots.weight_shape[1]

    assert len(records) == snapshots.num_snapshots
    assert series.sensitivity is not None
    for record in records:
        assert record.num_active == 205
        assert record.num_active + record.num_reservoir + record.num_dormant == total
        assert 0.0 <= record.active_fraction <= 1.0
        assert 0.0 <= record.reservoir_fraction <= 1.0
        if record.active_jaccard is not None:
            assert 0.0 <= record.active_jaccard <= 1.0
            assert 0.0 <= record.reservoir_jaccard <= 1.0
        assert 0.0 <= record.transition("reservoir_to_active").source_fraction <= 1.0


def test_load_t_schedule_requires_hparams(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_t_schedule(tmp_path)
    with pytest.raises(FileNotFoundError):
        load_hparams(tmp_path / "nope")


def test_load_t_schedule_honours_overrides():
    run_dir = pathlib.Path(REFERENCE_RUN).parent
    if not run_dir.is_dir():
        pytest.skip("the reference run is not present")

    schedule = load_t_schedule(run_dir, t_start=0.42)
    assert schedule[0] == pytest.approx(0.42, abs=1e-6) or schedule[0] == pytest.approx(1.0)