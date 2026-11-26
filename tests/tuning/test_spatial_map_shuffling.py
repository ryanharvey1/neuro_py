import nelpy as nel
import numpy as np
import pytest
from joblib import Parallel

from neuro_py.stats.stats import get_significant_events
from neuro_py.tuning.maps import SpatialMap


def make_short_map(dim: int) -> SpatialMap:
    timestamps = np.arange(40) / 10
    positions = np.vstack(
        (
            np.linspace(0, 50, 40),
            30 + 20 * np.sin(timestamps),
            45 + 30 * np.cos(timestamps),
        )
    )[:dim]
    pos = nel.AnalogSignalArray(data=positions, timestamps=timestamps)
    spikes = nel.SpikeTrainArray(
        time=[np.array([0.3, 1.1, 2.2]), np.array([0.5, 0.9, 1.7, 2.7])], fs=1000
    )
    return SpatialMap(
        pos=pos,
        st=spikes,
        speed_thres=0,
        s_binsize=10,
        tuning_curve_sigma=0,
        max_gap=0.11,
        parallel_shuff=False,
    )


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("parallel", [False, True])
def test_short_recordings_score_only_generated_shuffles(monkeypatch, dim, parallel):
    spatial = make_short_map(dim)
    spatial.parallel_shuff = parallel
    original_map = spatial.ratemap.copy()
    original_spikes = [times.copy() for times in spatial.st.data]
    scored_counts = []

    def score(observed, shuffled):
        scored_counts.append(shuffled.shape)
        return get_significant_events(observed, shuffled)

    monkeypatch.setattr("neuro_py.tuning.maps.get_significant_events", score)
    monkeypatch.setattr("neuro_py.tuning.maps.multiprocessing.cpu_count", lambda: 2)
    # Exercise the real Parallel task iterator without starting subprocesses in tests.
    monkeypatch.setattr(
        "neuro_py.tuning.maps.Parallel",
        lambda n_jobs: Parallel(n_jobs=n_jobs, prefer="threads"),
    )
    result = spatial.shuffle_spatial_information()

    assert scored_counts == [(40, 2)]
    assert result.shape == (2,)
    assert np.all((result >= 1 / 41) & (result <= 1))
    assert spatial.n_shuff == 500
    np.testing.assert_array_equal(spatial.ratemap, original_map)
    for actual, expected in zip(spatial.st.data, original_spikes):
        np.testing.assert_array_equal(actual, expected)


def test_serial_and_parallel_shuffling_agree_for_same_draws(monkeypatch):
    spatial = make_short_map(1)
    spatial.n_shuff = 5
    monkeypatch.setattr("neuro_py.tuning.maps.multiprocessing.cpu_count", lambda: 2)
    monkeypatch.setattr(
        "neuro_py.tuning.maps.Parallel",
        lambda n_jobs: Parallel(n_jobs=n_jobs, prefer="threads"),
    )
    state = np.random.get_state()
    try:
        np.random.seed(13)
        serial = spatial.shuffle_spatial_information().copy()
        np.random.seed(13)
        spatial.parallel_shuff = True
        parallel = spatial.shuffle_spatial_information()
    finally:
        np.random.set_state(state)
    np.testing.assert_allclose(serial, parallel)
    assert spatial.n_shuff == 5


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_invalid_shuffle_counts_are_rejected(count):
    spatial = make_short_map(1)
    spatial.n_shuff = count
    with pytest.raises(ValueError, match="positive integer"):
        spatial.shuffle_spatial_information()


def test_no_running_position_samples_is_explicit():
    spatial = make_short_map(1)
    spatial.run_epochs = nel.EpochArray([[10, 11]])
    with pytest.raises(ValueError, match="running position samples"):
        spatial.shuffle_spatial_information()
