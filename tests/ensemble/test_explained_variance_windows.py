import nelpy as nel
import numpy as np
import pytest

from neuro_py.ensemble.explained_variance import ExplainedVariance


def make_ev(matching, window=20, slideby=None, control=None, bin_size=0.2):
    rng = np.random.default_rng(92)
    spikes = nel.SpikeTrainArray(
        time=[np.sort(rng.uniform(0, 120, 400)) for _ in range(4)],
        fs=30000,
        support=nel.EpochArray([[0, 120]]),
    )
    return ExplainedVariance(
        spikes,
        nel.EpochArray([[20, 40]]),
        nel.EpochArray(matching),
        nel.EpochArray([[0, 20]] if control is None else control),
        bin_size=bin_size,
        window=window,
        slideby=slideby,
    )


@pytest.mark.parametrize("slideby", [None, 10])
def test_complete_final_windows_have_requested_duration(slideby):
    ev = make_ev([[40, 100]], slideby=slideby)
    expected_starts = np.arange(40, 81, 20 if slideby is None else slideby)
    np.testing.assert_allclose(
        ev.matching_windows, np.column_stack((expected_starts, expected_starts + 20))
    )
    np.testing.assert_allclose(np.diff(ev.matching_windows, axis=1), 20)
    assert len(ev.ev) == len(expected_starts)


@pytest.mark.parametrize("slideby", [None, 10])
def test_disjoint_epochs_never_create_windows_across_gaps(slideby):
    ev = make_ev([[40, 60], [80, 100]], slideby=slideby)
    np.testing.assert_allclose(ev.matching_windows, [[40, 60], [80, 100]])
    np.testing.assert_allclose(ev.matching_time, [50, 90])
    for index, window in enumerate(ev.matching_windows):
        expected = np.corrcoef(ev.bst[nel.EpochArray(window)].data)
        expected = expected[np.tril_indices(4, k=-1)]
        np.testing.assert_allclose(ev.matching_paircorr[index], expected)


def test_incomplete_trailing_portions_are_not_reported_as_full_windows():
    ev = make_ev([[40, 95]])
    np.testing.assert_allclose(ev.matching_windows, [[40, 60], [60, 80]])


def test_whole_epoch_mode_uses_union_and_preserves_fractional_times():
    matching = [[40.1, 41.7], [80.2, 81.3]]
    ev = make_ev(matching, window=None, control=[[0.1, 0.9]], bin_size=0.05)
    assert ev.ev.shape == (1,)
    np.testing.assert_allclose(ev.matching_windows, [[40.1, 81.3]])
    np.testing.assert_allclose(ev.control_windows, [[0.1, 0.9]])
    actual_bins = ev.bst[nel.EpochArray(matching)].data
    expected = np.corrcoef(actual_bins)[np.tril_indices(4, k=-1)]
    np.testing.assert_allclose(ev.matching_paircorr[0], expected)


def test_overlapping_components_do_not_duplicate_bins_or_windows():
    ev = make_ev([[40, 70], [60, 100]])
    np.testing.assert_allclose(ev.matching_windows, [[40, 60], [60, 80], [80, 100]])


def test_component_shorter_than_window_does_not_borrow_time_from_gaps():
    with pytest.raises(ValueError, match="No complete window"):
        make_ev([[40, 50], [80, 90]])


@pytest.mark.parametrize("kwargs", [{"window": 0}, {"slideby": 0}, {"bin_size": 0}])
def test_nonpositive_window_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError, match="positive"):
        make_ev([[40, 100]], **kwargs)
