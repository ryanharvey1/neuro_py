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


@pytest.mark.parametrize("period", ["matching", "control"])
def test_ignored_sub_bin_component_does_not_prevent_valid_windows(period):
    epochs = [[40, 60], [80.21, 80.25]]
    ev = make_ev(
        epochs if period == "matching" else [[40, 60]],
        control=epochs if period == "control" else None,
    )
    expected = np.corrcoef(ev.bst[nel.EpochArray([[40, 60]])].data)
    actual = ev.matching_paircorr if period == "matching" else ev.control_paircorr
    np.testing.assert_allclose(actual[0], expected[np.tril_indices(4, k=-1)])


@pytest.mark.parametrize("window,slideby", [(None, None), (20, None), (20, 10)])
def test_window_correlations_match_nelpy_complete_bin_selection(window, slideby):
    epochs = nel.EpochArray([[40.13, 60.37], [80.17, 110.41]])
    ev = make_ev(epochs.data, window=window, slideby=slideby)
    restricted = ev.bst[epochs]
    for index, bounds in enumerate(ev.matching_windows):
        data = restricted[nel.EpochArray(bounds)].data
        expected = np.corrcoef(data)[np.tril_indices(4, k=-1)]
        np.testing.assert_allclose(ev.matching_paircorr[index], expected)


def test_window_selection_does_not_deepcopy_recording(monkeypatch):
    import copy

    from nelpy.core._eventarray import BinnedEventArray

    ev = make_ev([[40, 100]])
    original = copy.deepcopy

    def check_copy(obj, *args, **kwargs):
        assert not isinstance(obj, BinnedEventArray), "full recording copied per window"
        return original(obj, *args, **kwargs)

    monkeypatch.setattr(copy, "deepcopy", check_copy)
    actual = ev._ExplainedVariance__time_resolved_correlation(
        ev.matching_windows, ev.matching
    )
    np.testing.assert_allclose(actual, ev.matching_paircorr)
    # A connected window uses a view, leaving the original recording untouched.
    data = ev._ExplainedVariance__get_window_data(nel.EpochArray([[40, 60]]))
    assert np.shares_memory(data, ev.bst.data)


def test_window_data_preserves_recording_support_gaps_and_empty_selections():
    ev = make_ev([[40, 100]])
    ev.bst = ev.st[nel.EpochArray([[0, 60], [80, 120]])].bin(ds=ev.bin_size)
    for bounds in ([[40.13, 100.37]], [[60, 80]], [[130, 140]]):
        epochs = nel.EpochArray(bounds)
        data = ev._ExplainedVariance__get_window_data(epochs)
        np.testing.assert_array_equal(data, ev.bst[epochs].data)
    # Entirely sub-bin components contribute zero samples, alongside valid ones.
    data = ev._ExplainedVariance__get_window_data(
        nel.EpochArray([[40, 60], [80.21, 80.25]])
    )
    np.testing.assert_array_equal(data, ev.bst[nel.EpochArray([[40, 60]])].data)


@pytest.mark.parametrize("kwargs", [{"window": 0}, {"slideby": 0}, {"bin_size": 0}])
def test_nonpositive_window_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError, match="positive"):
        make_ev([[40, 100]], **kwargs)
