import numpy as np
import pytest

from neuro_py.process.peri_event import (
    compute_psth,
    count_in_interval,
    crossCorr,
    event_spiking_threshold,
    event_triggered_average_irregular_sample,
    get_raster_points,
    joint_peth,
    peth_matrix,
)


def test_crosscorr_even_nbins_is_adjusted_and_values_are_expected():
    t1 = np.array([1.0, 2.0])
    t2 = np.array([1.1, 1.3, 2.1, 2.3])

    result = crossCorr(t1, t2, binsize=0.2, nbins=4)

    assert result.shape == (5,)
    np.testing.assert_allclose(result, np.array([0.0, 0.0, 0.0, 5.0, 2.5]))


def test_compute_psth_with_nonsymmetric_window_crops_to_original_range():
    data = np.array([[0.95, 1.05, 1.25], [0.90, 1.10, 1.30]], dtype=float)
    event = np.array([1.0])

    psth = compute_psth(data, event, bin_width=0.1, window=[-0.1, 0.3])

    assert psth.shape == (4, 2)
    np.testing.assert_allclose(
        psth.index.values,
        np.array([-0.1, 0.0, 0.1, 0.2]),
        atol=1e-12,
    )


def test_joint_peth_identical_constant_peths_give_zero_difference():
    peth = np.array([[1.0, 2.0], [1.0, 2.0]])

    joint, expected, difference = joint_peth(peth, peth, smooth_std=0)

    assert joint.shape == (2, 2)
    assert expected.shape == (2, 2)
    assert difference.shape == (2, 2)
    np.testing.assert_allclose(difference, np.zeros((2, 2)), atol=1e-12)


def test_get_raster_points_returns_expected_offsets_and_indices():
    data = np.array([0.90, 1.00, 1.10, 1.90, 2.05])
    time_ref = np.array([1.0, 2.0])

    x, y, times = get_raster_points(
        data, time_ref, bin_width=0.1, window=np.array([-0.15, 0.15])
    )

    np.testing.assert_allclose(times, np.array([-0.15, -0.05, 0.05, 0.15]))
    np.testing.assert_allclose(x, np.array([-0.10, 0.0, 0.10, -0.10, 0.05]))
    np.testing.assert_allclose(y, np.array([0.0, 0.0, 0.0, 1.0, 1.0]))


def test_peth_matrix_returns_count_matrix_and_expected_time_centers():
    data = np.array([0.9, 1.1, 2.1, 2.3])
    time_ref = np.array([1.0, 2.0])

    h, times = peth_matrix(data, time_ref, bin_width=0.2, n_bins=5)

    np.testing.assert_allclose(times, np.array([-0.4, -0.2, 0.0, 0.2, 0.4]), atol=1e-12)
    expected = np.column_stack(
        [
            crossCorr(np.array([time_ref[0]]), data, binsize=0.2, nbins=5) * 0.2,
            crossCorr(np.array([time_ref[1]]), data, binsize=0.2, nbins=5) * 0.2,
        ]
    )
    np.testing.assert_allclose(h, expected)


def test_event_triggered_average_irregular_sample_matches_expected_bin_stats():
    timestamps = np.array([0.90, 0.95, 1.05, 1.10, 1.90, 2.00, 2.10])
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    time_ref = np.array([1.0, 2.0])

    avg, std = event_triggered_average_irregular_sample(
        timestamps, data, time_ref, bin_width=0.1, window=(-0.1, 0.1)
    )

    np.testing.assert_allclose(avg.index.values, np.array([-0.05, 0.05, 0.15]))
    np.testing.assert_allclose(avg[0].values, np.array([4.0, 4.5, 5.5]))
    np.testing.assert_allclose(
        std[0].values,
        np.array(
            [
                np.std([2.0, 6.0], ddof=1),
                np.std([6.0, 3.0], ddof=1),
                np.std([4.0, 7.0], ddof=1),
            ]
        ),
    )


def test_count_in_interval_counts_binary_and_firing_rate():
    unit_1 = np.array([0.1, 0.3, 0.5, 1.2, 1.8])
    unit_2 = np.array([0.2, 0.8, 1.5])
    data = np.array([unit_1, unit_2], dtype=object)
    event_starts = np.array([0.0, 1.0])
    event_stops = np.array([0.7, 2.0])

    counts = count_in_interval(data, event_starts, event_stops, par_type="counts")
    binary = count_in_interval(data, event_starts, event_stops, par_type="binary")
    firing_rate = count_in_interval(
        data, event_starts, event_stops, par_type="firing_rate"
    )

    np.testing.assert_allclose(counts, np.array([[3.0, 2.0], [1.0, 1.0]]))
    np.testing.assert_allclose(binary, np.array([[1.0, 1.0], [1.0, 1.0]]))
    np.testing.assert_allclose(
        firing_rate,
        np.array([[3.0 / 0.7, 2.0], [1.0 / 0.7, 1.0]]),
    )


def test_event_spiking_threshold_returns_all_true_when_not_enough_units():
    SpikeTrainArray = pytest.importorskip("nelpy").SpikeTrainArray
    spikes = SpikeTrainArray([np.array([0.5, 1.0]), np.array([0.7, 1.2])], fs=1000)
    events = np.array([0.8, 1.0, 1.2])

    valid_events = event_spiking_threshold(spikes, events, min_units=3)

    np.testing.assert_array_equal(valid_events, np.array([True, True, True]))


def test_event_spiking_threshold_filters_events_by_population_response():
    SpikeTrainArray = pytest.importorskip("nelpy").SpikeTrainArray
    burst_near_first_event = [
        np.array([0.95, 0.99, 1.00, 1.01, 1.05, 3.8]) for _ in range(6)
    ]
    spikes = SpikeTrainArray(burst_near_first_event, fs=1000)
    events = np.array([1.0, 3.0])

    valid_events = event_spiking_threshold(
        spikes,
        events,
        window=[-0.5, 0.5],
        event_size=0.1,
        spiking_thres=0.0,
        binsize=0.01,
        sigma=0.02,
        min_units=6,
        show_fig=False,
    )

    assert valid_events.shape == (2,)
    assert valid_events.dtype == bool
    assert not valid_events[1]
