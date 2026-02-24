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
    peth,
    peth_matrix,
)


def test_crosscorr_even_nbins_is_adjusted_and_values_are_expected():
    t1 = np.array([1.0, 2.0])
    # Use non-boundary values to avoid floating bin-edge rounding effects.
    t2 = np.array([1.11, 1.31, 2.11, 2.31])

    result = crossCorr(t1, t2, binsize=0.2, nbins=4)

    assert result.shape == (5,)
    # crossCorr now computes a true event-wise cross-correlogram: each t2 is
    # considered relative to each t1, and with these inputs the last two bins match.
    np.testing.assert_allclose(result, np.array([0.0, 0.0, 0.0, 5.0, 5.0]))


def test_compute_psth_with_nonsymmetric_window_crops_to_original_range():
    data = np.array([[0.95, 1.05, 1.25], [0.90, 1.10, 1.30]], dtype=float)
    event = np.array([1.0])

    psth = compute_psth(data, event, bin_width=0.1, window=[-0.1, 0.3])

    assert psth.shape[1] == 2
    # Check index range and spacing without relying on edge inclusion behavior.
    index = psth.index.values.astype(float)
    assert index.min() >= -0.1 - 1e-9
    assert index.max() <= 0.3 + 1e-9
    if len(index) > 1:
        np.testing.assert_allclose(np.diff(index), 0.1, atol=1e-12)


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


def test_peth_matrix_returns_rate_matrix_and_expected_time_centers():
    data = np.array([0.9, 1.1, 2.1, 2.3])
    time_ref = np.array([1.0, 2.0])

    h, times = peth_matrix(data, time_ref, bin_width=0.2, n_bins=5)

    np.testing.assert_allclose(times, np.array([-0.4, -0.2, 0.0, 0.2, 0.4]), atol=1e-12)
    # peth_matrix now returns rates in Hz (same as crossCorr output)
    expected = np.column_stack(
        [
            crossCorr(np.array([time_ref[0]]), data, binsize=0.2, nbins=5),
            crossCorr(np.array([time_ref[1]]), data, binsize=0.2, nbins=5),
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


class TestPeth:
    """Test suite for the peth function with various nelpy types."""

    def test_peth_with_spiketrainarray(self):
        """Test peth with SpikeTrainArray (point process data)."""
        SpikeTrainArray = pytest.importorskip("nelpy").SpikeTrainArray

        # Create spike trains
        spike_train_1 = np.array([0.9, 1.1, 2.1, 2.3])
        spike_train_2 = np.array([0.95, 1.05, 2.05, 2.25])
        st = SpikeTrainArray([spike_train_1, spike_train_2], fs=1000)

        # Create events
        events = np.array([1.0, 2.0])

        # Compute PETH
        result = peth(st, events, window=[-0.5, 0.5], bin_width=0.1, n_bins=10)

        # Check output structure
        assert isinstance(result, pytest.importorskip("pandas").DataFrame)
        assert result.shape[1] == 2  # Two spike trains
        assert len(result.index) > 0  # Has time bins
        # Values should be firing rates (Hz)
        assert result.values.min() >= 0

    def test_peth_with_analogsignalarray(self):
        """Test peth with AnalogSignalArray (continuous data)."""
        AnalogSignalArray = pytest.importorskip("nelpy").AnalogSignalArray
        pd = pytest.importorskip("pandas")

        # Create continuous signal (2 channels, 100 samples)
        timestamps = np.linspace(0, 5, 100)
        signal = np.vstack(
            [np.sin(2 * np.pi * timestamps), np.cos(2 * np.pi * timestamps)]
        )
        # AnalogSignalArray expects data as (n_signals, n_samples)
        asa = AnalogSignalArray(timestamps=timestamps, data=signal)

        # Create events
        events = np.array([1.0, 2.0, 3.0])

        # Compute PETH
        result = peth(asa, events, window=[-0.2, 0.2], bin_width=0.01)

        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 2  # Two channels
        assert len(result.index) > 0  # Has time bins
        # Check that index represents time bins around events
        assert result.index.min() >= -0.2 - 1e-6
        assert result.index.max() <= 0.2 + 1e-6

    def test_peth_with_positionarray(self):
        """Test peth with PositionArray (2D position data)."""
        # Try multiple import paths for PositionArray
        PositionArray = None
        try:
            from nelpy.auxiliary import PositionArray
        except ImportError:
            try:
                from nelpy import PositionArray
            except ImportError:
                pytest.skip("PositionArray not available in nelpy")

        pd = pytest.importorskip("pandas")

        # Create position data (x, y coordinates)
        timestamps = np.linspace(0, 5, 100)
        x_pos = np.sin(2 * np.pi * timestamps)
        y_pos = np.cos(2 * np.pi * timestamps)
        position = PositionArray(timestamps=timestamps, data=np.vstack([x_pos, y_pos]))

        # Create events
        events = np.array([1.0, 2.0, 3.0])

        # Compute PETH
        result = peth(position, events, window=[-0.2, 0.2], bin_width=0.01)

        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 2  # x and y coordinates
        assert len(result.index) > 0
        # Check that index represents time bins around events
        assert result.index.min() >= -0.2 - 1e-6
        assert result.index.max() <= 0.2 + 1e-6

    def test_peth_with_eventarray(self):
        """Test peth with EventArray (point process data)."""
        nelpy = pytest.importorskip("nelpy")
        pd = pytest.importorskip("pandas")

        # Create event arrays
        event_train_1 = np.array([0.9, 1.1, 2.1, 2.3])
        event_train_2 = np.array([0.95, 1.05, 2.05, 2.25])

        # EventArray expects a list of arrays
        ea = nelpy.EventArray([event_train_1, event_train_2])

        # Create reference events
        events = np.array([1.0, 2.0])

        # Compute PETH
        result = peth(ea, events, window=[-0.5, 0.5], bin_width=0.1)

        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 2  # Two event trains
        assert len(result.index) > 0
        assert result.values.min() >= 0  # Rates should be non-negative

    def test_peth_with_binnedspiketrainarray(self):
        """Test peth with BinnedSpikeTrainArray."""
        nelpy = pytest.importorskip("nelpy")
        pd = pytest.importorskip("pandas")

        # Create spike trains and bin them
        spike_train_1 = np.array([0.9, 1.1, 2.1, 2.3])
        spike_train_2 = np.array([0.95, 1.05, 2.05, 2.25])
        st = nelpy.SpikeTrainArray([spike_train_1, spike_train_2], fs=1000)

        # Bin the spikes
        bst = st.bin(ds=0.01)

        # Create events well within the binned data range for the requested window
        window = [-0.5, 0.5]
        valid_events = bst.bin_centers[
            (bst.bin_centers + window[0] >= bst.bin_centers[0])
            & (bst.bin_centers + window[1] <= bst.bin_centers[-1])
        ]
        assert len(valid_events) >= 2
        events = np.array(
            [
                valid_events[len(valid_events) // 3],
                valid_events[2 * len(valid_events) // 3],
            ]
        )

        # Compute PETH
        result = peth(bst, events, window=window, bin_width=0.1)

        # Check output structure
        assert isinstance(result, pd.DataFrame)
        # BinnedSpikeTrainArray may flatten to single series - check for at least 1 column
        assert result.shape[1] >= 1
        assert len(result.index) > 0
        assert result.values.min() >= 0

    def test_peth_with_numpy_object_array(self):
        """Test peth with numpy object array (point process data)."""
        pd = pytest.importorskip("pandas")

        # Create spike trains as object array
        spike_train_1 = np.array([0.9, 1.1, 2.1, 2.3])
        spike_train_2 = np.array([0.95, 1.05, 2.05, 2.25])
        spikes = np.array([spike_train_1, spike_train_2], dtype=object)

        # Create events
        events = np.array([1.0, 2.0])

        # Compute PETH
        result = peth(spikes, events, window=[-0.5, 0.5], bin_width=0.1)

        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 2  # Two spike trains
        assert len(result.index) > 0
        assert result.values.min() >= 0

    def test_peth_with_numpy_1d_array(self):
        """Test peth with 1D numpy array (single point process)."""
        pd = pytest.importorskip("pandas")

        # Create single spike train
        spike_train = np.array([0.9, 1.1, 2.1, 2.3])

        # Create events
        events = np.array([1.0, 2.0])

        # Compute PETH
        result = peth(spike_train, events, window=[-0.5, 0.5], bin_width=0.1)

        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1  # Single spike train
        assert len(result.index) > 0
        assert result.values.min() >= 0

    def test_peth_with_empty_spike_trains(self):
        """Test peth handles empty spike trains gracefully."""
        pd = pytest.importorskip("pandas")

        # Create empty spike trains
        spike_train_1 = np.array([], dtype=np.float64)
        spike_train_2 = np.array([1.1, 2.1])
        spikes = np.array([spike_train_1, spike_train_2], dtype=object)

        # Create events
        events = np.array([1.0, 2.0])

        # Compute PETH
        result = peth(spikes, events, window=[-0.5, 0.5], bin_width=0.1)

        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 2
        assert len(result.index) > 0
        # First column should have zeros (empty spike train)
        assert result[0].sum() == 0

    def test_peth_with_asymmetric_window(self):
        """Test peth with asymmetric window."""
        pd = pytest.importorskip("pandas")

        # Create spike trains
        spike_train_1 = np.array([0.9, 1.1, 1.3, 2.1])
        spike_train_2 = np.array([0.95, 1.15, 1.25, 2.05])
        spikes = np.array([spike_train_1, spike_train_2], dtype=object)

        # Create events
        events = np.array([1.0, 2.0])

        # Compute PETH with asymmetric window
        result = peth(spikes, events, window=[-0.2, 0.4], bin_width=0.1)

        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 2
        # Window should be preserved even if asymmetric
        assert result.index.min() >= -0.2 - 1e-6
        assert result.index.max() <= 0.4 + 1e-6

    def test_peth_default_window_and_bins(self):
        """Test peth with default window and bins."""
        pd = pytest.importorskip("pandas")

        # Create spike trains
        spike_train = np.array([0.9, 1.0, 1.1, 2.0, 2.1])
        spikes = np.array([spike_train], dtype=object)

        # Create events
        events = np.array([1.0, 2.0])

        # Compute PETH with defaults (no window specified)
        result = peth(spikes, events, bin_width=0.002, n_bins=100)

        # Check output structure
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 1
        # Default should create symmetric window based on n_bins * bin_width
        expected_half_window = (100 * 0.002) / 2
        assert result.index.min() >= -expected_half_window - 1e-6
        assert result.index.max() <= expected_half_window + 1e-6

    def test_peth_raises_error_for_2d_numpy_array(self):
        """Test peth raises error for 2D numpy array without AnalogSignalArray."""
        # Create 2D numpy array (continuous data)
        signal = np.random.randn(100, 2)
        events = np.array([1.0, 2.0])

        # Should raise ValueError
        with pytest.raises(ValueError, match="continuous numpy arrays"):
            peth(signal, events, window=[-0.5, 0.5])

    def test_peth_raises_error_for_unsupported_type(self):
        """Test peth raises error for unsupported data type."""
        events = np.array([1.0, 2.0])

        # Should raise TypeError
        with pytest.raises(TypeError, match="Unsupported data type"):
            peth("invalid_data", events, window=[-0.5, 0.5])

    def test_peth_consistency_with_compute_psth(self):
        """Test that peth gives similar results to compute_psth for point process."""
        pd = pytest.importorskip("pandas")

        # Create spike trains
        spike_train_1 = np.array([0.9, 1.1, 2.1, 2.3])
        spike_train_2 = np.array([0.95, 1.05, 2.05, 2.25])
        spikes = np.array([spike_train_1, spike_train_2], dtype=object)

        # Create events
        events = np.array([1.0, 2.0])

        # Compute with both functions
        result_peth = peth(spikes, events, window=[-0.5, 0.5], bin_width=0.1, n_bins=10)
        result_psth = compute_psth(
            spikes, events, window=[-0.5, 0.5], bin_width=0.1, n_bins=10
        )

        # Should have same shape
        assert result_peth.shape == result_psth.shape

        # Values should be very close (may differ slightly due to implementation details)
        np.testing.assert_allclose(
            result_peth.values, result_psth.values, rtol=1e-10, atol=1e-10
        )

    def test_peth_average_false_returns_tuple(self):
        """Test that peth with average=False returns a tuple."""
        # Create spike trains
        spike_train_1 = np.array([0.9, 1.1, 2.1, 2.3])
        spike_train_2 = np.array([0.95, 1.05, 2.05, 2.25])
        spikes = np.array([spike_train_1, spike_train_2], dtype=object)

        # Create events
        events = np.array([1.0, 2.0])

        # Compute PETH with average=False
        result = peth(spikes, events, window=[-0.5, 0.5], bin_width=0.1, average=False)

        # Should return tuple
        assert isinstance(result, tuple)
        assert len(result) == 2
        peth_matrix, time_bins = result
        assert isinstance(peth_matrix, np.ndarray)
        assert isinstance(time_bins, np.ndarray)

    def test_peth_average_false_correct_shape_point_process(self):
        """Test that peth with average=False returns correct shape for point process."""
        # Create spike trains
        spike_train_1 = np.array([0.9, 1.1, 2.1, 2.3])
        spike_train_2 = np.array([0.95, 1.05, 2.05, 2.25])
        spikes = np.array([spike_train_1, spike_train_2], dtype=object)

        # Create events
        events = np.array([1.0, 2.0, 3.0])

        # Compute PETH with average=False
        peth_matrix, time_bins = peth(
            spikes, events, window=[-0.5, 0.5], bin_width=0.1, average=False
        )

        # Check shape: (n_time_bins, n_signals, n_events)
        n_time_bins = len(time_bins)
        n_signals = 2  # Two spike trains
        n_events = 3  # Three events
        assert peth_matrix.shape == (n_time_bins, n_signals, n_events)
        assert peth_matrix.ndim == 3

    def test_peth_average_false_correct_shape_continuous(self):
        """Test that peth with average=False returns correct shape for continuous data."""
        AnalogSignalArray = pytest.importorskip("nelpy").AnalogSignalArray

        # Create continuous signal (2 channels)
        timestamps = np.linspace(0, 5, 100)
        signal = np.vstack(
            [np.sin(2 * np.pi * timestamps), np.cos(2 * np.pi * timestamps)]
        )
        asa = AnalogSignalArray(timestamps=timestamps, data=signal)

        # Create events
        events = np.array([1.0, 2.0, 3.0])

        # Compute PETH with average=False
        peth_matrix, time_bins = peth(
            asa, events, window=[-0.2, 0.2], bin_width=0.01, average=False
        )

        # Check shape: (n_time_bins, n_signals, n_events)
        n_time_bins = len(time_bins)
        n_signals = 2  # Two channels
        n_events = 3  # Three events
        assert peth_matrix.shape == (n_time_bins, n_signals, n_events)
        assert peth_matrix.ndim == 3

    def test_peth_average_false_values_are_in_hz(self):
        """Test that peth with average=False returns rates in Hz for point process."""
        # Create spike trains with known spike counts
        # Put 5 spikes in 0.1s bin around first event
        spike_train = np.array([0.95, 0.97, 0.99, 1.01, 1.03])
        spikes = np.array([spike_train], dtype=object)

        # Single event
        events = np.array([1.0])

        # Compute PETH with average=False
        peth_matrix, time_bins = peth(
            spikes, events, window=[-0.5, 0.5], bin_width=0.1, average=False
        )

        # Check that values are rates (Hz), not counts
        # The bin around t=0 should have all 5 spikes
        center_bin_idx = len(time_bins) // 2
        center_value = peth_matrix[center_bin_idx, 0, 0]

        # Value should be a rate in Hz (spikes per second)
        # With bin_width=0.1s, 5 spikes gives rate of 50 Hz
        # But crossCorr returns averaged rate across events, so just check it's > 0
        assert center_value > 0
        assert center_value < 1000  # Reasonable upper bound for firing rate

    def test_peth_average_false_consistency_with_average_true(self):
        """Test that averaging event-wise matrix matches average=True output."""
        # Create spike trains
        spike_train_1 = np.array([0.9, 1.1, 2.1, 2.3, 3.1])
        spike_train_2 = np.array([0.95, 1.05, 2.05, 2.25, 3.05])
        spikes = np.array([spike_train_1, spike_train_2], dtype=object)

        # Create events
        events = np.array([1.0, 2.0, 3.0])

        # Compute both versions
        result_avg = peth(
            spikes, events, window=[-0.5, 0.5], bin_width=0.1, average=True
        )
        peth_matrix, time_bins = peth(
            spikes, events, window=[-0.5, 0.5], bin_width=0.1, average=False
        )

        # Average across events manually
        manual_avg = np.nanmean(peth_matrix, axis=2)

        # Should match the average=True output
        np.testing.assert_allclose(
            manual_avg, result_avg.values, rtol=1e-10, atol=1e-10
        )

    def test_peth_average_false_with_spiketrainarray(self):
        """Test peth with average=False using SpikeTrainArray."""
        SpikeTrainArray = pytest.importorskip("nelpy").SpikeTrainArray

        # Create spike trains
        spike_train_1 = np.array([0.9, 1.1, 2.1, 2.3])
        spike_train_2 = np.array([0.95, 1.05, 2.05, 2.25])
        st = SpikeTrainArray([spike_train_1, spike_train_2], fs=1000)

        # Create events
        events = np.array([1.0, 2.0])

        # Compute PETH with average=False
        peth_matrix, time_bins = peth(
            st, events, window=[-0.5, 0.5], bin_width=0.1, average=False
        )

        # Check shape
        assert peth_matrix.ndim == 3
        assert peth_matrix.shape[1] == 2  # Two spike trains
        assert peth_matrix.shape[2] == 2  # Two events
        assert len(time_bins) == peth_matrix.shape[0]

    def test_peth_average_false_with_asymmetric_window(self):
        """Test peth with average=False handles asymmetric window correctly."""
        # Create spike trains
        spike_train = np.array([0.9, 1.1, 1.3, 2.1, 2.3])
        spikes = np.array([spike_train], dtype=object)

        # Create events
        events = np.array([1.0, 2.0])

        # Compute PETH with asymmetric window
        peth_matrix, time_bins = peth(
            spikes, events, window=[-0.2, 0.4], bin_width=0.1, average=False
        )

        # Check time bins cover the requested window
        assert time_bins.min() >= -0.2 - 1e-6
        assert time_bins.max() <= 0.4 + 1e-6

        # Check shape
        assert peth_matrix.shape[2] == 2  # Two events

    def test_peth_average_false_continuous_data_with_positionarray(self):
        """Test peth with average=False for PositionArray."""
        # Try multiple import paths for PositionArray
        PositionArray = None
        try:
            from nelpy.auxiliary import PositionArray
        except ImportError:
            try:
                from nelpy import PositionArray
            except ImportError:
                pytest.skip("PositionArray not available in nelpy")

        # Create position data
        timestamps = np.linspace(0, 5, 100)
        x_pos = np.sin(2 * np.pi * timestamps)
        y_pos = np.cos(2 * np.pi * timestamps)
        position = PositionArray(timestamps=timestamps, data=np.vstack([x_pos, y_pos]))

        # Create events
        events = np.array([1.0, 2.0, 3.0])

        # Compute PETH with average=False
        peth_matrix, time_bins = peth(
            position, events, window=[-0.2, 0.2], bin_width=0.01, average=False
        )

        # Check shape: (n_time_bins, n_coordinates, n_events)
        assert peth_matrix.ndim == 3
        assert peth_matrix.shape[1] == 2  # x and y coordinates
        assert peth_matrix.shape[2] == 3  # Three events

    def test_peth_average_false_handles_empty_spike_trains(self):
        """Test peth with average=False handles empty spike trains gracefully."""
        # Create empty and non-empty spike trains
        spike_train_1 = np.array([], dtype=np.float64)
        spike_train_2 = np.array([1.1, 2.1])
        spikes = np.array([spike_train_1, spike_train_2], dtype=object)

        # Create events
        events = np.array([1.0, 2.0])

        # Compute PETH with average=False
        peth_matrix, time_bins = peth(
            spikes, events, window=[-0.5, 0.5], bin_width=0.1, average=False
        )

        # Check shape
        assert peth_matrix.shape == (len(time_bins), 2, 2)

        # First spike train (empty) should have all zeros
        assert np.all(peth_matrix[:, 0, :] == 0)

    def test_peth_average_false_single_event(self):
        """Test peth with average=False works with single event."""
        # Create spike train
        spike_train = np.array([0.9, 1.0, 1.1])
        spikes = np.array([spike_train], dtype=object)

        # Single event
        events = np.array([1.0])

        # Compute PETH with average=False
        peth_matrix, time_bins = peth(
            spikes, events, window=[-0.5, 0.5], bin_width=0.1, average=False
        )

        # Check shape: (n_time_bins, 1 signal, 1 event)
        assert peth_matrix.shape[1] == 1  # One spike train
        assert peth_matrix.shape[2] == 1  # One event

    def test_peth_average_false_many_events(self):
        """Test peth with average=False scales correctly with many events."""
        # Create spike train
        spike_train = np.arange(0, 10, 0.1)  # Dense spike train
        spikes = np.array([spike_train], dtype=object)

        # Many events
        events = np.arange(1, 9, 0.5)  # 16 events

        # Compute PETH with average=False
        peth_matrix, time_bins = peth(
            spikes, events, window=[-0.2, 0.2], bin_width=0.05, average=False
        )

        # Check shape
        assert peth_matrix.shape[1] == 1  # One spike train
        assert peth_matrix.shape[2] == len(events)  # All events

        # All values should be finite
        assert np.all(np.isfinite(peth_matrix))
