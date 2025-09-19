import logging

import nelpy as nel
import numpy as np
import pytest

from neuro_py.tuning.maps import SpatialMap
import numpy as np

# Ensure that the logging module captures warnings
logging.captureWarnings(True)


def generate_position_data(duration=1000, sampling_rate=50):
    """
    Generate synthetic 2D position data over time.
    """
    time = np.linspace(0, duration, int(duration * sampling_rate))
    x_position = np.sin(time / 100) * 50 + 50  # Oscillates between 0 and 100
    y_position = np.cos(time / 100) * 50 + 50  # Oscillates between 0 and 100
    return time, x_position, y_position


def generate_spike_train_data(n_units, time, rate=5, fs=20_000):
    """
    Generate synthetic spike train data for multiple units.
    """
    spike_trains = []
    for _ in range(n_units):
        spikes = np.random.poisson(rate / len(time), size=len(time))
        spike_trains.append(np.where(spikes > 0)[0] / fs)
    return np.array(spike_trains, dtype=object)


@pytest.fixture
def spatial_map_fixture():
    """
    Fixture to create a SpatialMap instance with generated position and spike train data.
    """
    # Generate synthetic data
    time, x_position, y_position = generate_position_data()
    spike_trains = generate_spike_train_data(n_units=5, time=time)

    pos = nel.AnalogSignalArray(np.array([x_position, y_position]), time=time, fs=50)
    st = nel.SpikeTrainArray(spike_trains, fs=20_000)

    # Create SpatialMap instance
    spatial_map = SpatialMap(
        pos, st, x_minmax=(0, 100), y_minmax=(0, 100), speed_thres=0
    )
    return spatial_map


def test_spatial_map_properties(spatial_map_fixture):
    """
    Test the properties of SpatialMap.
    """
    spatial_map = spatial_map_fixture

    assert spatial_map.is2d is True
    assert spatial_map.n_units == 5
    assert spatial_map.occupancy.shape == (34, 34)
    assert spatial_map.ratemap.shape == (5, 34, 34)  # 5 units with 34x34 maps
    assert spatial_map.isempty is False
    assert spatial_map.occupancy.sum() > 0  # Ensure occupancy was computed
    assert spatial_map.ratemap.min() >= 0  # Rates cannot be negative
    assert spatial_map.ratemap.max() > 0  # Some firing should be present


@pytest.fixture
def spatial_map_no_spike_fixture():
    """
    Fixture to create a SpatialMap instance with generated position and spike train data.
    """
    # Generate synthetic data
    time, x_position, y_position = generate_position_data()
    spike_trains = generate_spike_train_data(n_units=5, time=time)

    pos = nel.AnalogSignalArray(np.array([x_position, y_position]), time=time, fs=50)

    # remove spikes from cell 1
    spike_trains[1] = np.array([])
    st = nel.SpikeTrainArray(spike_trains, fs=20_000)

    # Create SpatialMap instance
    spatial_map = SpatialMap(
        pos, st, x_minmax=(0, 100), y_minmax=(0, 100), speed_thres=0
    )
    return spatial_map


def test_spatial_map_properties_no_spike(spatial_map_no_spike_fixture):
    """
    Test the properties of SpatialMap.
    """
    spatial_map = spatial_map_no_spike_fixture

    assert spatial_map.is2d is True
    assert spatial_map.n_units == 5
    assert spatial_map.occupancy.shape == (34, 34)
    assert spatial_map.ratemap.shape == (5, 34, 34)  # 5 units with 34x34 maps
    assert spatial_map.isempty is False
    assert spatial_map.occupancy.sum() > 0  # Ensure occupancy was computed
    assert spatial_map.ratemap[1].sum() == 0  # Ensure no spikes for cell 1
    assert spatial_map.ratemap.min() >= 0  # Rates cannot be negative
    assert spatial_map.ratemap.max() > 0  # Some firing should be present


@pytest.fixture
def spatial_map_nan_pos_fixture():
    """
    Fixture to create a SpatialMap instance with generated position and spike train data.
    """
    # Generate synthetic data
    time, x_position, y_position = generate_position_data()
    spike_trains = generate_spike_train_data(n_units=5, time=time)

    nan_idx = np.arange(0, 49900)
    x_position[nan_idx] = np.nan
    y_position[nan_idx] = np.nan
    pos = nel.AnalogSignalArray(np.array([x_position, y_position]), time=time, fs=50)

    st = nel.SpikeTrainArray(spike_trains, fs=20_000)

    # Create SpatialMap instance
    # Capture the warning
    with pytest.warns(Warning) as record:
        spatial_map = SpatialMap(
            pos, st, x_minmax=(0, 100), y_minmax=(0, 100), speed_thres=0.1
        )
    # Check that the warning message is as expected
    assert len(record) > 0  # Ensure that at least one warning was raised
    assert "No spike trains during running epochs" in str(
        record[0].message
    )  # Verify the warning message

    return spatial_map


def test_spatial_map_properties_nan_pos_with_speed_thres(spatial_map_nan_pos_fixture):
    """
    Test the properties of SpatialMap.
    """
    spatial_map = spatial_map_nan_pos_fixture

    assert spatial_map.is2d is True
    assert spatial_map.n_units == 5
    assert spatial_map.occupancy.shape == (34, 34)
    assert spatial_map.ratemap.shape == (5, 34, 34)  # 5 units with 34x34 maps
    assert spatial_map.isempty is False
    assert spatial_map.occupancy.sum() > 0  # Ensure occupancy was computed
    assert spatial_map.ratemap.min() >= 0  # Rates cannot be negative
    assert spatial_map.ratemap.max() == 0


def spatial_map_nan_pos_w_no_speed_thresh():
    """
    Fixture to create a SpatialMap instance with generated position and spike train data.
    """
    # Generate synthetic data
    time, x_position, y_position = generate_position_data()
    spike_trains = generate_spike_train_data(n_units=5, time=time)

    nan_idx = np.arange(0, 100)
    x_position[nan_idx] = np.nan
    y_position[nan_idx] = np.nan
    pos = nel.AnalogSignalArray(np.array([x_position, y_position]), time=time, fs=50)

    st = nel.SpikeTrainArray(spike_trains, fs=20_000)

    # Create SpatialMap instance
    spatial_map = SpatialMap(
        pos, st, x_minmax=(0, 100), y_minmax=(0, 100), speed_thres=0
    )

    assert spatial_map.is2d is True
    assert spatial_map.n_units == 5
    assert spatial_map.occupancy.shape == (34, 34)
    assert spatial_map.ratemap.shape == (5, 34, 34)  # 5 units with 34x34 maps
    assert spatial_map.isempty is False
    assert spatial_map.occupancy.sum() > 0  # Ensure occupancy was computed
    assert spatial_map.ratemap.min() >= 0  # Rates cannot be negative
    assert spatial_map.ratemap.max() > 0


def spatial_map_nan_pos_w_no_speed_thresh_1d():
    """
    Fixture to create a SpatialMap instance with generated position and spike train data.
    """
    # Generate synthetic data
    time, x_position, y_position = generate_position_data()
    spike_trains = generate_spike_train_data(n_units=5, time=time)

    nan_idx = np.arange(0, 100)
    x_position[nan_idx] = np.nan
    pos = nel.AnalogSignalArray(np.array(x_position), time=time, fs=50)

    st = nel.SpikeTrainArray(spike_trains, fs=20_000)

    # Create SpatialMap instance
    spatial_map = SpatialMap(pos, st, x_minmax=(0, 100), speed_thres=0)

    assert spatial_map.is2d is False
    assert spatial_map.n_units == 5
    assert spatial_map.occupancy.shape == (34,)
    assert spatial_map.ratemap.shape == (5, 34)  # 5 units with 34x34 maps
    assert spatial_map.isempty is False
    assert spatial_map.occupancy.sum() > 0  # Ensure occupancy was computed
    assert spatial_map.ratemap.min() >= 0  # Rates cannot be negative
    assert spatial_map.ratemap.max() > 0


def test_spatial_map_invalid_pos_and_st():
    """
    Test for invalid position and spike train data.
    """
    # Invalid position data (e.g., all NaNs)
    time, x_position, y_position = generate_position_data()
    x_position[:] = np.nan
    y_position[:] = np.nan
    pos = nel.AnalogSignalArray(np.array([x_position, y_position]), time=time, fs=50)
    spike_trains = generate_spike_train_data(n_units=5, time=time)
    st = nel.SpikeTrainArray(spike_trains, fs=20_000)

    try:
        SpatialMap(pos, st, x_minmax=(0, 100), y_minmax=(0, 100), speed_thres=0)
    except ValueError as e:
        assert str(e) == "Position data cannot contain all NaN values"


def test_spatial_map_too_high_speed_thresh():
    """
    Test for invalid position and spike train data.
    """
    # Invalid position data (e.g., all NaNs)
    time, x_position, y_position = generate_position_data()
    pos = nel.AnalogSignalArray(np.array([x_position, y_position]), time=time, fs=50)
    spike_trains = generate_spike_train_data(n_units=5, time=time)
    st = nel.SpikeTrainArray(spike_trains, fs=20_000)

    # Capture the warning
    with pytest.warns(Warning) as record:
        SpatialMap(pos, st, x_minmax=(0, 100), y_minmax=(0, 100), speed_thres=1000)

    # Check that the warning message is as expected
    assert len(record) > 0  # Ensure that at least one warning was raised
    assert "No spike trains during running epochs" in str(
        record[0].message
    )  # Verify the warning message


def spatial_map_continuous():
    """
    Fixture to create a SpatialMap instance with generated position and spike train data.
    """

    # Generate synthetic data
    time, x_position, y_position = generate_position_data()

    pos = nel.AnalogSignalArray(np.array([x_position, y_position]), time=time, fs=50)
    con_signal = nel.AnalogSignalArray(
        np.array([x_position, y_position, x_position, y_position, y_position]),
        time=time,
        fs=50,
    )

    # Create SpatialMap instance
    spatial_map = SpatialMap(
        pos, con_signal, x_minmax=(0, 100), y_minmax=(0, 100), speed_thres=0
    )

    assert spatial_map.is2d is True
    assert spatial_map.n_units == 5
    assert spatial_map.occupancy.shape == (34, 34)
    assert spatial_map.ratemap.shape == (5, 34, 34)  # 5 units with 34x34 maps
    assert spatial_map.isempty is False
    assert spatial_map.occupancy.sum() > 0  # Ensure occupancy was computed
    assert spatial_map.ratemap.min() >= 0  # Rates cannot be negative
    assert spatial_map.ratemap.max() > 0  # Some firing should be present


def spatial_map_continuous_1d():
    """
    Fixture to create a SpatialMap instance with generated position and spike train data.
    """

    # Generate synthetic data
    time, x_position, y_position = generate_position_data()

    pos = nel.AnalogSignalArray(np.array([x_position]), time=time, fs=50)
    con_signal = nel.AnalogSignalArray(
        np.array([x_position, y_position, x_position, y_position, y_position]),
        time=time,
        fs=50,
    )

    # Create SpatialMap instance
    spatial_map = SpatialMap(pos, con_signal, x_minmax=(0, 100), speed_thres=0)

    assert spatial_map.is2d is False
    assert spatial_map.n_units == 5
    assert spatial_map.occupancy.shape == (34,)
    assert spatial_map.ratemap.shape == (5, 34)  # 5 units with 34x34 maps
    assert spatial_map.isempty is False
    assert spatial_map.occupancy.sum() > 0  # Ensure occupancy was computed
    assert spatial_map.ratemap.min() >= 0  # Rates cannot be negative
    assert spatial_map.ratemap.max() > 0  # Some firing should be present


def test_spatial_map_multidimensional():
    """
    Test SpatialMap with more than 2 dimensions using the new NDimensionalBinner base class.
    """
    # Generate synthetic 3D position data
    time = np.linspace(0, 1000, 50000)
    x_position = np.sin(time / 100) * 50 + 50  # Oscillates between 0 and 100
    y_position = np.cos(time / 100) * 50 + 50  # Oscillates between 0 and 100
    z_position = np.sin(time / 200) * 30 + 30  # Oscillates between 0 and 60

    # Create 3D position data
    pos_3d = nel.AnalogSignalArray(
        np.array([x_position, y_position, z_position]), time=time, fs=50
    )

    # Generate synthetic spike train data
    spike_trains = generate_spike_train_data(n_units=3, time=time)
    st = nel.SpikeTrainArray(spike_trains, fs=20_000)

    # Create SpatialMap instance for 3D data
    spatial_map_3d = SpatialMap(
        pos_3d,
        st,
        speed_thres=0,  # No speed threshold
        s_binsize=10,  # Larger bins for 3D
    )

    # Test basic properties
    assert spatial_map_3d.dim == 3
    assert spatial_map_3d.n_units == 3
    assert len(spatial_map_3d.ratemap.shape) == 4  # (n_units, x_bins, y_bins, z_bins)
    assert spatial_map_3d.isempty is False
    assert spatial_map_3d.occupancy.sum() > 0
    assert spatial_map_3d.ratemap.min() >= 0
    assert spatial_map_3d.ratemap.max() > 0

    # Test that tuning curve has spatial information methods
    if hasattr(spatial_map_3d.tc, "spatial_information"):
        spatial_info = spatial_map_3d.tc.spatial_information()
        assert len(spatial_info) == 3  # One value per unit
        assert all(
            info >= 0 for info in spatial_info
        )  # Information should be non-negative


def test_spatial_map_base_class_compatibility():
    """
    Test that the new base class methods produce equivalent results for 1D and 2D data.
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate test data
    time, x_position, y_position = generate_position_data()
    spike_trains = generate_spike_train_data(n_units=2, time=time)

    # Test 1D compatibility
    pos_1d = nel.AnalogSignalArray(np.array([x_position]), time=time, fs=50)
    st = nel.SpikeTrainArray(spike_trains, fs=20_000)

    spatial_map_1d = SpatialMap(pos_1d, st, x_minmax=(0, 100), speed_thres=0)

    # Test both old and new methods for 1D
    tc_old, _ = spatial_map_1d.map_1d(use_base_class=False)
    tc_new, _ = spatial_map_1d.map_1d(use_base_class=True)

    # Results should be very similar (allowing for small numerical differences)
    assert tc_old.ratemap.shape == tc_new.ratemap.shape

    # Check that the implementations produce very similar results
    # Allow for small numerical differences due to floating point precision
    assert np.allclose(
        tc_old.ratemap, tc_new.ratemap, rtol=5e-3, atol=5e-3
    )  # Also check that the overall statistics are the same
    assert np.isclose(np.sum(tc_old.ratemap), np.sum(tc_new.ratemap), rtol=1e-10)
    assert np.count_nonzero(tc_old.ratemap) == np.count_nonzero(tc_new.ratemap)

    # Test 2D compatibility
    pos_2d = nel.AnalogSignalArray(np.array([x_position, y_position]), time=time, fs=50)
    spatial_map_2d = SpatialMap(
        pos_2d, st, x_minmax=(0, 100), y_minmax=(0, 100), speed_thres=0
    )

    # Test both old and new methods for 2D
    tc_old_2d, _ = spatial_map_2d.map_2d(use_base_class=False)
    tc_new_2d, _ = spatial_map_2d.map_2d(use_base_class=True)

    # Results should be very similar
    assert tc_old_2d.ratemap.shape == tc_new_2d.ratemap.shape

    # Check that the implementations produce very similar results
    # Allow for small numerical differences due to floating point precision
    assert np.allclose(tc_old_2d.ratemap, tc_new_2d.ratemap, rtol=5e-3, atol=5e-3)


def test_ndimensional_binner_direct():
    """
    Test the NDimensionalBinner class directly.
    """
    from neuro_py.tuning.maps import NDimensionalBinner

    # Create test data
    time = np.linspace(0, 100, 5000)
    pos_data = np.array(
        [
            np.sin(time / 10) * 10 + 10,  # x dimension
            np.cos(time / 10) * 10 + 10,  # y dimension
            np.sin(time / 20) * 5 + 5,  # z dimension
        ]
    )

    pos = nel.AnalogSignalArray(pos_data, time=time, fs=50)

    # Create simple spike train
    spike_times = np.random.choice(time, size=100, replace=False)
    st = nel.SpikeTrainArray([spike_times], fs=50)

    # Create binner and test
    binner = NDimensionalBinner()

    # Define bin edges for 3D
    bin_edges = [
        np.arange(0, 21, 2),  # x bins
        np.arange(0, 21, 2),  # y bins
        np.arange(0, 11, 1),  # z bins
    ]

    tc, occupancy, ratemap = binner.create_nd_tuning_curve(
        st_data=st, pos_data=pos, bin_edges=bin_edges, min_duration=0.01, minbgrate=0
    )

    # Test results
    assert ratemap.shape[0] == 1  # One unit
    assert len(ratemap.shape) == 4  # (units, x_bins, y_bins, z_bins)
    assert occupancy.shape == (10, 10, 10)  # Bins for each dimension
    assert occupancy.sum() > 0
    assert ratemap.min() >= 0


def test_s_binsize_array():
    """Test that s_binsize accepts arrays for dimension-specific bin sizes."""

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    timestamps = np.linspace(0, 100, n_samples)

    # Create 3D position data with different scales for each dimension
    pos_data = np.array(
        [
            np.sin(timestamps * 0.1) * 50,  # x: -50 to 50
            np.cos(timestamps * 0.1) * 30,  # y: -30 to 30
            timestamps * 0.5,  # z: 0 to 50
        ]
    )

    pos = nel.AnalogSignalArray(data=pos_data, timestamps=timestamps)

    # Create synthetic spike train data
    spike_times = np.sort(np.random.uniform(0, 100, 200))
    st = nel.SpikeTrainArray(time=spike_times, fs=1000.0)

    # Test with scalar s_binsize (backward compatibility)
    spatial_map_scalar = SpatialMap(pos=pos, st=st, s_binsize=5.0, speed_thres=0)

    # Verify scalar behavior
    assert spatial_map_scalar.s_binsize == 5.0
    assert np.all(spatial_map_scalar.s_binsize_array == 5.0)
    assert len(spatial_map_scalar.s_binsize_array) == 3

    # Test with array s_binsize (different bin sizes for each dimension)
    bin_sizes = [2.0, 3.0, 5.0]
    spatial_map_array = SpatialMap(pos=pos, st=st, s_binsize=bin_sizes, speed_thres=0)

    # Verify array behavior
    assert (
        spatial_map_array.s_binsize == 2.0
    )  # First dimension for backward compatibility
    assert np.array_equal(spatial_map_array.s_binsize_array, np.array(bin_sizes))

    # Check that bin edges have correct spacing
    x_spacing = np.diff(spatial_map_array.x_edges)[0]
    y_spacing = np.diff(spatial_map_array.y_edges)[0]

    assert np.isclose(x_spacing, 2.0), f"Expected X spacing 2.0, got {x_spacing}"
    assert np.isclose(y_spacing, 3.0), f"Expected Y spacing 3.0, got {y_spacing}"

    # Test with 2D data
    pos_2d = nel.AnalogSignalArray(data=pos_data[:2], timestamps=timestamps)
    bin_sizes_2d = [1.5, 4.0]
    spatial_map_2d = SpatialMap(
        pos=pos_2d, st=st, s_binsize=bin_sizes_2d, speed_thres=0
    )

    x_spacing_2d = np.diff(spatial_map_2d.x_edges)[0]
    y_spacing_2d = np.diff(spatial_map_2d.y_edges)[0]

    assert np.isclose(x_spacing_2d, 1.5), f"Expected X spacing 1.5, got {x_spacing_2d}"
    assert np.isclose(y_spacing_2d, 4.0), f"Expected Y spacing 4.0, got {y_spacing_2d}"

    # Test error case: wrong array length
    with pytest.raises(
        ValueError,
        match="Length of s_binsize array .* must match number of position dimensions",
    ):
        wrong_sizes = [2.0, 3.0]  # Only 2 bin sizes for 3D data
        SpatialMap(pos=pos, st=st, s_binsize=wrong_sizes, speed_thres=0)

    # Test with list input (should work the same as array)
    spatial_map_list = SpatialMap(
        pos=pos_2d, st=st, s_binsize=[2.5, 3.5], speed_thres=0
    )
    assert np.array_equal(spatial_map_list.s_binsize_array, np.array([2.5, 3.5]))


def test_s_binsize_array_backward_compatibility():
    """Test that existing functionality still works with new s_binsize array support."""

    # Generate 2D synthetic data
    time, x_position, y_position = generate_position_data()
    spike_trains = generate_spike_train_data(n_units=2, time=time)

    pos = nel.AnalogSignalArray(np.array([x_position, y_position]), time=time, fs=50)
    st = nel.SpikeTrainArray(spike_trains, fs=20_000)

    # Test that old code still works (scalar s_binsize)
    spatial_map_old = SpatialMap(pos, st, s_binsize=3.0, speed_thres=0)

    # Test that new array functionality produces same results when all bin sizes are equal
    spatial_map_new = SpatialMap(pos, st, s_binsize=[3.0, 3.0], speed_thres=0)

    # Compare results with tolerance for numerical differences
    np.testing.assert_allclose(
        spatial_map_old.tc.ratemap,
        spatial_map_new.tc.ratemap,
        rtol=1e-10,
        err_msg="Ratemap should be identical for equivalent bin sizes",
    )

    np.testing.assert_allclose(
        spatial_map_old.tc.occupancy,
        spatial_map_new.tc.occupancy,
        rtol=1e-10,
        err_msg="Occupancy should be identical for equivalent bin sizes",
    )


def test_dim_minmax():
    """Test that dim_minmax parameter works for dimension-specific min/max values."""

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    timestamps = np.linspace(0, 100, n_samples)

    # Create 3D position data
    pos_data = np.array(
        [
            np.sin(timestamps * 0.1) * 50,  # x: -50 to 50
            np.cos(timestamps * 0.1) * 30,  # y: -30 to 30
            timestamps * 0.5,  # z: 0 to 50
        ]
    )

    pos = nel.AnalogSignalArray(data=pos_data, timestamps=timestamps)

    # Create synthetic spike train data
    spike_times = np.sort(np.random.uniform(0, 100, 200))
    st = nel.SpikeTrainArray(time=spike_times, fs=1000.0)

    # Test with dim_minmax parameter
    custom_minmax = [[-60, 60], [-40, 40], [-10, 60]]
    spatial_map = SpatialMap(
        pos=pos, st=st, s_binsize=5.0, dim_minmax=custom_minmax, speed_thres=0
    )

    # Verify that custom values are used
    expected_minmax = np.array(custom_minmax)
    assert np.array_equal(spatial_map.dim_minmax_array, expected_minmax)

    # Check that x_minmax and y_minmax are properly set from dim_minmax
    assert spatial_map.x_minmax == [-60, 60]
    assert spatial_map.y_minmax == [-40, 40]

    # Test backward compatibility - x_minmax and y_minmax should still work
    spatial_map_compat = SpatialMap(
        pos=pos,
        st=st,
        s_binsize=5.0,
        x_minmax=[-70, 70],
        y_minmax=[-35, 35],
        speed_thres=0,
    )

    assert spatial_map_compat.dim_minmax_array[0, 0] == -70
    assert spatial_map_compat.dim_minmax_array[0, 1] == 70
    assert spatial_map_compat.dim_minmax_array[1, 0] == -35
    assert spatial_map_compat.dim_minmax_array[1, 1] == 35

    # Test that dim_minmax takes precedence over x_minmax and y_minmax
    spatial_map_precedence = SpatialMap(
        pos=pos,
        st=st,
        s_binsize=5.0,
        x_minmax=[-100, 100],
        y_minmax=[-100, 100],  # Should be overridden
        dim_minmax=[[-80, 80], [-45, 45], [-5, 55]],  # Should be used
        speed_thres=0,
    )

    assert spatial_map_precedence.x_minmax == [-80, 80]
    assert spatial_map_precedence.y_minmax == [-45, 45]

    # Test error case: wrong dim_minmax shape
    with pytest.raises(ValueError, match="dim_minmax must be a list of .* pairs"):
        wrong_minmax = [[-60, 60], [-40, 40]]  # Only 2 dimensions for 3D data
        SpatialMap(
            pos=pos, st=st, s_binsize=5.0, dim_minmax=wrong_minmax, speed_thres=0
        )

    # Test with 2D data
    pos_2d = nel.AnalogSignalArray(data=pos_data[:2], timestamps=timestamps)
    dim_minmax_2d = [[-80, 80], [-60, 60]]
    spatial_map_2d = SpatialMap(
        pos=pos_2d, st=st, dim_minmax=dim_minmax_2d, speed_thres=0
    )

    assert spatial_map_2d.dim_minmax_array.shape == (2, 2)
    assert np.array_equal(spatial_map_2d.dim_minmax_array, np.array(dim_minmax_2d))


def test_tuning_curve_sigma_array():
    """Test tuning_curve_sigma array functionality for dimension-specific smoothing."""
    # Create 3D position data
    n_samples = 1000
    timestamps = np.linspace(0, 100, n_samples)
    x_pos = np.random.uniform(-50, 50, n_samples)
    y_pos = np.random.uniform(-30, 30, n_samples)
    z_pos = np.random.uniform(-10, 50, n_samples)
    pos_data = np.vstack([x_pos, y_pos, z_pos])

    pos = nel.AnalogSignalArray(data=pos_data, timestamps=timestamps)

    # Create spike data
    spike_times_list = []
    for unit in range(3):
        spike_times = np.sort(np.random.uniform(0, 100, 50))
        spike_times_list.append(spike_times)

    st = nel.SpikeTrainArray(spike_times_list)

    # Test with scalar tuning_curve_sigma (backward compatibility)
    spatial_map_scalar = SpatialMap(
        pos=pos,
        st=st,
        s_binsize=5.0,
        tuning_curve_sigma=2.5,
        speed_thres=0,
    )

    # Should create array with same value for all dimensions
    assert hasattr(spatial_map_scalar, "tuning_curve_sigma_array")
    assert spatial_map_scalar.tuning_curve_sigma_array.shape == (3,)
    assert np.all(spatial_map_scalar.tuning_curve_sigma_array == 2.5)
    assert spatial_map_scalar.tuning_curve_sigma == 2.5  # Backward compatibility

    # Test with array tuning_curve_sigma
    sigma_array = [1.0, 2.0, 3.0]  # Different sigma for each dimension
    spatial_map_array = SpatialMap(
        pos=pos,
        st=st,
        s_binsize=5.0,
        tuning_curve_sigma=sigma_array,
        speed_thres=0,
    )

    assert hasattr(spatial_map_array, "tuning_curve_sigma_array")
    assert spatial_map_array.tuning_curve_sigma_array.shape == (3,)
    assert np.array_equal(
        spatial_map_array.tuning_curve_sigma_array, np.array(sigma_array)
    )
    assert (
        spatial_map_array.tuning_curve_sigma == 1.0
    )  # Should use first dimension for backward compatibility

    # Test with numpy array
    sigma_numpy = np.array([0.5, 1.5, 2.5])
    spatial_map_numpy = SpatialMap(
        pos=pos,
        st=st,
        s_binsize=5.0,
        tuning_curve_sigma=sigma_numpy,
        speed_thres=0,
    )

    assert np.array_equal(spatial_map_numpy.tuning_curve_sigma_array, sigma_numpy)

    # Test error case: wrong array length
    with pytest.raises(
        ValueError, match="Length of tuning_curve_sigma array .* must match"
    ):
        wrong_sigma = [1.0, 2.0]  # Only 2 values for 3D data
        SpatialMap(
            pos=pos, st=st, s_binsize=5.0, tuning_curve_sigma=wrong_sigma, speed_thres=0
        )

    # Test with 2D data
    pos_2d = nel.AnalogSignalArray(data=pos_data[:2], timestamps=timestamps)
    sigma_2d = [1.5, 2.5]
    spatial_map_2d = SpatialMap(
        pos=pos_2d, st=st, tuning_curve_sigma=sigma_2d, speed_thres=0
    )

    assert spatial_map_2d.tuning_curve_sigma_array.shape == (2,)
    assert np.array_equal(spatial_map_2d.tuning_curve_sigma_array, np.array(sigma_2d))

    # Test that N-dimensional mapping works with dimension-specific smoothing
    tc_3d, st_run_3d = spatial_map_array.map_nd()

    # Should return a TuningCurveND object
    assert isinstance(tc_3d, nel.TuningCurveND)
    assert tc_3d.n_units == st.n_units
    # Check that it has the expected number of dimensions in the ratemap shape
    assert len(tc_3d.ratemap.shape) == 4  # (n_units, dim1, dim2, dim3)
    assert tc_3d.ratemap.shape[0] == st.n_units


def test_spatial_map_continuous_multidimensional():
    """Test SpatialMap with continuous signals (AnalogSignalArray) and multi-dimensional positions."""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create 3D position data
    n_samples = 1000
    timestamps = np.linspace(0, 100, n_samples)
    x_pos = np.sin(timestamps * 0.1) * 50 + 50  # x: 0 to 100
    y_pos = np.cos(timestamps * 0.1) * 30 + 30  # y: 0 to 60
    z_pos = np.sin(timestamps * 0.05) * 20 + 20  # z: 0 to 40
    pos_data = np.vstack([x_pos, y_pos, z_pos])

    pos = nel.AnalogSignalArray(data=pos_data, timestamps=timestamps)

    # Create continuous signal data (AnalogSignalArray) with multiple units
    # Use different patterns for each unit to test mapping
    n_units = 3
    signal_data = []
    for unit in range(n_units):
        # Create different signal patterns for each unit
        signal = np.sin(timestamps * (0.1 + unit * 0.05)) * (unit + 1) + (unit + 1)
        signal_data.append(signal)

    signal_data = np.vstack(signal_data)
    continuous_signals = nel.AnalogSignalArray(data=signal_data, timestamps=timestamps)

    # Create SpatialMap with 3D position and continuous signals
    spatial_map_3d_continuous = SpatialMap(
        pos=pos,
        st=continuous_signals,  # Using continuous signals instead of spike trains
        s_binsize=10.0,  # Use larger bins for 3D
        speed_thres=0,  # No speed threshold
        tuning_curve_sigma=0,  # No smoothing for cleaner test
    )

    # Test basic properties
    assert spatial_map_3d_continuous.dim == 3
    assert spatial_map_3d_continuous.n_units == n_units
    assert (
        len(spatial_map_3d_continuous.ratemap.shape) == 4
    )  # (n_units, x_bins, y_bins, z_bins)
    assert spatial_map_3d_continuous.isempty is False
    assert spatial_map_3d_continuous.occupancy.sum() > 0
    assert spatial_map_3d_continuous.ratemap.min() >= 0
    assert spatial_map_3d_continuous.ratemap.max() > 0

    # Test that the tuning curve is properly created
    assert isinstance(spatial_map_3d_continuous.tc, nel.TuningCurveND)
    assert spatial_map_3d_continuous.tc.n_units == n_units

    # Test that ratemap has non-zero values (continuous signals should create activity)
    assert np.count_nonzero(spatial_map_3d_continuous.ratemap) > 0

    # Test that different units have different activity patterns
    unit_max_rates = [
        np.max(spatial_map_3d_continuous.ratemap[i]) for i in range(n_units)
    ]
    assert len(set(unit_max_rates)) > 1, (
        "Different units should have different max rates"
    )

    # Test that occupancy is computed correctly for 3D
    # For 3D, we need to calculate the expected shape from the position data bounds and bin sizes
    # Since z_edges isn't stored as an attribute, we calculate it directly
    z_min, z_max = spatial_map_3d_continuous.dim_minmax_array[2]
    z_binsize = spatial_map_3d_continuous.s_binsize_array[2]
    z_bins = len(np.arange(z_min, z_max + z_binsize, z_binsize)) - 1

    expected_occupancy_shape = (
        len(spatial_map_3d_continuous.x_edges) - 1,
        len(spatial_map_3d_continuous.y_edges) - 1,
        z_bins,
    )
    assert spatial_map_3d_continuous.occupancy.shape == expected_occupancy_shape

    # Test with dimension-specific bin sizes for continuous signals
    bin_sizes_3d = [5.0, 8.0, 12.0]  # Different bin size for each dimension
    spatial_map_3d_varied_bins = SpatialMap(
        pos=pos,
        st=continuous_signals,
        s_binsize=bin_sizes_3d,
        speed_thres=0,
        tuning_curve_sigma=0,
    )

    # Verify that different bin sizes are used
    assert np.isclose(np.diff(spatial_map_3d_varied_bins.x_edges)[0], 5.0), (
        "X bin size should be 5.0"
    )
    assert np.isclose(np.diff(spatial_map_3d_varied_bins.y_edges)[0], 8.0), (
        "Y bin size should be 8.0"
    )
    # For z dimension, calculate the edges manually since z_edges isn't stored as an attribute
    z_min_varied, z_max_varied = spatial_map_3d_varied_bins.dim_minmax_array[2]
    z_binsize_varied = spatial_map_3d_varied_bins.s_binsize_array[2]
    z_edges_varied = np.arange(
        z_min_varied, z_max_varied + z_binsize_varied, z_binsize_varied
    )
    assert np.isclose(np.diff(z_edges_varied)[0], 12.0), "Z bin size should be 12.0"

    # Test that ratemap shapes are different due to different bin sizes
    assert (
        spatial_map_3d_continuous.ratemap.shape
        != spatial_map_3d_varied_bins.ratemap.shape
    )


# --- Shuffling tests ---
def test_spatial_map_shuffle_1d():
    # Generate 1D synthetic data
    n_samples = 1000
    timestamps = np.linspace(0, 100, n_samples)
    x_pos = np.sin(timestamps * 0.1) * 50 + 50
    pos = nel.AnalogSignalArray(data=np.array([x_pos]), timestamps=timestamps)
    spike_times = np.sort(np.random.uniform(0, 100, 200))
    st = nel.SpikeTrainArray(time=spike_times, fs=1000.0)
    spatial_map = SpatialMap(pos=pos, st=st, s_binsize=5.0, speed_thres=0)
    pvals = spatial_map.shuffle_spatial_information()
    assert pvals.shape[0] == spatial_map.n_units
    assert np.all((pvals >= 0) & (pvals <= 1))


def test_spatial_map_shuffle_2d():
    # Generate 2D synthetic data
    n_samples = 1000
    timestamps = np.linspace(0, 100, n_samples)
    x_pos = np.sin(timestamps * 0.1) * 50 + 50
    y_pos = np.cos(timestamps * 0.1) * 30 + 30
    pos = nel.AnalogSignalArray(data=np.array([x_pos, y_pos]), timestamps=timestamps)
    spike_times = np.sort(np.random.uniform(0, 100, 200))
    st = nel.SpikeTrainArray(time=spike_times, fs=1000.0)
    spatial_map = SpatialMap(pos=pos, st=st, s_binsize=[5.0, 5.0], speed_thres=0)
    pvals = spatial_map.shuffle_spatial_information()
    assert pvals.shape[0] == spatial_map.n_units
    assert np.all((pvals >= 0) & (pvals <= 1))


def test_spatial_map_shuffle_nd():
    # Generate 3D synthetic data
    n_samples = 1000
    timestamps = np.linspace(0, 100, n_samples)
    x_pos = np.sin(timestamps * 0.1) * 50 + 50
    y_pos = np.cos(timestamps * 0.1) * 30 + 30
    z_pos = np.sin(timestamps * 0.05) * 20 + 20
    pos = nel.AnalogSignalArray(
        data=np.array([x_pos, y_pos, z_pos]), timestamps=timestamps
    )
    spike_times = np.sort(np.random.uniform(0, 100, 200))
    st = nel.SpikeTrainArray(time=spike_times, fs=1000.0)
    spatial_map = SpatialMap(pos=pos, st=st, s_binsize=[5.0, 8.0, 12.0], speed_thres=0)
    pvals = spatial_map.shuffle_spatial_information()
    assert pvals.shape[0] == spatial_map.n_units
    assert np.all((pvals >= 0) & (pvals <= 1))


def test_spatial_map_max_gap_changes_ratemap():
    """Check that max_gap impacts the final ratemap (sums should differ).

    This test is a behavioral check — it will skip if the SpatialMap API doesn't
    accept `max_gap` or if `ratemap` is not available.
    """
    time = np.concatenate([np.linspace(0, 10, 101), np.linspace(100, 110, 101)])
    x_pos = np.linspace(0, 1, len(time))

    pos = nel.AnalogSignalArray(np.array([x_pos]), time=time, fs=50)
    spike_times = np.linspace(0, 110, 200)
    st = nel.SpikeTrainArray([spike_times], fs=1000.0)

    try:
        sm_strict = SpatialMap(pos=pos, st=st, speed_thres=0, max_gap=1.0)
        sm_loose = SpatialMap(pos=pos, st=st, speed_thres=0, max_gap=200.0)
    except TypeError:
        pytest.skip("SpatialMap does not accept max_gap parameter in this version")

    if not (hasattr(sm_strict, "ratemap") and hasattr(sm_loose, "ratemap")):
        pytest.skip(
            "SpatialMap does not expose ratemap attribute; cannot compare outputs"
        )

    # Compare the total activity in the ratemaps
    sum_strict = np.nansum(sm_strict.ratemap)
    sum_loose = np.nansum(sm_loose.ratemap)

    # They should differ when interpolation/extrapolation across the gap is handled differently
    assert not np.isclose(sum_strict, sum_loose)


def test_max_gap_clamped_and_attribute_exposed():
    """Test that a too-small max_gap is clamped and `_min_allowed_gap` is exposed."""
    # create timestamps with nominal fs=50 (dt ~ 0.020)
    time = np.linspace(0, 10, 501)  # fs ~= 50.0 (exact 500 intervals)
    x_pos = np.linspace(0, 1, len(time))
    pos = nel.AnalogSignalArray(np.array([x_pos]), time=time, fs=50)

    spike_times = np.linspace(0, 10, 20)
    st = nel.SpikeTrainArray([spike_times], fs=1000.0)

    # ask for a max_gap far smaller than sampling interval
    sm = SpatialMap(pos=pos, st=st, speed_thres=0, max_gap=1e-6)

    # attribute should exist and be >= 1/pos.fs
    assert hasattr(sm, "_min_allowed_gap")
    assert sm.max_gap >= sm._min_allowed_gap


def test_clamping_changes_behavior_not_raising():
    """Constructing with tiny max_gap should not raise and should clamp."""
    time = np.linspace(0, 1, 101)
    x_pos = np.linspace(0, 1, len(time))
    pos = nel.AnalogSignalArray(np.array([x_pos]), time=time, fs=100)
    spike_times = np.linspace(0, 1, 10)
    st = nel.SpikeTrainArray([spike_times], fs=1000.0)

    # should not raise despite small max_gap; will be clamped internally
    sm = SpatialMap(pos=pos, st=st, speed_thres=0, max_gap=1e-6)
    assert sm.max_gap >= (1.0 / pos.fs)


def test_shuffle_detects_spatial_and_nonspatial_cells():
    """Simulate one strongly spatial unit and one non-spatial (uniform) unit.

    Use shuffle_spatial_information to ensure the spatial unit is significant
    and the non-spatial unit is not. Make RNG deterministic and disable
    parallel shuffling for test stability.
    """
    np.random.seed(0)

    # position: 1D linear track 0..200 over 100s with high sampling
    duration = 100.0
    fs_pos = 50.0
    n_samples = int(duration * fs_pos)
    timestamps = np.linspace(0, duration, n_samples)
    x_pos = np.linspace(0, 200, n_samples)

    # spatial unit: place field at 100 with gaussian tuning
    center = 100.0
    sigma = 5.0
    peak_rate = 40.0  # Hz (increase to make place field stronger)
    base_rate = 0.01
    dt = timestamps[1] - timestamps[0]
    rate_spatial = base_rate + peak_rate * np.exp(
        -0.5 * ((x_pos - center) / sigma) ** 2
    )

    # non-spatial unit: uniform low rate
    rate_nonspatial = np.full_like(rate_spatial, 0.2)

    # generate spikes
    def make_spikes(rate):
        counts = np.random.poisson(rate * dt)
        spike_times = []
        for t, c in zip(timestamps, counts):
            if c > 0:
                # spread multiple spikes uniformly within the bin
                spike_times.extend((t + np.random.rand(c) * dt).tolist())
        if len(spike_times) == 0:
            return np.array([])
        return np.sort(np.array(spike_times))

    # Create a strongly spatial unit by placing spikes preferentially within a tight
    # place field around `center`. Use a Bernoulli draw per sample in-field so the
    # spatial signal is robust and reproducible under the RNG seed.
    # widen field and increase per-sample spike probability to make unit strongly spatial
    in_field = np.abs(x_pos - center) <= (sigma * 4.0)
    p_spike_in_field = 0.8  # per-sample probability of a spike when in the field
    spikes_spatial = []
    for t, inf in zip(timestamps, in_field):
        if inf and np.random.rand() < p_spike_in_field:
            spikes_spatial.append(t + np.random.rand() * dt)
    spikes_spatial = np.sort(np.array(spikes_spatial))

    # Non-spatial unit remains Poisson with uniform rate
    spikes_nonspatial = make_spikes(rate_nonspatial)

    st = nel.SpikeTrainArray([spikes_spatial, spikes_nonspatial], fs=1000.0)
    pos = nel.AnalogSignalArray(np.array([x_pos]), timestamps=timestamps, fs=fs_pos)

    sm = SpatialMap(pos=pos, st=st, s_binsize=5.0, speed_thres=0)
    # reduce shuffles for test speed and avoid multiprocessing variability
    # increase shuffles for more stable p-value estimation in CI
    sm.n_shuff = 500
    sm.parallel_shuff = False

    # deterministic shuffles
    np.random.seed(0)
    pvals = sm.shuffle_spatial_information()

    assert pvals.shape[0] == 2
    # spatial cell should be significant at alpha=0.05
    assert pvals[0] < 0.05
    # non-spatial cell should not be significant
    assert pvals[1] > 0.05


def test_continuous_signal_binned_mean_high_bin():
    """
    Simulate position and continuous signal where high values occur
    within a small spatial region, then confirm the binned mean
    ratemap has its maximum at the expected spatial bin.
    """
    fs_pos = 50.0
    duration_s = 20.0
    timestamps = np.linspace(0, duration_s, int(duration_s * fs_pos))

    # x moves linearly 0..100, y fixed
    x = np.linspace(0, 100, timestamps.size)
    y = np.full_like(x, 50.0)

    # continuous signal: 5 channels, channel 0 has high values when x in [60,70]
    n_ch = 5
    rng = np.random.default_rng(0)
    con = rng.normal(scale=0.1, size=(n_ch, timestamps.size))
    high_mask = (x >= 60) & (x <= 70)
    con[0, high_mask] += 10.0

    pos = nel.AnalogSignalArray(np.vstack([x, y]), time=timestamps, fs=fs_pos)
    con_signal = nel.AnalogSignalArray(con, time=timestamps, fs=fs_pos)

    # Use 10 cm bins so 60-70 region maps to bin index 6 (0-based)
    sm = SpatialMap(pos=pos, st=con_signal, x_minmax=(0, 100), y_minmax=(0, 100), s_binsize=10, speed_thres=0)

    # ratemap shape: (n_units/n_channels, x_bins, y_bins)
    rm = sm.ratemap[0]
    occ = sm.occupancy

    # ignore empty bins when finding max
    rm_masked = np.where(occ > 0, rm, -np.inf)
    max_idx = np.unravel_index(np.nanargmax(rm_masked), rm_masked.shape)

    # Compute expected x bin based on SpatialMap's x_edges (robust to edge inclusion)
    center_high = x[high_mask].mean()
    expected_x_bin = np.searchsorted(sm.x_edges, center_high, side="right") - 1
    # `rm` shape is (x_bins, y_bins) so first index is x bin
    assert max_idx[0] == expected_x_bin, f"Expected peak x-bin {expected_x_bin}, got {max_idx[0]}"


def test_continuous_signal_binned_mean_high_bin_1d():
    """
    1D: continuous signal high values in a small region should map to expected bin.
    """
    fs_pos = 50.0
    duration_s = 10.0
    timestamps = np.linspace(0, duration_s, int(duration_s * fs_pos))

    # x moves linearly 0..100
    x = np.linspace(0, 100, timestamps.size)

    # continuous signal: 3 channels, channel 1 has high values when x in [30,40]
    n_ch = 3
    rng = np.random.default_rng(1)
    con = rng.normal(scale=0.05, size=(n_ch, timestamps.size))
    high_mask = (x >= 30) & (x <= 40)
    con[1, high_mask] += 5.0

    pos = nel.AnalogSignalArray(np.array([x]), time=timestamps, fs=fs_pos)
    con_signal = nel.AnalogSignalArray(con, time=timestamps, fs=fs_pos)

    sm = SpatialMap(pos=pos, st=con_signal, x_minmax=(0, 100), s_binsize=10, speed_thres=0)

    rm = sm.ratemap[1]  # channel 1
    occ = sm.occupancy

    rm_masked = np.where(occ > 0, rm, -np.inf)
    max_idx = np.unravel_index(np.nanargmax(rm_masked), rm_masked.shape)

    center_high = x[high_mask].mean()
    expected_x_bin = np.searchsorted(sm.x_edges, center_high, side="right") - 1
    # 1D ratemap shape is (x_bins,) so max_idx is a scalar tuple
    assert max_idx[0] == expected_x_bin, f"Expected peak x-bin {expected_x_bin}, got {max_idx[0]}"


def test_continuous_signal_binned_mean_high_bin_2d():
    """
    2D: continuous signal high values in a localized rectangle should map to expected x/y bins.
    """
    fs_pos = 50.0
    duration_s = 12.0
    timestamps = np.linspace(0, duration_s, int(duration_s * fs_pos))

    # x moves linearly 0..100, y oscillates between 0..100
    x = np.linspace(0, 100, timestamps.size)
    y = 50.0 + 20.0 * np.sin(timestamps / 2.0)

    n_ch = 4
    rng = np.random.default_rng(2)
    con = rng.normal(scale=0.1, size=(n_ch, timestamps.size))
    # create a guaranteed non-empty high region by picking a center x and a small window
    center_x_val = 75.0
    center_idx = np.abs(x - center_x_val).argmin()
    window = max(1, int(fs_pos * 0.2))
    start = max(0, center_idx - window)
    end = min(timestamps.size, center_idx + window)
    high_mask = np.zeros_like(x, dtype=bool)
    high_mask[start:end] = True
    con[2, high_mask] += 8.0

    pos = nel.AnalogSignalArray(np.vstack([x, y]), time=timestamps, fs=fs_pos)
    con_signal = nel.AnalogSignalArray(con, time=timestamps, fs=fs_pos)

    sm = SpatialMap(pos=pos, st=con_signal, x_minmax=(0, 100), y_minmax=(0, 100), s_binsize=10, speed_thres=0)

    rm = sm.ratemap[2]
    occ = sm.occupancy

    rm_masked = np.where(occ > 0, rm, -np.inf)
    max_idx = np.unravel_index(np.nanargmax(rm_masked), rm_masked.shape)

    # Determine which bins contain the high-value samples (robust to edge/rounding)
    x_bins_for_high = np.searchsorted(sm.x_edges, x[high_mask], side="right") - 1
    y_bins_for_high = np.searchsorted(sm.y_edges, y[high_mask], side="right") - 1
    # Remove out-of-range bins
    valid_mask = (
        (x_bins_for_high >= 0)
        & (x_bins_for_high < rm.shape[0])
        & (y_bins_for_high >= 0)
        & (y_bins_for_high < rm.shape[1])
    )
    bins_pairs = set(map(tuple, np.column_stack([x_bins_for_high[valid_mask], y_bins_for_high[valid_mask]])))

    assert (max_idx[0], max_idx[1]) in bins_pairs, f"Peak bin {(max_idx[0], max_idx[1])} not in high-value bins {bins_pairs}"


def test_continuous_signal_binned_mean_high_bin_3d():
    """
    3D: continuous signal high values in a localized volume should map to expected x/y/z bins.
    """
    fs_pos = 50.0
    duration_s = 16.0
    timestamps = np.linspace(0, duration_s, int(duration_s * fs_pos))

    x = np.linspace(0, 100, timestamps.size)
    y = 50.0 + 10.0 * np.cos(timestamps / 3.0)
    z = 20.0 + 15.0 * np.sin(timestamps / 4.0)

    n_ch = 3
    rng = np.random.default_rng(3)
    con = rng.normal(scale=0.05, size=(n_ch, timestamps.size))
    # create a guaranteed non-empty high region by picking a center x and a small window
    center_x_val = 15.0
    center_idx = np.abs(x - center_x_val).argmin()
    window = max(1, int(fs_pos * 0.2))
    start = max(0, center_idx - window)
    end = min(timestamps.size, center_idx + window)
    high_mask = np.zeros_like(x, dtype=bool)
    high_mask[start:end] = True
    con[0, high_mask] += 6.0

    pos = nel.AnalogSignalArray(np.vstack([x, y, z]), time=timestamps, fs=fs_pos)
    con_signal = nel.AnalogSignalArray(con, time=timestamps, fs=fs_pos)

    sm = SpatialMap(pos=pos, st=con_signal, s_binsize=[10, 10, 10], speed_thres=0)

    rm = sm.ratemap[0]
    occ = sm.occupancy

    rm_masked = np.where(occ > 0, rm, -np.inf)
    max_idx = np.unravel_index(np.nanargmax(rm_masked), rm_masked.shape)

    center_x = x[high_mask].mean()
    center_y = y[high_mask].mean()
    center_z = z[high_mask].mean()

    # Compute bin indices for each high-value sample
    x_bins_for_high = np.searchsorted(sm.x_edges, x[high_mask], side="right") - 1
    y_bins_for_high = np.searchsorted(sm.y_edges, y[high_mask], side="right") - 1
    z_min, z_max = sm.dim_minmax_array[2]
    z_binsize = sm.s_binsize_array[2]
    z_edges = np.arange(z_min, z_max + z_binsize, z_binsize)
    z_bins_for_high = np.searchsorted(z_edges, z[high_mask], side="right") - 1

    # Filter to valid bin indices
    valid_mask = (
        (x_bins_for_high >= 0)
        & (x_bins_for_high < rm.shape[0])
        & (y_bins_for_high >= 0)
        & (y_bins_for_high < rm.shape[1])
        & (z_bins_for_high >= 0)
        & (z_bins_for_high < rm.shape[2])
    )

    bins_triplets = set(
        map(
            tuple,
            np.column_stack([x_bins_for_high[valid_mask], y_bins_for_high[valid_mask], z_bins_for_high[valid_mask]]),
        )
    )

    assert (max_idx[0], max_idx[1], max_idx[2]) in bins_triplets, f"Peak bin {(max_idx[0], max_idx[1], max_idx[2])} not in high-value bins {bins_triplets}"
