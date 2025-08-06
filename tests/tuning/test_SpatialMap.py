import logging

import nelpy as nel
import numpy as np
import pytest

from neuro_py.tuning.maps import SpatialMap

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

    # Also check that the overall statistics are the same for 2D
    assert np.isclose(np.sum(tc_old_2d.ratemap), np.sum(tc_new_2d.ratemap), rtol=1e-10)
    assert np.count_nonzero(tc_old_2d.ratemap) == np.count_nonzero(tc_new_2d.ratemap)


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
