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


@pytest.fixture
def spatial_map_nan_pos_w_no_speed_thresh_fixture():
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
    return spatial_map


def test_spatial_map_properties_nan_pos(spatial_map_nan_pos_w_no_speed_thresh_fixture):
    """
    Test the properties of SpatialMap.
    """
    spatial_map = spatial_map_nan_pos_w_no_speed_thresh_fixture

    assert spatial_map.is2d is True
    assert spatial_map.n_units == 5
    assert spatial_map.occupancy.shape == (34, 34)
    assert spatial_map.ratemap.shape == (5, 34, 34)  # 5 units with 34x34 maps
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
