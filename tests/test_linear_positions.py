import nelpy as nel
import numpy as np
import pandas as pd
import pytest

# Assuming the functions are in a module named `linear_track`
from neuro_py.behavior.linear_positions import (
    __find_good_laps,
    __find_laps,
    __peakdetz,
    find_good_lap_epochs,
    get_linear_track_lap_epochs,
    linearize_position,
)


# Test data
@pytest.fixture
def position_data():
    # Simulate a linear track with laps
    track_length = 100  # Length of the linear track
    num_points = 500  # Total number of data points
    num_laps = 5  # Number of laps to simulate

    # Generate position data that oscillates between 0 and track_length
    x = np.linspace(0, track_length, num_points)

    # Simulate laps by repeating the track
    x = np.tile(x, num_laps)

    y = np.zeros_like(x)  # y-coordinates can be zero for simplicity

    return x, y


@pytest.fixture
def timestamp_data():
    # Simulate timestamps for the position data
    num_points = 500  # Number of data points per lap
    num_laps = 5  # Number of laps
    total_points = num_points * num_laps
    timestamps = np.linspace(0, 100, total_points)  # Timestamps from 0 to 100 seconds
    return timestamps


@pytest.fixture
def lap_data():
    return pd.DataFrame(
        {
            "start_ts": [0, 2, 4, 6, 8],
            "pos": [1, 3, 5, 7, 9],
            "start_idx": [0, 2, 4, 6, 8],
            "direction": [1, -1, 1, -1, 1],
        }
    )


@pytest.fixture
def analog_signal_array(position_data, timestamp_data):
    data, _ = position_data
    time = timestamp_data
    return nel.AnalogSignalArray(data=data, timestamps=time)


@pytest.fixture
def epoch_array():
    return nel.EpochArray(np.array([[0, 2], [2, 4], [4, 6], [6, 8], [8, 10]]) * 10)


# Test cases
def test_linearize_position(position_data):
    x, y = position_data
    linear_x, linear_y = linearize_position(x, y)
    assert linear_x.shape == x.shape
    assert linear_y.shape == y.shape
    assert not np.isnan(linear_x).any()
    assert not np.isnan(linear_y).any()


def test_find_laps(timestamp_data, position_data):
    ts = timestamp_data
    x, _ = position_data
    laps = __find_laps(ts, x)
    assert isinstance(laps, pd.DataFrame)
    assert not laps.empty
    assert "start_ts" in laps.columns
    assert "pos" in laps.columns
    assert "start_idx" in laps.columns
    assert "direction" in laps.columns


def test_peakdetz():
    v = np.array([1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2, 1])
    delta = 0.5
    maxtab, mintab = __peakdetz(v, delta)

    # Check peaks (adjust for 0-based indexing)
    assert maxtab == [(2, 3), (6, 3), (10, 3)]

    # Check troughs (adjust for 0-based indexing)
    assert mintab == [(4, 1), (8, 1), (12, 1)]


def test_find_good_laps(timestamp_data, position_data, lap_data):
    ts = timestamp_data
    x, _ = position_data
    laps = lap_data
    good_laps = __find_good_laps(ts, x, laps)
    assert isinstance(good_laps, pd.DataFrame)
    assert not good_laps.empty


def test_get_linear_track_lap_epochs(timestamp_data, position_data):
    ts = timestamp_data
    x, _ = position_data
    outbound_epochs, inbound_epochs = get_linear_track_lap_epochs(ts, x)
    assert isinstance(outbound_epochs, nel.EpochArray)
    assert isinstance(inbound_epochs, nel.EpochArray)
    assert outbound_epochs.n_intervals > 0
    assert inbound_epochs.n_intervals > 0


def test_find_good_lap_epochs(analog_signal_array, epoch_array):
    pos = analog_signal_array
    dir_epoch = epoch_array
    good_laps = find_good_lap_epochs(pos, dir_epoch, min_laps=0)
    assert isinstance(good_laps, nel.EpochArray)
    assert good_laps.n_intervals == 5


# Edge cases
def test_linearize_position_with_nans():
    x = np.array([1, 2, np.nan, 4, 5])
    y = np.array([1, 2, 3, np.nan, 5])
    linear_x, linear_y = linearize_position(x, y)
    assert np.isnan(linear_x).any()
    assert np.isnan(linear_y).any()


def test_find_laps_with_empty_data():
    ts = np.array([])
    x = np.array([])
    laps = __find_laps(ts, x)
    assert laps.empty


def test_peakdetz_with_no_peaks():
    v = np.array([1, 1, 1, 1, 1])
    delta = 0.5
    maxtab, mintab = __peakdetz(v, delta)
    assert len(maxtab) == 0
    assert len(mintab) == 0


def test_find_good_laps_with_all_bad_laps(timestamp_data, position_data, lap_data):
    ts = timestamp_data
    x, _ = position_data
    laps = lap_data
    laps["pos"] = np.nan  # Make all laps bad
    good_laps = __find_good_laps(ts, x, laps)
    assert good_laps.empty


def test_get_linear_track_lap_epochs_with_no_laps(timestamp_data):
    ts = timestamp_data
    x = np.ones_like(ts)  # No laps
    outbound_epochs, inbound_epochs = get_linear_track_lap_epochs(ts, x)
    assert outbound_epochs.n_intervals == 0
    assert inbound_epochs.n_intervals == 0


def test_find_good_lap_epochs_with_no_good_laps(analog_signal_array, epoch_array):
    pos = analog_signal_array
    dir_epoch = epoch_array
    good_laps = find_good_lap_epochs(pos, dir_epoch, thres=1.0)  # Impossible threshold
    assert good_laps.n_intervals == 0
