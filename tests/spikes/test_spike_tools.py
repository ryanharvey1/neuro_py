import numpy as np
import pandas as pd
import pytest
from neuro_py.spikes import spike_tools


def test_get_spindices_basic():
    spike_trains = [np.array([0.1, 0.2, 0.4]), np.array([0.15, 0.35])]
    df = spike_tools.get_spindices(spike_trains)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"spike_times", "spike_id"}
    assert np.all(np.diff(df["spike_times"]) >= 0)  # Sorted
    assert set(df["spike_id"]) == {0, 1}


def test_spindices_to_ndarray_basic():
    spike_trains = [np.array([0.1, 0.2]), np.array([0.3, 0.4, 0.5])]
    df = spike_tools.get_spindices(spike_trains)
    out = spike_tools.spindices_to_ndarray(df)
    assert isinstance(out, list)
    assert len(out) == 2
    assert np.allclose(out[0], [0.1, 0.2])
    assert np.allclose(out[1], [0.3, 0.4, 0.5])
    # Test spike_id filtering
    out2 = spike_tools.spindices_to_ndarray(df, spike_id=[1])
    assert len(out2) == 1
    assert np.allclose(out2[0], [0.3, 0.4, 0.5])


def test_BurstIndex_Royer_2012_basic():
    idx = np.linspace(0, 0.06, 61)
    # Simulate autocorr with a peak at 0.005 and baseline at 0.045
    autocorr = pd.DataFrame(
        {
            "cell1": np.exp(-(((idx - 0.005) / 0.002) ** 2)) + 0.1,
            "cell2": np.exp(-(((idx - 0.045) / 0.002) ** 2)) + 0.2,
        },
        index=idx,
    )
    burst_idx = spike_tools.BurstIndex_Royer_2012(autocorr)
    assert isinstance(burst_idx, list)
    assert len(burst_idx) == 2
    assert burst_idx[0] > 0  # cell1 is bursty
    assert burst_idx[1] < 0  # cell2 is non-bursty


def test_select_burst_spikes_bursts():
    spikes = np.array([0, 0.001, 0.002, 0.01, 0.011, 0.012])
    selected = spike_tools.select_burst_spikes(spikes, mode="bursts", isiBursts=0.006)
    assert isinstance(selected, np.ndarray)
    assert selected.dtype == bool
    # Should select spikes in bursts (first three and last three)
    assert np.sum(selected) >= 2


def test_select_burst_spikes_single():
    spikes = np.array([0, 0.01, 0.03, 0.06, 0.09])
    selected = spike_tools.select_burst_spikes(spikes, mode="single", isiSpikes=0.02)
    assert isinstance(selected, np.ndarray)
    assert selected.dtype == bool
    # Should select spikes with large ISIs
    assert np.sum(selected) >= 1


def test_select_burst_spikes_edge_cases():
    # Empty input
    spikes = np.array([])
    selected = spike_tools.select_burst_spikes(spikes)
    assert isinstance(selected, np.ndarray)
    # Single spike
    spikes = np.array([0.1])
    selected = spike_tools.select_burst_spikes(spikes)
    assert isinstance(selected, np.ndarray)
