import nelpy as nel
import numpy as np

from neuro_py.detectors.up_down_state import (
    detect_up_down_states,
    detect_up_down_states_bimodal_thresh,
)


def test_detect_up_down_states_detection():
    # Mock data generation
    down_state_times = [4, 7, 15]

    spikes_upstate = []
    for _ in range(10):
        up_spikes = np.random.poisson(0.25, 20_000)
        spikes = np.where(up_spikes > 0)[0] * 0.001
        # silence around down states
        for down_state_time in down_state_times:
            spikes = spikes[
                ~((spikes > down_state_time - 0.1) & (spikes < down_state_time + 0.1))
            ]
        spikes_upstate.append(spikes)

    nrem_epochs = nel.EpochArray([np.array([[0, 20]])])
    st = nel.SpikeTrainArray(timestamps=spikes_upstate, fs=1000, support=nrem_epochs)

    # Run the function with mock data
    down_state_epochs, up_state_epochs = detect_up_down_states(
        st=st,
        nrem_epochs=nrem_epochs,
        save_mat=False,  # Disable saving
    )

    # Assertions to check detection accuracy
    assert down_state_epochs is not None, "Expected detected DOWN states, got None."
    assert up_state_epochs is not None, "Expected detected UP states, got None."

    # Check DOWN states detected in low activity periods
    down_state_intervals = down_state_epochs.time
    for down_state_time in down_state_times:
        assert any(
            (down_state_time >= down_state_intervals[:, 0])
            & (down_state_time <= down_state_intervals[:, 1])
        ), f"DOWN state at {down_state_time} not detected."

    # Check UP states detected in high activity periods
    up_state_intervals = up_state_epochs.time
    for down_state_time in down_state_times:
        assert not any(
            (down_state_time >= up_state_intervals[:, 0])
            & (down_state_time <= up_state_intervals[:, 1])
        ), f"UP state at {down_state_time} detected."

    # Check UP states detected in high activity periods
    up_state_intervals = up_state_epochs.time
    for down_state_time in down_state_times:
        assert not any(
            (down_state_time >= up_state_intervals[:, 0])
            & (down_state_time <= up_state_intervals[:, 1])
        ), f"UP state at {down_state_time} detected."

    # Check that DOWN states are not detected in UP states
    assert (
        up_state_epochs & down_state_epochs
    ).isempty, "DOWN state detected in UP state."


def test_detect_up_down_states_epoch_by_epoch():
    # Create two sleep epochs and embed clear low-activity (DOWN) windows in each
    down_state_times = [4.0, 15.0]  # one in each behavioral epoch

    spikes_upstate = []
    for _ in range(10):
        # background firing during UP (Poisson), then silence around DOWN times
        up_spikes = np.random.poisson(0.25, 20_000)
        spikes = np.where(up_spikes > 0)[0] * 0.001  # seconds

        # carve out silences (DOWN states) around the specified times
        for t in down_state_times:
            spikes = spikes[~((spikes > t - 0.1) & (spikes < t + 0.1))]

        spikes_upstate.append(spikes)

    # One NREM block spanning full duration; behavior epochs split into two halves
    nrem_epochs = nel.EpochArray([np.array([[0.0, 20.0]])])
    beh_epochs = nel.EpochArray([np.array([[0.0, 10.0], [10.0, 20.0]])])
    st = nel.SpikeTrainArray(timestamps=spikes_upstate, fs=1000, support=nrem_epochs)

    down_state_epochs, up_state_epochs = detect_up_down_states(
        st=st,
        nrem_epochs=nrem_epochs,
        beh_epochs=beh_epochs,
        epoch_by_epoch=True,
        save_mat=False,
    )

    # basic existence
    assert (
        down_state_epochs is not None
    ), "Expected DOWN states with epoch_by_epoch=True."
    assert up_state_epochs is not None, "Expected UP states with epoch_by_epoch=True."

    # Verify detected DOWN states occur near the silenced windows and stay within behavior epochs
    d_intervals = down_state_epochs.time
    for t in down_state_times:
        # Found as a DOWN state somewhere
        assert any(
            (t >= d_intervals[:, 0]) & (t <= d_intervals[:, 1])
        ), f"DOWN state at {t} s not detected in epoch_by_epoch mode."

    # Ensure results respect epoch boundaries: every interval should be fully inside one beh epoch
    beh_intervals = beh_epochs.time
    for start, stop in d_intervals:
        # contained within at least one behavior epoch
        assert any(
            (start >= beh_intervals[:, 0]) & (stop <= beh_intervals[:, 1])
        ), f"Detected DOWN interval [{start}, {stop}] crosses behavior epoch boundary."

    # No overlap between up and down outputs
    assert (
        up_state_epochs & down_state_epochs
    ).isempty, "DOWN state detected within UP state for epoch_by_epoch mode."


def test_detect_up_down_states_bimodal_thresh_basic():
    """Test bimodal_thresh detector with synthetic bimodal-ish data."""
    # Create data with two activity levels: low (DOWN-like) and high (UP-like)
    rng = np.random.default_rng(42)

    # Generate firing rates that cycle between low and high states
    low_rates = rng.normal(2, 0.5, 1000)
    high_rates = rng.normal(8, 1.0, 1000)
    firing_rate_series = np.concatenate([low_rates, high_rates, low_rates, high_rates])

    # Convert firing rate series to spike times for 10 neurons
    nrem_epochs = nel.EpochArray([np.array([[0, 40]])])
    spikes_list = []
    for neuron_id in range(10):
        spike_times = []
        t = 0
        bin_size = 0.01
        for rate in firing_rate_series:
            n_spikes = np.random.poisson(rate * bin_size)
            if n_spikes > 0:
                spike_times.extend(np.random.uniform(t, t + bin_size, n_spikes))
            t += bin_size
        spikes_list.append(np.sort(spike_times))

    st = nel.SpikeTrainArray(timestamps=spikes_list, fs=1000, support=nrem_epochs)

    # Run bimodal_thresh detector with force_bimodal to ensure detection
    down_state_epochs, up_state_epochs = detect_up_down_states_bimodal_thresh(
        st=st,
        nrem_epochs=nrem_epochs,
        save_mat=False,
        force_bimodal=True,
        nboot=10,  # Use few bootstraps for speed in test
    )

    # Basic checks
    assert down_state_epochs is not None, "Expected DOWN states, got None."
    assert up_state_epochs is not None, "Expected UP states, got None."

    # Should have detected some states
    assert down_state_epochs.n_epochs > 0, "No DOWN states detected."
    assert up_state_epochs.n_epochs > 0, "No UP states detected."

    # DOWN and UP should not overlap
    assert (down_state_epochs & up_state_epochs).isempty, "DOWN and UP states overlap."

    # All epochs should be within NREM support
    for epoch in [down_state_epochs, up_state_epochs]:
        for start, stop in epoch.time:
            assert (
                start >= nrem_epochs.starts[0] and stop <= nrem_epochs.stops[0]
            ), f"State interval [{start}, {stop}] outside NREM bounds."


def test_detect_up_down_states_bimodal_thresh_epoch_by_epoch():
    """Test bimodal_thresh detector in epoch-by-epoch mode."""
    rng = np.random.default_rng(123)

    # Create two separate sleep epochs with low-high activity cycles
    low_rates = rng.normal(2, 0.4, 800)
    high_rates = rng.normal(8, 0.8, 800)
    epoch1_rates = np.concatenate([low_rates, high_rates])
    epoch2_rates = np.concatenate([high_rates, low_rates])

    # Convert to spike times for 10 neurons
    nrem_epochs = nel.EpochArray([np.array([[0, 20], [20, 40]])])
    beh_epochs = nel.EpochArray([np.array([[0, 20], [20, 40]])])

    spikes_list = []
    for neuron_id in range(10):
        spike_times = []
        t = 0
        bin_size = 0.01
        all_rates = np.concatenate([epoch1_rates, epoch2_rates])
        for rate in all_rates:
            n_spikes = np.random.poisson(rate * bin_size)
            if n_spikes > 0:
                spike_times.extend(np.random.uniform(t, t + bin_size, n_spikes))
            t += bin_size
        spikes_list.append(np.sort(spike_times))

    st = nel.SpikeTrainArray(timestamps=spikes_list, fs=1000, support=nrem_epochs)

    # Run with epoch_by_epoch=True
    down_state_epochs, up_state_epochs = detect_up_down_states_bimodal_thresh(
        st=st,
        nrem_epochs=nrem_epochs,
        beh_epochs=beh_epochs,
        epoch_by_epoch=True,
        save_mat=False,
        force_bimodal=True,
        nboot=10,
    )

    assert (
        down_state_epochs is not None
    ), "Expected DOWN states with epoch_by_epoch=True."
    assert up_state_epochs is not None, "Expected UP states with epoch_by_epoch=True."
    assert (
        down_state_epochs.n_epochs > 0
    ), "No DOWN states detected in epoch_by_epoch mode."

    # All states should respect NREM boundaries
    for start, stop in down_state_epochs.time:
        assert any(
            (start >= nrem_epochs.starts) & (stop <= nrem_epochs.stops)
        ), f"DOWN interval [{start}, {stop}] not contained in any NREM epoch."
