import nelpy as nel
import numpy as np

from neuro_py.detectors.up_down_state import detect_up_down_states


def test_detect_up_down_states_detection():
    # Mock data generation
    down_state_times = [4, 7, 15]

    spikes_upstate = []
    for i in range(10):
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
    assert (up_state_epochs & down_state_epochs).isempty, (
        "DOWN state detected in UP state."
    )
