import numpy as np

from neuro_py.lfp.spectral import event_triggered_wavelet


def test_event_triggered_wavelet():
    # Test 1, basic data
    # Create some example data for testing
    np.random.seed(0)
    signal = np.random.rand(1000)
    timestamps = np.linspace(0, 10, 1000)
    events = np.random.choice(timestamps, size=100, replace=False)

    # Test the function with example data
    mwt, sigs, times, freqs = event_triggered_wavelet(signal, timestamps, events)

    # Assert that the output shapes are correct
    assert mwt.shape == (
        24,
        200,
    )  # Adjust according to your frequency range and max_lag
    assert sigs.shape == (200,)
    assert times.shape == (200,)
    np.testing.assert_allclose(np.diff(times), timestamps[1] - timestamps[0])
    assert freqs.shape == (24,)

    # Assert that the average signal is within expected range
    assert np.all(np.all(sigs >= 0) and np.all(sigs <= 1))

    # Assert that the average wavelet transform is within expected range
    assert np.all(mwt >= 0)

    # Test 2, different sample rate
    # Create some example data for testing
    np.random.seed(0)
    signal = np.random.rand(1000)
    timestamps = np.linspace(0, 11, 1000)
    events = np.random.choice(timestamps, size=100, replace=False)

    # Test the function with example data
    mwt, sigs, times, freqs = event_triggered_wavelet(signal, timestamps, events)

    # Assert that the output shapes are correct
    assert mwt.shape == (
        24,
        182,
    )  # Adjust according to your frequency range and max_lag
    assert sigs.shape == (182,)
    assert times.shape == (182,)
    np.testing.assert_allclose(np.diff(times), timestamps[1] - timestamps[0])
    assert freqs.shape == (24,)

    # Assert that the average signal is within expected range
    assert np.all(np.all(sigs >= 0) and np.all(sigs <= 1))

    # Assert that the average wavelet transform is within expected range
    assert np.all(mwt >= 0)
