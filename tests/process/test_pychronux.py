import re

import numpy as np
import pandas as pd
import pytest
from scipy.signal.windows import dpss

from neuro_py.process import pychronux as px


def test_pychronux():
    # Test the function
    assert px is not None


def test_getfgrid():
    Fs = 1000
    nfft = 1024
    fpass = [0, 500]

    expected_df = Fs / nfft
    expected_f = np.arange(0, Fs + expected_df, expected_df)
    expected_f = expected_f[0:nfft]
    expected_findx = (expected_f >= fpass[0]) & (expected_f <= fpass[-1])
    expected_f = expected_f[expected_findx]

    f, findx = px.getfgrid(Fs, nfft, fpass)

    assert np.allclose(f, expected_f), (
        "Frequency vector does not match expected values."
    )
    assert np.array_equal(findx, expected_findx), (
        "Index array does not match expected values."
    )


def test_dpsschk():
    tapers = (2.5, 4)
    N = 256
    Fs = 1000.0

    # Generate expected DPSS tapers using scipy
    expected_tapers, _ = dpss(
        N, NW=tapers[0], Kmax=tapers[1], sym=False, return_ratios=True
    )
    expected_tapers = expected_tapers * np.sqrt(Fs)
    expected_tapers = expected_tapers.T

    # Call the function
    result_tapers = px.dpsschk(tapers, N, Fs)

    # Check if the result matches the expected tapers
    assert np.allclose(result_tapers, expected_tapers), (
        "DPSS tapers do not match expected values."
    )


def test_get_tapers():
    N = 256
    bandwidth = 4.0
    fs = 1000.0
    min_lambda = 0.95

    # Generate expected DPSS tapers using scipy
    NW = bandwidth * N / fs
    K = int(np.ceil(2 * NW)) - 1
    expected_tapers, expected_lambdas = dpss(
        N, NW=NW, Kmax=K, sym=False, norm=2, return_ratios=True
    )
    mask = expected_lambdas > min_lambda
    expected_tapers = expected_tapers[mask]
    expected_lambdas = expected_lambdas[mask]

    # Ensure n_tapers does not exceed the number of tapers that satisfy the criteria
    n_tapers = min(5, expected_tapers.shape[0])

    if n_tapers is not None:
        expected_tapers = expected_tapers[:n_tapers]
        expected_lambdas = expected_lambdas[:n_tapers]

    # Call the function
    tapers, lambdas = px.get_tapers(
        N, bandwidth, fs=fs, min_lambda=min_lambda, n_tapers=n_tapers
    )

    # Check if the result matches the expected tapers and lambdas
    assert np.allclose(tapers, expected_tapers), "Tapers do not match expected values."
    assert np.allclose(lambdas, expected_lambdas), (
        "Lambdas do not match expected values."
    )


def test_get_tapers_value_error():
    N = 256
    bandwidth = 0.1
    fs = 1000.0
    min_lambda = 0.95

    with pytest.raises(ValueError, match="Not enough tapers"):
        px.get_tapers(N, bandwidth, fs=fs, min_lambda=min_lambda)

    bandwidth = 4.0
    n_tapers = 10

    with pytest.raises(
        ValueError,
        match="None of the tapers satisfied the minimum energy concentration",
    ):
        px.get_tapers(N, bandwidth, fs=fs, min_lambda=1.0)

    with pytest.raises(ValueError, match="'n_tapers' of 10 is greater than the"):
        px.get_tapers(N, bandwidth, fs=fs, min_lambda=min_lambda, n_tapers=n_tapers)


def test_mtfftpt():
    # Test Case 1: Basic functionality
    data = np.array([0.1, 0.2, 0.4, 0.5])  # Spike times
    tapers = np.array(
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.7, 0.8],
        ]
    )  # DPSS tapers
    nfft = 8
    t = np.linspace(0, 1, 4)  # Time vector
    f = np.linspace(0, 10, 5)  # Frequency vector

    # Update findx to match the size of nfft
    findx = np.zeros(nfft, dtype=bool)
    findx[: len(f)] = True

    # Call the function
    J, Msp, Nsp = px.mtfftpt(data, tapers, nfft, t, f, findx)

    # Assertions
    assert J.shape == (5, 2)  # Ensure correct shape of J
    assert np.isclose(Msp, 4 / len(t))  # Verify mean spike rate
    assert Nsp == 4  # Verify total spike count

    # Test Case 2: Empty spike times
    data_empty = np.array([])
    J_empty, Msp_empty, Nsp_empty = px.mtfftpt(data_empty, tapers, nfft, t, f, findx)

    # Assertions
    assert J_empty.shape == (5, 2)
    assert np.allclose(J_empty, 0)  # J should be zeros
    assert Msp_empty == 0  # Mean spike rate should be zero
    assert Nsp_empty == 0  # Total spike count should be zero

    # Test Case 3: No valid spike times in range
    data_out_of_range = np.array([-0.5, 1.5])
    J_out, Msp_out, Nsp_out = px.mtfftpt(data_out_of_range, tapers, nfft, t, f, findx)

    # Assertions
    assert J_out.shape == (5, 2)
    assert np.allclose(J_out, 0)  # J should be zeros
    assert Msp_out == 0  # Mean spike rate should be zero
    assert Nsp_out == 0  # Total spike count should be zero

    # Test Case 4: Single spike time
    data_single = np.array([0.2])
    J_single, Msp_single, Nsp_single = px.mtfftpt(
        data_single, tapers, nfft, t, f, findx
    )

    # Assertions
    assert J_single.shape == (5, 2)  # Ensure correct shape of J
    assert Msp_single == 1 / len(t)  # Verify mean spike rate
    assert Nsp_single == 1  # Verify total spike count

    # Test Case 5: Multiple tapers and frequencies
    data_multiple = np.array([0.1, 0.25, 0.5])
    tapers_multiple = np.random.rand(4, 3)  # Random tapers (4 time points, 3 tapers)
    nfft_multiple = 16
    t_multiple = np.linspace(0, 1, 4)
    f_multiple = np.linspace(0, 20, 8)

    # Update findx for this case
    findx_multiple = np.zeros(nfft_multiple, dtype=bool)
    findx_multiple[: len(f_multiple)] = True

    J_mult, Msp_mult, Nsp_mult = px.mtfftpt(
        data_multiple,
        tapers_multiple,
        nfft_multiple,
        t_multiple,
        f_multiple,
        findx_multiple,
    )

    # Assertions
    assert J_mult.shape == (8, 3)  # Ensure correct shape of J
    assert Msp_mult == len(data_multiple) / len(t_multiple)
    assert Nsp_mult == len(data_multiple)


def simulate_bursting_cell(frequency, duration, burst_rate, Fs):
    """
    Simulate spike times for a bursting cell.

    Parameters
    ----------
    frequency : float
        Frequency of bursts in Hz (e.g., 8 Hz).
    duration : float
        Duration of the simulation in seconds.
    burst_rate : float
        Rate of spikes within each burst (e.g., 50 Hz).
    Fs : int
        Sampling frequency for the time vector.

    Returns
    -------
    spike_times : np.ndarray
        Array of spike times in seconds.
    """
    burst_times = np.arange(0, duration, 1 / frequency)
    spike_times = []
    for bt in burst_times:
        burst_spikes = bt + np.random.rand(int(burst_rate / frequency)) / burst_rate
        spike_times.extend(burst_spikes)
    spike_times = np.array(spike_times)
    spike_times = spike_times[spike_times < duration]  # Clip to duration
    return spike_times


def test_mtspectrumpt():
    # Test Case 1: Basic functionality
    data = np.array(
        [
            np.array([0.1, 0.2, 0.4, 0.6, 0.77, 1, 1.1, 1.11]),
            np.array([0.3, 0.5, 0.7, 0.73, 1, 2, 2.1]),
        ],
        dtype=object,
    )  # Spike times (2 channels)
    Fs = 1000  # Sampling frequency in Hz
    fpass = [1, 50]  # Frequency range to evaluate
    NW = 2.5  # Time-bandwidth product
    n_tapers = 3  # Number of tapers

    spectrum_df = px.mtspectrumpt(data, Fs, fpass, NW, n_tapers)

    # Assertions
    assert isinstance(spectrum_df, pd.DataFrame), "Output should be a pandas DataFrame."
    assert len(spectrum_df) > 0, "Spectrum DataFrame should not be empty."
    assert all(fpass[0] <= f <= fpass[1] for f in spectrum_df.index), (
        "Frequencies should lie within fpass."
    )

    # Test Case 2: Precomputed tapers from scipy
    mintime = np.min(np.concatenate(data))
    maxtime = np.max(np.concatenate(data))
    dt = 1 / Fs
    tapers_ts = np.arange(mintime - dt, maxtime + dt, dt)

    tapers, _ = dpss(len(tapers_ts), NW, n_tapers, return_ratios=True)

    spectrum_df_precomputed = px.mtspectrumpt(
        data, Fs, fpass, NW, n_tapers, tapers=tapers.T
    )

    # Assertions
    assert isinstance(spectrum_df_precomputed, pd.DataFrame), (
        "Output with precomputed tapers should be a DataFrame."
    )

    # Test Case 3: Empty data
    empty_data = np.array([])
    spectrum_df_empty = px.mtspectrumpt(empty_data, Fs, fpass, NW, n_tapers)

    # Assertions
    assert spectrum_df_empty.empty, (
        "Output for empty data should be an empty DataFrame."
    )

    # Test Case 4: Invalid frequency range
    invalid_fpass = [50, 1]  # Invalid frequency range
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Invalid frequency range: fpass[0] should be less than fpass[1]."
        ),
    ):
        px.mtspectrumpt(data, Fs, invalid_fpass, NW, n_tapers)

    # Test Case 5: Single channel data
    single_channel_data = np.array([[0.1, 0.3, 0.5]])  # Single channel of spike times
    spectrum_df_single = px.mtspectrumpt(single_channel_data, Fs, fpass, NW, n_tapers)

    # Assertions
    assert isinstance(spectrum_df_single, pd.DataFrame), (
        "Output should be a pandas DataFrame for single channel."
    )
    assert spectrum_df_single.shape[1] == 1, (
        "Output DataFrame should have one column for single channel."
    )

    # Test Case 6: Custom time support
    time_support = [0.0, 0.8]  # Custom time range
    spectrum_df_support = px.mtspectrumpt(
        data, Fs, fpass, NW, n_tapers, time_support=time_support
    )

    # Assertions
    assert isinstance(spectrum_df_support, pd.DataFrame), (
        "Output with custom time support should be a DataFrame."
    )

    # Test Case 7: validate frequency peak
    # Parameters
    Fs = 1000  # Sampling frequency (Hz)
    duration = 10  # Duration of simulation (seconds)
    burst_frequency = 8  # Burst frequency (Hz)
    burst_rate = 50  # Rate of spikes within each burst (Hz)

    # Simulate spike times
    spike_times = simulate_bursting_cell(burst_frequency, duration, burst_rate, Fs)

    spec = px.mtspectrumpt(
        [spike_times],
        1250,
        [1, 40],
        NW=3,
        n_tapers=5,
        time_support=[0, duration],
    )
    # peak is at 8 Hz
    assert spec.index[np.argmax(spec.values)].astype(int) == 8


def test_mtfftc():
    # Generate mock data
    data = np.random.randn(256)
    N = data.shape[0]
    bandwidth = 4.0
    fs = 1000.0
    nfft = 512

    # Generate tapers using scipy
    NW = bandwidth * N / fs
    K = int(np.ceil(2 * NW)) - 1
    tapers, _ = dpss(N, NW=NW, Kmax=K, sym=False, norm=2, return_ratios=True)

    # Ensure tapers have the correct shape (N, K)
    tapers = tapers.T

    # Call the function
    J = px.mtfftc(data, tapers, nfft, fs)

    # Check the shape of the output
    assert J.shape == (nfft, K), "Shape of J does not match expected values."

    # Check if the function raises a ValueError for incompatible lengths
    with pytest.raises(
        ValueError, match="Length of tapers is incompatible with length of data."
    ):
        px.mtfftc(data[:-1], tapers, nfft, fs)

    # Test Case 2: Silumated data
    # Generate a sinusoidal signal at 8Hz for 10 seconds
    Fs = 1000
    duration = 100
    t = np.linspace(0, duration, duration * Fs)
    data_simulated = np.sin(2 * np.pi * 8 * t)

    # Generate tapers using scipy
    N = len(data_simulated)
    NW = 3
    K = int(np.ceil(2 * NW)) - 1
    tapers, _ = dpss(N, NW=NW, Kmax=K, sym=False, norm=2, return_ratios=True)

    # Ensure tapers have the correct shape (N, K)
    tapers = tapers.T

    # Call the function
    J_simulated = px.mtfftc(data_simulated, tapers, nfft, Fs)

    # Check the shape of the output
    assert J_simulated.shape == (nfft, K), "Shape of J does not match expected values."

    f, findx = px.getfgrid(Fs, nfft, [0, 20])
    J_simulated = J_simulated[findx, :]
    S = np.real(np.mean(np.conj(J_simulated) * J_simulated, 1))

    # Check the peak frequency is 8 Hz
    peak_freq = f[np.argmax(S)]
    assert np.isclose(peak_freq, 8, atol=0.2), (
        f"Peak frequency {peak_freq} does not match expected value of 8 Hz."
    )


def test_mtspectrumc():
    Fs = 1000
    duration = 100
    t = np.linspace(0, duration, duration * Fs)
    data_simulated = np.sin(2 * np.pi * 8 * t)

    NW = 3
    K = int(np.ceil(2 * NW)) - 1

    df = px.mtspectrumc(data_simulated, Fs, [0, 20], [NW, K])

    # Check the peak frequency is 8 Hz
    peak_freq = df.index[np.argmax(df.values)]
    assert np.isclose(peak_freq, 8, atol=0.2), (
        f"Peak frequency {peak_freq} does not match expected value of 8 Hz."
    )


def test_point_spectra():
    """
    Test the point_spectra function.
    """
    # Parameters
    Fs = 1250  # Sampling frequency
    freq_range = [1, 20]  # Frequency range of interest
    tapers0 = [3, 5]  # Tapers configuration
    pad = 0  # No padding

    np.random.seed(42)
    times = np.sort(np.random.uniform(0, 10, 100))

    # Run the function
    spectra, freqs = px.point_spectra(
        times=times,
        Fs=Fs,
        freq_range=freq_range,
        tapers0=tapers0,
        pad=pad,
    )

    # Assertions
    assert isinstance(spectra, np.ndarray), "Spectra should be a numpy array."
    assert isinstance(freqs, np.ndarray), "Frequencies should be a numpy array."
    assert len(spectra) == len(freqs), (
        "Spectra and frequencies should have the same length."
    )
    assert len(freqs) > 0, "There should be frequencies in the output."
    assert freq_range[0] <= freqs[0], (
        "Frequencies should start at or above freq_range[0]."
    )
    assert freq_range[1] >= freqs[-1], (
        "Frequencies should end at or below freq_range[1]."
    )
    assert np.all(spectra >= 0), "Power spectrum values should be non-negative."

    # Test Case 2: validate frequency peak
    # Parameters
    Fs = 1000  # Sampling frequency (Hz)
    duration = 10  # Duration of simulation (seconds)
    burst_frequency = 8  # Burst frequency (Hz)
    burst_rate = 50  # Rate of spikes within each burst (Hz)

    # Simulate spike times
    times = simulate_bursting_cell(burst_frequency, duration, burst_rate, Fs)

    spectra, freqs = px.point_spectra(
        times=times,
        Fs=Fs,
        freq_range=freq_range,
        tapers0=tapers0,
        pad=pad,
    )
    # peak is at 8 Hz
    peak_freq = freqs[np.argmax(spectra)]

    assert np.isclose(peak_freq, 8, atol=0.2), (
        f"Peak frequency {peak_freq} does not match expected value of 8 Hz."
    )


def test_mtcsdpt():
    # Generate mock spike times for two signals
    data1 = np.sort(np.random.uniform(0, 10, 100))
    data2 = np.sort(np.random.uniform(0, 10, 100))
    Fs = 1000
    fpass = [0, 50]
    NW = 2.5
    n_tapers = 4

    # Call the function
    csd_df = px.mtcsdpt(data1, data2, Fs, fpass, NW, n_tapers)

    # Check the shape of the output
    assert isinstance(csd_df, pd.DataFrame), "Output is not a DataFrame."
    assert "CSD" in csd_df.columns, "CSD column is missing in the output DataFrame."
    assert len(csd_df) > 0, "Output DataFrame is empty."

    # Check if the frequency range is correct
    assert csd_df.index.min() >= fpass[0], "Minimum frequency is less than fpass[0]."
    assert csd_df.index.max() <= fpass[1], "Maximum frequency is greater than fpass[1]."

    # Check if the cross-spectral density values are real numbers
    assert np.all(np.isreal(csd_df["CSD"])), "CSD values are not real numbers."


def test_mtcoherencept():
    # Generate mock spike times for two signals
    data1 = np.sort(np.random.uniform(0, 10, 100))
    data2 = np.sort(np.random.uniform(0, 10, 100))
    Fs = 1000
    fpass = [0, 50]
    NW = 2.5
    n_tapers = 4

    # Call the function
    coherence_df = px.mtcoherencept(data1, data2, Fs, fpass, NW, n_tapers)

    # Check the shape of the output
    assert isinstance(coherence_df, pd.DataFrame), "Output is not a DataFrame."
    assert "Coherence" in coherence_df.columns, (
        "Coherence column is missing in the output DataFrame."
    )
    assert len(coherence_df) > 0, "Output DataFrame is empty."

    # Check if the frequency range is correct
    assert coherence_df.index.min() >= fpass[0], (
        "Minimum frequency is less than fpass[0]."
    )
    assert coherence_df.index.max() <= fpass[1], (
        "Maximum frequency is greater than fpass[1]."
    )

    # Check if the coherence values are between 0 and 1
    assert np.all(
        (coherence_df["Coherence"] >= 0) & (coherence_df["Coherence"] <= 1)
    ), "Coherence values are not between 0 and 1."

    # Test Case 2: validate frequency peak
    # Parameters
    Fs = 1000  # Sampling frequency (Hz)
    duration = 10  # Duration of simulation (seconds)
    burst_frequency = 8  # Burst frequency (Hz)
    burst_rate = 50  # Rate of spikes within each burst (Hz)

    # Simulate spike times
    times_1 = simulate_bursting_cell(burst_frequency, duration, burst_rate, Fs)
    times_2 = simulate_bursting_cell(burst_frequency, duration, burst_rate, Fs)

    coherence_df = px.mtcoherencept(times_1, times_2, Fs, fpass, NW, n_tapers)
    # peak is at 8 Hz
    peak_freq = coherence_df.index[np.argmax(coherence_df["Coherence"])]
    assert np.isclose(peak_freq, 8, atol=0.2), (
        f"Peak frequency {peak_freq} does not match expected value of 8 Hz."
    )
