import warnings
from typing import List, Optional, Tuple, Union

import bottleneck as bn
import numpy as np
import pandas as pd
from nelpy import EpochArray
from nelpy.core._eventarray import SpikeTrainArray
from numba import jit, prange
from scipy import stats
from scipy.linalg import toeplitz
from scipy.ndimage import gaussian_filter1d

from neuro_py.process.intervals import in_intervals, split_epoch_by_width


@jit(nopython=True)
def crossCorr(
    t1: np.ndarray,
    t2: np.ndarray,
    binsize: float,
    nbins: int,
) -> np.ndarray:
    """
    Perform the discrete cross-correlogram of two time series.

    This function calculates the firing rate of the series 't2' relative to the timings of 't1'.
    The units should be in seconds for all arguments.

    Parameters
    ----------
    t1 : np.ndarray
        First time series.
    t2 : np.ndarray
        Second time series.
    binsize : float
        Size of the bin in seconds.
    nbins : int
        Number of bins.

    Returns
    -------
    np.ndarray
        Cross-correlogram of the two time series.

    Notes
    -----
    This implementation is based on the work of Guillaume Viejo.
    References:
    - https://github.com/PeyracheLab/StarterPack/blob/master/python/main6_autocorr.py
    - https://github.com/pynapple-org/pynapple/blob/main/pynapple/process/correlograms.py
    """
    # Calculate the length of the input time series
    nt1 = len(t1)
    nt2 = len(t2)

    # Ensure that 'nbins' is an odd number
    if np.floor(nbins / 2) * 2 == nbins:
        nbins = nbins + 1

    # Calculate the half-width of the cross-correlogram window
    w = (nbins / 2) * binsize
    C = np.zeros(nbins)
    i2 = 1

    # Iterate through the first time series
    for i1 in range(nt1):
        lbound = t1[i1] - w

        # Find the index of the first element in 't2' that is within 'lbound'
        while i2 < nt2 and t2[i2] < lbound:
            i2 = i2 + 1

        # Find the index of the last element in 't2' that is within 'lbound'
        while i2 > 1 and t2[i2 - 1] > lbound:
            i2 = i2 - 1

        rbound = lbound
        last_index = i2

        # Calculate the cross-correlogram values for each bin
        for j in range(nbins):
            k = 0
            rbound = rbound + binsize

            # Count the number of elements in 't2' that fall within the bin
            while last_index < nt2 and t2[last_index] < rbound:
                last_index = last_index + 1
                k = k + 1

            C[j] += k

    # Normalize the cross-correlogram by dividing by the total observation time and bin size
    C = C / (nt1 * binsize)

    return C


def compute_psth(
    spikes: np.ndarray,
    event: np.ndarray,
    bin_width: float = 0.002,
    n_bins: int = 100,
    window: list = None,
) -> pd.DataFrame:
    """
    Compute the Peri-Stimulus Time Histogram (PSTH) from spike trains.

    This function calculates the PSTH for a given set of spike times aligned to specific events.
    The PSTH provides a histogram of spike counts in response to the events over a defined time window.

    Parameters
    ----------
    spikes : np.ndarray
        An array of spike times for multiple trials, with each trial in a separate row.
    event : np.ndarray
        An array of event times to which the spikes are aligned.
    bin_width : float, optional
        Width of each time bin in seconds (default is 0.002 seconds).
    n_bins : int, optional
        Number of bins to create for the histogram (default is 100).
    window : list, optional
        Time window around each event to consider for the PSTH. If None, a symmetric window is created based on `n_bins` and `bin_width`.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the PSTH, indexed by time bins and columns representing each trial's PSTH.

    Notes
    -----
    If the specified window is not symmetric around 0, it is adjusted to be symmetric.

    Examples
    -------
    >>> spikes = np.array([[0.1, 0.15, 0.2], [0.1, 0.12, 0.13]])
    >>> event = np.array([0.1, 0.3])
    >>> psth = compute_psth(spikes, event)
    """
    if window is not None:
        window_original = None
        # check if window is symmetric around 0, if not make it so
        if ((window[1] - window[0]) / 2 != window[1]) | (
            (window[1] - window[0]) / -2 != window[0]
        ):
            window_original = np.array(window)
            window = [-np.max(np.abs(window)), np.max(np.abs(window))]

        times = np.arange(window[0], window[1] + bin_width / 2, bin_width)
        n_bins = len(times) - 1
    else:
        times = np.linspace(
            -(n_bins * bin_width) / 2, (n_bins * bin_width) / 2, n_bins + 1
        )

    ccg = pd.DataFrame(index=times, columns=np.arange(len(spikes)))
    # Now we can iterate over spikes
    for i, s in enumerate(spikes):
        ccg[i] = crossCorr(event, s, bin_width, n_bins)

    # if window was not symmetric, remove the extra bins
    if window is not None:
        if window_original is not None:
            ccg = ccg.loc[window_original[0] : window_original[1], :]
    return ccg


def joint_peth(
    peth_1: np.ndarray, peth_2: np.ndarray, smooth_std: float = 2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    joint_peth - produce a joint histogram for the co-occurrence of two sets of signals around events.

    This analysis tests for interactions. For example, the interaction of
    ripples and spindles around the occurrence of delta waves. It is a good way
    to control whether the relationships between two variables is entirely explained
    by a third variable (the events serving as basis for the PETHs).

    Parameters
    ----------
    peth_1 : np.ndarray
        The first peri-event time histogram (PETH) signal, shape (n_events, n_time).
    peth_2 : np.ndarray
        The second peri-event time histogram (PETH) signal, shape (n_events, n_time).
    smooth_std : float, optional
        The standard deviation of the Gaussian smoothing kernel (default is 2).

    Returns
    -------
    joint : np.ndarray
        The joint histogram of the two PETH signals (n_time, n_time).
    expected : np.ndarray
        The expected histogram of the two PETH signals (n_time, n_time).
    difference : np.ndarray
        The difference between the joint and expected histograms of the two PETH signals (n_time, n_time).

    Examples
    -------
    >>> from neuro_py.process.peri_event import joint_peth, peth_matrix, joint_peth
    >>> from neuro_py.spikes.spike_tools import get_spindices
    >>> from neuro_py.io import loading

    >>> # load ripples, delta waves, and PFC pyramidal cell spikes from basepath

    >>> basepath = r"Z:\\Data\\HMC1\\day8"

    >>> ripples = loading.load_ripples_events(basepath, return_epoch_array=True)
    >>> delta_waves = loading.load_events(basepath, epoch_name="deltaWaves")
    >>> st,cm = loading.load_spikes(basepath,brainRegion="PFC",putativeCellType="Pyr")

    >>> # flatten spikes (nelpy has .flatten(), but get_spindices is much faster)
    >>> spikes = get_spindices(st.data)

    >>> # create peri-event time histograms (PETHs) for the three signals
    >>> window=[-1,1]
    >>> labels = ["spikes", "ripple", "delta"]
    >>> peth_1,ts = peth_matrix(spikes.spike_times.values, delta_waves.starts, bin_width=0.02, n_bins=101)
    >>> peth_2,ts = peth_matrix(ripples.starts, delta_waves.starts, bin_width=0.02, n_bins=101)

    >>> # calculate the joint, expected, and difference histograms
    >>> joint, expected, difference = joint_peth(peth_1.T, peth_2.T, smooth_std=2)

    Notes
    -----
    Note: sometimes the difference between "joint" and "expected" may be dominated due to
    brain state effects (e.g. if both ripples are spindles are more common around delta
    waves taking place in early SWS and have decreased rates around delta waves in late
    SWS, then all the values of "joint" would be larger than the value of "expected".
    In such a case, to investigate the timing effects in particular and ignore such
    global changes (correlations across the rows of "PETH1" and "PETH2"), consider
    normalizing the rows of the PETHs before calling joint_peth.

    See Sirota et al. (2003)

    Adapted from JointPETH.m, Copyright (C) 2018-2022 by Ralitsa Todorova
    """
    from scipy.ndimage import gaussian_filter

    # make inputs np.ndarrays
    peth_1 = np.array(peth_1)
    peth_2 = np.array(peth_2)

    # calculate the joint histogram
    joint = peth_1.T @ peth_2

    # smooth the 2d joint histogram
    joint = gaussian_filter(joint, smooth_std)

    # calculate the expected histogram
    expected = np.tile(np.nanmean(peth_1, axis=0), [peth_1.shape[0], 1]).T @ np.tile(
        np.nanmean(peth_2, axis=0), [peth_2.shape[0], 1]
    )

    # smooth the 2d expected histogram
    expected = gaussian_filter(expected, smooth_std)

    # normalize the joint and expected histograms
    joint = joint / peth_1.shape[0]
    expected = expected / peth_1.shape[0]

    # square root the joint and expected histograms so result is Hz
    joint = np.sqrt(joint)
    expected = np.sqrt(expected)

    # calculate the difference between the joint and expected histograms
    difference = joint - expected

    return joint, expected, difference


def deconvolve_peth(
    signal: np.ndarray, events: np.ndarray, bin_width: float = 0.002, n_bins: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform deconvolution of a peri-event time histogram (PETH) signal.

    This function calculates the deconvolved signal based on the input signal and events.

    Parameters
    ----------
    signal : np.ndarray
        An array representing the discrete events.
    events : np.ndarray
        An array representing the discrete events.
    bin_width : float, optional
        The width of a time bin in seconds (default is 0.002 seconds).
    n_bins : int, optional
        The number of bins to use in the PETH (default is 100 bins).

    Returns
    -------
    deconvolved : np.ndarray
        An array representing the deconvolved signal.
    times : np.ndarray
        An array representing the time points corresponding to the bins.

    Notes
    -----
    Based on DeconvolvePETH.m from https://github.com/ayalab1/neurocode/blob/master/spikes/DeconvolvePETH.m
    """

    # calculate time lags for peth
    times = np.linspace(-(n_bins * bin_width) / 2, (n_bins * bin_width) / 2, n_bins + 1)

    # Calculate the autocorrelogram of the signal and the PETH of the events and the signal
    autocorrelogram = crossCorr(signal, signal, bin_width, n_bins * 2)
    raw_peth = crossCorr(events, signal, bin_width, n_bins * 2)

    # If raw_peth all zeros, return zeros
    if not raw_peth.any():
        return np.zeros(len(times)), times

    # Subtract the mean value from the raw_peth
    const = np.mean(raw_peth)
    raw_peth = raw_peth - const

    # Calculate the Toeplitz matrix using the autocorrelogram and
    #   the cross-correlation of the autocorrelogram
    T0 = toeplitz(
        autocorrelogram,
        np.hstack([autocorrelogram[0], np.zeros(len(autocorrelogram) - 1)]),
    )
    T = T0[n_bins:, : n_bins + 1]

    # Calculate the deconvolved signal by solving a linear equation
    deconvolved = np.linalg.solve(
        T, raw_peth[int(n_bins / 2) : int(n_bins / 2 * 3 + 1)].T + const / len(events)
    )

    return deconvolved, times


@jit(nopython=True)
def get_raster_points(
    data: np.ndarray,
    time_ref: np.ndarray,
    bin_width: float = 0.002,
    n_bins: int = 100,
    window: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate points for a raster plot centered around each reference time in the `time_ref` array.

    Parameters
    ----------
    data : ndarray
        A 1D array of time values.
    time_ref : ndarray
        A 1D array of reference times.
    bin_width : float, optional
        The width of each bin in the raster plot, in seconds. Default is 0.002 seconds.
    n_bins : int, optional
        The number of bins in the raster plot. Default is 100.
    window : tuple, optional
        A tuple containing the start and end times of the window to be plotted around each reference time.
        If not provided, the window will be centered around each reference time and have a width of `n_bins * bin_width` seconds.

    Returns
    -------
    x : ndarray
        A 1D array of x values representing the time offsets of each data point relative to the corresponding reference time.
    y : ndarray
        A 1D array of y values representing the reference times.
    times : ndarray
        A 1D array of time values corresponding to the bins in the raster plot.
    """
    if window is not None:
        times = np.arange(window[0], window[1] + bin_width / 2, bin_width)
    else:
        times = np.linspace(
            -(n_bins * bin_width) / 2, (n_bins * bin_width) / 2, n_bins + 1
        )

    x = np.empty(0)
    y = np.empty(0)
    for i, r in enumerate(time_ref):
        idx = (data > r + times.min()) & (data < r + times.max())
        cur_data = data[idx]
        x = np.concatenate((x, cur_data - r))
        y = np.concatenate((y, np.ones_like(cur_data) * i))

    return x, y, times


@jit(nopython=True, parallel=True)
def peth_matrix(
    data: np.ndarray,
    time_ref: np.ndarray,
    bin_width: float = 0.002,
    n_bins: int = 100,
    window: Union[list, None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a peri-event time histogram (PETH) matrix.

    Parameters
    ----------
    data : ndarray
        A 1D array of time values.
    time_ref : ndarray
        A 1D array of reference times.
    bin_width : float, optional
        The width of each bin in the PETH matrix, in seconds. Default is 0.002 seconds.
    n_bins : int, optional
        The number of bins in the PETH matrix. Default is 100.
    window : tuple, optional
        A tuple containing the start and end times of the window to be plotted around each reference time.
        If not provided, the window will be centered around each reference time and have a width of `n_bins * bin_width` seconds.

    Returns
    -------
    H : ndarray
        A 2D array representing the PETH matrix.
    t : ndarray
        A 1D array of time values corresponding to the bins in the PETH matrix.

    """
    if window is not None:
        times = np.arange(window[0], window[1] + bin_width / 2, bin_width)
        n_bins = len(times) - 1
    else:
        times = (
            np.arange(0, bin_width * n_bins, bin_width)
            - (bin_width * n_bins) / 2
            + bin_width / 2
        )

    H = np.zeros((len(times), len(time_ref)))

    for event_i in prange(len(time_ref)):
        H[:, event_i] = crossCorr([time_ref[event_i]], data, bin_width, n_bins)

    return H * bin_width, times


def event_triggered_average_irregular_sample(
    timestamps: np.ndarray,
    data: np.ndarray,
    time_ref: np.ndarray,
    bin_width: float = 0.002,
    n_bins: int = 100,
    window: Union[tuple, None] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute the average and standard deviation of data values within a window around
    each reference time, specifically for irregularly sampled data.

    Parameters
    ----------
    timestamps : np.ndarray
        A 1D array of times associated with data.
    data : np.ndarray
        A 1D array of data values.
    time_ref : np.ndarray
        A 1D array of reference times.
    bin_width : float, optional
        The width of each bin in the window, in seconds. Default is 0.002 seconds.
    n_bins : int, optional
        The number of bins in the window. Default is 100.
    window : Union[tuple, None], optional
        A tuple containing the start and end times of the window to be plotted around each reference time.
        If not provided, the window will be centered around each reference time and have a
        width of `n_bins * bin_width` seconds.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Two DataFrames: the first containing the average values, the second the
        standard deviation of data values within the window around each reference time.
    """

    if window is not None:
        times = np.arange(window[0], window[1] + bin_width, bin_width)
    else:
        times = np.linspace(
            -(n_bins * bin_width) / 2, (n_bins * bin_width) / 2, n_bins + 1
        )
    x = []
    y = []
    for i, r in enumerate(time_ref):
        idx = (timestamps > r + times.min()) & (timestamps < r + times.max())
        x.append((timestamps - r)[idx])
        y.append(data[idx])

    temp_df = pd.DataFrame()
    if len(x) == 0:
        return temp_df, temp_df
    temp_df["time"] = np.hstack(x)
    temp_df["data"] = np.hstack(y)
    temp_df = temp_df.sort_values(by="time", ascending=True)

    average_val = np.zeros(len(times) - 1)
    std_val = np.zeros(len(times) - 1)
    for i in range(len(times) - 1):
        average_val[i] = temp_df[
            temp_df.time.between(times[i], times[i + 1])
        ].data.mean()
        std_val[i] = temp_df[temp_df.time.between(times[i], times[i + 1])].data.std()

    avg = pd.DataFrame(index=times[:-1] + bin_width / 2)
    avg[0] = average_val

    std = pd.DataFrame(index=times[:-1] + bin_width / 2)
    std[0] = std_val

    return avg, std


def event_triggered_average(
    timestamps: np.ndarray,
    signal: np.ndarray,
    events: Union[np.ndarray, List[np.ndarray]],
    sampling_rate: Union[float, None] = None,
    window: List[float] = [-0.5, 0.5],
    return_average: bool = True,
    return_pandas: bool = False,
) -> Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray]:
    """
    Calculates the event-triggered averages of signals in a time window
    relative to the event times of corresponding events for multiple signals.

    Parameters
    ----------
    timestamps : np.ndarray
        A 1D array of timestamps corresponding to the signal samples.
    signal : np.ndarray
        A 2D array of shape (n_samples, n_signals) containing the signal values.
    events : Union[np.ndarray, List[np.ndarray]]
        One or more 1D arrays of event times.
    sampling_rate : Union[float, None], optional
        The sampling rate of the signal. If not provided, it will be calculated
        based on the timestamps.
    window : List[float], optional
        A list containing two elements: the start and stop times relative to an event
        for the time interval of signal averaging. Default is [-0.5, 0.5].
    return_average : bool, optional
        Whether to return the average of the event-triggered average. Defaults to True.
        If False, returns the full event-triggered average matrix (n_samples x n_signals x n_events).
    return_pandas : bool, optional
        If True, return the result as a Pandas DataFrame. Default is False.

    Returns
    -------
    Union[np.ndarray, pd.DataFrame]
        If `return_average` is True, returns the event-triggered averages of the signals
        (n_samples, n_signals) or a Pandas DataFrame if `return_pandas` is True.
        If `return_average` is False, returns the full event-triggered average matrix
        (n_samples, n_signals, n_events).
    np.ndarray
        An array of time lags corresponding to the event-triggered averages.

    Notes
    -----
    - The function filters out events that do not fit within the valid range of the signal
    considering the specified window size.
    - If the `sampling_rate` is not provided, it is calculated based on the timestamps.
    - The function handles both regular and irregular sampling of the signal.

    Examples
    --------
    >>> peth_avg, time_lags = event_triggered_average(
    ...    timestamps, signal, events, window=[-0.5, 0.5]
    ... )
    >>> # Get individual event responses
    >>> peth_matrix, time_lags = event_triggered_average(
    ...    timestamps, signal, events, window=[-0.5, 0.5], return_average=False
    ... )
    """
    # Basic input validation
    if len(window) != 2 or window[0] > window[1]:
        raise ValueError("'window' must be [start, stop] with start < stop")

    if len(signal.shape) == 1:
        signal = signal.reshape(-1, 1)

    if sampling_rate is None:
        sampling_rate = 1 / stats.mode(np.diff(timestamps), keepdims=True)[0][0]

    if isinstance(events, list):
        events = np.array(events)

    window_starttime, window_stoptime = window
    window_bins = int(np.ceil(((window_stoptime - window_starttime) * sampling_rate)))
    time_lags = np.linspace(window_starttime, window_stoptime, window_bins)

    # Filter events that fit within the signal range
    min_timestamp, max_timestamp = timestamps[0], timestamps[-1]
    valid_mask = (events + window_starttime >= min_timestamp) & (
        events + window_stoptime <= max_timestamp
    )

    if not np.any(valid_mask):
        warnings.warn("No events found within the valid signal range")
        empty_shape = (window_bins, signal.shape[1])
        if return_average:
            result = np.zeros(empty_shape)
            return (
                pd.DataFrame(result, index=time_lags) if return_pandas else result
            ), time_lags
        else:
            return np.full(empty_shape + (len(events),), np.nan), time_lags

    # Initialize result matrix: (window_bins, n_signals, n_events) - keep all events
    result_matrix = np.full((window_bins, signal.shape[1], len(events)), np.nan)

    # For regular sampling, use fast indexing approach similar to event_triggered_average_fast
    dt = np.median(np.diff(timestamps))
    is_regular_sampling = np.allclose(np.diff(timestamps), dt, rtol=1e-3)

    if is_regular_sampling:
        # Fast path: regular sampling - use direct indexing like event_triggered_average_fast
        # Match the exact indexing logic from event_triggered_average_fast
        start_time = timestamps[0]  # Cache start time for efficiency
        for i, event in enumerate(events):
            if not valid_mask[i]:  # Skip invalid events (already filled with NaN)
                continue

            # Convert event time to sample indices, accounting for timestamp start time
            event_sample = np.round((event - start_time) * sampling_rate)
            ts_idx = np.arange(
                event_sample - window_bins / 2,
                event_sample + window_bins / 2,
            ).astype(int)

            # Check bounds
            if np.min(ts_idx) >= 0 and np.max(ts_idx) < len(signal):
                result_matrix[:, :, i] = signal[ts_idx, :]
            # If bounds check fails, keep as NaN (already initialized)
    else:
        # Slow path: irregular sampling - use interpolation but vectorized
        target_times_template = np.linspace(
            window_starttime, window_stoptime, window_bins
        )

        for i, event in enumerate(events):
            if not valid_mask[i]:  # Skip invalid events (already filled with NaN)
                continue

            target_times = target_times_template + event

            # Find the range of timestamps that covers our target times
            start_search = np.searchsorted(
                timestamps, target_times[0] - dt, side="left"
            )
            stop_search = np.searchsorted(
                timestamps, target_times[-1] + dt, side="right"
            )

            if start_search >= stop_search:
                # Keep as NaN (already initialized)
                continue

            # Extract relevant data for this event
            event_timestamps = timestamps[start_search:stop_search]
            event_signal = signal[start_search:stop_search, :]

            # Vectorized interpolation for all channels at once
            if len(event_timestamps) > 1:
                for j in range(signal.shape[1]):
                    result_matrix[:, j, i] = np.interp(
                        target_times, event_timestamps, event_signal[:, j]
                    )
            # If interpolation fails, keep as NaN (already initialized)

    # Return results
    if return_average:
        result_avg = bn.nanmean(result_matrix, axis=2)
        if return_pandas:
            return pd.DataFrame(
                result_avg, index=time_lags, columns=np.arange(signal.shape[1])
            )
        return result_avg, time_lags
    else:
        return result_matrix, time_lags


def event_triggered_average_fast(
    signal: np.ndarray,
    events: np.ndarray,
    sampling_rate: int,
    window: Union[list, Tuple[float, float]] = [-0.5, 0.5],
    return_average: bool = True,
    return_pandas: bool = False,
) -> Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray]:
    """
    Calculate the event-triggered average of a signal.

    Parameters
    ----------
    signal : np.ndarray
        A 2D array of signal data with shape (channels, timepoints).

    events : np.ndarray
        A 1D array of event times.

    sampling_rate : int
        The sampling rate of the signal in Hz.

    window : Union[list, Tuple[float, float]], optional
        A list or tuple specifying the time window (in seconds) to average the signal
        around each event. Defaults to [-0.5, 0.5].

    return_average : bool, optional
        Whether to return the average of the event-triggered average. Defaults to True.
        If False, returns the full event-triggered average matrix (channels x timepoints x events).

    return_pandas : bool, optional
        If True, returns the average as a Pandas DataFrame. Defaults to False.

    Returns
    -------
    Union[np.ndarray, pd.DataFrame]
        If `return_average` is True, returns the event-triggered average of the signal
        (channels x timepoints) or a Pandas DataFrame if `return_pandas` is True.
        If `return_average` is False, returns the full event-triggered average matrix (channels x timebins x events).

    np.ndarray
        An array of time lags corresponding to the event-triggered averages.

    Notes
    -----
    - The function filters out events that do not fit within the valid range of the signal
    considering the specified window size.
    - Assumes the signal starts at time 0.
    """

    window_starttime, window_stoptime = window
    window_bins = int(np.ceil(((window_stoptime - window_starttime) * sampling_rate)))
    time_lags = np.linspace(window_starttime, window_stoptime, window_bins)

    # Create valid mask instead of filtering events
    valid_mask = (events * sampling_rate > len(time_lags) / 2 + 1) & (
        events * sampling_rate < signal.shape[1] - len(time_lags) / 2 + 1
    )

    # Initialize result matrix with all events, filled with NaN
    avg_signal = np.full(
        [signal.shape[0], len(time_lags), len(events)], np.nan, dtype=signal.dtype
    )

    # Process only valid events
    for i, event in enumerate(events):
        if not valid_mask[i]:  # Skip invalid events (already filled with NaN)
            continue

        ts_idx = np.arange(
            np.round(event * sampling_rate) - len(time_lags) / 2,
            np.round(event * sampling_rate) + len(time_lags) / 2,
        ).astype(int)
        avg_signal[:, :, i] = signal[:, ts_idx]

    if return_pandas and return_average:
        return pd.DataFrame(
            index=time_lags,
            columns=np.arange(signal.shape[0]),
            data=bn.nanmean(avg_signal, axis=2).T,
        )

    if return_average:
        return bn.nanmean(avg_signal, axis=2), time_lags
    else:
        return avg_signal, time_lags


def count_in_interval(
    st: np.ndarray,
    event_starts: np.ndarray,
    event_stops: np.ndarray,
    par_type: str = "counts",
) -> np.ndarray:
    """
    Count timestamps in specified intervals and return a matrix where each
    column represents the counts for each spike train over given event epochs.

    Parameters
    ----------
    st : np.ndarray
        A 1D array where each element is a spike train for a unit.

    event_starts : np.ndarray
        A 1D array containing the start times of events.

    event_stops : np.ndarray
        A 1D array containing the stop times of events.

    par_type : str, optional
        The type of count calculation to perform:
        - 'counts': returns raw counts of spikes in the intervals.
        - 'binary': returns a binary matrix indicating presence (1) or absence (0) of spikes.
        - 'firing_rate': returns the firing rate calculated as counts divided by the interval duration.
        Defaults to 'binary'.

    Returns
    -------
    np.ndarray
        A 2D array (n units x n epochs) where each column shows the counts (or binary values or firing rates)
        per unit for each epoch.
    """
    # convert to numpy array
    event_starts, event_stops = np.array(event_starts), np.array(event_stops)

    # initialize matrix
    unit_mat = np.zeros((len(st), (len(event_starts))))

    # loop over units and bin spikes into epochs
    for i, s in enumerate(st):
        idx1 = np.searchsorted(s, event_starts, "right")
        idx2 = np.searchsorted(s, event_stops, "left")
        unit_mat[i, :] = idx2 - idx1

    par_type_funcs = {
        "counts": lambda x: x,
        "binary": lambda x: (x > 0) * 1,
        "firing_rate": lambda x: x / (event_stops - event_starts),
    }
    calc_func = par_type_funcs[par_type]
    unit_mat = calc_func(unit_mat)

    return unit_mat


# the function name "get_participation" is depreciated, but kept for backwards compatibility
get_participation = count_in_interval


def get_rank_order(
    st: SpikeTrainArray,  # Assuming 'nelpy.array' is a custom type
    epochs: EpochArray,
    method: str = "peak_fr",  # 'first_spike' or 'peak_fr'
    ref: str = "cells",  # 'cells' or 'epoch'
    padding: float = 0.05,
    dt: float = 0.001,
    sigma: float = 0.01,
    min_units: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the rank order of spike trains within specified epochs.

    Parameters
    ----------
    st : np.ndarray or nelpy.array
        Spike train data. Can be a nelpy array containing spike times.

    epochs : nelpy.EpochArray
        An object containing the epochs (windows) in which to calculate the rank order.

    method : str, optional
        Method to calculate rank order. Choices are 'first_spike' or 'peak_fr'.
        Defaults to 'peak_fr'.

    ref : str, optional
        Reference frame for rank order. Choices are 'cells' or 'epoch'.
        Defaults to 'cells'.

    padding : float, optional
        Padding (in seconds) to apply to the epochs. Defaults to 0.05 seconds.

    dt : float, optional
        Bin width (in seconds) for finding relative time in the epoch reference.
        Defaults to 0.001 seconds.

    sigma : float, optional
        Smoothing sigma (in seconds) for the 'peak_fr' method. Defaults to 0.01 seconds.

    min_units : int, optional
        Minimum number of active units required to compute the rank order. Defaults to 5.

    Returns
    -------
    median_rank : np.ndarray
        The median rank order across all epochs, normalized between 0 and 1.

    rank_order : np.ndarray
        A 2D array of rank orders, where each column corresponds to an epoch,
        and each row corresponds to a cell, normalized between 0 and 1.

    Examples
    --------
    >>> st, _ = loading.load_spikes(basepath, putativeCellType='Pyr')
    >>> forward_replay = nel.EpochArray(np.array([starts, stops]).T)
    >>> median_rank, rank_order = get_rank_order(st, forward_replay)
    """
    # filter out specific warnings
    warnings.filterwarnings(
        "ignore", message="ignoring events outside of eventarray support"
    )
    warnings.filterwarnings("ignore", message="Mean of empty slice")

    if method not in ["first_spike", "peak_fr"]:
        raise Exception("method " + method + " not implemented")
    if ref not in ["cells", "epoch"]:
        raise Exception("ref " + ref + " not implemented")

    def get_min_ts(st_temp):
        min_ts = []
        for ts in st_temp.data:
            # nan if no spikes
            if len(ts) == 0:
                min_ts.append(np.nan)
            else:
                min_ts.append(np.nanmin(ts))
        return min_ts

    def rank_order_first_spike(st_epoch, epochs, dt, min_units, ref):
        # set up empty matrix for rank order
        rank_order = np.ones([st_epoch.data.shape[0], epochs.n_intervals]) * np.nan

        unit_id = np.arange(st_epoch.data.shape[0])
        st_epoch._abscissa.support = epochs

        # iter over every event
        for event_i, st_temp in enumerate(st_epoch):
            if ref == "cells":
                # get firing order
                idx = np.array(st_temp.get_event_firing_order()) - 1
                # reorder unit ids by order and remove non-active
                units = unit_id[idx][st_temp.n_events[idx] > 0]
                # how many are left?
                nUnits = len(units)

                if nUnits < min_units:
                    rank_order[:, event_i] = np.nan
                else:
                    # arange 1 to n units in order of units
                    rank_order[units, event_i] = np.arange(nUnits)
                    # normalize by n units
                    rank_order[units, event_i] = rank_order[units, event_i] / nUnits
            elif ref == "epoch":
                # find first spike time for each cell
                min_ts = get_min_ts(st_temp)
                # make time stamps for interpolation
                epoch_ts = np.arange(epochs[event_i].start, epochs[event_i].stop, dt)
                # make normalized range 0-1
                norm_range = np.linspace(0, 1, len(epoch_ts))
                # get spike order relative to normalized range
                if len(min_ts) < min_units:
                    rank_order[:, event_i] = np.nan
                else:
                    rank_order[:, event_i] = np.interp(min_ts, epoch_ts, norm_range)
        return rank_order

    def rank_order_fr(st, epochs, dt, sigma, min_units, ref):
        # set up empty matrix for rank order
        rank_order = np.zeros([st.data.shape[0], epochs.n_intervals]) * np.nan

        unit_id = np.arange(st.data.shape[0])

        edges = split_epoch_by_width(epochs.data, dt)

        z_t = count_in_interval(st.data, edges[:, 0], edges[:, 1], par_type="counts")
        _, interval_id = in_intervals(edges[:, 0], epochs.data, return_interval=True)

        # iter over epochs
        for event_i, epochs_temp in enumerate(epochs):
            # smooth spike train in order to estimate peak
            # z_t_temp.smooth(sigma=sigma, inplace=True)
            z_t_temp = z_t[:, interval_id == event_i]
            # smooth spike train in order to estimate peak
            z_t_temp = gaussian_filter1d(z_t_temp, sigma / dt, axis=1)
            if ref == "cells":
                # find loc of each peak and get sorted idx of active units
                idx = np.argsort(np.argmax(z_t_temp, axis=1))
                # reorder unit ids by order and remove non-active
                units = unit_id[idx][np.sum(z_t_temp[idx, :] > 0, axis=1) > 0]

                nUnits = len(units)

                if nUnits < min_units:
                    rank_order[:, event_i] = np.nan
                else:
                    # arange 1 to n units in order of units
                    rank_order[units, event_i] = np.arange(nUnits)
                    # normalize by n units
                    rank_order[units, event_i] = rank_order[units, event_i] / nUnits
            elif ref == "epoch":
                # iterate over each cell
                for cell_i, unit in enumerate(z_t_temp):
                    # if the cell is not active apply nan
                    if not np.any(unit > 0):
                        rank_order[cell_i, event_i] = np.nan
                    else:
                        # calculate normalized rank order (0-1)
                        rank_order[cell_i, event_i] = np.argmax(unit) / len(unit)
        return rank_order

    # expand epochs by padding amount
    epochs = epochs.expand(padding)

    # check if no active cells
    if st.n_active == 0:
        return np.tile(np.nan, st.data.shape), np.tile(
            np.nan, (st.data.shape[0], epochs.n_intervals)
        )

    # check if there are any spikes in the epoch
    st_epoch = count_in_interval(
        st.data, epochs.starts, epochs.stops, par_type="counts"
    )

    # if no spikes in epoch, break out
    if (st_epoch == 0).all():
        return np.tile(np.nan, st.data.shape), np.tile(
            np.nan, (st.data.shape[0], epochs.n_intervals)
        )

    # set up empty matrix for rank order
    if method == "peak_fr":
        rank_order = rank_order_fr(st, epochs, dt, sigma, min_units, ref)
    elif method == "first_spike":
        rank_order = rank_order_first_spike(st[epochs], epochs, dt, min_units, ref)
    else:
        raise Exception("method " + method + " not implemented")

    return np.nanmedian(rank_order, axis=1), rank_order


def count_events(
    events: np.ndarray, time_ref: np.ndarray, time_range: Tuple[float, float]
) -> np.ndarray:
    """
    Count the number of events that occur within a given time range after each reference event.

    Parameters
    ----------
    events : np.ndarray
        A 1D array of event times.
    time_ref : np.ndarray
        A 1D array of reference times.
    time_range : tuple of (float, float)
        A tuple containing the start and end times of the time range.

    Returns
    -------
    counts : np.ndarray
        A 1D array of event counts, one for each reference time (same length as time_ref).
    """
    # Initialize an array to store the event counts
    counts = np.zeros_like(time_ref)

    # Iterate over the reference times
    for i, r in enumerate(time_ref):
        # Check if any events occur within the time range
        idx = (events > r + time_range[0]) & (events < r + time_range[1])
        # Increment the event count if any events are found
        counts[i] = len(events[idx])

    return counts


@jit(nopython=True)
def relative_times(
    t: np.ndarray, intervals: np.ndarray, values: np.ndarray = np.array([0, 1])
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate relative times and interval IDs for a set of time points.

    Parameters
    ----------
    t : np.ndarray
        An array of time points.
    intervals : np.ndarray
        An array of time intervals, represented as pairs of start and end times.
    values : np.ndarray, optional
        An array of values to assign to interval bounds. The default is [0,1].

    Returns
    -------
    rt : np.ndarray
        An array of relative times, one for each time point (same len as t).
    intervalID : np.ndarray
        An array of interval IDs, one for each time point (same len as t).

    Examples
    --------
    >>> t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> intervals = np.array([[1, 3], [4, 6], [7, 9]])
    >>> relative_times(t, intervals)
        (array([nan, 0. , 0.5, 1. , 0. , 0.5, 1. , 0. , 0.5, 1. ]),
        array([nan,  0.,  0.,  0.,  1.,  1.,  1.,  2.,  2.,  2.]))

    >>> t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> intervals = np.array([[1, 3], [4, 6], [7, 9]])
    >>> values = np.array([0, 2*np.pi])
    >>> relative_times(t, intervals, values)
        (array([       nan, 0.        , 3.14159265, 6.28318531, 0.        ,
                3.14159265, 6.28318531, 0.        , 3.14159265, 6.28318531]),
        array([nan,  0.,  0.,  0.,  1.,  1.,  1.,  2.,  2.,  2.]))

    Notes
    -----
    Intervals are defined as pairs of start and end times. The relative time is the time
    within the interval, normalized to the interval duration. The interval ID is the index
    of the interval in the intervals array. The values array can be used to assign a value
    to each interval.

    By Ryan H, based on RelativeTimes.m by Ralitsa Todorova

    """

    rt = np.zeros(len(t), dtype=np.float64) * np.nan
    intervalID = np.zeros(len(t), dtype=np.float64) * np.nan

    start_times = intervals[:, 0]
    end_times = intervals[:, 1]
    values_diff = values[1] - values[0]
    intervals_diff = end_times - start_times
    intervals_scale = values_diff / intervals_diff

    for i in range(len(t)):
        idx = np.searchsorted(start_times, t[i])
        if idx > 0 and t[i] <= end_times[idx - 1]:
            interval_i = idx - 1
        elif idx < len(start_times) and t[i] == start_times[idx]:
            interval_i = idx
        else:
            continue

        scale = intervals_scale[interval_i]
        rt[i] = ((t[i] - start_times[interval_i]) * scale) + values[0]
        intervalID[i] = interval_i

    return rt, intervalID


def nearest_event_delay(
    ts_1: np.ndarray, ts_2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return for each timestamp in ts_1 the nearest timestamp in ts_2 and the delay between the two.

    Parameters
    ----------
    ts_1 : np.ndarray
        1D array of timestamps.
    ts_2 : np.ndarray
        1D array of timestamps (must be monotonically increasing).

    Returns
    -------
    nearest_ts : np.ndarray
        Nearest timestamps in ts_2 for each timestamp in ts_1.
    delays : np.ndarray
        Delays between ts_1 and nearest_ts.
    nearest_index : np.ndarray
        Index of nearest_ts in ts_2.

    Raises
    ------
    ValueError
        If ts_1 or ts_2 are empty or not monotonically increasing.

    Notes
    -----
    Both ts_1 and ts_2 must be monotonically increasing arrays of timestamps.
    """
    ts_1, ts_2 = np.array(ts_1), np.array(ts_2)

    if not np.all(np.diff(ts_2) > 0):
        raise ValueError("ts_2 must be monotonically increasing")

    if not np.all(np.diff(ts_1) > 0):
        raise ValueError("ts_1 must be monotonically increasing")
    # check if empty
    if len(ts_1) == 0:
        raise ValueError("ts_1 is empty")
    if len(ts_2) == 0:
        raise ValueError("ts_2 is empty")

    # Use searchsorted to find the indices where elements of ts_1 should be inserted
    nearest_indices = np.searchsorted(ts_2, ts_1, side="left")

    # Calculate indices for the elements before and after the insertion points
    before = np.maximum(nearest_indices - 1, 0)
    after = np.minimum(nearest_indices, len(ts_2) - 1)

    # Determine the nearest timestamp for each element in ts_1
    nearest_ts = np.where(
        np.abs(ts_1 - ts_2[before]) < np.abs(ts_1 - ts_2[after]),
        ts_2[before],
        ts_2[after],
    )

    # Calculate delays between ts_1 and nearest_ts
    delays = ts_1 - nearest_ts

    # Find the nearest_index using the absolute difference
    absolute_diff_before = np.abs(ts_1 - ts_2[before])
    absolute_diff_after = np.abs(ts_1 - ts_2[after])
    nearest_index = np.where(absolute_diff_before < absolute_diff_after, before, after)

    return nearest_ts, delays, nearest_index


def event_spiking_threshold(
    spikes: SpikeTrainArray,
    events: np.ndarray,
    window: list = [-0.5, 0.5],
    event_size: float = 0.1,
    spiking_thres: float = 0,
    binsize: float = 0.01,
    sigma: float = 0.02,
    min_units: int = 6,
    show_fig: bool = False,
) -> np.ndarray:
    """
    event_spiking_threshold: filter events based on spiking threshold

    Parameters
    ----------
    spikes : nel.SpikeTrainArray
        Spike train array of neurons.
    events : np.ndarray
        Event times in seconds.
    window : list of float, optional
        Time window (in seconds) to compute event-triggered average, by default [-0.5, 0.5].
    event_size : float, optional
        Time window (in seconds) around event to measure firing response, by default 0.1.
    spiking_thres : float, optional
        Spiking threshold in z-score units, by default 0.
    binsize : float, optional
        Bin size (in seconds) for time-binning the spike trains, by default 0.01.
    sigma : float, optional
        Standard deviation (in seconds) for Gaussian smoothing of spike counts, by default 0.02.
    min_units : int, optional
        Minimum number of units required to compute event-triggered average, by default 6.
    show_fig : bool, optional
        If True, plots the figure of event-triggered spiking activity, by default False.

    Returns
    -------
    np.ndarray
        Boolean array indicating valid events that meet the spiking threshold.

    Examples
    -------
    >>> basepath = r"U:\\data\\hpc_ctx_project\\HP04\\day_32_20240430"
    >>> ripples = loading.load_ripples_events(basepath, return_epoch_array=False)
    >>> st, cell_metrics = loading.load_spikes(
            basepath,
            brainRegion="CA1",
            support=nel.EpochArray([0, loading.load_epoch(basepath).iloc[-1].stopTime])
        )
    >>> idx = event_spiking_threshold(st, ripples.peaks.values, show_fig=True)
    >>> print(f"Number of valid ripples: {idx.sum()} out of {len(ripples)}")
    Number of valid ripples: 9244 out of 12655

    """

    # check if there are enough units to compute a confident event triggered average
    if spikes.n_active < min_units:
        return np.ones(len(events), dtype=bool)

    # bin spikes
    bst = spikes.bin(ds=binsize).smooth(sigma=sigma)
    # sum over all neurons and zscore
    bst = bst.data.sum(axis=0)
    bst = (bst - bst.mean()) / bst.std()
    # get event triggered average
    avg_signal, time_lags = event_triggered_average_fast(
        bst[np.newaxis, :],
        events,
        sampling_rate=int(1 / binsize),
        window=window,
        return_average=False,
    )
    # get the event response within the event size
    idx = (time_lags >= -event_size) & (time_lags <= event_size)
    event_response = avg_signal[0, idx, :].mean(axis=0)

    # get events that are above threshold
    valid_events = event_response > spiking_thres

    if show_fig:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sorted_idx = np.argsort(event_response)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        ax[0].imshow(
            avg_signal[0, :, sorted_idx],
            aspect="auto",
            extent=[time_lags[0], time_lags[-1], 0, len(event_response)],
            vmin=-2,
            vmax=2,
            origin="lower",
            interpolation="nearest",
        )
        ax[0].axhline(
            np.where(event_response[sorted_idx] > spiking_thres)[0][0],
            color="r",
            linestyle="--",
        )
        ax[1].plot(event_response[sorted_idx], np.arange(len(event_response)))
        ax[1].axvline(spiking_thres, color="r", linestyle="--")
        ax[0].set_xlabel("Time from event (s)")
        ax[0].set_ylabel("Event index")
        ax[1].set_xlabel("Average response")
        ax[1].set_ylabel("Event index")
        sns.despine()

    return valid_events
