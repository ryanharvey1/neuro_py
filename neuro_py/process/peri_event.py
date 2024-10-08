import warnings
from typing import Union

import numpy as np
import pandas as pd
from nelpy.core._eventarray import SpikeTrainArray
from numba import jit, prange
from scipy import stats
from scipy.linalg import toeplitz


@jit(nopython=True)
def crossCorr(
    t1: np.ndarray,
    t2: np.ndarray,
    binsize: float,
    nbins: int,
) -> np.ndarray:
    """
    Performs the discrete cross-correlogram of two time series.
    The units should be in s for all arguments.
    Return the firing rate of the series t2 relative to the timings of t1.

    crossCorr functions from Guillaume Viejo of Peyrache Lab
    https://github.com/PeyracheLab/StarterPack/blob/master/python/main6_autocorr.py
    https://github.com/pynapple-org/pynapple/blob/main/pynapple/process/correlograms.py

    Parameters
    ----------
    t1 : array
        First time series.
    t2 : array
        Second time series.
    binsize : float
        Size of the bin in seconds.
    nbins : int
        Number of bins.

    Returns
    -------
    C : array
        Cross-correlogram of the two time series.

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
):
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


def joint_peth(peth_1: np.ndarray, peth_2: np.ndarray, smooth_std: float = 2):
    """
    joint_peth - produce a joint histogram for the co-occurrence of two sets of signals around events.
    PETH1 and PETH2 should be in the format of PETH, see example usage below.
    This analysis tests for interactions. For example, the interaction of
    ripples and spindles around the occurrence of delta waves. It is a good way
    to control whether the relationships between two variables is entirely explained
    by a third variable (the events serving as basis for the PETHs). See Sirota et al. (2003)

    % Note: sometimes the difference between "joint" and "expected" may be dominated due to
    % brain state effects (e.g. if both ripples are spindles are more common around delta
    % waves taking place in early SWS and have decreased rates around delta waves in late
    % SWS, then all the values of "joint" would be larger than the value of "expected".
    % In such a case, to investigate the timing effects in particular and ignore such
    % global changes (correlations across the rows of "PETH1" and "PETH2"), consider
    % normalizing the rows of the PETHs before calling joint_peth.

    Adapted from JointPETH.m, Copyright (C) 2018-2022 by Ralitsa Todorova

    Parameters
    ----------
    peth_1 : array
        The first peri-event time histogram (PETH) signal, (n events x time).
    peth_2 : array
        The second peri-event time histogram (PETH) signal, (n events x time).
    smooth_std : float, optional
        The standard deviation of the Gaussian smoothing kernel (default value is 2).

    Returns
    -------
    joint : array
        The joint histogram of the two PETH signals (time x time).
    expected : array
        The expected histogram of the two PETH signals (time x time).
    difference : array
        The difference between the joint and expected histograms of the two PETH signals (time x time).

    Example
    -------
    from neuro_py.process.peri_event import joint_peth, peth_matrix, joint_peth
    from neuro_py.spikes.spike_tools import get_spindices
    from neuro_py.io import loading

    # load ripples, delta waves, and PFC pyramidal cell spikes from basepath

    basepath = r"Z:\Data\HMC1\day8"

    ripples = loading.load_ripples_events(basepath, return_epoch_array=True)
    delta_waves = loading.load_events(basepath, epoch_name="deltaWaves")
    st,cm = loading.load_spikes(basepath,brainRegion="PFC",putativeCellType="Pyr")

    # flatten spikes (nelpy has .flatten(), but get_spindices is much faster)
    spikes = get_spindices(st.data)

    # create peri-event time histograms (PETHs) for the three signals
    window=[-1,1]
    labels = ["spikes", "ripple", "delta"]
    peth_1,ts = peth_matrix(spikes.spike_times.values, delta_waves.starts, bin_width=0.02, n_bins=101)
    peth_2,ts = peth_matrix(ripples.starts, delta_waves.starts, bin_width=0.02, n_bins=101)

    # calculate the joint, expected, and difference histograms
    joint, expected, difference = joint_peth(peth_1.T, peth_2.T, smooth_std=2)


    also see plot_joint_peth in figure_helpers.py
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


def deconvolve_peth(signal, events, bin_width=0.002, n_bins=100):
    """
    This function performs deconvolution of a peri-event time histogram (PETH) signal.

    Parameters:
    signal (array): An array representing the discrete events.
    events (array): An array representing the discrete events.
    bin_width (float, optional): The width of a time bin in seconds (default value is 0.002 seconds).
    n_bins (int, optional): The number of bins to use in the PETH (default value is 100 bins).

    Returns:
    deconvolved (array): An array representing the deconvolved signal.
    times (array): An array representing the time points corresponding to the bins.

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
def get_raster_points(data, time_ref, bin_width=0.002, n_bins=100, window=None):
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
):
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
    timestamps, data, time_ref, bin_width=0.002, n_bins=100, window=None
):
    """
    Compute the average and standard deviation of data values within a window around
    each reference time.

    Specifically for irregularly sampled data

    Parameters
    ----------
    timestamps : ndarray
        A 1D array of times associated with data.
    data : ndarray
        A 1D array of data values.
    time_ref : ndarray
        A 1D array of reference times.
    bin_width : float, optional
        The width of each bin in the window, in seconds. Default is 0.002 seconds.
    n_bins : int, optional
        The number of bins in the window. Default is 100.
    window : tuple, optional
        A tuple containing the start and end times of the window to be plotted around each reference time.
        If not provided, the window will be centered around each reference time and have a
        width of `n_bins * bin_width` seconds.

    Returns
    -------
    pd.DataFrame, pd.DataFrame
        two dataframes, the first containing the average values, the second the
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
    events: np.ndarray,
    sampling_rate=None,
    window=[-0.5, 0.5],
    return_pandas: bool = False,
) -> np.ndarray:
    """
    Calculates the spike-triggered averages of signals in a time window
    relative to the event times of a corresponding events for multiple
    signals each. The function receives n signals and either one or
    n events. In case it is one event this one is muliplied n-fold
    and used for each of the n signals.

    adapted from elephant.sta.spike_triggered_average to be used with ndarray

    Parameters
    ----------
    timestamps : ndarray (n samples)

    signal : ndarray (n samples x n signals)

    events : one numpy ndarray or a list of n of either of these.

    window : tuple of 2.
        'window' is the start time and the stop time, relative to a event, of
        the time interval for signal averaging.
        If the window size is not a multiple of the sampling interval of the
        signal the window will be extended to the next multiple.

    Returns
    -------
    result_sta : ndarray
        'result_sta' contains the event-triggered averages of each of the
        signals with respect to the event in the corresponding
        events. The length of 'result_sta' is calculated as the number
        of bins from the given start and stop time of the averaging interval
        and the sampling rate of the signal. If for an signal
        no event was either given or all given events had to be ignored
        because of a too large averaging interval, the corresponding returned
        signal has all entries as nan.


    Examples
    --------

    >>> m1 = assembly_reactivation.AssemblyReact(basepath=r"Z:\Data\HMC2\day5")

    >>> m1.load_data()
    >>> m1.get_weights(epoch=m1.epochs[1])
    >>> assembly_act = m1.get_assembly_act()

    >>> peth_avg, time_lags = event_triggered_average(
    ...    assembly_act.abscissa_vals, assembly_act.data.T, m1.ripples.starts, window=[-0.5, 0.5]
    ... )

    >>> plt.plot(time_lags,peth_avg)
    >>> plt.show()
    """

    # check inputs
    if len(window) != 2:
        raise ValueError(
            "'window' must be a tuple of 2 elements, not {}".format(len(window))
        )

    if window[0] > window[1]:
        raise ValueError(
            "'window' first value must be less than second value, not {}".format(
                len(window)
            )
        )

    if not isinstance(timestamps, np.ndarray):
        raise ValueError(
            "'timestamps' must be a numpy ndarray, not {}".format(type(timestamps))
        )

    if not isinstance(signal, np.ndarray):
        raise ValueError(
            "'signal' must be a numpy ndarray, not {}".format(type(signal))
        )

    if not isinstance(events, (list, np.ndarray)):
        raise ValueError(
            "'events' must be a numpy ndarray or list, not {}".format(type(events))
        )

    if signal.shape[0] != timestamps.shape[0]:
        raise ValueError("'signal' and 'timestamps' must have the same number of rows")

    if len(timestamps.shape) > 1:
        raise ValueError(
            "'timestamps' must be a 1D array, not {}".format(len(timestamps.shape))
        )

    window_starttime, window_stoptime = window

    if len(signal.shape) == 1:
        signal = np.expand_dims(signal, -1)

    _, num_signals = signal.shape

    if sampling_rate is None:
        sampling_rate = 1 / stats.mode(np.diff(timestamps), keepdims=True)[0][0]

    # window_bins: number of bins of the chosen averaging interval
    window_bins = int(np.ceil(((window_stoptime - window_starttime) * sampling_rate)))
    # result_sta: array containing finally the spike-triggered averaged signal
    result_sta = np.zeros((window_bins, num_signals))
    # setting of correct times of the spike-triggered average
    # relative to the spike
    time_lags = np.linspace(window_starttime, window_stoptime, window_bins)

    used_events = np.zeros(num_signals, dtype=int)
    total_used_events = 0

    for i in range(num_signals):
        # summing over all respective signal intervals around spiketimes
        for event in events:
            # locate signal in time range
            idx = (timestamps >= event + window_starttime) & (
                timestamps <= event + window_stoptime
            )

            # for speed, instead of checking if we have enough time each iteration, just skip if we don't
            try:
                result_sta[:, i] += signal[idx, i]
            except Exception:
                continue
            # counting of the used event
            used_events[i] += 1

        # normalization
        result_sta[:, i] = result_sta[:, i] / used_events[i]

        total_used_events += used_events[i]

    if total_used_events == 0:
        warnings.warn("No events at all was either found or used for averaging")

    if return_pandas:
        return pd.DataFrame(
            index=time_lags, columns=np.arange(result_sta.shape[1]), data=result_sta
        )

    return result_sta, time_lags


def event_triggered_average_fast(
    signal: np.ndarray,
    events: np.ndarray,
    sampling_rate: int,
    window=[-0.5, 0.5],
    return_average: bool = True,
    return_pandas: bool = False,
):
    """
    event_triggered_average: Calculate the event triggered average of a signal

    Args:
        signal (np.ndarray): 2D array of signal data (channels x timepoints)
        events (np.ndarray): 1D array of event times
        sampling_rate (int): Sampling rate of signal.
        window (list, optional): Time window (seconds) to average signal around event. Defaults to [-0.5, 0.5].
        return_average (bool, optional): Whether to return the average of the event triggered average. Defaults to True.
            if False, returns the full event triggered average matrix (channels x timepoints x events)

    Returns:
        np.ndarray: Event triggered average of signal
        np.ndarray: Time lags of event triggered average

    note: This version assumes constant sampling rate, no missing data (time gaps), signal start time at 0
    """

    window_starttime, window_stoptime = window
    window_bins = int(np.ceil(((window_stoptime - window_starttime) * sampling_rate)))
    time_lags = np.linspace(window_starttime, window_stoptime, window_bins)

    events = events[
        (events * sampling_rate > len(time_lags) / 2 + 1)
        & (events * sampling_rate < signal.shape[1] - len(time_lags) / 2 + 1)
    ]

    avg_signal = np.zeros(
        [signal.shape[0], len(time_lags), len(events)], dtype=signal.dtype
    )

    for i, event in enumerate(events):
        ts_idx = np.arange(
            np.round(event * sampling_rate) - len(time_lags) / 2,
            np.round(event * sampling_rate) + len(time_lags) / 2,
        ).astype(int)
        avg_signal[:, :, i] = signal[:, ts_idx]

    if return_pandas and return_average:
        return pd.DataFrame(
            index=time_lags,
            columns=np.arange(signal.shape[0]),
            data=avg_signal.mean(axis=2).T,
        )
    if return_average:
        return avg_signal.mean(axis=2), time_lags
    else:
        return avg_signal, time_lags


def count_in_interval(
    st: np.ndarray,
    event_starts: np.ndarray,
    event_stops: np.ndarray,
    par_type: str = "binary",
) -> np.ndarray:
    """
    count_in_interval: count timestamps in intervals
    make matrix n rows (units) by n cols (ripple epochs)
    Input:
        st: spike train list
        event_starts: event starts
        event_stops: event stops
        par_type: count type (counts, binary (default), firing_rate)

        quick binning solution using searchsorted from:
        https://stackoverflow.com/questions/57631469/extending-histogram-function-to-overlapping-bins-and-bins-with-arbitrary-gap

    Output:
        unit_mat: matrix (n cells X n epochs) each column shows count per cell per epoch
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
    st,
    epochs,
    method="peak_fr",  # 'first_spike' or 'peak_fr'
    ref="cells",  # 'cells' or 'epochs'
    padding=0.05,
    dt=0.001,
    sigma=0.01,
    min_units=5,
):
    """
    get rank order of spike train within epoch
    Input:
        st: spike train nelpy array
        epochs: epoch array, windows in which to calculate rank order
        method: method of rank order 'first_spike' or 'peak_fr' (default: peak_fr)
        ref: frame of reference for rank order ('cells' or 'epoch') (default: cells)
        padding: +- padding for epochs
        dt: bin width (s) for finding relative time (epoch ref)
        sigma: smoothing sigma (s) (peak_fr method)
    Output:
        median_rank: median rank order over all epochs (0-1)
        rank_order: matrix (n cells X n epochs) each column shows rank per cell per epoch (0-1)

    Example:
        st,_ = loading.load_spikes(basepath,putativeCellType='Pyr')
        forward_replay = nel.EpochArray(np.array([starts,stops]).T)
        median_rank,rank_order = get_rank_order(st,forward_replay)
    """
    # filter out specific warnings
    warnings.filterwarnings(
        "ignore", message="ignoring events outside of eventarray support"
    )
    warnings.filterwarnings("ignore", message="Mean of empty slice")

    if method not in ["first_spike", "peak_fr"]:
        assert Exception("method " + method + " not implemented")
    if ref not in ["cells", "epoch"]:
        assert Exception("ref " + ref + " not implemented")

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
        rank_order = np.ones([st_epoch.data.shape[0], st_epoch.n_intervals]) * np.nan

        unit_id = np.arange(st_epoch.data.shape[0])

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

    def rank_order_fr(st_epoch, dt, sigma, min_units, ref):
        # set up empty matrix for rank order
        rank_order = np.zeros([st_epoch.data.shape[0], st_epoch.n_intervals]) * np.nan

        unit_id = np.arange(st_epoch.data.shape[0])

        # bin spike train here (smooth later per epoch to not have edge issues)
        z_t = st_epoch.bin(ds=dt)
        # iter over epochs
        for event_i, z_t_temp in enumerate(z_t):
            # smooth spike train in order to estimate peak
            z_t_temp.smooth(sigma=sigma, inplace=True)

            if ref == "cells":
                # find loc of each peak and get sorted idx of active units
                idx = np.argsort(np.argmax(z_t_temp.data, axis=1))
                # reorder unit ids by order and remove non-active
                units = unit_id[idx][np.sum(z_t_temp.data[idx, :] > 0, axis=1) > 0]

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
                for cell_i, unit in enumerate(z_t_temp.data):
                    # if the cell is not active apply nan
                    if not np.any(unit > 0):
                        rank_order[cell_i, event_i] = np.nan
                    else:
                        # calculate normalized rank order (0-1)
                        rank_order[cell_i, event_i] = np.argmax(unit) / len(unit)
        return rank_order

    # create epoched spike array
    st_epoch = st[epochs.expand(padding)]

    # if no spikes in epoch, break out
    if st_epoch.n_active == 0:
        return np.tile(np.nan, st.data.shape), None

    # set up empty matrix for rank order
    if method == "peak_fr":
        rank_order = rank_order_fr(st_epoch, dt, sigma, min_units, ref)
    elif method == "first_spike":
        rank_order = rank_order_first_spike(st_epoch, epochs, dt, min_units, ref)
    else:
        raise Exception("other method, " + method + " is not implemented")

    return np.nanmedian(rank_order, axis=1), rank_order


def count_events(events, time_ref, time_range):
    """
    Count the number of events that occur within a given time range after each reference event.
    Parameters
    ----------
    events : ndarray
        A 1D array of event times.
    time_ref : ndarray
        A 1D array of reference times.
    time_range : tuple
        A tuple containing the start and end times of the time range.
    Returns
    -------
    counts : ndarray
        A 1D array of event counts, one for each reference time (same len as time_ref).
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
def relative_times(t, intervals, values=np.array([0, 1])):
    """
    Calculate relative times and interval IDs for a set of time points.
    Intervals are defined as pairs of start and end times. The relative time is the time
    within the interval, normalized to the interval duration. The interval ID is the index
    of the interval in the intervals array. The values array can be used to assign a value
    to each interval.
    Parameters
    ----------
    t : ndarray
        An array of time points.
    intervals : ndarray
        An array of time intervals, represented as pairs of start and end times.
    values : ndarray, optional
        An array of values to assign to interval bounds. The default is [0,1].
    Returns
    -------
    rt : ndarray
        An array of relative times, one for each time point (same len as t).
    intervalID : ndarray
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


def nearest_event_delay(ts_1: np.ndarray, ts_2: np.ndarray) -> np.ndarray:
    """
    Return for each timestamp in ts_1 the nearest timestamp in ts_2 and the delay between the two

    Parameters
    ----------
    ts_1 : array-like
        timestamps
    ts_2 : array-like
        timestamps
    Returns
    -------
    nearest_ts : array-like
        nearest timestamps in ts_2
    delays : array-like
        delays between ts_1 and nearest_ts
    nearest_index : array-like
        index of nearest_ts in ts_2
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
):
    """
    event_spiking_threshold: filter events based on spiking threshold
    Parameters
    ----------
    spikes : nel.SpikeTrainArray
        spike train array
    events : np.ndarray
        event times
    window : list, optional
        window to compute the event triggered average, by default [-0.5, 0.5]
    event_size : float, optional
        event size in seconds (assumed firing increase location, +-), by default 0.1
    spiking_thres : float, optional
        spiking threshold in zscore units, by default 0
    binsize : float, optional
        bin size, by default 0.01
    sigma : float, optional
        sigma for smoothing binned spike train, by default 0.02
    min_units : int, optional
        minimum number of units to compute the event triggered average, by default 6
    show_fig : bool, optional
        show figure, by default False
    Returns
    -------
    np.ndarray
        valid events

    Example
    -------
    basepath = r"U:\data\hpc_ctx_project\HP04\day_32_20240430"
    ripples = loading.load_ripples_events(basepath, return_epoch_array=False)

    st, cell_metrics = loading.load_spikes(
        basepath,
        brainRegion="CA1",
        support=nel.EpochArray([0, loading.load_epoch(basepath).iloc[-1].stopTime]),
    )
    idx = event_spiking_threshold(st, ripples.peaks.values, show_fig=True)
    print(f"Number of valid ripples: {idx.sum()} out of {len(ripples)}")

    >>> Number of valid ripples: 9244 out of 12655

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
