import numpy as np
from concurrent.futures import ThreadPoolExecutor
from neurodsp.timefrequency.wavelets import compute_wavelet_transform
import pandas as pd
from scipy import linalg
from scipy import signal
from statsmodels.regression import yule_walker


def _yule_walker(X, order=1):
    """
    Compute the Yule-Walker equations to estimate the parameters of an autoregressive (AR) model.

    This function calculates the autocorrelation coefficients (rho) and the noise standard deviation
    (sigma) of an AR model of a given order for a 2D input array `X` using the Yule-Walker method.

    Parameters:
    -----------
    X : ndarray
        A 2D numpy array where each row represents a different time series.
        The number of columns is the length of each time series.

    order : int, optional (default=1)
        The order of the autoregressive model.

    Returns:
    --------
    rho : ndarray
        The estimated AR coefficients of length equal to `order`.

    sigma : float
        The estimated standard deviation of the noise in the AR model.
    """
    assert X.ndim == 2, "Input data X must be a 2-dimensional array."

    # Compute the denominator for the autocorrelation calculation
    denom = X.shape[-1] - np.arange(order + 1)  # [N, N-1, ..., N-order]

    # Initialize the autocorrelation array
    r = np.zeros(
        order + 1, np.float64
    )  # Holds autocorrelations up to the specified order

    # Loop over each row (time series) in X
    for di, d in enumerate(X):
        # Remove mean to center the time series
        d -= d.mean()
        # Autocorrelation at lag 0 (total variance)
        r[0] += np.dot(d, d)

        # Compute autocorrelation for each lag up to the specified order
        for k in range(1, order + 1):
            # Autocorrelation at lag k
            r[k] += np.dot(d[0:-k], d[k:])

    # Normalize by the number of observations and the denominator
    r /= denom * len(X)

    # Solve the Yule-Walker equations (Toeplitz system) to estimate AR coefficients
    rho = linalg.solve(linalg.toeplitz(r[:-1]), r[1:])

    # Estimate the variance of the noise
    sigmasq = r[0] - (r[1:] * rho).sum()

    # Return AR coefficients and noise standard deviation
    return rho, np.sqrt(sigmasq)


def whiten_lfp(lfp, order=2):
    """
    Perform temporal whitening of Local Field Potential (LFP) data using an Autoregressive (AR) model.

    This function applies temporal whitening to LFP data by fitting an AR model of the specified order
    and using the model to remove temporal correlations, resulting in a 'whitened' signal.

    Parameters:
    -----------
    lfp : ndarray
        A 1D numpy array containing the LFP data. 

    order : int, optional (default=6)
        The order of the AR model to be used for whitening the LFP data.

    Returns:
    --------
    whitened_lfp : ndarray
        The temporally whitened LFP data as a 1D numpy array.
    """

    rho, sigma = yule_walker(lfp, order=order)

    b, a = np.array([1.0]), np.concatenate(([1.0], -rho))

    # Apply the whitening filter to the LFP data and return the result as a 1D array
    return signal.convolve(lfp, a, "same")


def event_triggered_wavelet(
    signal: np.ndarray,
    timestamps: np.ndarray,
    events: np.ndarray,
    max_lag: float = 1,
    freq_min: float = 4,
    freq_max: float = 100,
    freq_step: float = 4,
    return_pandas: bool = False,
    parallel: bool = True,
    whiten: bool = True,
):
    """
    Compute the event-triggered wavelet transform of a signal.

    Parameters
    ----------
    signal : 1d array
        Time series.
    timestamps : 1d array
        Time points for each sample in the signal.
    events : 1d array
        Time points of events.
    max_lag : float
        Maximum lag to consider, in seconds.
    freq_min : float
        Minimum frequency to consider, in Hz.
    freq_max : float
        Maximum frequency to consider, in Hz.
    freq_step : float
        Step size for frequency range, in Hz.
    return_pandas : bool
        If True, return the output as pandas objects.

    Returns
    -------
    mwt : 2d array
        Time frequency representation of the input signal.
    sigs : 1d array
        Average signal.
    times : 1d array
        Time points for each sample in the output.
    freqs : 1d array
        Frequencies used in the wavelet transform.

    Examples:
    from neuro_py.lfp.spectral import event_triggered_wavelet

    basepath = r"Z:\Data\hpc_ctx_project\HP04\day_34_20240503"

    # load lfp
    nChannels, fs, _, _ = loading.loadXML(basepath)
    # Load the LFP data
    lfp, ts = loading.loadLFP(basepath, n_channels=nChannels,
                    channel=23,
                    frequency=fs)
    # load events
    opto = loading.load_events(basepath, epoch_name="optoStim")
    opto = opto.merge(gap=.1)

    # compute event triggered averate
    mwt, sigs, times, freqs = event_triggered_wavelet(
        lfp,
        ts,
        opto.starts,
    )

    # plot
    plt.figure(figsize=set_size("thesis", fraction=1, subplots=(1, 1)))

    im = plt.imshow(
        abs(mwt),
        aspect="auto",
        extent=[times[0], times[-1], freqs[-1], freqs[0]],
        cmap="magma",
        vmax=600,
        vmin=50,
    )
    plt.axhline(23, color="orange", linestyle="--", label="23hz")

    plt.yscale("log")
    # move legend outside of plot
    plt.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1), frameon=False)

    plt.gca().invert_yaxis()

    plt.colorbar(location="top", label="Power (uV^2)")
    # move colorbar more to the left
    plt.gcf().axes[1].set_position([0.5, 0.8, 0.4, 0.6])


    plt.gca().set_ylabel("Frequency (Hz)")

    plt.gca().set_xlabel("Time from opto stim (s)")

    plt.twinx()
    plt.yscale("linear")
    plt.axvline(0, color="k", linestyle="--")
    plt.axvline(0.5, color="k", linestyle="--")
    plt.plot(times, sigs, "w", linewidth=0.5)


    # plt.gca().set_xlabel('Time (s)')
    plt.gca().set_ylabel("Voltage (uV)")
    plt.gca().set_title("PFC during 23Hz stim in behavior", y=1)
    """

    signal_ = signal.copy()
    if whiten:
        signal_ = whiten_lfp(signal)

    # set up frequency range
    freqs = np.arange(freq_min, freq_max, freq_step)
    ds = timestamps[1] - timestamps[0]
    fs = 1 / ds
    times = np.linspace(-max_lag, max_lag, np.round((max_lag * 2) / ds).astype(int))

    n_freqs = len(freqs)
    n_samples = len(times)

    # set up mwt and sigs to store results
    mwt = np.zeros((n_freqs, n_samples))
    sigs = np.zeros(n_samples)

    event_i = 0

    def process_event(start):
        nonlocal event_i
        nonlocal mwt
        nonlocal sigs

        if start + max_lag > timestamps.max() or start - max_lag < timestamps.min():
            return None, None
        
        idx = (timestamps >= start - max_lag) & (timestamps < start + max_lag)
        sig = signal[idx]

        mwt_partial = np.abs(compute_wavelet_transform(signal_[idx], fs, freqs=freqs))

        return mwt_partial, sig

    if parallel:
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_event, events))

        for mwt_partial, sig_partial in results:
            if mwt_partial is not None:
                # samples might be missing if the event is too close to the edge
                if mwt_partial.shape[1] != n_samples:
                    continue
                mwt += mwt_partial
                sigs += sig_partial
                event_i += 1
    else:
        for start in events:
            mwt_partial, sig_partial = process_event(start)
            if mwt_partial is not None:
                mwt += mwt_partial
                sigs += sig_partial
                event_i += 1

    mwt /= event_i
    sigs /= event_i

    if return_pandas:
        mwt = pd.DataFrame(mwt.T, index=times, columns=freqs)
        sigs = pd.Series(sigs, index=times)
        return mwt, sigs
    else:
        return mwt, sigs, times, freqs
