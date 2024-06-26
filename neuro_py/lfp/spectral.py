import numpy as np
from concurrent.futures import ThreadPoolExecutor
from neurodsp.timefrequency.wavelets import compute_wavelet_transform
import pandas as pd


def event_triggered_wavelet(
    signal: np.ndarray,
    timestamps: np.ndarray,
    events: np.ndarray,
    max_lag: float = 1,
    freq_min: float = 4,
    freq_max: float = 100,
    freq_step: float = 4,
    return_pandas: bool = False,
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

    # set up frequency range
    freqs = np.arange(freq_min, freq_max, freq_step)
    ds = timestamps[1] - timestamps[0]
    fs = 1 / ds
    times = np.linspace(-max_lag, max_lag, int((max_lag * 2) / ds))

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

        sig = signal[(timestamps >= start - max_lag) & (timestamps < start + max_lag)]

        mwt_partial = np.abs(compute_wavelet_transform(sig, fs, freqs=freqs))

        return mwt_partial, sig

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

    mwt /= event_i
    sigs /= event_i

    if return_pandas:
        mwt = pd.DataFrame(mwt.T, index=times, columns=freqs)
        sigs = pd.Series(sigs, index=times)
        return mwt, sigs
    else:
        return mwt, sigs, times, freqs
