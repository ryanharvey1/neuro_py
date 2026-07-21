import os
import sys
from typing import Any, Optional, Sequence, Tuple

import nelpy as nel
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import signal
from scipy.io import savemat

from neuro_py.io import loading
from neuro_py.lfp.spectral import filter_signal
from neuro_py.process.intervals import find_interval

# The implementation below follows the public algorithms in Bycycle 1.2.0
# (Voytek Lab, Apache-2.0).
_BYCYCLE_COLUMNS = [
    "sample_peak",
    "sample_last_zerox_decay",
    "sample_zerox_decay",
    "sample_zerox_rise",
    "sample_last_trough",
    "sample_next_trough",
    "period",
    "time_peak",
    "time_trough",
    "volt_peak",
    "volt_trough",
    "time_decay",
    "time_rise",
    "volt_decay",
    "volt_rise",
    "volt_amp",
    "time_rdsym",
    "time_ptsym",
    "band_amp",
    "monotonicity",
    "amp_fraction",
    "amp_consistency",
    "period_consistency",
    "is_burst",
]


def _find_bycycle_flank_zerox(
    values: NDArray[np.float64], flank: str
) -> NDArray[np.int_]:
    """Return Bycycle's index-before-transition zero crossings."""
    positive = values <= 0 if flank == "rise" else values > 0
    crossings = np.flatnonzero(positive[:-1] & ~positive[1:])
    if crossings.size == 0:
        return np.array([len(values) // 2], dtype=int)
    return crossings


def _find_bycycle_extrema(
    values: NDArray[np.float64], fs: float, theta_freq: Tuple[int, int]
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Find extrema using Bycycle's padded narrow-band cycle definition."""
    filter_length = int(np.ceil(fs * 3 / theta_freq[0]))
    if filter_length % 2 == 0:
        filter_length += 1
    padding = int(np.ceil(filter_length / 2))
    padded = np.pad(values, padding, mode="constant")
    filtered = filter_signal(
        padded, fs, "bandpass", theta_freq, n_cycles=3, remove_edges=False
    )
    rise_crossings = _find_bycycle_flank_zerox(filtered, "rise")
    decay_crossings = _find_bycycle_flank_zerox(filtered, "decay")
    if rise_crossings.size < 2 or decay_crossings.size < 2:
        return np.array([], dtype=int), np.array([], dtype=int)

    peaks: list[int] = []
    troughs: list[int] = []
    for rise in rise_crossings:
        following = decay_crossings[decay_crossings > rise]
        if following.size:
            peaks.append(int(rise + np.argmax(padded[rise : following[0]])))
    for decay in decay_crossings:
        following = rise_crossings[rise_crossings > decay]
        if following.size:
            troughs.append(int(decay + np.argmin(padded[decay : following[0]])))
    peaks_array = np.asarray(peaks, dtype=int) - padding
    troughs_array = np.asarray(troughs, dtype=int) - padding
    peaks_array = peaks_array[(peaks_array > 0) & (peaks_array < len(values))]
    troughs_array = troughs_array[(troughs_array > 0) & (troughs_array < len(values))]
    if peaks_array.size == 0 or troughs_array.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if peaks_array[0] > troughs_array[0]:
        troughs_array = troughs_array[1:]
    if peaks_array.size and troughs_array.size and peaks_array[-1] > troughs_array[-1]:
        peaks_array = peaks_array[:-1]
    count = min(peaks_array.size, troughs_array.size)
    return peaks_array[:count], troughs_array[:count]


def _find_bycycle_zerox(
    values: NDArray[np.float64], peaks: NDArray[np.int_], troughs: NDArray[np.int_]
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Find midpoint crossings between the extrema used to define each cycle."""
    rises = np.empty(len(peaks) - 1, dtype=int)
    decays = np.empty(len(peaks), dtype=int)
    for idx, (trough, peak) in enumerate(zip(troughs[:-1], peaks[1:])):
        segment = values[trough : peak + 1]
        if np.sum(np.abs(segment)) == 0 or segment[0] > segment[-1]:
            rises[idx] = trough + len(segment) // 2
        else:
            midpoint = (segment[0] + segment[-1]) / 2
            rises[idx] = trough + int(
                np.median(_find_bycycle_flank_zerox(segment - midpoint, "rise"))
            )
    for idx, (peak, trough) in enumerate(zip(peaks, troughs)):
        segment = values[peak : trough + 1]
        if np.sum(np.abs(segment)) == 0 or segment[0] < segment[-1]:
            decays[idx] = peak + len(segment) // 2
        else:
            midpoint = (segment[0] + segment[-1]) / 2
            decays[idx] = peak + int(
                np.median(_find_bycycle_flank_zerox(segment - midpoint, "decay"))
            )
    return rises, decays


def _detect_bycycle_features(
    sig: NDArray[Any],
    fs: float,
    theta_freq: Tuple[int, int],
    thresholds: dict[str, Any],
) -> pd.DataFrame:
    """Compute the cycle fields consumed by the historical Bycycle workflow.

    This implementation follows the cycle definition and consistency thresholds
    used by Bycycle 1.2.0 (Voytek Lab, Apache-2.0). It is intentionally local
    to avoid making Bycycle and neurodsp installation requirements.
    """
    if theta_freq[0] <= 0 or theta_freq[1] >= fs / 2 or theta_freq[0] >= theta_freq[1]:
        raise ValueError("theta_freq must lie within (0, fs / 2).")
    required = {
        "amp_fraction",
        "amp_consistency",
        "period_consistency",
        "monotonicity",
        "min_n_cycles",
    }
    unknown = set(thresholds).difference(required)
    if unknown:
        raise ValueError(f"Unsupported theta-cycle thresholds: {sorted(unknown)}")
    values = np.asarray(sig, dtype=float)
    peaks, troughs = _find_bycycle_extrema(values, fs, theta_freq)
    if peaks.size < 2 or troughs.size < 2:
        return pd.DataFrame(columns=pd.Index(data=_BYCYCLE_COLUMNS))

    rises, decays = _find_bycycle_zerox(values, peaks, troughs)
    samples = pd.DataFrame(
        {
            "sample_peak": peaks[1:],
            "sample_last_zerox_decay": decays[:-1],
            "sample_zerox_decay": decays[1:],
            "sample_zerox_rise": rises,
            "sample_last_trough": troughs[:-1],
            "sample_next_trough": troughs[1:],
        }
    )
    if samples.empty:
        return pd.DataFrame(columns=pd.Index(data=_BYCYCLE_COLUMNS))

    peak_samples = samples["sample_peak"].to_numpy()
    last_troughs = samples["sample_last_trough"].to_numpy()
    next_troughs = samples["sample_next_trough"].to_numpy()
    period = next_troughs - last_troughs
    volt_rise = values[peak_samples] - values[last_troughs]
    volt_decay = values[peak_samples] - values[next_troughs]
    filtered = filter_signal(
        values, fs, "bandpass", theta_freq, n_cycles=3, remove_edges=False
    )
    analytic_amplitude = np.abs(signal.hilbert(filtered))
    band_amp = np.array(
        [
            np.mean(analytic_amplitude[start:stop])
            for start, stop in zip(last_troughs, next_troughs)
        ]
    )
    monotonicity = np.array(
        [
            np.mean(
                [
                    np.mean(np.diff(values[last_trough : peak + 1]) > 0),
                    np.mean(np.diff(values[peak : next_trough + 1]) < 0),
                ]
            )
            for peak, last_trough, next_trough in zip(
                peak_samples, last_troughs, next_troughs
            )
        ]
    )
    features = samples.assign(
        period=period,
        time_peak=samples["sample_zerox_decay"] - samples["sample_zerox_rise"],
        time_trough=samples["sample_zerox_rise"] - samples["sample_last_zerox_decay"],
        volt_peak=values[peak_samples],
        volt_trough=values[last_troughs],
        time_decay=next_troughs - peak_samples,
        time_rise=peak_samples - last_troughs,
        volt_decay=volt_decay,
        volt_rise=volt_rise,
        volt_amp=(volt_rise + volt_decay) / 2,
        time_rdsym=(peak_samples - last_troughs) / period,
        time_ptsym=(
            (samples["sample_zerox_decay"] - samples["sample_zerox_rise"])
            / (samples["sample_zerox_decay"] - samples["sample_last_zerox_decay"])
        ),
        band_amp=band_amp,
        monotonicity=monotonicity,
    )
    if features.empty:
        features["is_burst"] = pd.Series(dtype=bool)
        return features
    # Bycycle's cycle-consistency burst features (v1.2.0): amplitude is
    # ranked across cycles, while the consistency metrics compare both adjacent
    # cycles and the rise/decay pair in the current cycle.
    features["amp_fraction"] = features["volt_amp"].rank() / len(features)
    amp_consistency = np.full(len(features), np.nan)
    period_consistency = np.full(len(features), np.nan)
    rises = features["volt_rise"].to_numpy()
    decays = features["volt_decay"].to_numpy()
    periods = features["period"].to_numpy()
    for cycle in range(1, len(features) - 1):
        with np.errstate(invalid="ignore", divide="ignore"):
            amp_ratios = np.array(
                [
                    min(rises[cycle], decays[cycle]) / max(rises[cycle], decays[cycle]),
                    min(rises[cycle], decays[cycle - 1])
                    / max(rises[cycle], decays[cycle - 1]),
                    min(rises[cycle + 1], decays[cycle])
                    / max(rises[cycle + 1], decays[cycle]),
                ]
            )
        amp_consistency[cycle] = np.nanmin(amp_ratios)
        period_consistency[cycle] = min(periods[cycle - 1], periods[cycle]) / max(
            periods[cycle - 1], periods[cycle]
        )
        period_consistency[cycle] = min(
            period_consistency[cycle],
            min(periods[cycle + 1], periods[cycle])
            / max(periods[cycle + 1], periods[cycle]),
        )
    features["amp_consistency"] = amp_consistency
    features["period_consistency"] = period_consistency
    is_burst = (
        (features["amp_fraction"] > thresholds["amp_fraction"])
        & (features["amp_consistency"] > thresholds["amp_consistency"])
        & (features["period_consistency"] > thresholds["period_consistency"])
        & (features["monotonicity"] > thresholds["monotonicity"])
    ).to_numpy(dtype=bool, copy=True)
    is_burst[[0, -1]] = False
    starts = np.flatnonzero(np.diff(np.r_[False, is_burst, False].astype(int)) == 1)
    stops = np.flatnonzero(np.diff(np.r_[False, is_burst, False].astype(int)) == -1)
    for start, stop in zip(starts, stops):
        if stop - start < thresholds["min_n_cycles"]:
            is_burst[start:stop] = False
    features["is_burst"] = is_burst
    return features


def get_theta_channel(basepath: str, tag: str = "CA1so") -> Optional[int]:
    """
    Get the theta channel for the specified brain region. First looks in channel_tags, then in brain regions.
    If not found or all channels are bad, returns None.

    Parameters
    ----------
    basepath : str
        The base path for loading data.
    tag : str, optional
        The tag identifying the brain region. Default is "CA1so".

    Returns
    -------
    int or None
        The index of the theta channel (0-based), or None if not found or all channels are bad.
    """
    brain_region = loading.load_brain_regions(basepath)
    channel_tags = loading.load_channel_tags(basepath)
    if not isinstance(brain_region, dict):
        return None

    # First, check in channel_tags
    if tag in channel_tags:
        ch = channel_tags[tag]["channels"] - 1  # correct for 0-based indexing
        if isinstance(ch, (np.ndarray, list)) and len(ch) > 1:
            print(
                f"Multiple theta channels found for {tag} in {basepath}. Using the first one."
            )
            ch = ch[0]
        return int(ch)

    # Then try brain_region
    if tag in brain_region:
        print(
            f"Input tag: {tag} not found in {basepath} channel_tags. Looking in brain regions."
        )
        region_chan = brain_region[tag]["channels"] - 1  # 0-based indexing

        # Ensure iterable
        if isinstance(region_chan, (int, np.integer)):
            region_chan = np.array([region_chan])
        else:
            region_chan = np.asarray(region_chan)

        bad_ch = channel_tags.get("Bad", {}).get("channels", [])
        bad_ch = np.asarray(bad_ch)

        for chan in region_chan:
            if chan not in bad_ch:
                print(
                    f"Multiple theta channels found for {tag} in {basepath}. Using the first good one."
                )
                return int(chan)

        print(f"Input tag: {tag} found in brain regions but all channels are bad.")
        return None

    print(f"Input tag: {tag} not found in {basepath} channel_tags or brain regions.")
    return None


def process_lfp(basepath: str) -> Tuple[NDArray[Any], NDArray[Any], float]:
    """
    Process and load Local Field Potential (LFP) data.

    Parameters
    ----------
    basepath : str
        The base path for loading LFP data.

    Returns
    -------
    tuple
        A tuple containing the LFP data, timestamps, and sampling frequency.
    """
    xml_data = loading.loadXML(basepath)
    if xml_data is None:
        raise FileNotFoundError(f"Could not load XML metadata from {basepath}")
    nChannels, fs, _, _ = xml_data

    lfp, ts = loading.loadLFP(
        basepath, n_channels=nChannels, channel=None, frequency=fs
    )
    return lfp, ts, fs


def get_ep_from_df(df: pd.DataFrame, ts: NDArray[Any]) -> nel.EpochArray:
    """
    Extract epochs of theta oscillations from a bycycle dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing burst detection results.
    ts : np.ndarray
        Timestamps of the LFP data.

    Returns
    -------
    nel.EpochArray
        An array of theta epochs.
    """
    index_for_oscilation_epoch = find_interval(df.is_burst)
    start = []
    stop = []
    for idx in index_for_oscilation_epoch or []:
        start.append(df.sample_peak[idx[0]])
        stop.append(df.sample_peak[idx[1]])

    # convert list to array
    start = np.array(start)
    stop = np.array(stop)

    # index ts get get start and end ts for each oscillation epoch

    start_ts = ts[start]
    stop_ts = ts[stop]

    theta_epoch = nel.EpochArray([np.array([start_ts, stop_ts]).T])

    return theta_epoch


def save_theta_cycles(
    df: pd.DataFrame,
    ts: NDArray[Any],
    basepath: str,
    detection_params: dict[str, Any],
    ch: int,
    event_name: str = "thetacycles",
    detection_name: Optional[str] = None,
) -> None:
    """
    Save theta cycles detected using bycycle to a .mat file in the cell explorer format.

    Parameters
    ----------
    df : pd.DataFrame
        The bycycle dataframe containing theta cycle features.
    ts : np.ndarray
        Timestamps of the LFP data.
    basepath : str
        Base path to save the file to.
    detection_params : dict
        Dictionary of detection parameters.
    ch : int
        Channel used for theta detection.
    event_name : str, optional
        Name of the events (default is "thetacycles").
    detection_name : str or None, optional
        Name of the detection (default is None).
    """
    filename = os.path.join(
        basepath, os.path.basename(basepath) + "." + event_name + ".events.mat"
    )
    data: dict[str, Any] = {}
    data[event_name] = {}

    # create variables that will be saved
    timestamps = np.array(
        [ts[df.sample_peak.values[:-1]], ts[df.sample_peak.values[1:]]]
    )
    peaks = ts[df.sample_last_trough.values[1:]]
    amplitudes = df.band_amp.values[1:]
    duration = np.diff(
        np.array([ts[df.sample_peak.values[:-1]], ts[df.sample_peak.values[1:]]]),
        axis=0,
    )
    center = np.median(
        np.array([ts[df.sample_peak.values[:-1]], ts[df.sample_peak.values[1:]]]),
        axis=0,
    )

    # limit to cycles using is_burst
    timestamps = timestamps[:, df.is_burst.values[1:]]
    peaks = peaks[df.is_burst.values[1:]]
    amplitudes = amplitudes[df.is_burst.values[1:]]
    duration = duration[:, df.is_burst.values[1:]]
    center = center[df.is_burst.values[1:]]

    # save start_ts and stop_ts as 2d array
    data[event_name]["timestamps"] = timestamps.T
    data[event_name]["peaks"] = peaks.T
    data[event_name]["amplitudes"] = amplitudes.T
    data[event_name]["amplitudeUnits"] = "mV"
    data[event_name]["eventID"] = []
    data[event_name]["eventIDlabels"] = []
    data[event_name]["eventIDbinary"] = []

    # check if only single epoch
    data[event_name]["duration"] = duration.T

    data[event_name]["center"] = center.T
    data[event_name]["detectorinfo"] = {}
    if detection_name is None:
        data[event_name]["detectorinfo"]["detectorname"] = []
    else:
        data[event_name]["detectorinfo"]["detectorname"] = detection_name
    data[event_name]["detectorinfo"]["detectionparms"] = detection_params
    data[event_name]["detectorinfo"]["detectionintervals"] = []
    data[event_name]["detectorinfo"]["theta_channel"] = ch

    savemat(filename, data, long_field_names=True)


def get_theta_cycles(
    basepath: str,
    theta_freq: Tuple[int, int] = (6, 10),
    lowpass: int = 48,
    detection_params: Optional[dict[str, Any]] = None,
    ch: Optional[int] = None,
    tag: Sequence[str] = ("CA1so", "CA1sp"),
) -> None:
    """
    Detect theta cycles in LFP data and save the results.

    Parameters
    ----------
    basepath : str
        The base path for loading LFP data.
    theta_freq : tuple, optional
        Frequency range for theta detection (default is (6, 10)).
    lowpass : int, optional
        Cut-off frequency for low-pass filtering (default is 48).
    detection_params : dict or None, optional
        Parameters for theta detection (default is None).
    ch : int or None, optional
        Channel used for theta detection (default is None).
    tag : list, optional
        List of tags to identify the theta channel (default is ["CA1so", "CA1sp"]).
        The function will first try to find the channel using the first tag, then the second.

    Returns
    -------
    None
    """
    # load lfp as memmap
    lfp, ts, fs = process_lfp(basepath)

    # get theta channel - default chooses CA1so
    if ch is None:
        ch = get_theta_channel(basepath, tag=tag[0])

    if ch is None:
        ch = get_theta_channel(basepath, tag=tag[1])

    if ch is None:
        Warning("No theta channel found")
        return None

    # per bycycle documentation, low-pass filter signal before running bycycle 4x the frequency of interest
    filt_sig = filter_signal(lfp[:, ch], fs, "lowpass", lowpass, remove_edges=False)

    # for detecting theta epochs
    if detection_params is None:
        thresholds = {
            "amp_fraction": 0.1,
            "amp_consistency": 0.4,
            "period_consistency": 0.5,
            "monotonicity": 0.6,
            "min_n_cycles": 3,
        }
    else:
        thresholds = detection_params

    features = _detect_bycycle_features(filt_sig, fs, theta_freq, thresholds)
    save_theta_cycles(features, ts, basepath, detection_params=thresholds, ch=ch)


# to run on cmd
if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) == 2:
        get_theta_cycles(sys.argv[1])
    elif len(sys.argv) == 3:
        get_theta_cycles(sys.argv[1], ch=int(sys.argv[2]))
