"""Cortical delta waves from trough--peak--trough LFP morphology."""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import nelpy as nel
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage, signal
from scipy.io import savemat

from neuro_py.io import loading
from neuro_py.io.matlab import load_mat
from neuro_py.process.intervals import in_intervals

_REGIONS = ("PFC", "MEC", "LEC", "EC", "ILA", "PL", "Cortex", "CTX")


def _epochs(value: nel.EpochArray | ArrayLike) -> nel.EpochArray:
    """Validate seconds-based intervals and consolidate their union."""
    if isinstance(value, nel.EpochArray):
        return value.merge()
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return nel.EpochArray(empty=True)
    if array.ndim == 1 and array.size == 2:
        array = array.reshape(1, 2)
    if (
        array.ndim != 2
        or array.shape[1] != 2
        or not np.isfinite(array).all()
        or np.any(array[:, 1] <= array[:, 0])
    ):
        raise ValueError("Intervals must have shape (n, 2) with finite start < stop.")
    return nel.EpochArray(array).merge()


def _bounds(mask: NDArray[np.bool_]) -> NDArray[np.int64]:
    """Find half-open contiguous sample ranges."""
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return edges.reshape(-1, 2)


def _segments(
    timestamps: NDArray[np.float64],
    epochs: nel.EpochArray,
    fs: float,
) -> list[tuple[int, int]]:
    """Split restrictions at timestamp gaps without concatenating across them."""
    if timestamps.size == 0 or epochs.isempty:
        return []
    mask = np.asarray(in_intervals(timestamps, epochs.data), dtype=bool)
    gaps = np.flatnonzero(np.diff(timestamps) > 1.5 / fs) + 1
    result: list[tuple[int, int]] = []
    for start, stop in _bounds(mask):
        cuts = gaps[(gaps > start) & (gaps < stop)]
        points = np.r_[start, cuts, stop]
        result.extend((int(a), int(b)) for a, b in zip(points[:-1], points[1:]))
    return result


def _channel_candidates(
    session: dict[str, Any],
    channel: int | Sequence[int] | None,
    brain_regions: Sequence[str],
    n_channels: int,
) -> list[int]:
    """Translate MATLAB tags once; public channel arguments are zero-based."""
    tags = session.get("channelTags", {})
    if channel is not None:
        raw = np.atleast_1d(channel)
    else:
        tagged = [
            v.get("channels", []) for k, v in tags.items() if k.lower() == "delta"
        ]
        if tagged and np.asarray(tagged[0]).size:
            raw = np.atleast_1d(tagged[0]).astype(float) - 1
        else:
            regions = session.get("brainRegions", {})
            matches = [
                np.atleast_1d(value.get("channels", [])).astype(float) - 1
                for key, value in regions.items()
                if any(region.lower() in key.lower() for region in brain_regions)
            ]
            raw = np.concatenate(matches) if matches else np.array([])
    raw = np.asarray(raw, dtype=float)
    if (
        raw.ndim != 1
        or not np.isfinite(raw).all()
        or np.any(raw != np.floor(raw))
        or np.any(raw < 0)
        or np.any(raw >= n_channels)
    ):
        raise ValueError("channel must contain zero-indexed integer LFP channels.")
    bad = {
        int(c) - 1
        for key, value in tags.items()
        if key.lower() == "bad"
        for c in np.atleast_1d(value.get("channels", []))
    }
    candidates = [int(c) for c in np.unique(raw) if int(c) not in bad]
    if not candidates:
        raise ValueError(
            "No usable delta channels; pass channel or check session metadata."
        )
    return candidates


def _select_channel(
    data: NDArray[np.generic],
    candidates: list[int],
    segments: list[tuple[int, int]],
    fs: float,
    passband: tuple[float, float],
) -> int:
    """Compare weighted delta/broadband Welch power, one channel at a time."""
    if len(candidates) == 1:
        return candidates[0]
    scores = []
    for channel in candidates:
        band_power = total_power = 0.0
        for start, stop in segments:
            # Bound temporary arrays even for an hours-long uninterrupted recording.
            for left in range(start, stop, max(1, int(60 * fs))):
                values = np.asarray(
                    data[left : min(left + int(60 * fs), stop), channel], dtype=float
                )
                if values.size < max(32, int(2 * fs)) or not np.isfinite(values).all():
                    continue
                freq, power = signal.welch(
                    values, fs=fs, nperseg=min(values.size, int(4 * fs))
                )
                band_power += (
                    float(power[(freq >= passband[0]) & (freq <= passband[1])].sum())
                    * values.size
                )
                total_power += (
                    float(power[(freq >= 1) & (freq < fs / 2)].sum()) * values.size
                )
        scores.append(band_power / total_power if total_power > 0 else -np.inf)
    if not np.isfinite(scores).any():
        raise ValueError("No finite, nonconstant data available for channel selection.")
    return candidates[int(np.argmax(scores))]


def _clean_segments(
    data: NDArray[np.generic],
    segments: list[tuple[int, int]],
    fs: float,
    reject_artifacts: bool,
) -> list[tuple[int, NDArray[np.float64]]]:
    """Exclude nonfinite values and padded large/fast artifacts before filtering."""
    count, mean, m2 = 0, 0.0, 0.0
    # Combine moments without a recording-sized concatenation or catastrophic
    # cancellation when the recording has a large DC offset.
    for start, stop in segments:
        values = np.asarray(data[start:stop], dtype=float)
        values = values[np.isfinite(values)]
        n = values.size
        if n:
            delta = float(values.mean()) - mean
            m2 += float(
                np.sum((values - values.mean()) ** 2)
            ) + delta**2 * count * n / (count + n)
            mean += delta * n / (count + n)
            count += n
    std = np.sqrt(m2 / (count - 1)) if count > 1 else 0.0
    result = []
    for start, stop in segments:
        values = np.asarray(data[start:stop], dtype=float)
        good = np.isfinite(values)
        if reject_artifacts and std > 0:
            z = (values - mean) / std
            large = np.abs(z) > 6
            fast = np.r_[np.abs(np.diff(z)) > 10, False]
            # Range expansion is O(samples + artifacts), independent of padding.
            for bad, padding in ((large, 0.5), (fast, 0.1)):
                indices = np.flatnonzero(bad)
                radius = int(np.ceil(padding * fs))
                edges = np.zeros(values.size + 1, dtype=np.int64)
                np.add.at(edges, np.maximum(indices - radius, 0), 1)
                np.add.at(edges, np.minimum(indices + radius + 1, values.size), -1)
                good &= np.cumsum(edges[:-1]) == 0
        for left, right in _bounds(good):
            if right - left > max(30, int(fs * 0.15)):
                result.append((start + int(left), values[left:right]))
    return result


def _lowpass(
    values: NDArray[np.float64], cutoff: float, fs: float, order: int
) -> NDArray[np.float64]:
    """Apply the reference Chebyshev-II lowpass using stable second-order sections."""
    sos = signal.cheby2(order, 20, cutoff, btype="lowpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, values)


def _find_waves(
    segments: list[tuple[int, NDArray[np.float64]]],
    timestamps: NDArray[np.float64],
    fs: float,
    filter_band: tuple[float, float],
    threshold: float,
    edge_buffer: float,
    *,
    low_peak_threshold: float,
    high_peak_threshold: float,
    low_trough_threshold: float,
    high_trough_threshold: float,
    min_duration: float,
    max_duration: float,
) -> tuple[pd.DataFrame, NDArray[np.float64]]:
    """Detect extrema independently; standardize amplitudes over all usable samples."""
    filtered_segments = []
    count, mean, m2 = 0, 0.0, 0.0
    for start, values in segments:
        filtered = _lowpass(
            values - _lowpass(values, filter_band[0], fs, 5), filter_band[1], fs, 5
        )
        n = filtered.size
        delta = float(filtered.mean()) - mean
        m2 += float(
            np.sum((filtered - filtered.mean()) ** 2)
        ) + delta**2 * count * n / (count + n)
        mean += delta * n / (count + n)
        count += n
        filtered_segments.append((start, filtered))
    std = np.sqrt(m2 / (count - 1)) if count > 1 else 0.0
    records = []
    intervals = []
    for start, filtered in filtered_segments:
        stop = start + filtered.size
        if timestamps[stop - 1] - timestamps[start] <= 2 * edge_buffer:
            continue
        intervals.append(
            [timestamps[start] + edge_buffer, timestamps[stop - 1] - edge_buffer]
        )
        if std == 0:
            continue
        derivative = _lowpass(np.r_[np.diff(filtered), 0.0], filter_band[1], fs, 8)
        derivative -= derivative.mean()
        # Ignore exact zeros when locating sign changes, so flat extrema are safe.
        nonzero = np.flatnonzero(derivative)
        left, right = nonzero[:-1], nonzero[1:]
        up = left[(derivative[left] < 0) & (derivative[right] > 0)]
        down = left[(derivative[left] > 0) & (derivative[right] < 0)]
        if up.size < 2 or down.size == 0:
            continue
        positions = np.searchsorted(down, up[:-1], side="right")
        valid = positions < down.size
        first, last = up[:-1][valid], up[1:][valid]
        peak = down[positions[valid]]
        valid = peak < last
        where = np.column_stack((first[valid], peak[valid], last[valid]))
        times = timestamps[start + where]
        z = (filtered[where] - mean) / std
        length = times[:, 2] - times[:, 0]
        keep = (
            (length >= min_duration)
            & (length <= max_duration)
            & (
                ((z[:, 1] > high_peak_threshold) & (z[:, 2] <= -low_trough_threshold))
                | ((z[:, 1] >= low_peak_threshold) & (z[:, 2] < -high_trough_threshold))
            )
            & (z[:, 1] > z[:, 0])
            & (z[:, 1] - z[:, 2] > threshold)
            & (times[:, 0] >= intervals[-1][0])
            & (times[:, 2] <= intervals[-1][1])
        )
        records.extend(np.column_stack((times[keep], z[keep])).tolist())
    waves = np.asarray(records, dtype=float).reshape(-1, 6)
    events = pd.DataFrame(
        {
            "start": waves[:, 0],
            "stop": waves[:, 2],
            "peaks": waves[:, 1],
            "amplitude": waves[:, 4] - waves[:, 5],
            "center": (waves[:, 0] + waves[:, 2]) / 2,
            "duration": waves[:, 2] - waves[:, 0],
            "start_value": waves[:, 3],
            "peak_value": waves[:, 4],
            "stop_value": waves[:, 5],
        }
    )
    return events, np.asarray(intervals, dtype=float).reshape(-1, 2)


def _verify_firing(
    events: pd.DataFrame,
    spike_times: Sequence[ArrayLike],
    threshold: float,
) -> tuple[pd.DataFrame, float, bool]:
    """Raise the amplitude cutoff using amplitude-ordered population suppression."""
    trains = [np.asarray(train, dtype=float).reshape(-1) for train in spike_times]
    if events.empty or not trains or not any(train.size for train in trains):
        warnings.warn(
            "No events or cortical spikes available to verify firing.", stacklevel=2
        )
        return events, threshold, False
    if any(not np.isfinite(train).all() for train in trains):
        raise ValueError("spike_times must be finite timestamps in seconds.")
    spikes = np.sort(np.concatenate(trains))
    # Population PETH for each event, in 10 ms bins over +/- 0.5 s.
    bins = np.linspace(-0.5, 0.5, 101)
    edges = events.peaks.to_numpy()[:, None] + bins
    counts = np.diff(np.searchsorted(spikes, edges), axis=1).astype(float)
    smooth = ndimage.gaussian_filter1d(counts, 1.0, axis=1)
    std = smooth.std(axis=1, ddof=1, keepdims=True)
    z = np.divide(
        smooth - smooth.mean(axis=1, keepdims=True),
        std,
        out=np.zeros_like(smooth),
        where=std > 0,
    )
    centers = (bins[:-1] + bins[1:]) / 2
    response = z[:, np.abs(centers) < 0.05].mean(axis=1)
    order = np.argsort(-events.amplitude.to_numpy(), kind="stable")
    width = min(100, len(order))
    ordered_response = ndimage.uniform_filter1d(
        response[order], size=width, mode="nearest"
    )
    suppressed = np.flatnonzero(ordered_response <= -0.5)
    if not suppressed.size:
        warnings.warn(
            "Unable to verify firing suppression; retaining the LFP threshold.",
            stacklevel=2,
        )
        return events, threshold, False
    cutoff = max(threshold, float(events.amplitude.iloc[order[suppressed[-1]]]))
    selected = events.loc[events.amplitude > cutoff].reset_index(drop=True)
    return selected, cutoff, not selected.empty


def _event_path(basepath: str | Path, event_name: str) -> Path:
    """Require an event name that is also a MATLAB struct identifier."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,62}", event_name):
        raise ValueError(
            "event_name must be a MATLAB identifier (at most 63 characters)."
        )
    path = Path(basepath)
    return path / f"{path.name}.{event_name}.events.mat"


def save_delta_wave_events(
    events: pd.DataFrame,
    basepath: str | Path,
    event_name: str = "deltaWaves",
) -> str:
    """Save delta waves in CellExplorer format, refusing to overwrite any file.

    Parameters
    ----------
    events : pandas.DataFrame
        Table returned by :func:`detect_delta_waves`, including detector metadata
        in ``events.attrs['detectorinfo']``.
    basepath : str or pathlib.Path
        Existing session directory.
    event_name : str, optional
        MATLAB variable and filename component; defaults to ``deltaWaves``.

    Returns
    -------
    str
        Saved filename.

    Raises
    ------
    FileExistsError
        The destination exists. Choose another event_name to retain both files.
    """
    path = _event_path(basepath, event_name)
    data = {
        "timestamps": events[["start", "stop"]].to_numpy(dtype=float),
        "amplitudeUnits": "zscore",
        "eventIDlabels": np.array(["delta"], dtype=object),
        "eventIDbinary": False,
        "detectorinfo": events.attrs["detectorinfo"],
        "values": events[["start_value", "peak_value", "stop_value"]].to_numpy(
            dtype=float
        ),
    }
    for field in ("peaks", "amplitude", "center", "duration", "eventID"):
        data[field] = events[field].to_numpy().reshape(-1, 1)
    # Serialize before creating the destination so serialization errors leave no
    # partial event file. Exclusive creation also closes the preflight race.
    payload = BytesIO()
    savemat(payload, {event_name: data}, long_field_names=True)
    with path.open("xb") as stream:
        stream.write(payload.getbuffer())
    return str(path)


def detect_delta_waves(
    basepath: str | Path | None = None,
    *,
    lfp: ArrayLike | None = None,
    fs: float | None = None,
    timestamps: ArrayLike | None = None,
    channel: int | Sequence[int] | None = None,
    brain_regions: str | Sequence[str] = _REGIONS,
    detection_epochs: nel.EpochArray | ArrayLike | None = None,
    nrem_restrict: bool = True,
    emg_threshold: float = 0.6,
    ignore_intervals: nel.EpochArray | ArrayLike | None = None,
    threshold: float = 3.0,
    low_peak_threshold: float = 1.0,
    high_peak_threshold: float = 3.0,
    low_trough_threshold: float = 0.0,
    high_trough_threshold: float = 1.5,
    min_duration: float = 0.15,
    max_duration: float = 0.65,
    filter_band: tuple[float, float] = (0.5, 9.0),
    passband: tuple[float, float] = (1.0, 6.0),
    reject_artifacts: bool = True,
    edge_buffer: float = 1.0,
    verify_firing: bool = False,
    spike_times: Sequence[ArrayLike] | None = None,
    save_mat: bool = True,
    event_name: str = "deltaWaves",
) -> pd.DataFrame:
    """Detect cortical delta waves and optionally save a CellExplorer event file.

    Parameters
    ----------
    basepath : str or pathlib.Path, optional
        Session directory. Required for file-backed LFP, NREM/EMG metadata, or saving.
    lfp : array_like, optional
        Single-channel LFP, shape (n_samples,), in arbitrary consistent voltage units.
        Otherwise memory-map the session int16 .lfp (or .eeg) using session metadata.
    fs : float, optional
        Sampling rate in Hz. Required with lfp; otherwise read srLfp/srLFP from session.
    timestamps : array_like, optional
        Strictly increasing sample timestamps in seconds. Defaults to arange(n)/fs.
        Gaps are allowed, but sampling within each segment must match fs.
    channel : int or sequence of int, optional
        Zero-indexed detection channel, or candidates ranked by relative delta power.
        Defaults to a Delta channel tag, then matching brain-region channels. Bad
        channels are excluded. Metadata channel numbers are converted from MATLAB.
    brain_regions : str or sequence of str, optional
        Region substrings for automatic channel selection and optional cortical spikes.
    detection_epochs : nel.EpochArray or array_like, optional
        Detection intervals in seconds, intersected with NREM and low EMG if enabled.
        Empty intervals explicitly request no detection.
    nrem_restrict : bool, optional
        Restrict to SleepState.ints.NREMstate (default True). Requires basepath.
    emg_threshold : float, optional
        Keep EMG < 0.6 for at least one second. Set np.inf to disable; otherwise
        requires basepath and an EMGFromLFP.LFP.mat file.
    ignore_intervals : nel.EpochArray or array_like, optional
        Excluded intervals in seconds. Events cannot cross these intervals.
    threshold : float, optional
        Strict minimum peak-to-final-trough amplitude in z units (default 3).
    low_peak_threshold : float, optional
        Minimum peak in z units when the final trough is below
        -high_trough_threshold. Inclusive; default 1.0.
    high_peak_threshold : float, optional
        Minimum peak in z units when the final trough is at or below
        -low_trough_threshold. Strict; default 3.0.
    low_trough_threshold : float, optional
        Minimum absolute negative final-trough amplitude in z units for the
        high-peak condition. Inclusive; default 0.0.
    high_trough_threshold : float, optional
        Minimum absolute negative final-trough amplitude in z units for the
        low-peak condition. Strict; default 1.5. Both peak and trough threshold
        pairs must be nonnegative and ordered low <= high.
    min_duration : float, optional
        Inclusive minimum trough-to-trough duration in seconds; default 0.15.
    max_duration : float, optional
        Inclusive maximum trough-to-trough duration in seconds; default 0.65.
        Must exceed min_duration.
    filter_band : tuple of float, optional
        Morphology filter cutoffs in Hz (0.5, 9). A fifth-order Chebyshev-II lowpass
        is subtracted for highpassing, followed by lowpassing, as in FilterLFP.
    passband : tuple of float, optional
        Band for automatic relative-power channel selection only (1, 6 Hz).
    reject_artifacts : bool, optional
        Exclude raw amplitudes > 6 SD and raw differences > 10 SD of the signal,
        padded by 0.5 and 0.1 seconds respectively. Default True.
    edge_buffer : float, optional
        Exclude one second at each filtered segment edge to limit filter transients.
    verify_firing : bool, optional
        Optionally raise the amplitude threshold using cortical firing suppression.
        Default False, matching the MATLAB wrapper's documented default. Uses a
        population PETH (+/-0.5 s, 10 ms bins, Gaussian sigma one bin), per-event
        z scores within +/-50 ms, and a 100-event amplitude-ordered moving mean
        <= -0.5. This is an explicit heuristic, not a statistical significance test.
    spike_times : sequence of array_like, optional
        Spike timestamps in seconds per cortical unit for firing verification.
        Otherwise load cortical cell metrics/spikes from basepath when requested.
    save_mat : bool, optional
        Save a CellExplorer event file by default. Requires basepath. Existing
        files always raise FileExistsError; use a distinct event_name to compare runs.
    event_name : str, optional
        Event struct and filename component, default deltaWaves.

    Returns
    -------
    pandas.DataFrame
        One row per event with start, stop, timestamps, peaks, amplitude (zscore),
        amplitudeUnits, values (start/peak/final trough z scores), eventID,
        eventIDlabels, eventIDbinary, center and duration (seconds), plus individual
        start_value, peak_value, stop_value columns. Shared detectorinfo is in attrs,
        including exact usable intervals, parameters, and both channel conventions.
        attrs['event_file'] contains the output path when saved. Empty results retain
        the same columns and metadata.

    Notes
    -----
    Based on neurocode detect_delta_waves.m / FindDeltaWaves.m. Derivative extrema
    use an eighth-order 9 Hz Chebyshev-II lowpass and sample-SD normalization.
    Unlike the MATLAB implementation, filtering stays at the native sampling rate,
    uses stable second-order sections, and never interpolates across recording gaps,
    ignored epochs, NaNs, or artifacts. Z scores are pooled across usable segments.
    Missing required metadata raises an actionable error rather than silently
    dropping a requested restriction. No interactive channel picker is launched.

    Examples
    --------
    >>> events = detect_delta_waves(basepath, event_name="deltaWavesPython")
    >>> events = detect_delta_waves(lfp=trace, fs=1250, nrem_restrict=False,
    ...                             emg_threshold=np.inf, save_mat=False)
    """
    if save_mat:
        if basepath is None:
            raise ValueError("basepath is required when save_mat=True.")
        destination = _event_path(basepath, event_name)
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite {destination}; choose another event_name."
            )
    if (nrem_restrict or emg_threshold < np.inf) and basepath is None:
        raise ValueError(
            "NREM/EMG restrictions require basepath; disable them for array-only detection."
        )
    if np.isnan(emg_threshold) or emg_threshold == -np.inf:
        raise ValueError("emg_threshold must be finite or positive infinity.")
    if isinstance(brain_regions, str):
        brain_regions = (brain_regions,)
    session = {}
    if lfp is None:
        if basepath is None:
            raise ValueError("Provide lfp or basepath.")
        path = Path(basepath)
        session = load_mat(path / f"{path.name}.session.mat")["session"]
        extracellular = session["extracellular"]
        if fs is None:
            fs = float(extracellular.get("srLfp", extracellular.get("srLFP", np.nan)))
        n_channels = int(extracellular["nChannels"])
        candidates = _channel_candidates(session, channel, brain_regions, n_channels)
        loaded = loading.loadLFP(str(path), n_channels=n_channels, frequency=fs)
        if loaded is None:
            raise FileNotFoundError(f"No .lfp or .eeg file in {path}.")
        data, sample_times = loaded
        if timestamps is not None:
            raise ValueError("timestamps may only be supplied with lfp.")
    else:
        data = np.asarray(lfp)
        if data.ndim != 1:
            raise ValueError("lfp must be a one-dimensional single-channel signal.")
        sample_times = (
            np.asarray(timestamps, dtype=float) if timestamps is not None else None
        )
        if channel is not None and (
            not isinstance(channel, (int, np.integer)) or channel < 0
        ):
            raise ValueError(
                "Array input requires a single nonnegative integer channel label."
            )
        candidates = [int(channel)] if channel is not None else []
    if fs is None or not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be a finite positive sampling rate in Hz.")
    for name, band in (("filter_band", filter_band), ("passband", passband)):
        if len(band) != 2 or not (0 < band[0] < band[1] < fs / 2):
            raise ValueError(f"{name} must satisfy 0 < low < high < fs/2.")
    for name, value in (
        ("low_peak_threshold", low_peak_threshold),
        ("high_peak_threshold", high_peak_threshold),
        ("low_trough_threshold", low_trough_threshold),
        ("high_trough_threshold", high_trough_threshold),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if low_peak_threshold > high_peak_threshold:
        raise ValueError("low_peak_threshold must be <= high_peak_threshold.")
    if low_trough_threshold > high_trough_threshold:
        raise ValueError("low_trough_threshold must be <= high_trough_threshold.")
    for name, value in (("min_duration", min_duration), ("max_duration", max_duration)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, in seconds.")
    if min_duration >= max_duration:
        raise ValueError("min_duration must be less than max_duration.")
    if (
        not np.isfinite(threshold)
        or threshold < 0
        or not np.isfinite(edge_buffer)
        or edge_buffer < 0
    ):
        raise ValueError("threshold and edge_buffer must be finite and nonnegative.")
    times = (
        np.arange(data.shape[0], dtype=float) / fs
        if sample_times is None
        else np.asarray(sample_times, dtype=float)
    )
    dt = np.diff(times)
    if (
        times.ndim != 1
        or times.size != data.shape[0]
        or not np.isfinite(times).all()
        or np.any(dt <= 0)
        or np.any((dt <= 1.5 / fs) & ~np.isclose(dt, 1 / fs, rtol=1e-3, atol=1e-9))
    ):
        raise ValueError(
            "timestamps must be finite, increasing, match lfp length and fs (gaps allowed)."
        )
    epochs = (
        _epochs([[times[0], times[-1]]])
        if times.size > 1
        else nel.EpochArray(empty=True)
    )
    if detection_epochs is not None:
        epochs = epochs & _epochs(detection_epochs)
    if nrem_restrict:
        path = Path(str(basepath))
        sleep = load_mat(path / f"{path.name}.SleepState.states.mat")["SleepState"]
        epochs = epochs & _epochs(sleep["ints"]["NREMstate"])
    if emg_threshold < np.inf:
        _, _, low_emg = loading.load_emg(str(basepath), threshold=emg_threshold)
        epochs = epochs & low_emg[low_emg.durations >= 1.0]
    if ignore_intervals is not None:
        ignored = _epochs(ignore_intervals)
        if not ignored.isempty:
            epochs = epochs & ~ignored
    segments = _segments(times, epochs, fs)
    detection_channel = candidates[0] if candidates else None
    if data.ndim == 2:
        if segments:
            detection_channel = _select_channel(
                data, candidates, segments, fs, passband
            )
        data = data[:, detection_channel]
    cleaned = _clean_segments(data, segments, fs, reject_artifacts)
    events, intervals = _find_waves(
        cleaned,
        times,
        fs,
        filter_band,
        threshold,
        edge_buffer,
        low_peak_threshold=low_peak_threshold,
        high_peak_threshold=high_peak_threshold,
        low_trough_threshold=low_trough_threshold,
        high_trough_threshold=high_trough_threshold,
        min_duration=min_duration,
        max_duration=max_duration,
    )
    effective_threshold, firing_verified = threshold, False
    if verify_firing:
        if spike_times is None and basepath is not None:
            spikes, _ = loading.load_spikes(
                str(basepath), brainRegion=list(brain_regions)
            )
            spike_times = [] if spikes is None or spikes.isempty else list(spikes.data)
        events, effective_threshold, firing_verified = _verify_firing(
            events, spike_times if spike_times is not None else [], threshold
        )
    params = {
        "channel": candidates,
        "threshold": threshold,
        "low_peak_threshold": low_peak_threshold,
        "high_peak_threshold": high_peak_threshold,
        "low_trough_threshold": low_trough_threshold,
        "high_trough_threshold": high_trough_threshold,
        "min_duration": min_duration,
        "max_duration": max_duration,
        "filter_band": filter_band,
        "passband": passband,
        "fs": fs,
        "nrem_restrict": nrem_restrict,
        "emg_threshold": emg_threshold,
        "reject_artifacts": reject_artifacts,
        "edge_buffer": edge_buffer,
        "brain_regions": list(brain_regions),
        "verify_firing": verify_firing,
        "firing_verified": firing_verified,
        "ignore_intervals": _epochs(ignore_intervals).data
        if ignore_intervals is not None
        else np.empty((0, 2)),
    }
    info = {
        "detectorname": "neuro_py.detectors.detect_delta_waves",
        "detectiondate": datetime.now(timezone.utc).isoformat(),
        "detectionintervals": intervals,
        "detectionparms": params,
        "detectionchannel": [] if detection_channel is None else detection_channel,
        "detectionchannel1": [] if detection_channel is None else detection_channel + 1,
        "threshold": effective_threshold,
    }
    events["timestamps"] = list(map(tuple, events[["start", "stop"]].to_numpy()))
    events["values"] = list(
        map(tuple, events[["start_value", "peak_value", "stop_value"]].to_numpy())
    )
    events["amplitudeUnits"] = "zscore"
    events["eventID"] = np.ones(len(events), dtype=np.int64)
    events["eventIDlabels"] = "delta"
    events["eventIDbinary"] = False
    events.attrs["detectorinfo"] = info
    if save_mat:
        events.attrs["event_file"] = save_delta_wave_events(
            events, str(basepath), event_name
        )
    return events
