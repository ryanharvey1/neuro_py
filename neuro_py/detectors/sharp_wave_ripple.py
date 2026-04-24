from __future__ import annotations

import os
from datetime import datetime
from typing import Optional, Union

import nelpy as nel
import numpy as np
import pandas as pd
from scipy import ndimage, signal
from scipy.io import savemat

from neuro_py.io import loading
from neuro_py.process.intervals import find_interval, in_intervals


def _sanitize_for_matlab(value: object) -> object:
    """Convert Python-only values into MATLAB-writeable equivalents."""
    if value is None:
        return []
    if isinstance(value, dict):
        return {key: _sanitize_for_matlab(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_matlab(item) for item in value]
    return value


def _zscore(values: np.ndarray) -> np.ndarray:
    """Z-score a 1-D array while preserving NaNs."""
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if np.isnan(std) or std == 0:
        return np.zeros_like(values, dtype=float)
    return (values - mean) / std


def _coerce_interval_array(
    detection_epochs: Optional[Union[nel.EpochArray, np.ndarray]],
) -> np.ndarray:
    """Normalize interval inputs to an ``(n_intervals, 2)`` array."""
    if detection_epochs is None:
        return np.empty((0, 2), dtype=float)

    if isinstance(detection_epochs, nel.EpochArray):
        return np.asarray(detection_epochs.data, dtype=float)

    intervals = np.asarray(detection_epochs, dtype=float)
    if intervals.ndim == 1:
        if intervals.size != 2:
            raise ValueError(
                "1-D detection_epochs inputs must have exactly two values: [start, stop]."
            )
        intervals = intervals[np.newaxis, :]

    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError(
            "detection_epochs must be an EpochArray or an array with shape (n_intervals, 2)."
        )
    return intervals


def _get_ripple_channel(basepath: str) -> int:
    """Infer the ripple channel from CellExplorer channel tags."""
    channel_tags = loading.load_channel_tags(basepath)
    for key in ("ripple", "Ripple", "CA1sp", "ca1sp"):
        if key in channel_tags:
            channels = np.atleast_1d(channel_tags[key]["channels"]).astype(int) - 1
            return int(channels[0])

    raise ValueError(
        "Could not infer a ripple channel from session channel tags. "
        "Pass `ripple_channel` explicitly."
    )


def _get_sharp_wave_channel(basepath: str) -> Optional[int]:
    """Infer the sharp-wave channel from CellExplorer channel tags."""
    channel_tags = loading.load_channel_tags(basepath)
    for key in ("SharpWave", "sharpwave", "sharp_wave", "CA1sr", "ca1sr"):
        if key in channel_tags:
            channels = np.atleast_1d(channel_tags[key]["channels"]).astype(int) - 1
            return int(channels[0])
    return None


def _get_noise_channel(basepath: str) -> Optional[int]:
    """Infer a noise channel from CellExplorer ``Bad`` channel tags."""
    channel_tags = loading.load_channel_tags(basepath)
    for key in ("Bad", "bad"):
        if key in channel_tags:
            channels = np.atleast_1d(channel_tags[key]["channels"]).astype(int) - 1
            if channels.size > 0:
                return int(channels[0])
    return None


def _load_signals(
    basepath: str,
    ripple_channel: Optional[int],
    sharp_wave_channel: Optional[int],
    noise_channel: Optional[int],
    detection_epochs: Optional[Union[nel.EpochArray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, float, int, dict[str, Optional[np.ndarray]]]:
    """Load one or more LFP channels for ripple detection."""
    if ripple_channel is None:
        ripple_channel = _get_ripple_channel(basepath)
    if sharp_wave_channel is None:
        sharp_wave_channel = _get_sharp_wave_channel(basepath)
    if noise_channel is None:
        noise_channel = _get_noise_channel(basepath)

    channels = [ripple_channel]
    channel_map = {"ripple_signal": 0}

    if sharp_wave_channel is not None:
        channel_map["sharp_wave_signal"] = len(channels)
        channels.append(sharp_wave_channel)

    if noise_channel is not None:
        channel_map["noise_signal"] = len(channels)
        channels.append(noise_channel)

    lfp = loading.LFPLoader(
        basepath=basepath,
        channels=channels,
        ext="lfp",
        epoch=detection_epochs,
    )
    signals = {
        name: np.asarray(lfp.data[idx], dtype=float)
        for name, idx in channel_map.items()
    }
    return (
        signals["ripple_signal"],
        np.asarray(lfp.abscissa_vals, dtype=float),
        float(lfp.fs),
        int(ripple_channel),
        {
            "sharp_wave_signal": signals.get("sharp_wave_signal"),
            "noise_signal": signals.get("noise_signal"),
        },
    )


def _compute_envelope(
    signal_in: np.ndarray,
    fs: float,
    ripple_band: tuple[float, float],
    smooth_sigma: float,
    filter_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return filtered ripple-band signal, smoothed envelope, and instantaneous frequency."""
    sos = signal.butter(
        filter_order,
        np.asarray(ripple_band, dtype=float),
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    filtered = signal.sosfiltfilt(sos, np.asarray(signal_in, dtype=float))
    analytic = signal.hilbert(filtered)
    envelope = np.abs(analytic)
    sigma_samples = smooth_sigma * fs
    if sigma_samples > 0:
        envelope = ndimage.gaussian_filter1d(
            envelope,
            sigma=sigma_samples,
            mode="nearest",
        )
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.gradient(phase) * fs / (2 * np.pi)
    return filtered, envelope, inst_freq


def _filter_signal(
    signal_in: np.ndarray,
    fs: float,
    passband: tuple[float, float],
    filter_order: int,
) -> np.ndarray:
    """Apply a zero-phase band-pass filter and return the filtered trace."""
    sos = signal.butter(
        filter_order,
        np.asarray(passband, dtype=float),
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    return signal.sosfiltfilt(sos, np.asarray(signal_in, dtype=float))


def _compute_sharp_wave_difference(
    ripple_signal: np.ndarray,
    sharp_wave_signal: np.ndarray,
    fs: float,
    sharp_wave_band: tuple[float, float],
    filter_order: int,
    smooth_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a filtered sharp-wave difference trace and its z-scored feature."""
    ripple_low = _filter_signal(
        signal_in=ripple_signal,
        fs=fs,
        passband=sharp_wave_band,
        filter_order=filter_order,
    )
    sharp_low = _filter_signal(
        signal_in=sharp_wave_signal,
        fs=fs,
        passband=sharp_wave_band,
        filter_order=filter_order,
    )
    sharp_wave_diff = ripple_low - sharp_low
    sigma_samples = smooth_sigma * fs
    if sigma_samples > 0:
        sharp_wave_diff = ndimage.gaussian_filter1d(
            sharp_wave_diff,
            sigma=sigma_samples,
            mode="nearest",
        )
    return sharp_wave_diff, _zscore(sharp_wave_diff)


def _merge_bounds(bounds: list[tuple[int, int]], gap_samples: int) -> list[tuple[int, int]]:
    """Merge intervals separated by fewer than ``gap_samples`` samples."""
    if not bounds:
        return []

    merged = [bounds[0]]
    for start, stop in bounds[1:]:
        last_start, last_stop = merged[-1]
        if start - last_stop <= gap_samples:
            merged[-1] = (last_start, max(last_stop, stop))
        else:
            merged.append((start, stop))
    return merged


def _bound_containing_index(
    bounds: list[tuple[int, int]],
    index: int,
) -> Optional[tuple[int, int]]:
    """Return the interval that contains ``index``."""
    for start, stop in bounds:
        if start <= index <= stop:
            return (start, stop)
    return None


def _nearest_trough(
    filtered_signal: np.ndarray,
    center_idx: int,
    fs: float,
    window: float = 0.010,
) -> int:
    """Return the index of the nearest ripple trough around ``center_idx``."""
    radius = max(1, int(round(window * fs)))
    start = max(0, center_idx - radius)
    stop = min(filtered_signal.size, center_idx + radius + 1)
    return int(start + np.argmin(filtered_signal[start:stop]))


def _events_to_dataframe(
    ripple_bounds: list[tuple[int, int]],
    ripple_power: np.ndarray,
    ripple_envelope: np.ndarray,
    ripple_filtered: np.ndarray,
    inst_freq: np.ndarray,
    timestamps: np.ndarray,
    fs: float,
    ripple_min_duration: float,
    ripple_max_duration: float,
    ripple_high_threshold: float,
    ripple_channel: Optional[int],
    sharp_wave_bounds: Optional[list[tuple[int, int]]] = None,
    sharp_wave_power: Optional[np.ndarray] = None,
    sharp_wave_trace: Optional[np.ndarray] = None,
    sharp_wave_high_threshold: Optional[float] = None,
    sharp_wave_min_duration: Optional[float] = None,
    sharp_wave_max_duration: Optional[float] = None,
    search_window: float = 0.050,
    boundary_mode: str = "sharp_wave",
    noise_power: Optional[np.ndarray] = None,
    noise_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Convert joint ripple and sharp-wave detections into a CellExplorer-style table."""
    records: list[dict[str, float]] = []
    dt = 1.0 / fs
    search_radius = max(1, int(round(search_window * fs)))

    for ripple_start, ripple_stop in ripple_bounds:
        segment = ripple_power[ripple_start : ripple_stop + 1]
        if segment.size == 0:
            continue

        ripple_peak_idx = int(ripple_start + np.nanargmax(segment))
        ripple_peak_power = float(ripple_power[ripple_peak_idx])
        if ripple_peak_power < ripple_high_threshold:
            continue

        ripple_start_time = float(timestamps[ripple_start])
        ripple_stop_time = float(timestamps[ripple_stop] + dt)
        ripple_duration = ripple_stop_time - ripple_start_time
        if ripple_duration < ripple_min_duration or ripple_duration > ripple_max_duration:
            continue

        sharp_wave_peak_idx = None
        sharp_wave_peak_power = np.nan
        sharp_wave_duration = np.nan
        event_start_idx, event_stop_idx = ripple_start, ripple_stop

        if sharp_wave_power is not None:
            if sharp_wave_bounds is None:
                continue

            window_start = max(0, ripple_peak_idx - search_radius)
            window_stop = min(sharp_wave_power.size, ripple_peak_idx + search_radius + 1)
            local_sharp_wave = sharp_wave_power[window_start:window_stop]
            if local_sharp_wave.size == 0:
                continue

            sharp_wave_peak_idx = int(window_start + np.nanargmax(local_sharp_wave))
            sharp_wave_peak_power = float(sharp_wave_power[sharp_wave_peak_idx])
            if (
                sharp_wave_high_threshold is not None
                and sharp_wave_peak_power < sharp_wave_high_threshold
            ):
                continue

            sharp_wave_interval = _bound_containing_index(sharp_wave_bounds, sharp_wave_peak_idx)
            if sharp_wave_interval is None:
                continue

            sharp_wave_start, sharp_wave_stop = sharp_wave_interval
            sharp_wave_duration = float(
                (timestamps[sharp_wave_stop] + dt) - timestamps[sharp_wave_start]
            )
            if (
                sharp_wave_min_duration is not None
                and sharp_wave_duration < sharp_wave_min_duration
            ):
                continue
            if (
                sharp_wave_max_duration is not None
                and sharp_wave_duration > sharp_wave_max_duration
            ):
                continue

            if boundary_mode == "sharp_wave":
                event_start_idx, event_stop_idx = sharp_wave_start, sharp_wave_stop
            elif boundary_mode == "union":
                event_start_idx = min(ripple_start, sharp_wave_start)
                event_stop_idx = max(ripple_stop, sharp_wave_stop)
            else:
                raise ValueError("`boundary_mode` must be either 'sharp_wave' or 'union'.")

        event_peak_idx = _nearest_trough(
            filtered_signal=ripple_filtered,
            center_idx=ripple_peak_idx,
            fs=fs,
        )

        if noise_power is not None and noise_threshold is not None:
            noise_peak = float(np.nanmax(noise_power[event_start_idx : event_stop_idx + 1]))
            if noise_peak >= noise_threshold:
                continue
        else:
            noise_peak = np.nan

        start = float(timestamps[event_start_idx])
        stop = float(timestamps[event_stop_idx] + dt)
        duration = stop - start

        record = {
            "start": start,
            "stop": stop,
            "peaks": float(timestamps[event_peak_idx]),
            "center": float(start + (duration / 2.0)),
            "duration": float(duration),
            "amplitude": float(ripple_envelope[event_peak_idx]),
            "frequency": float(np.nanmedian(inst_freq[event_start_idx : event_stop_idx + 1])),
            "peakNormedPower": ripple_peak_power,
            "ripple_channel": np.nan if ripple_channel is None else int(ripple_channel),
            "noise_peakNormedPower": noise_peak,
            "ripple_duration": float(ripple_duration),
            "sharp_wave_peakNormedPower": sharp_wave_peak_power,
        }
        if sharp_wave_trace is not None:
            peak_index = sharp_wave_peak_idx if sharp_wave_peak_idx is not None else event_peak_idx
            record["sharp_wave_amplitude"] = float(sharp_wave_trace[peak_index])
            record["sharp_wave_duration"] = sharp_wave_duration
        records.append(record)

    if not records:
        return pd.DataFrame(
            columns=[
                "start",
                "stop",
                "peaks",
                "center",
                "duration",
                "amplitude",
                "frequency",
                "peakNormedPower",
                "ripple_channel",
                "noise_peakNormedPower",
                "ripple_duration",
                "sharp_wave_peakNormedPower",
            ]
        )

    return (
        pd.DataFrame.from_records(records)
        .drop_duplicates(subset=["start", "stop", "peaks"])
        .sort_values("start")
        .reset_index(drop=True)
    )


def save_ripple_events(
    events: pd.DataFrame,
    basepath: str,
    detection_name: str = "detect_sharp_wave_ripples",
    detection_params: Optional[dict] = None,
    ripple_channel: Optional[int] = None,
    detection_epochs: Optional[Union[nel.EpochArray, np.ndarray]] = None,
    event_name: str = "ripples",
    amplitude_units: str = "a.u.",
) -> str:
    """
    Save ripple events to a CellExplorer ``*.events.mat`` file.

    Parameters
    ----------
    events : pd.DataFrame
        Ripple event table returned by :func:`detect_sharp_wave_ripples`.
    basepath : str
        Session folder where the event file will be written.
    detection_name : str, optional
        Name stored in ``detectorinfo.detectorname``.
    detection_params : dict, optional
        Detection parameters stored in ``detectorinfo.detectionparms``.
    ripple_channel : int, optional
        Zero-indexed ripple detection channel.
    detection_epochs : nel.EpochArray or np.ndarray, optional
        Detection intervals in seconds.
    event_name : str, optional
        Name of the CellExplorer event struct.
    amplitude_units : str, optional
        Units for the saved ripple amplitude.

    Returns
    -------
    str
        Path to the saved ``*.events.mat`` file.
    """
    filename = os.path.join(
        basepath,
        os.path.basename(basepath) + f".{event_name}.events.mat",
    )

    intervals = _coerce_interval_array(detection_epochs)

    if events.empty:
        timestamps = np.empty((0, 2), dtype=float)
        peaks = np.empty((0,), dtype=float)
        amplitude = np.empty((0,), dtype=float)
        duration = np.empty((0,), dtype=float)
        center = np.empty((0,), dtype=float)
        frequency = np.empty((0,), dtype=float)
        peak_normed_power = np.empty((0,), dtype=float)
    else:
        timestamps = events[["start", "stop"]].to_numpy(dtype=float)
        peaks = events["peaks"].to_numpy(dtype=float)
        amplitude = events["amplitude"].to_numpy(dtype=float)
        duration = events["duration"].to_numpy(dtype=float)
        center = events["center"].to_numpy(dtype=float)
        frequency = events["frequency"].to_numpy(dtype=float)
        peak_normed_power = events["peakNormedPower"].to_numpy(dtype=float)

    detectorinfo: dict[str, object] = {
        "detectorname": detection_name,
        "detectiondate": datetime.now().strftime("%Y-%m-%d"),
        "detectionintervals": intervals,
        "detectionparms": detection_params if detection_params is not None else {},
    }
    if ripple_channel is not None:
        detectorinfo["detectionchannel"] = int(ripple_channel)
        detectorinfo["detectionchannel1"] = int(ripple_channel) + 1

    data = {
        event_name: {
            "timestamps": timestamps,
            "peaks": peaks,
            "amplitude": amplitude,
            "amplitudeUnits": amplitude_units,
            "eventID": [],
            "eventIDlabels": [],
            "eventIDbinary": [],
            "center": center,
            "duration": duration,
            "frequency": frequency,
            "peakNormedPower": peak_normed_power,
            "detectorinfo": detectorinfo,
        }
    }
    savemat(filename, _sanitize_for_matlab(data), long_field_names=True)
    return filename


def detect_sharp_wave_ripples(
    basepath: Optional[str] = None,
    ripple_signal: Optional[np.ndarray] = None,
    fs: Optional[float] = None,
    timestamps: Optional[np.ndarray] = None,
    ripple_channel: Optional[int] = None,
    sharp_wave_signal: Optional[np.ndarray] = None,
    sharp_wave_channel: Optional[int] = None,
    noise_signal: Optional[np.ndarray] = None,
    noise_channel: Optional[int] = None,
    detection_epochs: Optional[Union[nel.EpochArray, np.ndarray]] = None,
    ripple_band: tuple[float, float] = (120.0, 250.0),
    sharp_wave_band: tuple[float, float] = (2.0, 50.0),
    smooth_sigma: float = 0.004,
    sharp_wave_smooth_sigma: float = 0.0,
    low_threshold: float = 2.0,
    high_threshold: float = 5.0,
    sharp_wave_low_threshold: float = 0.5,
    sharp_wave_high_threshold: float = 2.5,
    noise_threshold: Optional[float] = None,
    min_duration: float = 0.015,
    max_duration: float = 0.150,
    sharp_wave_min_duration: float = 0.020,
    sharp_wave_max_duration: float = 0.500,
    merge_gap: float = 0.020,
    peak_window: float = 0.050,
    boundary_mode: str = "sharp_wave",
    filter_order: int = 4,
    save_mat: bool = False,
    overwrite: bool = False,
    return_epoch_array: bool = False,
    event_name: str = "ripples",
) -> Union[pd.DataFrame, nel.EpochArray]:
    """
    Detect sharp wave ripple events from a ripple-band LFP channel.

    The detector follows a compact joint SWR workflow: detect candidate ripple
    intervals from ripple-band power, require a nearby sharp-wave event on a
    companion low-frequency channel difference, and then validate ripple and
    sharp-wave durations separately before returning final events.

    Parameters
    ----------
    basepath : str, optional
        Session folder used to load LFP data and optionally save a CellExplorer
        event file. Required when ``save_mat=True`` or when ``ripple_signal`` is
        not provided.
    ripple_signal : np.ndarray, optional
        One-dimensional ripple detection signal. If omitted, the signal is loaded
        from ``basepath`` using ``ripple_channel``.
    fs : float, optional
        Sampling rate in Hz. Required when ``ripple_signal`` is provided.
    timestamps : np.ndarray, optional
        Timestamps in seconds for ``ripple_signal``. If omitted, they are inferred
        from ``fs``.
    ripple_channel : int, optional
        Zero-indexed ripple channel. When omitted during file-backed detection, the
        detector tries to infer it from CellExplorer channel tags.
    sharp_wave_signal : np.ndarray, optional
        Optional companion signal used to measure low-frequency sharp-wave
        amplitude at ripple peaks.
    sharp_wave_channel : int, optional
        Zero-indexed sharp-wave channel used when loading from ``basepath``.
    noise_signal : np.ndarray, optional
        Optional ripple-band noise channel. Events are rejected when the noise-band
        power within the event exceeds ``noise_threshold``.
    noise_channel : int, optional
        Zero-indexed noise channel used when loading from ``basepath``.
    detection_epochs : nel.EpochArray or np.ndarray, optional
        Detection intervals in seconds. File-backed detection uses these intervals
        to restrict LFP loading; in-memory detection filters final events to these
        intervals.
    ripple_band : tuple of float, optional
        Ripple passband in Hz.
    sharp_wave_band : tuple of float, optional
        Sharp-wave passband in Hz used for the low-frequency difference signal.
    smooth_sigma : float, optional
        Gaussian smoothing width for the ripple envelope, in seconds.
    sharp_wave_smooth_sigma : float, optional
        Optional Gaussian smoothing width for the sharp-wave difference signal,
        in seconds.
    low_threshold : float, optional
        Lower z-scored ripple-envelope threshold used to mark candidate ripple
        boundaries.
    high_threshold : float, optional
        Peak z-scored ripple-envelope threshold required to accept an event.
    sharp_wave_low_threshold : float, optional
        Lower z-scored sharp-wave threshold used to mark candidate sharp-wave
        boundaries.
    sharp_wave_high_threshold : float, optional
        Peak z-scored sharp-wave threshold required to accept an event.
    noise_threshold : float, optional
        Maximum tolerated z-scored noise envelope for accepted events.
    min_duration : float, optional
        Minimum ripple duration in seconds.
    max_duration : float, optional
        Maximum ripple duration in seconds.
    sharp_wave_min_duration : float, optional
        Minimum sharp-wave duration in seconds.
    sharp_wave_max_duration : float, optional
        Maximum sharp-wave duration in seconds.
    merge_gap : float, optional
        Merge candidate events separated by less than this gap, in seconds.
    peak_window : float, optional
        Search window around the ripple peak, in seconds, used to find the
        associated sharp-wave peak.
    boundary_mode : {"sharp_wave", "union"}, optional
        Whether final event boundaries follow the sharp-wave interval or the
        union of ripple and sharp-wave intervals.
    filter_order : int, optional
        Butterworth filter order for ripple-band filtering.
    save_mat : bool, optional
        If True, save a CellExplorer event file to ``basepath``.
    overwrite : bool, optional
        If False and the target event file already exists, load and return the
        existing file instead of redetecting.
    return_epoch_array : bool, optional
        If True, return the detected events as a ``nel.EpochArray``.
    event_name : str, optional
        Name of the CellExplorer event struct written to disk.

    Returns
    -------
    pandas.DataFrame or nel.EpochArray
        Detected ripple events.
    """
    if basepath is None and ripple_signal is None:
        raise ValueError("Provide either `basepath` or `ripple_signal` for detection.")

    if save_mat and basepath is None:
        raise ValueError("`basepath` is required when `save_mat=True`.")

    if save_mat and basepath is not None and not overwrite:
        event_path = os.path.join(
            basepath,
            os.path.basename(basepath) + f".{event_name}.events.mat",
        )
        if os.path.exists(event_path):
            if event_name == "ripples":
                existing = loading.load_ripples_events(basepath)
            else:
                existing = loading.load_events(basepath, event_name, load_pandas=True)
                if existing is None:
                    existing = pd.DataFrame()
                else:
                    existing = existing.rename(columns={"starts": "start", "stops": "stop"})
            if return_epoch_array:
                if existing.empty:
                    return nel.EpochArray(np.empty((0, 2), dtype=float))
                return nel.EpochArray(existing[["start", "stop"]].to_numpy(dtype=float))
            return existing

    if ripple_signal is None:
        ripple_signal, timestamps, fs, ripple_channel, loaded_signals = _load_signals(
            basepath=basepath,
            ripple_channel=ripple_channel,
            sharp_wave_channel=sharp_wave_channel,
            noise_channel=noise_channel,
            detection_epochs=detection_epochs,
        )
        sharp_wave_signal = loaded_signals["sharp_wave_signal"]
        noise_signal = loaded_signals["noise_signal"]
    else:
        if fs is None:
            raise ValueError("`fs` is required when `ripple_signal` is provided.")
        ripple_signal = np.asarray(ripple_signal, dtype=float)
        if ripple_signal.ndim != 1:
            raise ValueError("`ripple_signal` must be one-dimensional.")
        if timestamps is None:
            timestamps = np.arange(ripple_signal.size, dtype=float) / float(fs)
        else:
            timestamps = np.asarray(timestamps, dtype=float)
        if timestamps.shape[0] != ripple_signal.shape[0]:
            raise ValueError("`timestamps` must have the same length as `ripple_signal`.")
        if sharp_wave_signal is not None:
            sharp_wave_signal = np.asarray(sharp_wave_signal, dtype=float)
        if noise_signal is not None:
            noise_signal = np.asarray(noise_signal, dtype=float)

    ripple_filtered, envelope, inst_freq = _compute_envelope(
        signal_in=ripple_signal,
        fs=float(fs),
        ripple_band=ripple_band,
        smooth_sigma=smooth_sigma,
        filter_order=filter_order,
    )
    power = _zscore(envelope)

    candidate_bounds = find_interval(power >= low_threshold)
    candidate_bounds = _merge_bounds(
        candidate_bounds,
        gap_samples=max(1, int(round(merge_gap * float(fs)))),
    )

    sharp_wave_trace = None
    sharp_wave_power = None
    sharp_wave_bounds = None
    if sharp_wave_signal is not None:
        sharp_wave_trace, sharp_wave_power = _compute_sharp_wave_difference(
            ripple_signal=ripple_signal,
            sharp_wave_signal=sharp_wave_signal,
            fs=float(fs),
            sharp_wave_band=sharp_wave_band,
            filter_order=filter_order,
            smooth_sigma=sharp_wave_smooth_sigma,
        )
        sharp_wave_bounds = find_interval(sharp_wave_power >= sharp_wave_low_threshold)
        sharp_wave_bounds = _merge_bounds(
            sharp_wave_bounds,
            gap_samples=max(1, int(round(merge_gap * float(fs)))),
        )

    noise_power = None
    if noise_signal is not None:
        _, noise_envelope, _ = _compute_envelope(
            signal_in=noise_signal,
            fs=float(fs),
            ripple_band=ripple_band,
            smooth_sigma=smooth_sigma,
            filter_order=filter_order,
        )
        noise_power = _zscore(noise_envelope)

    events = _events_to_dataframe(
        ripple_bounds=candidate_bounds,
        ripple_power=power,
        ripple_envelope=envelope,
        ripple_filtered=ripple_filtered,
        inst_freq=inst_freq,
        timestamps=timestamps,
        fs=float(fs),
        ripple_min_duration=min_duration,
        ripple_max_duration=max_duration,
        ripple_high_threshold=high_threshold,
        ripple_channel=ripple_channel,
        sharp_wave_bounds=sharp_wave_bounds,
        sharp_wave_power=sharp_wave_power,
        sharp_wave_trace=sharp_wave_trace,
        sharp_wave_high_threshold=sharp_wave_high_threshold,
        sharp_wave_min_duration=sharp_wave_min_duration,
        sharp_wave_max_duration=sharp_wave_max_duration,
        search_window=peak_window,
        boundary_mode=boundary_mode,
        noise_power=noise_power,
        noise_threshold=noise_threshold,
    )

    intervals = _coerce_interval_array(detection_epochs)
    if intervals.size > 0 and not events.empty:
        keep = (
            in_intervals(events["start"].to_numpy(dtype=float), intervals)
            & in_intervals(events["stop"].to_numpy(dtype=float), intervals)
        )
        events = events.loc[keep].reset_index(drop=True)

    if save_mat:
        detection_params = {
            "ripple_band": np.asarray(ripple_band, dtype=float),
            "sharp_wave_band": np.asarray(sharp_wave_band, dtype=float),
            "smooth_sigma": float(smooth_sigma),
            "sharp_wave_smooth_sigma": float(sharp_wave_smooth_sigma),
            "low_threshold": float(low_threshold),
            "high_threshold": float(high_threshold),
            "sharp_wave_low_threshold": float(sharp_wave_low_threshold),
            "sharp_wave_high_threshold": float(sharp_wave_high_threshold),
            "noise_threshold": noise_threshold,
            "min_duration": float(min_duration),
            "max_duration": float(max_duration),
            "sharp_wave_min_duration": float(sharp_wave_min_duration),
            "sharp_wave_max_duration": float(sharp_wave_max_duration),
            "merge_gap": float(merge_gap),
            "peak_window": float(peak_window),
            "boundary_mode": boundary_mode,
            "filter_order": int(filter_order),
        }
        if ripple_channel is not None:
            detection_params["channel"] = int(ripple_channel)

        save_ripple_events(
            events=events,
            basepath=basepath,
            detection_name="detect_sharp_wave_ripples",
            detection_params=detection_params,
            ripple_channel=ripple_channel,
            detection_epochs=detection_epochs,
            event_name=event_name,
        )

    if return_epoch_array:
        if events.empty:
            return nel.EpochArray(np.empty((0, 2), dtype=float))
        return nel.EpochArray(events[["start", "stop"]].to_numpy(dtype=float))

    return events
