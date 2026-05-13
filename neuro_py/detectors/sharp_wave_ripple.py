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
from neuro_py.process.intervals import in_intervals_interval


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


def _fill_nonfinite_for_filter(values: np.ndarray) -> np.ndarray:
    """Fill non-finite samples so filtering does not poison the full trace."""
    values = np.asarray(values, dtype=float)
    if np.all(np.isfinite(values)):
        return values

    filled = values.copy()
    finite = np.isfinite(filled)
    if not finite.any():
        return np.zeros_like(filled, dtype=float)

    sample_idx = np.arange(filled.size)
    filled[~finite] = np.interp(sample_idx[~finite], sample_idx[finite], filled[finite])
    return filled


def _find_true_bounds(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive bounds for contiguous True segments."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []

    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    starts = edges[0::2]
    stops = edges[1::2] - 1
    return list(zip(starts.tolist(), stops.tolist()))


def _bounds_to_array(bounds: Optional[list[tuple[int, int]]]) -> np.ndarray:
    """Convert interval bounds to an ``(n_bounds, 2)`` integer array."""
    if not bounds:
        return np.empty((0, 2), dtype=int)
    return np.asarray(bounds, dtype=int)


def _bound_containing_index(
    bounds: np.ndarray,
    index: int,
) -> Optional[tuple[int, int]]:
    """Return the interval that contains ``index`` using binary search."""
    if bounds.size == 0:
        return None

    candidate = int(np.searchsorted(bounds[:, 0], index, side="right") - 1)
    if candidate < 0:
        return None

    start, stop = bounds[candidate]
    if start <= index <= stop:
        return int(start), int(stop)
    return None


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


def _filter_events_to_detection_epochs(
    events: pd.DataFrame,
    detection_epochs: Optional[Union[nel.EpochArray, np.ndarray]],
) -> pd.DataFrame:
    """Keep events fully contained within one detection epoch."""
    intervals = _coerce_interval_array(detection_epochs)
    if intervals.size == 0 or events.empty:
        return events

    start_interval = in_intervals_interval(
        events["start"].to_numpy(dtype=float), intervals
    )
    stop_interval = in_intervals_interval(
        events["stop"].to_numpy(dtype=float), intervals
    )
    keep = (
        ~np.isnan(start_interval)
        & ~np.isnan(stop_interval)
        & (start_interval == stop_interval)
    )
    return events.loc[keep].reset_index(drop=True)


def _enforce_min_inter_event_interval(
    events: pd.DataFrame,
    min_interval: float,
) -> pd.DataFrame:
    """Keep strongest events while removing direct minimum-interval conflicts."""
    if events.empty or min_interval <= 0:
        return events

    events_for_sort = events.copy()
    if "sharp_wave_peakNormedPower" in events_for_sort.columns:
        events_for_sort["_sharp_wave_score"] = events_for_sort[
            "sharp_wave_peakNormedPower"
        ].fillna(-np.inf)
    else:
        events_for_sort["_sharp_wave_score"] = -np.inf

    events_by_strength = events_for_sort.sort_values(
        by=["peakNormedPower", "_sharp_wave_score"],
        ascending=[False, False],
        na_position="last",
    )
    keep_indices: list[int] = []
    kept_peaks: list[float] = []

    for row_index, row in events_by_strength.iterrows():
        peak = float(row["peaks"])
        if any(abs(peak - kept_peak) < min_interval for kept_peak in kept_peaks):
            continue
        keep_indices.append(row_index)
        kept_peaks.append(peak)

    return (
        events.loc[keep_indices]
        .sort_values("start")
        .reset_index(drop=True)
    )


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
    """Return filtered ripple-band signal, smoothed envelope, and unwrapped phase."""
    sos = signal.butter(
        filter_order,
        np.asarray(ripple_band, dtype=float),
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    filtered = signal.sosfiltfilt(sos, _fill_nonfinite_for_filter(signal_in))
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
    return filtered, envelope, phase


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
    return signal.sosfiltfilt(sos, _fill_nonfinite_for_filter(signal_in))


def _compute_sharp_wave_difference(
    ripple_signal: np.ndarray,
    sharp_wave_signal: np.ndarray,
    fs: float,
    sharp_wave_band: tuple[float, float],
    filter_order: int,
    smooth_sigma: float,
    sharp_wave_polarity: str,
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
    if sharp_wave_polarity == "negative":
        sharp_wave_feature = sharp_wave_diff
    elif sharp_wave_polarity == "positive":
        sharp_wave_feature = -sharp_wave_diff
    elif sharp_wave_polarity == "both":
        sharp_wave_feature = np.abs(sharp_wave_diff)
    else:
        raise ValueError(
            "`sharp_wave_polarity` must be 'negative', 'positive', or 'both'."
        )
    sigma_samples = smooth_sigma * fs
    if sigma_samples > 0:
        sharp_wave_feature = ndimage.gaussian_filter1d(
            sharp_wave_feature,
            sigma=sigma_samples,
            mode="nearest",
        )
    return sharp_wave_feature, _zscore(sharp_wave_feature)


def _merge_bounds(
    bounds: list[tuple[int, int]], gap_samples: int
) -> list[tuple[int, int]]:
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


def _nearest_trough(
    filtered_signal: np.ndarray,
    center_idx: int,
    fs: float,
    window: float = 0.010,
    min_idx: Optional[int] = None,
    max_idx: Optional[int] = None,
) -> int:
    """Return the index of the nearest ripple trough around ``center_idx``."""
    radius = max(1, int(round(window * fs)))
    lower = 0 if min_idx is None else int(min_idx)
    upper = filtered_signal.size - 1 if max_idx is None else int(max_idx)
    start = max(0, lower, center_idx - radius)
    stop = min(filtered_signal.size, upper + 1, center_idx + radius + 1)
    if stop <= start:
        return int(np.clip(center_idx, lower, upper))
    return int(start + np.argmin(filtered_signal[start:stop]))


def _median_frequency_from_phase(
    phase: np.ndarray,
    start_idx: int,
    stop_idx: int,
    fs: float,
) -> float:
    """Return median instantaneous frequency in an event window."""
    event_phase = phase[start_idx : stop_idx + 1]
    if event_phase.size < 2:
        return np.nan
    return float(np.nanmedian(np.diff(event_phase) * fs / (2 * np.pi)))


def _local_zscore_at(
    values: np.ndarray,
    index: int,
    fs: float,
    local_window: float,
) -> float:
    """Return a local median/std z-score at one sample index."""
    radius = max(1, int(round(local_window * fs)))
    start = max(0, index - radius)
    stop = min(values.size, index + radius + 1)
    local_values = values[start:stop]
    median = np.nanmedian(local_values)
    std = np.nanstd(local_values)
    if np.isnan(std) or std == 0:
        return np.nan
    return float((values[index] - median) / std)


def _edge_rejects_event(
    event_start_idx: int,
    event_stop_idx: int,
    n_samples: int,
    edge_samples: int,
) -> bool:
    """Return True when an event is too close to the signal edge."""
    if edge_samples <= 0:
        return False
    return event_start_idx < edge_samples or event_stop_idx > (n_samples - edge_samples - 1)


def _window_has_artifact(
    signals: list[np.ndarray],
    start_idx: int,
    stop_idx: int,
    saturation_fraction: float,
    flat_std_threshold: float,
) -> bool:
    """Return True if any signal window is non-finite, saturated, or flat."""
    def _longest_run(mask: np.ndarray) -> int:
        if mask.size == 0:
            return 0
        bounds = _find_true_bounds(mask)
        if not bounds:
            return 0
        return max(stop - start + 1 for start, stop in bounds)

    for signal_values in signals:
        window = np.asarray(signal_values[start_idx : stop_idx + 1], dtype=float)
        if window.size == 0 or not np.all(np.isfinite(window)):
            return True

        window_std = float(np.nanstd(window))
        if window_std <= flat_std_threshold:
            return True

        min_value = np.nanmin(window)
        max_value = np.nanmax(window)
        if min_value == max_value:
            return True

        longest_clip = max(
            _longest_run(window == min_value),
            _longest_run(window == max_value),
        )
        clipped = longest_clip / window.size
        if longest_clip >= 3 and clipped >= saturation_fraction:
            return True
    return False


def _find_local_peaks(
    values: np.ndarray,
    start_idx: int,
    stop_idx: int,
) -> np.ndarray:
    """Return local peak indices, falling back to the window maximum."""
    if stop_idx < start_idx:
        return np.empty((0,), dtype=int)

    segment = np.asarray(values[start_idx : stop_idx + 1], dtype=float)
    if segment.size == 0 or np.all(~np.isfinite(segment)):
        return np.empty((0,), dtype=int)

    finite_segment = np.where(np.isfinite(segment), segment, -np.inf)
    local_peaks, _ = signal.find_peaks(finite_segment)
    if local_peaks.size == 0:
        return np.asarray([start_idx + int(np.nanargmax(segment))], dtype=int)
    return local_peaks.astype(int) + start_idx


def _localize_sharp_wave_interval(
    sharp_wave_power: np.ndarray,
    parent_start: int,
    parent_stop: int,
    peak_idx: int,
    low_threshold: float,
) -> tuple[int, int]:
    """Split a broad sharp-wave interval around a selected local peak."""
    local_peaks = _find_local_peaks(sharp_wave_power, parent_start, parent_stop)
    local_peaks = np.unique(np.append(local_peaks, peak_idx)).astype(int)
    local_peaks.sort()

    peak_position = int(np.searchsorted(local_peaks, peak_idx))
    previous_peak = local_peaks[peak_position - 1] if peak_position > 0 else None
    next_peak = (
        local_peaks[peak_position + 1]
        if peak_position < local_peaks.size - 1
        else None
    )

    split_start = parent_start
    split_stop = parent_stop
    if previous_peak is not None:
        split_start = max(split_start, int(np.floor((previous_peak + peak_idx) / 2)) + 1)
    if next_peak is not None:
        split_stop = min(split_stop, int(np.ceil((peak_idx + next_peak) / 2)) - 1)

    if split_stop < split_start:
        return int(peak_idx), int(peak_idx)

    above_threshold = (
        np.isfinite(sharp_wave_power[split_start : split_stop + 1])
        & (sharp_wave_power[split_start : split_stop + 1] >= low_threshold)
    )
    peak_offset = peak_idx - split_start
    if 0 <= peak_offset < above_threshold.size and above_threshold[peak_offset]:
        left = peak_offset
        right = peak_offset
        while left > 0 and above_threshold[left - 1]:
            left -= 1
        while right < above_threshold.size - 1 and above_threshold[right + 1]:
            right += 1
        split_start += left
        split_stop = split_start + (right - left)

    return int(split_start), int(split_stop)


def _select_sharp_wave_partner(
    sharp_wave_power: np.ndarray,
    sharp_wave_bounds: np.ndarray,
    ripple_start: int,
    ripple_stop: int,
    ripple_peak_idx: int,
    search_radius: int,
    low_threshold: float,
) -> Optional[tuple[int, int, int, float]]:
    """Select the nearest/overlapping sharp-wave sub-event for a ripple."""
    if sharp_wave_bounds.size == 0:
        return None

    association_start = max(0, ripple_start - search_radius)
    association_stop = min(sharp_wave_power.size - 1, ripple_stop + search_radius)
    if association_stop < association_start:
        return None

    candidate_rows = sharp_wave_bounds[
        (sharp_wave_bounds[:, 1] >= association_start)
        & (sharp_wave_bounds[:, 0] <= association_stop)
    ]
    best_partner: Optional[
        tuple[tuple[float, int, float], tuple[int, int, int, float]]
    ] = None

    for parent_start, parent_stop in candidate_rows:
        clipped_start = max(int(parent_start), association_start)
        clipped_stop = min(int(parent_stop), association_stop)
        local_peaks = _find_local_peaks(sharp_wave_power, clipped_start, clipped_stop)
        for peak_idx in local_peaks:
            peak_power = float(sharp_wave_power[peak_idx])
            if not np.isfinite(peak_power):
                continue

            localized_start, localized_stop = _localize_sharp_wave_interval(
                sharp_wave_power,
                int(parent_start),
                int(parent_stop),
                int(peak_idx),
                low_threshold,
            )
            overlap = max(
                0,
                min(ripple_stop, localized_stop)
                - max(ripple_start, localized_start)
                + 1,
            )
            score = (
                abs(int(peak_idx) - ripple_peak_idx),
                -overlap,
                -peak_power,
            )
            partner = (localized_start, localized_stop, int(peak_idx), peak_power)
            if best_partner is None or score < best_partner[0]:
                best_partner = (score, partner)

    if best_partner is None:
        return None
    return best_partner[1]


def _events_to_dataframe(
    ripple_bounds: list[tuple[int, int]],
    ripple_power: np.ndarray,
    ripple_envelope: np.ndarray,
    ripple_filtered: np.ndarray,
    ripple_phase: np.ndarray,
    ripple_signal: np.ndarray,
    timestamps: np.ndarray,
    fs: float,
    ripple_min_duration: float,
    ripple_max_duration: float,
    ripple_high_threshold: float,
    ripple_channel: Optional[int],
    sharp_wave_bounds: Optional[list[tuple[int, int]]] = None,
    sharp_wave_power: Optional[np.ndarray] = None,
    sharp_wave_trace: Optional[np.ndarray] = None,
    sharp_wave_low_threshold: float = 0.5,
    sharp_wave_high_threshold: Optional[float] = None,
    sharp_wave_min_duration: Optional[float] = None,
    sharp_wave_max_duration: Optional[float] = None,
    search_window: float = 0.050,
    boundary_mode: str = "sharp_wave",
    noise_power: Optional[np.ndarray] = None,
    noise_signal: Optional[np.ndarray] = None,
    noise_threshold: Optional[float] = None,
    sharp_wave_signal: Optional[np.ndarray] = None,
    threshold_mode: str = "global",
    local_window: float = 5.0,
    reject_edge_events: bool = True,
    edge_buffer: float = 0.050,
    reject_artifacts: bool = True,
    saturation_fraction: float = 0.05,
    flat_std_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Convert joint ripple and sharp-wave detections into a CellExplorer-style table."""
    records: list[dict[str, float]] = []
    dt = 1.0 / fs
    search_radius = max(1, int(round(search_window * fs)))
    edge_samples = max(0, int(round(edge_buffer * fs)))
    flat_std = (
        np.finfo(float).eps
        if flat_std_threshold is None
        else float(flat_std_threshold)
    )
    sharp_wave_bounds_array = _bounds_to_array(sharp_wave_bounds)

    for ripple_start, ripple_stop in ripple_bounds:
        if reject_edge_events and _edge_rejects_event(
            ripple_start, ripple_stop, ripple_power.size, edge_samples
        ):
            continue

        segment = ripple_power[ripple_start : ripple_stop + 1]
        if segment.size == 0:
            continue

        ripple_peak_idx = int(ripple_start + np.nanargmax(segment))
        ripple_peak_power = float(ripple_power[ripple_peak_idx])
        if threshold_mode == "local":
            ripple_peak_power = _local_zscore_at(
                ripple_envelope, ripple_peak_idx, fs, local_window
            )
        if not np.isfinite(ripple_peak_power):
            continue
        if ripple_peak_power < ripple_high_threshold:
            continue

        ripple_start_time = float(timestamps[ripple_start])
        ripple_stop_time = float(timestamps[ripple_stop] + dt)
        ripple_duration = ripple_stop_time - ripple_start_time
        if (
            ripple_duration < ripple_min_duration
            or ripple_duration > ripple_max_duration
        ):
            continue

        sharp_wave_peak_idx = None
        sharp_wave_peak_power = np.nan
        sharp_wave_duration = np.nan
        event_start_idx, event_stop_idx = ripple_start, ripple_stop

        if sharp_wave_power is not None:
            if sharp_wave_bounds_array.size == 0:
                continue

            sharp_wave_partner = _select_sharp_wave_partner(
                sharp_wave_power=sharp_wave_power,
                sharp_wave_bounds=sharp_wave_bounds_array,
                ripple_start=ripple_start,
                ripple_stop=ripple_stop,
                ripple_peak_idx=ripple_peak_idx,
                search_radius=search_radius,
                low_threshold=sharp_wave_low_threshold,
            )
            if sharp_wave_partner is None:
                continue

            sharp_wave_start, sharp_wave_stop, sharp_wave_peak_idx, _ = (
                sharp_wave_partner
            )
            sharp_wave_peak_power = float(sharp_wave_power[sharp_wave_peak_idx])
            if threshold_mode == "local":
                sharp_wave_peak_power = _local_zscore_at(
                    sharp_wave_trace, sharp_wave_peak_idx, fs, local_window
                )
            if not np.isfinite(sharp_wave_peak_power):
                continue
            if (
                sharp_wave_high_threshold is not None
                and sharp_wave_peak_power < sharp_wave_high_threshold
            ):
                continue

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
                raise ValueError(
                    "`boundary_mode` must be either 'sharp_wave' or 'union'."
                )

        if reject_edge_events and _edge_rejects_event(
            event_start_idx, event_stop_idx, ripple_power.size, edge_samples
        ):
            continue

        event_segment = ripple_power[event_start_idx : event_stop_idx + 1]
        if event_segment.size == 0:
            continue

        event_ripple_peak_idx = int(event_start_idx + np.nanargmax(event_segment))
        event_ripple_peak_power = float(ripple_power[event_ripple_peak_idx])
        if threshold_mode == "local":
            event_ripple_peak_power = _local_zscore_at(
                ripple_envelope, event_ripple_peak_idx, fs, local_window
            )
        if not np.isfinite(event_ripple_peak_power):
            continue
        if event_ripple_peak_power < ripple_high_threshold:
            continue

        event_peak_idx = _nearest_trough(
            filtered_signal=ripple_filtered,
            center_idx=event_ripple_peak_idx,
            fs=fs,
            min_idx=event_start_idx,
            max_idx=event_stop_idx,
        )
        if not event_start_idx <= event_peak_idx <= event_stop_idx:
            continue

        if reject_artifacts:
            artifact_signals = [ripple_signal]
            if sharp_wave_signal is not None:
                artifact_signals.append(sharp_wave_signal)
            if noise_signal is not None:
                artifact_signals.append(noise_signal)
            if _window_has_artifact(
                artifact_signals,
                event_start_idx,
                event_stop_idx,
                saturation_fraction=saturation_fraction,
                flat_std_threshold=flat_std,
            ):
                continue

        if noise_power is not None and noise_threshold is not None:
            noise_peak = float(
                np.nanmax(noise_power[event_start_idx : event_stop_idx + 1])
            )
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
            "frequency": _median_frequency_from_phase(
                ripple_phase,
                event_start_idx,
                event_stop_idx,
                fs,
            ),
            "peakNormedPower": event_ripple_peak_power,
            "ripple_channel": np.nan if ripple_channel is None else int(ripple_channel),
            "noise_peakNormedPower": noise_peak,
            "ripple_duration": float(ripple_duration),
            "sharp_wave_peakNormedPower": sharp_wave_peak_power,
        }
        if sharp_wave_trace is not None:
            peak_index = (
                sharp_wave_peak_idx
                if sharp_wave_peak_idx is not None
                else event_peak_idx
            )
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
    ripple_band: tuple[float, float] = (80.0, 250.0),
    sharp_wave_band: tuple[float, float] = (2.0, 50.0),
    smooth_sigma: float = 0.004,
    sharp_wave_smooth_sigma: float = 0.0,
    low_threshold: float = 1.0,
    high_threshold: float = 2.5,
    sharp_wave_low_threshold: float = 0.4,
    sharp_wave_high_threshold: float = 2.5,
    noise_threshold: Optional[float] = None,
    min_duration: float = 0.015,
    max_duration: float = 0.250,
    sharp_wave_min_duration: float = 0.020,
    sharp_wave_max_duration: float = 0.500,
    min_inter_event_interval: float = 0.025,
    merge_gap: float = 0.001,
    peak_window: float = 0.150,
    boundary_mode: str = "union",
    filter_order: int = 4,
    threshold_mode: str = "global",
    local_window: float = 5.0,
    reject_edge_events: bool = True,
    edge_buffer: Optional[float] = None,
    reject_artifacts: bool = True,
    saturation_fraction: float = 0.05,
    flat_std_threshold: Optional[float] = None,
    sharp_wave_polarity: str = "negative",
    require_sharp_wave: bool = True,
    save_mat: bool = True,
    overwrite: bool = False,
    return_epoch_array: bool = False,
    event_name: str = "ripples",
) -> Union[pd.DataFrame, nel.EpochArray]:
    """
    Detect sharp wave ripple events from a ripple-band LFP channel.

    The detector follows a compact joint SWR workflow by default: detect
    candidate ripple intervals from ripple-band power, require a nearby
    sharp-wave event on a companion low-frequency channel difference, and then
    validate ripple and sharp-wave durations separately before returning final
    events. Set ``require_sharp_wave=False`` to run explicit ripple-only
    detection when no sharp-wave channel is available.

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
    min_inter_event_interval : float, optional
        Minimum time between accepted event peaks in seconds. Stronger events
        are kept first and weaker direct conflicts are removed. Set to 0 to
        disable.
    merge_gap : float, optional
        Merge candidate events separated by less than this gap, in seconds.
    peak_window : float, optional
        Maximum association window around the ripple candidate, in seconds,
        used to find the nearest/overlapping sharp-wave partner.
    boundary_mode : {"sharp_wave", "union"}, optional
        Whether final event boundaries follow the sharp-wave interval or the
        union of ripple and sharp-wave intervals.
    filter_order : int, optional
        Butterworth filter order for ripple-band filtering.
    threshold_mode : {"global", "local"}, optional
        Use global z-scored features or MATLAB-like local median/std validation
        around each candidate event.
    local_window : float, optional
        Half-window in seconds used for local threshold validation when
        ``threshold_mode="local"``.
    reject_edge_events : bool, optional
        If True, reject candidate events too close to signal boundaries.
    edge_buffer : float, optional
        Boundary buffer in seconds. If omitted, uses at least ``peak_window`` and
        uses ``local_window`` when local thresholds are enabled.
    reject_artifacts : bool, optional
        If True, reject event windows with non-finite, saturated, or flat
        required signals.
    saturation_fraction : float, optional
        Maximum tolerated fraction of event-window samples at the local minimum
        or maximum before the window is treated as clipped.
    flat_std_threshold : float, optional
        Minimum allowed event-window standard deviation. Defaults to a
        near-zero variation check.
    sharp_wave_polarity : {"negative", "positive", "both"}, optional
        Polarity of sharp-wave deflections. The default expects downward
        sharp waves and scores them positively. Ripples remain polarity
        independent because they are detected from envelope power.
    require_sharp_wave : bool, optional
        If True, require a sharp-wave signal or inferable sharp-wave channel for
        joint SWR detection. If False, allow ripple-only detection when
        sharp-wave data are unavailable. Ripple-only detections should be
        interpreted cautiously because the sharp-wave criterion helps reject
        ripple-band noise.
    save_mat : bool, optional
        If True and ``basepath`` is provided, save a CellExplorer event file.
        In-memory detections without a ``basepath`` still run normally but do
        not write an event file.
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

    Examples
    --------
    Detect joint SWRs from a CellExplorer session folder and save the default
    ``*.ripples.events.mat`` file. The ripple, sharp-wave, and noise channels
    are inferred from channel tags when available.

    >>> from neuro_py.detectors.sharp_wave_ripple import detect_sharp_wave_ripples
    >>> ripples = detect_sharp_wave_ripples(
    ...     basepath=r"S:\\data\\HMC\\HMC1\\day8",
    ...     low_threshold=0.75,
    ...     high_threshold=2.5,
    ...     sharp_wave_low_threshold=0.4,
    ...     sharp_wave_high_threshold=2.5,
    ...     overwrite=True,
    ... )

    Restrict detection to a time interval and return only the event table
    without writing a CellExplorer file.

    >>> ripples = detect_sharp_wave_ripples(
    ...     basepath=r"S:\\data\\HMC\\HMC1\\day8",
    ...     detection_epochs=np.array([[250.0, 350.0]]),
    ...     save_mat=False,
    ... )

    Run on in-memory LFP arrays. This is useful for simulations or when data
    have already been loaded by another pipeline.

    >>> ripples = detect_sharp_wave_ripples(
    ...     ripple_signal=ripple_lfp,
    ...     sharp_wave_signal=sharp_wave_lfp,
    ...     fs=1250.0,
    ...     timestamps=timestamps,
    ...     save_mat=False,
    ... )

    If no sharp-wave channel is available, ripple-only detection must be
    requested explicitly.

    >>> ripples = detect_sharp_wave_ripples(
    ...     ripple_signal=ripple_lfp,
    ...     fs=1250.0,
    ...     save_mat=False,
    ...     require_sharp_wave=False,
    ... )
    """
    if basepath is None and ripple_signal is None:
        raise ValueError("Provide either `basepath` or `ripple_signal` for detection.")
    if boundary_mode not in {"sharp_wave", "union"}:
        raise ValueError("`boundary_mode` must be either 'sharp_wave' or 'union'.")
    if threshold_mode not in {"global", "local"}:
        raise ValueError("`threshold_mode` must be either 'global' or 'local'.")
    if sharp_wave_polarity not in {"negative", "positive", "both"}:
        raise ValueError(
            "`sharp_wave_polarity` must be 'negative', 'positive', or 'both'."
        )
    if min_inter_event_interval < 0:
        raise ValueError("`min_inter_event_interval` must be non-negative.")
    if local_window <= 0:
        raise ValueError("`local_window` must be positive.")
    if not 0 < saturation_fraction <= 1:
        raise ValueError("`saturation_fraction` must be in the interval (0, 1].")
    if edge_buffer is not None and edge_buffer < 0:
        raise ValueError("`edge_buffer` must be non-negative.")

    should_save_mat = bool(save_mat and basepath is not None)

    if should_save_mat and not overwrite:
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
                    existing = existing.rename(
                        columns={"starts": "start", "stops": "stop"}
                    )
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
            raise ValueError(
                "`timestamps` must have the same length as `ripple_signal`."
            )
        if sharp_wave_signal is not None:
            sharp_wave_signal = np.asarray(sharp_wave_signal, dtype=float)
        if noise_signal is not None:
            noise_signal = np.asarray(noise_signal, dtype=float)

    if require_sharp_wave and sharp_wave_signal is None:
        raise ValueError(
            "Joint SWR detection requires a sharp-wave signal. Pass "
            "`sharp_wave_signal`, provide `sharp_wave_channel`, add a "
            "CellExplorer SharpWave channel tag, or set "
            "`require_sharp_wave=False` for explicit ripple-only detection."
        )

    ripple_filtered, envelope, ripple_phase = _compute_envelope(
        signal_in=ripple_signal,
        fs=float(fs),
        ripple_band=ripple_band,
        smooth_sigma=smooth_sigma,
        filter_order=filter_order,
    )
    power = _zscore(envelope)

    candidate_bounds = _find_true_bounds(power >= low_threshold)
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
            sharp_wave_polarity=sharp_wave_polarity,
        )
        sharp_wave_bounds = _find_true_bounds(
            sharp_wave_power >= sharp_wave_low_threshold
        )
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

    effective_edge_buffer = peak_window if edge_buffer is None else float(edge_buffer)
    if threshold_mode == "local":
        effective_edge_buffer = max(effective_edge_buffer, float(local_window))

    events = _events_to_dataframe(
        ripple_bounds=candidate_bounds,
        ripple_power=power,
        ripple_envelope=envelope,
        ripple_filtered=ripple_filtered,
        ripple_phase=ripple_phase,
        ripple_signal=ripple_signal,
        timestamps=timestamps,
        fs=float(fs),
        ripple_min_duration=min_duration,
        ripple_max_duration=max_duration,
        ripple_high_threshold=high_threshold,
        ripple_channel=ripple_channel,
        sharp_wave_bounds=sharp_wave_bounds,
        sharp_wave_power=sharp_wave_power,
        sharp_wave_trace=sharp_wave_trace,
        sharp_wave_low_threshold=sharp_wave_low_threshold,
        sharp_wave_high_threshold=sharp_wave_high_threshold,
        sharp_wave_min_duration=sharp_wave_min_duration,
        sharp_wave_max_duration=sharp_wave_max_duration,
        search_window=peak_window,
        boundary_mode=boundary_mode,
        noise_power=noise_power,
        noise_signal=noise_signal,
        noise_threshold=noise_threshold,
        sharp_wave_signal=sharp_wave_signal,
        threshold_mode=threshold_mode,
        local_window=local_window,
        reject_edge_events=reject_edge_events,
        edge_buffer=effective_edge_buffer,
        reject_artifacts=reject_artifacts,
        saturation_fraction=saturation_fraction,
        flat_std_threshold=flat_std_threshold,
    )

    events = _filter_events_to_detection_epochs(events, detection_epochs)
    events = _enforce_min_inter_event_interval(events, min_inter_event_interval)

    if should_save_mat:
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
            "min_inter_event_interval": float(min_inter_event_interval),
            "merge_gap": float(merge_gap),
            "peak_window": float(peak_window),
            "boundary_mode": boundary_mode,
            "threshold_mode": threshold_mode,
            "local_window": float(local_window),
            "reject_edge_events": bool(reject_edge_events),
            "edge_buffer": effective_edge_buffer,
            "reject_artifacts": bool(reject_artifacts),
            "saturation_fraction": float(saturation_fraction),
            "flat_std_threshold": flat_std_threshold,
            "sharp_wave_polarity": sharp_wave_polarity,
            "require_sharp_wave": bool(require_sharp_wave),
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
