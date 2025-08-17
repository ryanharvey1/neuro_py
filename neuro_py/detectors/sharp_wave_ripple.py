import logging

import matplotlib.pyplot as plt
import nelpy as nel
import numpy as np
import scipy.signal
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
from numba import jit
from scipy.signal import find_peaks, hilbert
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE, SpectralEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap
import neuro_py as npy


def filter_signal(signal, lowcut, highcut, fs, order=4):
    """Bandpass filter the signal."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy.signal.butter(order, [low, high], btype="band")
    return scipy.signal.filtfilt(b, a, signal)


# def detect_events(signal, threshold_std):
#     """Detect events where the signal exceeds a threshold."""
#     threshold = np.mean(signal) + threshold_std * np.std(signal)
#     above_threshold = signal > threshold
#     events = np.where(np.diff(above_threshold.astype(int)) == 1)[0]
#     return events


# @jit(nopython=True)
# def expand_intervals(signal, events, threshold_std):
#     """Expand intervals to where the signal passes 1 std above the mean."""
#     threshold = np.mean(signal) + threshold_std * np.std(signal)
#     intervals = []
#     for event in events:
#         start = np.where(signal[:event] < threshold)[0]
#         start = start[-1] if len(start) > 0 else 0
#         end = np.where(signal[event:] < threshold)[0]
#         end = end[0] + event if len(end) > 0 else len(signal)
#         intervals.append((start, end))
#     return intervals


# @jit(nopython=True)
# def overlap_intervals(intervals_a, intervals_b):
#     """Find overlapping intervals between two lists of intervals."""
#     overlaps = []
#     for a in intervals_a:
#         for b in intervals_b:
#             if a[1] > b[0] and a[0] < b[1]:
#                 overlaps.append((max(a[0], b[0]), min(a[1], b[1])))
#     return overlaps


# @jit(nopython=True)
# def extract_peak_data(signal, intervals, fs):
#     """Extract peak times and amplitudes from intervals."""
#     peak_times = []
#     peak_amplitudes = []
#     for start, end in intervals:
#         segment = signal[start:end]
#         peak_idx = np.argmax(segment)
#         peak_times.append((start + peak_idx) / fs)
#         peak_amplitudes.append(segment[peak_idx])
#     return peak_times, peak_amplitudes


def detect(signal, threshold_1, fs, min_duration=0.02, max_duration=0.2):
    # min and max time width (converted to samples for find_peaks)
    time_widths = [
        int(min_duration * fs),
        int(max_duration * fs),
    ]

    PrimaryThreshold = signal.mean() + threshold_1 * signal.std()

    peaks, properties = find_peaks(
        signal,
        height=PrimaryThreshold,
        width=time_widths,
    )

    peaks = peaks / fs
    peak_val = properties["peak_heights"]

    # create EpochArray with bounds
    detected_epoch = np.array([properties["left_ips"], properties["right_ips"]]).T / fs

    return peaks, detected_epoch, peak_val


def reduce_dimensionality(data):
    # """Reduce dimensionality using PCA."""
    # pca = PCA(n_components=2)
    # return pca.fit_transform(data)
    # isomap

    return Isomap(n_components=2).fit_transform(data)


def interactive_curation(
    points,
    ripple_signal,
    sharp_wave_signal,
    fs,
    peak_times,
    raw_ripple_signal,
    raw_sharp_wave_signal,
):
    """Interactive window for manual curation with dark mode and visual enhancements."""

    # close all previous figures
    plt.close("all")

    # Set dark mode style
    plt.style.use("dark_background")

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6), facecolor="#1e1e1e", gridspec_kw={"width_ratios": [2, 3]}
    )
    fig.subplots_adjust(wspace=0.3)

    # Scatter plot for points
    scatter = ax1.scatter(
        points[:, 0],
        points[:, 1],
        picker=True,
        c="cyan",
        edgecolors="white",
        alpha=0.8,
        s=50,
    )
    ax1.set_title(
        "Click to select points. Close window when done.",
        color="white",
        fontsize=12,
        pad=10,
    )
    ax1.set_xlabel("PCA Component 1", color="white", fontsize=10)
    ax1.set_ylabel("PCA Component 2", color="white", fontsize=10)
    ax1.grid(color="gray", linestyle="--", alpha=0.3)
    ax1.set_facecolor("#2d2d2d")

    # Placeholder for raw signal plot
    ax2.set_title("Raw Signals", color="white", fontsize=12, pad=10)
    ax2.set_xlabel("Time (ms)", color="white", fontsize=10)
    ax2.set_ylabel("Amplitude", color="white", fontsize=10)
    ax2.grid(color="gray", linestyle="--", alpha=0.3)
    ax2.set_facecolor("#2d2d2d")
    (line_ripple,) = ax2.plot([], [], label="Raw Ripple", color="lime", linewidth=1.5)
    (line_sharp_wave,) = ax2.plot(
        [], [], label="Raw Sharp Wave", color="magenta", linewidth=1.5
    )
    ax2.legend(facecolor="#2d2d2d", edgecolor="none", fontsize=10, labelcolor="white")

    # Store selected points
    selected_indices = []

    def onpick(event):
        idx = event.ind[0]
        start = max(0, int((peak_times[idx] - 0.25) * fs))
        end = min(len(ripple_signal), int((peak_times[idx] + 0.25) * fs))

        # Update raw signal plot
        time_axis = np.linspace(-0.25, 0.25, end - start)
        line_ripple.set_data(time_axis, raw_ripple_signal[start:end])
        line_sharp_wave.set_data(time_axis, raw_sharp_wave_signal[start:end])

        # Rescale the axes
        ax2.relim()
        ax2.autoscale_view()

        # Redraw the figure
        fig.canvas.draw()

    def onselect(verts):
        """Callback function for LassoSelector."""
        path = Path(verts)
        selected_indices.clear()
        for i, point in enumerate(points):
            if path.contains_point(point):
                selected_indices.append(i)

        # Highlight selected points
        scatter.set_edgecolors(
            ["red" if i in selected_indices else "white" for i in range(len(points))]
        )
        fig.canvas.draw_idle()

    # Add LassoSelector
    LassoSelector(ax1, onselect, props={"color": "red", "linewidth": 2, "alpha": 0.8})

    fig.canvas.mpl_connect("pick_event", onpick)
    plt.show(block=True)

    # Return selected indices
    return selected_indices


# @jit(nopython=True)
def extract_fixed_length_segments(signal, peak_times, fs, window_size=0.05):
    """Extract fixed-length segments around peak times."""
    fixed_length = int(window_size * fs)
    segments = []
    for peak_time in peak_times:
        center = int(peak_time * fs)
        start = center - fixed_length // 2
        end = center + fixed_length // 2
        if start < 0 or end > len(signal):
            # Pad with zeros if the segment is out of bounds
            segment = np.zeros(fixed_length)
            valid_start = max(0, start)
            valid_end = min(len(signal), end)
            segment[valid_start - start : valid_end - start] = signal[
                valid_start:valid_end
            ]
        else:
            segment = signal[start:end]
        segments.append(segment)
    return np.array(segments)


def compute_envelope(signal):
    """Compute the envelope of a signal using the Hilbert transform."""
    analytic_signal = hilbert(signal)
    return np.abs(analytic_signal)


def extract_features(ripple_segment, sharp_wave_segment):
    ripple_features = [
        np.max(ripple_segment),  # Peak amplitude
        len(ripple_segment),  # Duration
        np.mean(ripple_segment),  # Mean amplitude
    ]
    sharp_wave_features = [
        np.max(sharp_wave_segment),  # Peak amplitude
        len(sharp_wave_segment),  # Duration
        np.mean(sharp_wave_segment),  # Mean amplitude
    ]
    return np.hstack([ripple_features, sharp_wave_features])


def merge_overlapping_intervals(intervals):
    """
    Merge overlapping intervals and return the indices of intervals to merge.

    Parameters:
        intervals (list of tuples): List of intervals, where each interval is a tuple (start, end).

    Returns:
        list of lists: Each sublist contains the indices of intervals that were merged.
    """

    # Sort intervals based on start time
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged_indices = []
    current_merge_group = [0]  # Start with the first interval
    current_end = sorted_intervals[0][1]

    for i in range(1, len(sorted_intervals)):
        start, end = sorted_intervals[i]
        if start <= current_end:  # Overlapping interval
            current_merge_group.append(i)
            current_end = max(current_end, end)  # Extend the merged interval
        else:
            # No overlap, start a new merge group
            merged_indices.append(current_merge_group)
            current_merge_group = [i]
            current_end = end

    # Add the last merge group
    merged_indices.append(current_merge_group)

    return np.array(merged_indices, object)[
        np.arange(len(merged_indices))[
            np.array([len(elm) for elm in merged_indices]) > 1
        ]
    ]


def instant_frequnecy(signal, fs):
    """Compute the instantaneous frequency of a signal."""
    analytic_signal = hilbert(signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs
    return instantaneous_frequency


def swr_detector(
    ripple_signal,
    sharp_wave_signal,
    fs,
    noise_signal=None,
    min_duration=0.02,
    max_duration=0.2,
    ripple_band=(80, 250),
    sharp_wave_band=(2, 50),
    ripple_peak_threshold=1,
    sharp_wave_peak_threshold=1,
    verbose=False,
    clean_lfp=True,
):
    """Detect sharp wave-ripple (SWR) events from LFP signals."""

    # adjust peak threshold based on if lfp was cleaned
    if clean_lfp:
        original_std = [np.std(ripple_signal), np.std(sharp_wave_signal)]

        ts = np.arange(len(ripple_signal)) / fs
        ripple_signal_ = npy.lfp.clean_lfp(ripple_signal, ts)
        sharp_wave_signal_ = npy.lfp.clean_lfp(sharp_wave_signal, ts)

        new_std = [np.std(ripple_signal_), np.std(sharp_wave_signal_)]

        ripple_peak_threshold = ripple_peak_threshold * (new_std[0] / original_std[0])
        sharp_wave_peak_threshold = sharp_wave_peak_threshold * (
            new_std[1] / original_std[1]
        )

    # Filter signals
    ripple_filtered = filter_signal(ripple_signal, ripple_band[0], ripple_band[1], fs)
    sharp_wave_filtered = filter_signal(
        sharp_wave_signal, sharp_wave_band[0], sharp_wave_band[1], fs
    )
    if noise_signal is not None:
        noise_filtered = filter_signal(noise_signal, ripple_band[0], ripple_band[1], fs)

    logging.info("Signals filtered")

    # Compute envelopes
    ripple_envelope = compute_envelope(ripple_filtered)
    sharp_wave_envelope = compute_envelope(sharp_wave_filtered)
    if noise_signal is not None:
        noise_envelope = compute_envelope(noise_filtered)

    logging.info("Envelopes computed")

    # Detect events on the envelope
    ripple_events, ripple_intervals, ripple_amp = detect(
        ripple_envelope,
        ripple_peak_threshold,
        fs,
        min_duration=min_duration,
        max_duration=max_duration,
    )
    sharp_wave_events, sharp_wave_intervals, sharpwave_amp = detect(
        sharp_wave_envelope,
        sharp_wave_peak_threshold,
        fs,
        min_duration=min_duration,
        max_duration=max_duration,
    )
    # ensure sharp waves events are negative in the filtered signal
    idx = sharp_wave_filtered[(sharp_wave_events * fs).astype(int)] < 0
    sharp_wave_events = sharp_wave_events[idx]
    sharp_wave_intervals = sharp_wave_intervals[idx]
    sharpwave_amp = sharpwave_amp[idx]

    if noise_signal is not None:
        _, noise_intervals, _ = detect(
            noise_envelope,
            ripple_peak_threshold,
            fs,
            min_duration=min_duration,
            max_duration=max_duration,
        )

    # Find overlapping intervals
    idx, ind = npy.process.in_intervals(
        ripple_events, sharp_wave_intervals, return_interval=True
    )
    ripple_intervals = ripple_intervals[idx]
    ripple_amp = ripple_amp[idx]
    ripple_events = ripple_events[idx]

    ind = ind[~np.isnan(ind)].astype(int)
    sharp_wave_intervals = sharp_wave_intervals[ind]
    sharpwave_amp = sharpwave_amp[ind]
    sharp_wave_events = sharp_wave_events[ind]

    # swr_intervals = ripple_intervals.copy()
    # remove dentate spikes that are overlapping with noise spikes
    if noise_signal is not None:
        noise_label = npy.process.find_intersecting_intervals(
            nel.EpochArray(ripple_intervals),
            nel.EpochArray(noise_intervals),
            return_indices=True,
        )
        ripple_intervals = ripple_intervals[~noise_label]
        ripple_amp = ripple_amp[~noise_label]
        ripple_events = ripple_events[~noise_label]
        sharp_wave_intervals = sharp_wave_intervals[~noise_label]
        sharpwave_amp = sharpwave_amp[~noise_label]
        sharp_wave_events = sharp_wave_events[~noise_label]

    # merge overlapping intervals
    # merged_indices = merge_overlapping_intervals(ripple_intervals)
    # for indices in merged_indices:
    #     start

    # Extract peak data
    # peak_times, peak_amplitudes = extract_peak_data(
    #     sharp_wave_envelope, swr_intervals, fs
    # )

    logging.info("Peak data extracted")

    # Extract fixed-length segments around peaks
    ripple_segments = extract_fixed_length_segments(ripple_signal, ripple_events, fs)
    sharp_wave_segments = extract_fixed_length_segments(
        sharp_wave_signal, ripple_events, fs
    )

    ripple_freq_segments = extract_fixed_length_segments(
        instant_frequnecy(ripple_filtered, fs), ripple_events, fs
    )
    sharp_wave_freq_segments = extract_fixed_length_segments(
        instant_frequnecy(sharp_wave_filtered, fs), ripple_events, fs
    )

    # Combine ripple and sharp wave segments for dimensionality reduction
    combined_segments = np.hstack([ripple_segments, sharp_wave_segments])
    # normalize each segment
    combined_segments = (combined_segments - combined_segments.mean(axis=1)[:, None]) / combined_segments.std(axis=1)[:, None]

    # get features [max, duration, mean, max freq, mean freq] for each segment and each signal
    features = np.array([
        ripple_segments.max(axis=1),
        ripple_intervals[:, 1] - ripple_intervals[:, 0],
        ripple_segments.mean(axis=1),
        ripple_freq_segments.max(axis=1),
        ripple_freq_segments.mean(axis=1),
        sharp_wave_segments.max(axis=1),
        sharp_wave_intervals[:, 1] - sharp_wave_intervals[:, 0],
        sharp_wave_segments.mean(axis=1),
        sharp_wave_freq_segments.max(axis=1),
        sharp_wave_freq_segments.mean(axis=1),
    ]).T

    # normalize each feature 
    features = (features - features.mean(axis=0)) / features.std(axis=0)

    # lda = LinearDiscriminantAnalysis(n_components=2)
    # points = lda.fit_transform(features, noise_label * 1)
    points = umap.UMAP().fit_transform(combined_segments)

    # Reduce dimensionality
    # points = reduce_dimensionality(features)

    logging.info("Dimensionality reduced")

    # Interactive curation
    selected_indices = interactive_curation(
        points,
        ripple_filtered,
        sharp_wave_filtered,
        fs,
        sharp_wave_events,
        ripple_signal,
        sharp_wave_signal,
    )
    if len(selected_indices) > 0:
        # Filter valid events based on selected indices
        valid_swr_intervals = [sharp_wave_intervals[i] for i in selected_indices]
        valid_peak_times = [sharp_wave_events[i] for i in selected_indices]
        valid_peak_amplitudes = [ripple_amp[i] for i in selected_indices]
    else:
        valid_swr_intervals = sharp_wave_intervals
        valid_peak_times = sharp_wave_events
        valid_peak_amplitudes = ripple_amp

    # convert to seconds
    valid_swr_intervals = np.array(valid_swr_intervals)

    # Return valid events
    return valid_swr_intervals, valid_peak_times, valid_peak_amplitudes


# Example usage
# fs = 1000  # Sampling rate
# ripple_signal = np.random.randn(10 * fs)  # Replace with actual data
# sharp_wave_signal = np.random.randn(10 * fs)  # Replace with actual data
# noise_signal = np.random.randn(10 * fs)  # Optional

# swr_intervals, peak_times, peak_amplitudes = swr_detector(ripple_signal, sharp_wave_signal, fs, noise_signal)


# basepath = r"U:\data\hpc_ctx_project\HP15\hp15_day44_20250214"
# lfp = npy.io.LFPLoader(basepath)
# ripple_ch = 41
# sharp_wave_ch = 32
# noise_ch = 193
# fs = lfp.fs

# ripple_signal = lfp.lfp.data[ripple_ch, :].copy()
# sharp_wave_signal = lfp.lfp.data[sharp_wave_ch, :].copy()
# noise_signal = lfp.lfp.data[noise_ch, :].copy()

# print("data loaded")

# swr_intervals, peak_times, peak_amplitudes = swr_detector(
#     ripple_signal,
#     sharp_wave_signal,
#     fs,
#     noise_signal,
# )
# print(swr_intervals)
