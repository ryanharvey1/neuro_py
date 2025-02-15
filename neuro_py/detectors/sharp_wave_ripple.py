import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
from sklearn.decomposition import PCA
from scipy.signal import hilbert


def filter_signal(signal, lowcut, highcut, fs, order=4):
    """Bandpass filter the signal."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = scipy.signal.butter(order, [low, high], btype="band")
    return scipy.signal.filtfilt(b, a, signal)


def detect_events(signal, threshold_std):
    """Detect events where the signal exceeds a threshold."""
    threshold = np.mean(signal) + threshold_std * np.std(signal)
    above_threshold = signal > threshold
    events = np.where(np.diff(above_threshold.astype(int)) == 1)[0]
    return events


def expand_intervals(signal, events, threshold_std):
    """Expand intervals to where the signal passes 1 std above the mean."""
    threshold = np.mean(signal) + threshold_std * np.std(signal)
    intervals = []
    for event in events:
        start = np.where(signal[:event] < threshold)[0]
        start = start[-1] if len(start) > 0 else 0
        end = np.where(signal[event:] < threshold)[0]
        end = end[0] + event if len(end) > 0 else len(signal)
        intervals.append((start, end))
    return intervals


def overlap_intervals(intervals_a, intervals_b):
    """Find overlapping intervals between two lists of intervals."""
    overlaps = []
    for a in intervals_a:
        for b in intervals_b:
            if a[1] > b[0] and a[0] < b[1]:
                overlaps.append((max(a[0], b[0]), min(a[1], b[1])))
    return overlaps


def extract_peak_data(signal, intervals, fs):
    """Extract peak times and amplitudes from intervals."""
    peak_times = []
    peak_amplitudes = []
    for start, end in intervals:
        segment = signal[start:end]
        peak_idx = np.argmax(segment)
        peak_times.append((start + peak_idx) / fs)
        peak_amplitudes.append(segment[peak_idx])
    return peak_times, peak_amplitudes


def reduce_dimensionality(data):
    """Reduce dimensionality using PCA."""
    pca = PCA(n_components=2)
    return pca.fit_transform(data)


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
    # Set dark mode style
    plt.style.use("dark_background")

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor="#1e1e1e")
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
        start = max(0, int((peak_times[idx] - 0.05) * fs))
        end = min(len(ripple_signal), int((peak_times[idx] + 0.05) * fs))

        # Update raw signal plot
        time_axis = np.linspace(-50, 50, end - start)
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
    lasso = LassoSelector(
        ax1, onselect, props={"color": "red", "linewidth": 2, "alpha": 0.8}
    )

    fig.canvas.mpl_connect("pick_event", onpick)
    plt.show()

    # Return selected indices
    return selected_indices


def extract_fixed_length_segments(signal, peak_times, fs, window_size=0.1):
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


def swr_detector(ripple_signal, sharp_wave_signal, fs, noise_signal=None):
    # Store raw signals for plotting
    raw_ripple_signal = ripple_signal
    raw_sharp_wave_signal = sharp_wave_signal

    # Filter signals
    ripple_filtered = filter_signal(ripple_signal, 80, 250, fs)
    sharp_wave_filtered = filter_signal(sharp_wave_signal, 2, 50, fs)
    if noise_signal is not None:
        noise_filtered = filter_signal(noise_signal, 80, 250, fs)

    # Compute envelopes
    ripple_envelope = compute_envelope(ripple_filtered)
    sharp_wave_envelope = compute_envelope(sharp_wave_filtered)
    if noise_signal is not None:
        noise_envelope = compute_envelope(noise_filtered)

    # Detect events on the envelope
    ripple_events = detect_events(ripple_envelope, 2)
    sharp_wave_events = detect_events(sharp_wave_envelope, 2)
    if noise_signal is not None:
        noise_events = detect_events(noise_envelope, 2)

    # Expand intervals
    ripple_intervals = expand_intervals(ripple_envelope, ripple_events, 1)
    sharp_wave_intervals = expand_intervals(sharp_wave_envelope, sharp_wave_events, 1)
    if noise_signal is not None:
        noise_intervals = expand_intervals(noise_envelope, noise_events, 1)

    # Find overlapping intervals
    swr_intervals = overlap_intervals(ripple_intervals, sharp_wave_intervals)
    if noise_signal is not None:
        swr_intervals = [
            interval
            for interval in swr_intervals
            if not any(
                interval[0] < noise_end and interval[1] > noise_start
                for noise_start, noise_end in noise_intervals
            )
        ]

    # Extract peak data
    peak_times, peak_amplitudes = extract_peak_data(
        sharp_wave_envelope, swr_intervals, fs
    )

    # Extract fixed-length segments around peaks
    ripple_segments = extract_fixed_length_segments(ripple_filtered, peak_times, fs)
    sharp_wave_segments = extract_fixed_length_segments(
        sharp_wave_filtered, peak_times, fs
    )

    # Combine ripple and sharp wave segments for dimensionality reduction
    combined_segments = np.hstack([ripple_segments, sharp_wave_segments])

    # Reduce dimensionality
    points = reduce_dimensionality(combined_segments)

    # Interactive curation
    selected_indices = interactive_curation(
        points,
        ripple_filtered,
        sharp_wave_filtered,
        fs,
        peak_times,
        raw_ripple_signal,
        raw_sharp_wave_signal,
    )

    # Filter valid events based on selected indices
    valid_swr_intervals = [swr_intervals[i] for i in selected_indices]
    valid_peak_times = [peak_times[i] for i in selected_indices]
    valid_peak_amplitudes = [peak_amplitudes[i] for i in selected_indices]

    # Return valid events
    return valid_swr_intervals, valid_peak_times, valid_peak_amplitudes


# # Example usage
# fs = 1000  # Sampling rate
# ripple_signal = np.random.randn(10 * fs)  # Replace with actual data
# sharp_wave_signal = np.random.randn(10 * fs)  # Replace with actual data
# noise_signal = np.random.randn(10 * fs)  # Optional

# swr_intervals, peak_times, peak_amplitudes = swr_detector(ripple_signal, sharp_wave_signal, fs, noise_signal)


import neuro_py as npy

basepath = r"U:\data\hpc_ctx_project\HP15\hp15_day36_20250206"
lfp = npy.io.LFPLoader(basepath)
ripple_ch = 39
sharp_wave_ch = 32
noise_ch = 171
fs = lfp.fs
swr_intervals, peak_times, peak_amplitudes = swr_detector(
    lfp.lfp.data[ripple_ch, :],
    lfp.lfp.data[sharp_wave_ch, :],
    fs,
    lfp.lfp.data[noise_ch, :],
)
print(swr_intervals)
