# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.23.0",
# ]
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import html
    import math
    import random

    import marimo as mo

    return html, math, mo, random


@app.cell
def _(mo):
    mo.md(
        """
# Interactive simulated SWR detector

This browser-only demo simulates paired ripple and sharp-wave LFP channels,
then runs a compact joint detector. Green spans mark the known simulated SWR
windows; orange spans mark accepted detections.

This app intentionally avoids `neuro_py`, `nelpy`, SciPy, Matplotlib, and real
session data so the interactive example can load quickly in browser-WASM.
        """
    )
    return


@app.cell
def _(mo):
    ripple_amplitude = mo.ui.slider(
        0.5, 4.0, step=0.1, value=2.0, label="Ripple amplitude"
    )
    sharp_wave_amplitude = mo.ui.slider(
        0.5, 5.0, step=0.1, value=2.8, label="Sharp-wave amplitude"
    )
    noise_amplitude = mo.ui.slider(
        0.2, 2.0, step=0.1, value=1.1, label="Noise amplitude"
    )
    ripple_duration = mo.ui.slider(
        0.050, 0.150, step=0.005, value=0.100, label="Ripple duration (s)"
    )
    ripple_threshold = mo.ui.slider(
        0.5, 4.0, step=0.1, value=1.8, label="Ripple high threshold (z)"
    )
    sharp_wave_threshold = mo.ui.slider(
        0.5, 4.0, step=0.1, value=1.4, label="Sharp-wave high threshold (z)"
    )

    mo.vstack(
        [
            mo.md("### Simulation and detector controls"),
            mo.hstack(
                [ripple_amplitude, sharp_wave_amplitude, noise_amplitude],
                justify="start",
                gap=1,
            ),
            mo.hstack(
                [ripple_duration, ripple_threshold, sharp_wave_threshold],
                justify="start",
                gap=1,
            ),
        ]
    )
    return (
        noise_amplitude,
        ripple_amplitude,
        ripple_duration,
        ripple_threshold,
        sharp_wave_amplitude,
        sharp_wave_threshold,
    )


@app.cell
def _(math, random):
    FS = 1250.0
    DURATION = 1.5
    RIPPLE_FREQUENCY = 165.0
    SWR_TIMES = [0.45, 1.02]

    def zscore(values):
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / len(values)
        scale = math.sqrt(var)
        if scale == 0:
            return [0.0 for _ in values]
        return [(value - mean) / scale for value in values]

    def brown_noise(n_samples, seed):
        rng = random.Random(seed)
        value = 0.0
        values = []
        for _ in range(n_samples):
            value += rng.gauss(0.0, 1.0)
            values.append(value)
        return zscore(values)

    def moving_average(values, window):
        window = max(1, int(window))
        half = window // 2
        averaged = []
        for index in range(len(values)):
            start = max(0, index - half)
            stop = min(len(values), index + half + 1)
            averaged.append(sum(values[start:stop]) / (stop - start))
        return averaged

    def threshold_bounds(values, low_threshold):
        bounds = []
        start = None
        for index, value in enumerate(values):
            if value >= low_threshold and start is None:
                start = index
            elif value < low_threshold and start is not None:
                bounds.append((start, index - 1))
                start = None
        if start is not None:
            bounds.append((start, len(values) - 1))
        return bounds

    def merge_bounds(bounds, gap_samples):
        if not bounds:
            return []
        merged = [bounds[0]]
        for start, stop in bounds[1:]:
            prev_start, prev_stop = merged[-1]
            if start - prev_stop <= gap_samples:
                merged[-1] = (prev_start, max(prev_stop, stop))
            else:
                merged.append((start, stop))
        return merged

    def bound_containing_index(bounds, index):
        for start, stop in bounds:
            if start <= index <= stop:
                return start, stop
        return None

    return (
        DURATION,
        FS,
        RIPPLE_FREQUENCY,
        SWR_TIMES,
        bound_containing_index,
        brown_noise,
        merge_bounds,
        moving_average,
        threshold_bounds,
        zscore,
    )


@app.cell
def _(
    DURATION,
    FS,
    RIPPLE_FREQUENCY,
    SWR_TIMES,
    brown_noise,
    math,
    noise_amplitude,
    ripple_amplitude,
    ripple_duration,
    sharp_wave_amplitude,
):
    timestamps = [sample / FS for sample in range(int(DURATION * FS))]
    ripple_noise = brown_noise(len(timestamps), seed=7)
    sharp_noise = brown_noise(len(timestamps), seed=8)
    ripple_raw = [noise_amplitude.value * value for value in ripple_noise]
    sharp_wave_raw = [
        0.75 * noise_amplitude.value * value for value in sharp_noise
    ]

    sharp_wave_duration = 0.140
    for center in SWR_TIMES:
        for index, time in enumerate(timestamps):
            ripple_width = ripple_duration.value / 6.0
            sharp_width = sharp_wave_duration / 6.0
            ripple_envelope = math.exp(
                -((time - center) ** 2) / (2.0 * ripple_width**2)
            )
            sharp_wave_envelope = math.exp(
                -((time - center) ** 2) / (2.0 * sharp_width**2)
            )
            ripple_carrier = math.sin(
                2.0 * math.pi * RIPPLE_FREQUENCY * time
            )
            ripple_raw[index] += (
                ripple_amplitude.value * ripple_carrier * ripple_envelope
            )
            sharp_wave_raw[index] -= (
                sharp_wave_amplitude.value * sharp_wave_envelope
            )

    truth = [
        (
            center - ripple_duration.value / 2.0,
            center + ripple_duration.value / 2.0,
        )
        for center in SWR_TIMES
    ]
    return ripple_raw, sharp_wave_raw, timestamps, truth


@app.cell
def _(
    FS,
    bound_containing_index,
    merge_bounds,
    moving_average,
    ripple_raw,
    ripple_threshold,
    sharp_wave_raw,
    sharp_wave_threshold,
    threshold_bounds,
    timestamps,
    zscore,
):
    slow_ripple = moving_average(ripple_raw, window=int(0.030 * FS))
    ripple_high = [
        abs(raw - slow) for raw, slow in zip(ripple_raw, slow_ripple)
    ]
    ripple_feature = zscore(
        moving_average(ripple_high, window=int(0.012 * FS))
    )

    sharp_wave_feature = zscore(
        moving_average([-value for value in sharp_wave_raw], int(0.040 * FS))
    )

    ripple_bounds = merge_bounds(
        threshold_bounds(ripple_feature, ripple_threshold.value * 0.45),
        gap_samples=int(0.010 * FS),
    )
    sharp_wave_bounds = merge_bounds(
        threshold_bounds(sharp_wave_feature, sharp_wave_threshold.value * 0.45),
        gap_samples=int(0.010 * FS),
    )

    search_radius = int(0.060 * FS)
    detected_events = []
    for ripple_start, ripple_stop in ripple_bounds:
        ripple_peak = max(
            range(ripple_start, ripple_stop + 1),
            key=lambda index: ripple_feature[index],
        )
        if ripple_feature[ripple_peak] < ripple_threshold.value:
            continue

        search_start = max(0, ripple_peak - search_radius)
        search_stop = min(len(timestamps), ripple_peak + search_radius + 1)
        sharp_peak = max(
            range(search_start, search_stop),
            key=lambda index: sharp_wave_feature[index],
        )
        if sharp_wave_feature[sharp_peak] < sharp_wave_threshold.value:
            continue

        sharp_interval = bound_containing_index(sharp_wave_bounds, sharp_peak)
        if sharp_interval is None:
            continue

        sharp_start, sharp_stop = sharp_interval
        event_start = min(ripple_start, sharp_start)
        event_stop = max(ripple_stop, sharp_stop)
        detected_events.append(
            {
                "start": timestamps[event_start],
                "stop": timestamps[min(event_stop + 1, len(timestamps) - 1)],
                "peak": timestamps[ripple_peak],
                "ripple_z": ripple_feature[ripple_peak],
                "sharp_wave_z": sharp_wave_feature[sharp_peak],
            }
        )

    return detected_events, ripple_feature, sharp_wave_feature


@app.cell
def _(
    DURATION,
    detected_events,
    html,
    mo,
    ripple_feature,
    ripple_raw,
    sharp_wave_feature,
    sharp_wave_raw,
    timestamps,
    truth,
):
    def polyline(values, row_top, row_height, color, stroke_width=1.4):
        lo = min(values)
        hi = max(values)
        if hi == lo:
            hi = lo + 1.0
        points = []
        for time, value in zip(timestamps[::2], values[::2]):
            x = 70.0 + (time / DURATION) * 860.0
            y = row_top + row_height - ((value - lo) / (hi - lo)) * row_height
            points.append(f"{x:.1f},{y:.1f}")
        return (
            f'<polyline points="{" ".join(points)}" fill="none" '
            f'stroke="{color}" stroke-width="{stroke_width}" />'
        )

    def span_rect(start, stop, row_top, row_height, color, opacity):
        x = 70.0 + (start / DURATION) * 860.0
        width = max(1.0, ((stop - start) / DURATION) * 860.0)
        return (
            f'<rect x="{x:.1f}" y="{row_top:.1f}" width="{width:.1f}" '
            f'height="{row_height:.1f}" fill="{color}" opacity="{opacity}" />'
        )

    def row_label(text, row_top):
        return (
            f'<text x="18" y="{row_top + 18:.1f}" font-size="13" '
            f'font-weight="700" fill="#334155">{html.escape(text)}</text>'
        )

    rows = [
        ("Ripple LFP", ripple_raw, 28, 145, "#111827"),
        ("Sharp-wave LFP", sharp_wave_raw, 198, 145, "#047857"),
        ("Detector features", ripple_feature, 368, 145, "#991b1b"),
    ]
    overlays = []
    for row_top in [28, 198, 368]:
        overlays.extend(
            span_rect(start, stop, row_top, 145, "#16a34a", "0.16")
            for start, stop in truth
        )
        overlays.extend(
            span_rect(event["start"], event["stop"], row_top, 145, "#f59e0b", "0.28")
            for event in detected_events
        )

    backgrounds = []
    series = []
    for label, values, row_top, row_height, color in rows:
        backgrounds.append(
            f'<rect x="70" y="{row_top}" width="860" height="{row_height}" '
            'rx="8" fill="#f8fafc" stroke="#e2e8f0" />'
        )
        series.append(row_label(label, row_top))
        series.append(polyline(values, row_top + 8, row_height - 16, color))
    series.append(
        polyline(sharp_wave_feature, 376, 129, "#047857", stroke_width=1.1)
    )

    ticks = []
    for tick in [0.0, 0.5, 1.0, 1.5]:
        x = 70.0 + (tick / DURATION) * 860.0
        ticks.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="520" y2="526" '
            'stroke="#64748b" />'
        )
        ticks.append(
            f'<text x="{x:.1f}" y="546" text-anchor="middle" '
            f'font-size="12" fill="#475569">{tick:.1f}</text>'
        )

    peak_text = ", ".join(
        f"{event['peak']:.3f}s" for event in detected_events
    )
    if not peak_text:
        peak_text = "none"

    svg = f"""
<svg viewBox="0 0 980 585" width="100%" role="img"
     aria-label="Interactive simulated sharp wave ripple detector plot"
     style="font-family: ui-sans-serif, system-ui, sans-serif;">
  <rect x="0" y="0" width="980" height="585" rx="18" fill="#ffffff" />
  <text x="70" y="22" font-size="14" font-weight="700" fill="#0f172a">
    Simulated joint SWR detection
  </text>
  {"".join(backgrounds)}
  {"".join(overlays)}
  {"".join(series)}
  {"".join(ticks)}
  <text x="500" y="572" text-anchor="middle" font-size="13" fill="#475569">
    Time (s)
  </text>
  <rect x="695" y="12" width="14" height="14" fill="#16a34a" opacity="0.25" />
  <text x="715" y="24" font-size="12" fill="#475569">Simulated SWR</text>
  <rect x="810" y="12" width="14" height="14" fill="#f59e0b" opacity="0.38" />
  <text x="830" y="24" font-size="12" fill="#475569">Detected SWR</text>
</svg>
"""

    mo.vstack(
        [
            mo.md(
                f"**Detected SWRs:** {len(detected_events)}. "
                f"**Peak times:** {peak_text}."
            ),
            mo.Html(svg),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
