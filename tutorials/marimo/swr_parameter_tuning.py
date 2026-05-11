# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.23.0",
# ]
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import nelpy as nel
    import numpy as np
    import pandas as pd

    from neuro_py.detectors.sharp_wave_ripple import (
        _get_noise_channel,
        _get_ripple_channel,
        _get_sharp_wave_channel,
        detect_sharp_wave_ripples,
    )
    from neuro_py.io import loading

    return detect_sharp_wave_ripples, loading, mo, nel, np, os, pd, plt, time


@app.cell
def _(mo):
    mo.md(r"""
    # Local SWR parameter tuning

    Use this notebook when you want to tune the joint sharp wave ripple detector on
    real LFP data without paying the full-session runtime after every parameter
    change. The default loading mode samples a representative subset from every
    CellExplorer subsession crossed with the major sleep states.

    This is a **local Marimo notebook**, not a browser-WASM docs demo. It uses the
    real `neuro_py.detectors.sharp_wave_ripple.detect_sharp_wave_ripples` function
    and can read local or network paths such as `S:\data\HMC\HMC1\day8`.

    If startup is slow because of `nelpy` or `numba`, launch with JIT disabled:

    ```powershell
    $env:NUMBA_DISABLE_JIT="1"
    marimo edit tutorials/marimo/swr_parameter_tuning.py
    ```
    """)
    return


@app.cell
def _(mo):
    load_form = (
        mo.md(
            r"""
    ## 1. Load and cache tuning LFP

    Channels are zero-indexed. Leave a channel blank to infer it from CellExplorer
    channel tags where possible (`Ripple`, `SharpWave`, and `Bad`).

    {basepath}

    {ext}

    | Channel | Value |
    |---|---|
    | Ripple channel | {ripple_channel} |
    | Sharp-wave channel | {sharp_wave_channel} |
    | Noise / bad channel | {noise_channel} |

    | Tuning sample | Control |
    |---|---|
    | Sampling mode | {sampling_mode} |
    | Seconds per subsession x state | {seconds_per_combo} |
    | Sleep states | {sleep_states} |
    | Manual start time, seconds | {manual_start} |
    | Manual stop time, seconds | {manual_stop} |
            """
        )
        .batch(
            basepath=mo.ui.text(
                value=r"S:\data\HMC\HMC1\day8",
                label="Basepath",
                full_width=True,
            ),
            ext=mo.ui.dropdown(
                options=["lfp", "dat"],
                value="lfp",
                label="File extension",
            ),
            ripple_channel=mo.ui.text(
                value="",
                label="Ripple channel",
                placeholder="blank = infer",
            ),
            sharp_wave_channel=mo.ui.text(
                value="",
                label="Sharp-wave channel",
                placeholder="blank = infer",
            ),
            noise_channel=mo.ui.text(
                value="",
                label="Noise channel",
                placeholder="blank = infer Bad tag",
            ),
            sampling_mode=mo.ui.dropdown(
                options=[
                    "Representative subsession x state subset",
                    "Manual interval",
                ],
                value="Representative subsession x state subset",
                label="Sampling mode",
            ),
            seconds_per_combo=mo.ui.slider(
                10.0,
                300.0,
                step=10.0,
                value=60.0,
                label="Seconds per combination",
                include_input=True,
            ),
            sleep_states=mo.ui.multiselect(
                options=["WAKEstate", "NREMstate", "REMstate", "THETA", "nonTHETA"],
                value=["WAKEstate", "NREMstate", "REMstate"],
                label="Sleep states",
                full_width=True,
            ),
            manual_start=mo.ui.text(
                value="",
                label="Manual start",
                placeholder="used only for manual interval",
            ),
            manual_stop=mo.ui.text(
                value="",
                label="Manual stop",
                placeholder="used only for manual interval",
            ),
        )
        .form(
            submit_button_label="Load tuning subset",
            bordered=True,
        )
    )
    load_form
    return (load_form,)


@app.cell
def _(mo, np, pd):
    def _parse_optional_int(value):
        value = str(value).strip()
        if value == "":
            return None
        return int(value)

    def _parse_optional_float(value):
        value = str(value).strip()
        if value == "":
            return None
        return float(value)

    def _interval_intersections(first, second):
        first = np.asarray(first, dtype=float)
        second = np.asarray(second, dtype=float)
        if first.size == 0 or second.size == 0:
            return np.empty((0, 2), dtype=float)

        intersections = []
        for left_start, left_stop in np.atleast_2d(first):
            for right_start, right_stop in np.atleast_2d(second):
                start = max(float(left_start), float(right_start))
                stop = min(float(left_stop), float(right_stop))
                if stop > start:
                    intersections.append((start, stop))
        if not intersections:
            return np.empty((0, 2), dtype=float)
        return np.asarray(intersections, dtype=float)

    def _center_sample_interval(start, stop, max_duration):
        duration = min(float(max_duration), float(stop - start))
        center = (float(start) + float(stop)) / 2.0
        sampled_start = center - duration / 2.0
        sampled_stop = center + duration / 2.0
        return sampled_start, sampled_stop

    def _build_manual_window(start, stop):
        return pd.DataFrame(
            [
                {
                    "window_id": 0,
                    "subsession": "manual",
                    "state": "manual",
                    "source_start": float(start),
                    "source_stop": float(stop),
                    "start": float(start),
                    "stop": float(stop),
                    "duration": float(stop - start),
                    "sample_start": -1,
                    "sample_stop": -1,
                }
            ]
        )

    def _build_representative_windows(epoch_df, state_dict, selected_states, seconds):
        records = []
        skipped = []
        window_id = 0

        for epoch_idx, epoch in epoch_df.reset_index(drop=True).iterrows():
            epoch_start = float(epoch["startTime"])
            epoch_stop = float(epoch["stopTime"])
            epoch_name = epoch.get("name", f"epoch_{epoch_idx}")
            if pd.isna(epoch_name):
                epoch_name = f"epoch_{epoch_idx}"
            epoch_label = f"{epoch_idx}: {epoch_name}"
            epoch_interval = np.asarray([[epoch_start, epoch_stop]], dtype=float)

            for state_name in selected_states:
                state_epoch = state_dict.get(state_name)
                if state_epoch is None or state_epoch.isempty:
                    skipped.append(
                        {
                            "subsession": epoch_label,
                            "state": state_name,
                            "reason": "state unavailable",
                        }
                    )
                    continue

                intersections = _interval_intersections(epoch_interval, state_epoch.data)
                if intersections.size == 0:
                    skipped.append(
                        {
                            "subsession": epoch_label,
                            "state": state_name,
                            "reason": "no overlap",
                        }
                    )
                    continue

                durations = intersections[:, 1] - intersections[:, 0]
                order = np.argsort(durations)[::-1]
                remaining = float(seconds)
                for interval_idx in order:
                    if remaining <= 0:
                        break
                    source_start, source_stop = intersections[interval_idx]
                    sample_start, sample_stop = _center_sample_interval(
                        source_start,
                        source_stop,
                        remaining,
                    )
                    records.append(
                        {
                            "window_id": window_id,
                            "subsession": epoch_label,
                            "state": state_name,
                            "source_start": float(source_start),
                            "source_stop": float(source_stop),
                            "start": float(sample_start),
                            "stop": float(sample_stop),
                            "duration": float(sample_stop - sample_start),
                            "sample_start": -1,
                            "sample_stop": -1,
                        }
                    )
                    window_id += 1
                    remaining -= float(sample_stop - sample_start)

        return pd.DataFrame(records), pd.DataFrame(skipped)

    def _empty_events():
        return pd.DataFrame(
            columns=[
                "start",
                "stop",
                "peaks",
                "duration",
                "peakNormedPower",
                "sharp_wave_peakNormedPower",
                "window_id",
                "subsession",
                "state",
            ]
        )

    mo.md("Loaded helper functions for representative tuning windows.")
    return


@app.cell
def _(load_form, loading, mo, nel, np, os, pd, time):
    mo.stop(
        load_form.value is None,
        mo.md("Submit the **Load tuning subset** form to cache LFP windows."),
    )

    load_values = load_form.value
    basepath = os.path.normpath(load_values["basepath"])
    mo.stop(
        not os.path.isdir(basepath),
        mo.md(f"Basepath does not exist: `{basepath}`"),
    )

    ripple_channel = _parse_optional_int(load_values["ripple_channel"])
    sharp_wave_channel = _parse_optional_int(load_values["sharp_wave_channel"])
    noise_channel = _parse_optional_int(load_values["noise_channel"])

    if ripple_channel is None:
        ripple_channel = _get_ripple_channel(basepath)
    if sharp_wave_channel is None:
        sharp_wave_channel = _get_sharp_wave_channel(basepath)
    if noise_channel is None:
        noise_channel = _get_noise_channel(basepath)

    skipped_windows = pd.DataFrame(columns=["subsession", "state", "reason"])
    sampling_mode_name = load_values["sampling_mode"]
    sampling_mode = {
        "Representative subsession x state subset": "representative",
        "Manual interval": "manual",
    }[sampling_mode_name]

    if sampling_mode == "manual":
        manual_start = _parse_optional_float(load_values["manual_start"])
        manual_stop = _parse_optional_float(load_values["manual_stop"])
        mo.stop(
            manual_start is None or manual_stop is None,
            mo.md("Manual interval mode requires both start and stop times."),
        )
        mo.stop(
            manual_stop <= manual_start,
            mo.md("The manual stop time must be greater than the start time."),
        )
        window_table = _build_manual_window(manual_start, manual_stop)
    else:
        selected_states = list(load_values["sleep_states"])
        mo.stop(
            len(selected_states) == 0,
            mo.md("Select at least one sleep state for representative sampling."),
        )

        epoch_df = loading.load_epoch(basepath)
        mo.stop(
            epoch_df.empty,
            mo.md("Could not load CellExplorer session epochs from this basepath."),
        )

        state_dict = loading.load_SleepState_states(
            basepath,
            return_epoch_array=True,
            states_list=selected_states,
        )
        mo.stop(
            state_dict is None,
            mo.md("Could not load SleepState intervals from this basepath."),
        )

        window_table, skipped_windows = _build_representative_windows(
            epoch_df=epoch_df,
            state_dict=state_dict,
            selected_states=selected_states,
            seconds=float(load_values["seconds_per_combo"]),
        )

    mo.stop(
        window_table.empty,
        mo.md("No tuning windows were available for the selected sampling settings."),
    )

    channels = [int(ripple_channel)]
    channel_roles = {"ripple": 0}
    if sharp_wave_channel is not None:
        channel_roles["sharp_wave"] = len(channels)
        channels.append(int(sharp_wave_channel))
    if noise_channel is not None:
        channel_roles["noise"] = len(channels)
        channels.append(int(noise_channel))

    tuning_epoch = nel.EpochArray(window_table[["start", "stop"]].to_numpy(float))

    _load_start_time = time.perf_counter()
    lfp = loading.LFPLoader(
        basepath=basepath,
        channels=channels,
        ext=load_values["ext"],
        epoch=tuning_epoch,
    )
    load_seconds = time.perf_counter() - _load_start_time

    lfp_data = np.asarray(lfp.data, dtype=float)
    timestamps = np.asarray(lfp.abscissa_vals, dtype=float)

    indexed_windows = window_table.copy()
    sample_starts = []
    sample_stops = []
    for _window in indexed_windows.itertuples(index=False):
        indices = np.flatnonzero(
            (timestamps >= float(_window.start)) & (timestamps <= float(_window.stop))
        )
        sample_starts.append(int(indices[0]) if indices.size else -1)
        sample_stops.append(int(indices[-1] + 1) if indices.size else -1)
    indexed_windows["sample_start"] = sample_starts
    indexed_windows["sample_stop"] = sample_stops
    indexed_windows = indexed_windows[indexed_windows["sample_start"] >= 0].reset_index(
        drop=True
    )

    mo.stop(
        indexed_windows.empty,
        mo.md("The selected tuning windows did not contain any loaded LFP samples."),
    )

    cached_lfp = {
        "basepath": basepath,
        "basename": os.path.basename(basepath),
        "event_file": os.path.join(
            basepath,
            f"{os.path.basename(basepath)}.ripples.events.mat",
        ),
        "fs": float(lfp.fs),
        "timestamps": timestamps,
        "ripple_signal": lfp_data[channel_roles["ripple"]],
        "sharp_wave_signal": (
            lfp_data[channel_roles["sharp_wave"]]
            if "sharp_wave" in channel_roles
            else None
        ),
        "noise_signal": (
            lfp_data[channel_roles["noise"]] if "noise" in channel_roles else None
        ),
        "ripple_channel": int(ripple_channel),
        "sharp_wave_channel": (
            int(sharp_wave_channel) if sharp_wave_channel is not None else None
        ),
        "noise_channel": int(noise_channel) if noise_channel is not None else None,
        "channels": channels,
        "sampling_mode": sampling_mode,
        "window_table": indexed_windows,
        "skipped_windows": skipped_windows,
        "load_seconds": load_seconds,
    }

    total_duration = float(indexed_windows["duration"].sum())
    loaded_summary = pd.DataFrame(
        [
            ("basepath", cached_lfp["basepath"]),
            ("sampling_mode", cached_lfp["sampling_mode"]),
            ("sampled_windows", len(indexed_windows)),
            ("sampled_duration_s", f"{total_duration:.1f}"),
            ("samples", f"{timestamps.size:,}"),
            ("fs_hz", f"{cached_lfp['fs']:.3f}"),
            ("ripple_channel", cached_lfp["ripple_channel"]),
            ("sharp_wave_channel", cached_lfp["sharp_wave_channel"]),
            ("noise_channel", cached_lfp["noise_channel"]),
            ("load_seconds", f"{load_seconds:.2f}"),
            ("skipped_combinations", len(skipped_windows)),
        ],
        columns=["field", "value"],
    )

    mo.vstack(
        [
            mo.md("### Cached tuning subset"),
            mo.ui.table(loaded_summary, selection=None),
            mo.md("### Included tuning windows"),
            mo.ui.table(indexed_windows, selection=None, page_size=10),
            mo.md("### Skipped subsession x state combinations"),
            mo.ui.table(skipped_windows, selection=None, page_size=10),
            mo.md(
                "Detector tuning below runs independently inside each sampled "
                "window, so filters do not bleed across gaps."
            ),
        ]
    )
    return (cached_lfp,)


@app.cell
def _(cached_lfp, mo):
    detect_form = (
        mo.md(
            r"""
    ## 2. Tune detector parameters

    Submit this form to rerun `detect_sharp_wave_ripples` on the cached tuning
    windows. Changing sliders alone does not rerun detection.

    | Ripple detection | Control |
    |---|---|
    | Low threshold | {low_threshold} |
    | High threshold | {high_threshold} |
    | Minimum duration (s) | {min_duration} |
    | Maximum duration (s) | {max_duration} |
    | Merge gap (s) | {merge_gap} |

    | Sharp-wave detection | Control |
    |---|---|
    | Low threshold | {sharp_wave_low_threshold} |
    | High threshold | {sharp_wave_high_threshold} |
    | Polarity | {sharp_wave_polarity} |
    | Boundary mode | {boundary_mode} |
    | Require sharp wave | {require_sharp_wave} |

    | Robustness and artifacts | Control |
    |---|---|
    | Noise threshold | {noise_threshold} |
    | Minimum inter-event interval (s) | {min_inter_event_interval} |
    | Threshold mode | {threshold_mode} |
    | Local window (s) | {local_window} |
    | Reject edge events | {reject_edge_events} |
    | Reject artifacts | {reject_artifacts} |
            """
        )
        .batch(
            low_threshold=mo.ui.slider(
                0.1,
                3.0,
                step=0.05,
                value=0.75,
                label="Low threshold",
                include_input=True,
            ),
            high_threshold=mo.ui.slider(
                0.5,
                8.0,
                step=0.1,
                value=2.5,
                label="High threshold",
                include_input=True,
            ),
            min_duration=mo.ui.slider(
                0.005,
                0.080,
                step=0.005,
                value=0.015,
                label="Min duration",
                include_input=True,
            ),
            max_duration=mo.ui.slider(
                0.050,
                0.500,
                step=0.010,
                value=0.250,
                label="Max duration",
                include_input=True,
            ),
            merge_gap=mo.ui.slider(
                0.0,
                0.100,
                step=0.005,
                value=0.020,
                label="Merge gap",
                include_input=True,
            ),
            sharp_wave_low_threshold=mo.ui.slider(
                0.1,
                3.0,
                step=0.05,
                value=0.4,
                label="Sharp-wave low",
                include_input=True,
            ),
            sharp_wave_high_threshold=mo.ui.slider(
                0.5,
                8.0,
                step=0.1,
                value=2.5,
                label="Sharp-wave high",
                include_input=True,
            ),
            sharp_wave_polarity=mo.ui.dropdown(
                options=["negative", "positive", "both"],
                value="negative",
                label="Sharp-wave polarity",
            ),
            boundary_mode=mo.ui.dropdown(
                options=["sharp_wave", "union"],
                value="sharp_wave",
                label="Boundary mode",
            ),
            require_sharp_wave=mo.ui.checkbox(
                value=cached_lfp["sharp_wave_signal"] is not None,
                label="Require sharp wave",
            ),
            noise_threshold=mo.ui.number(
                start=0.0,
                stop=8.0,
                step=0.1,
                value=4.0 if cached_lfp["noise_signal"] is not None else None,
                label="Noise threshold",
            ),
            min_inter_event_interval=mo.ui.slider(
                0.0,
                0.250,
                step=0.005,
                value=0.050,
                label="Minimum inter-event interval",
                include_input=True,
            ),
            threshold_mode=mo.ui.dropdown(
                options=["global", "local"],
                value="global",
                label="Threshold mode",
            ),
            local_window=mo.ui.slider(
                0.5,
                20.0,
                step=0.5,
                value=5.0,
                label="Local window",
                include_input=True,
            ),
            reject_edge_events=mo.ui.checkbox(
                value=True,
                label="Reject edge events",
            ),
            reject_artifacts=mo.ui.checkbox(
                value=True,
                label="Reject artifacts",
            ),
        )
        .form(
            submit_button_label="Run tuning detection",
            bordered=True,
        )
    )
    detect_form
    return (detect_form,)


@app.cell
def _(detect_form, mo):
    mo.stop(
        detect_form.value is None,
        mo.md("Submit the **Run tuning detection** form to detect SWRs."),
    )

    _values = detect_form.value
    detector_params = {
        "low_threshold": float(_values["low_threshold"]),
        "high_threshold": float(_values["high_threshold"]),
        "sharp_wave_low_threshold": float(_values["sharp_wave_low_threshold"]),
        "sharp_wave_high_threshold": float(_values["sharp_wave_high_threshold"]),
        "noise_threshold": (
            None
            if _values["noise_threshold"] is None
            else float(_values["noise_threshold"])
        ),
        "min_duration": float(_values["min_duration"]),
        "max_duration": float(_values["max_duration"]),
        "merge_gap": float(_values["merge_gap"]),
        "min_inter_event_interval": float(_values["min_inter_event_interval"]),
        "threshold_mode": _values["threshold_mode"],
        "local_window": float(_values["local_window"]),
        "sharp_wave_polarity": _values["sharp_wave_polarity"],
        "boundary_mode": _values["boundary_mode"],
        "require_sharp_wave": bool(_values["require_sharp_wave"]),
        "reject_edge_events": bool(_values["reject_edge_events"]),
        "reject_artifacts": bool(_values["reject_artifacts"]),
    }
    return (detector_params,)


@app.cell
def _(cached_lfp, detect_sharp_wave_ripples, detector_params, mo, pd, time):
    _detect_start_time = time.perf_counter()
    window_events = []

    for _window in cached_lfp["window_table"].itertuples(index=False):
        start_idx = int(_window.sample_start)
        stop_idx = int(_window.sample_stop)
        if stop_idx - start_idx < 3:
            continue

        _sharp_wave_signal = cached_lfp["sharp_wave_signal"]
        _noise_signal = cached_lfp["noise_signal"]
        events = detect_sharp_wave_ripples(
            ripple_signal=cached_lfp["ripple_signal"][start_idx:stop_idx],
            sharp_wave_signal=(
                _sharp_wave_signal[start_idx:stop_idx]
                if _sharp_wave_signal is not None
                else None
            ),
            noise_signal=(
                _noise_signal[start_idx:stop_idx] if _noise_signal is not None else None
            ),
            fs=cached_lfp["fs"],
            timestamps=cached_lfp["timestamps"][start_idx:stop_idx],
            ripple_channel=cached_lfp["ripple_channel"],
            sharp_wave_channel=cached_lfp["sharp_wave_channel"],
            noise_channel=cached_lfp["noise_channel"],
            save_mat=False,
            **detector_params,
        )
        if events.empty:
            continue
        events = events.copy()
        events["window_id"] = int(_window.window_id)
        events["subsession"] = _window.subsession
        events["state"] = _window.state
        window_events.append(events)

    detection_seconds = time.perf_counter() - _detect_start_time

    if window_events:
        detected_swr = (
            pd.concat(window_events, ignore_index=True)
            .sort_values("start")
            .reset_index(drop=True)
        )
    else:
        detected_swr = _empty_events()

    if detected_swr.empty:
        summary_table = pd.DataFrame(
            [
                ("event_count", 0),
                ("detection_seconds", f"{detection_seconds:.2f}"),
                ("sampled_windows", len(cached_lfp["window_table"])),
                ("sampled_duration_s", f"{cached_lfp['window_table']['duration'].sum():.1f}"),
                ("save_status", "disabled for tuning subset"),
            ],
            columns=["metric", "value"],
        )
    else:
        duration = detected_swr["stop"] - detected_swr["start"]
        summary_rows = [
            ("event_count", len(detected_swr)),
            ("duration_median_s", f"{duration.median():.4f}"),
            ("duration_min_s", f"{duration.min():.4f}"),
            ("duration_max_s", f"{duration.max():.4f}"),
            (
                "ripple_peak_power_median",
                f"{detected_swr['peakNormedPower'].median():.3f}",
            ),
            ("detection_seconds", f"{detection_seconds:.2f}"),
            ("sampled_windows", len(cached_lfp["window_table"])),
            ("sampled_duration_s", f"{cached_lfp['window_table']['duration'].sum():.1f}"),
            ("save_status", "disabled for tuning subset"),
        ]
        if "sharp_wave_peakNormedPower" in detected_swr:
            summary_rows.insert(
                5,
                (
                    "sharp_wave_peak_power_median",
                    f"{detected_swr['sharp_wave_peakNormedPower'].median():.3f}",
                ),
            )
        summary_table = pd.DataFrame(summary_rows, columns=["metric", "value"])

    mo.vstack(
        [
            mo.md("### Tuning detection summary"),
            mo.ui.table(summary_table, selection=None),
            mo.md(
                "Tuning detection ran independently in each sampled window and "
                "did not save a CellExplorer file. Use the final full-session "
                "section below once the parameters look good."
            ),
        ]
    )
    return (detected_swr,)


@app.cell
def _(detected_swr, mo):
    gallery_form = (
        mo.md(
            r"""
    ## 3. Random SWR gallery

    Use this gallery to quickly audit detected events. Larger grids are helpful for
    screening, but `15 x 15` can take a few seconds to render.

    | Gallery setting | Control |
    |---|---|
    | Rows | {rows} |
    | Columns | {cols} |
    | Window around peak (s) | {window} |
    | Random seed | {seed} |
            """
        )
        .batch(
            rows=mo.ui.slider(
                1,
                15,
                step=1,
                value=5,
                label="Rows",
                include_input=True,
            ),
            cols=mo.ui.slider(
                1,
                15,
                step=1,
                value=5,
                label="Columns",
                include_input=True,
            ),
            window=mo.ui.slider(
                0.050,
                0.500,
                step=0.025,
                value=0.200,
                label="Window",
                include_input=True,
            ),
            seed=mo.ui.number(
                start=0,
                stop=1_000_000,
                step=1,
                value=0,
                label="Random seed",
            ),
        )
        .form(
            submit_button_label="Resample gallery",
            bordered=True,
        )
    )
    mo.stop(
        detected_swr.empty,
        mo.md("No detected SWRs to plot. Loosen thresholds or check channel selection."),
    )
    gallery_form
    return (gallery_form,)


@app.cell
def _(cached_lfp, detected_swr, gallery_form, mo, np, plt):
    gallery_values = gallery_form.value
    if gallery_values is None:
        gallery_values = {
            "rows": 5,
            "cols": 5,
            "window": 0.200,
            "seed": 0,
        }

    n_rows = int(gallery_values["rows"])
    n_cols = int(gallery_values["cols"])
    n_tiles = min(n_rows * n_cols, len(detected_swr))
    plot_window = float(gallery_values["window"])
    rng = np.random.default_rng(int(gallery_values["seed"]))
    event_indices = np.sort(rng.choice(len(detected_swr), size=n_tiles, replace=False))

    gallery_timestamps = cached_lfp["timestamps"]
    ripple_signal = cached_lfp["ripple_signal"]
    gallery_sharp_wave_signal = cached_lfp["sharp_wave_signal"]

    fig_width = max(8.0, min(22.0, 1.35 * n_cols))
    fig_height = max(5.0, min(22.0, 1.15 * n_rows))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharex=False,
        sharey=False,
    )

    for axis in axes.flat:
        axis.set_axis_off()

    for axis, event_idx in zip(axes.flat, event_indices):
        event = detected_swr.iloc[event_idx]
        peak = float(event["peaks"])
        start = float(event["start"])
        stop = float(event["stop"])
        plot_start = peak - plot_window / 2.0
        plot_stop = peak + plot_window / 2.0
        sample_mask = (gallery_timestamps >= plot_start) & (
            gallery_timestamps <= plot_stop
        )
        if not np.any(sample_mask):
            continue

        local_time = gallery_timestamps[sample_mask] - peak
        ripple = ripple_signal[sample_mask]
        ripple = ripple - np.nanmedian(ripple)

        axis.set_axis_on()
        axis.axvspan(start - peak, stop - peak, color="#f59e0b", alpha=0.25)
        axis.axvline(0.0, color="#111827", linewidth=0.8, alpha=0.7)
        axis.plot(local_time, ripple, color="#111827", linewidth=0.7)

        if gallery_sharp_wave_signal is not None:
            sharp_wave = gallery_sharp_wave_signal[sample_mask]
            sharp_wave = sharp_wave - np.nanmedian(sharp_wave)
            scale = np.nanstd(ripple) / np.nanstd(sharp_wave)
            if not np.isfinite(scale) or scale == 0:
                scale = 1.0
            offset = 3.0 * np.nanstd(ripple)
            axis.plot(
                local_time,
                sharp_wave * scale - offset,
                color="#047857",
                linewidth=0.7,
            )

        axis.set_title(
            f"{peak:.2f}s | {event['state']} | {event['subsession']}",
            fontsize=6,
        )
        axis.tick_params(labelsize=6, length=2)

    fig.suptitle(
        f"Random detected SWR gallery ({n_tiles} of {len(detected_swr)} events)",
        fontsize=14,
    )
    fig.tight_layout()
    mo.vstack(
        [
            mo.md(
                "Black: ripple channel. Green: sharp-wave channel, scaled and offset. "
                "Orange span: detected event interval. Vertical line: detected peak."
            ),
            fig,
        ]
    )
    return


@app.cell
def _(detector_params, mo):
    final_run_form = (
        mo.md(
            r"""
    ## 4. Final full-session run

    Once the tuning subset looks good, submit this form to run the same detector
    parameters on the full session. This step reloads the full LFP and may take
    substantially longer than tuning.

    | Final run option | Control |
    |---|---|
    | Save CellExplorer event file | {save_mat} |
    | Overwrite existing event file | {overwrite} |
            """
        )
        .batch(
            save_mat=mo.ui.checkbox(
                value=True,
                label="Save CellExplorer event file",
            ),
            overwrite=mo.ui.checkbox(
                value=False,
                label="Overwrite existing event file",
            ),
        )
        .form(
            submit_button_label="Run full-session detection",
            bordered=True,
        )
    )
    detector_params
    final_run_form
    return (final_run_form,)


@app.cell
def _(
    cached_lfp,
    detect_sharp_wave_ripples,
    detector_params,
    final_run_form,
    mo,
    os,
    pd,
    time,
):
    mo.stop(
        final_run_form.value is None,
        mo.md(
            "Submit the **Run full-session detection** form only after the tuning "
            "parameters look good."
        ),
    )

    final_values = final_run_form.value
    _full_start_time = time.perf_counter()
    full_session_swr = detect_sharp_wave_ripples(
        basepath=cached_lfp["basepath"],
        ripple_channel=cached_lfp["ripple_channel"],
        sharp_wave_channel=cached_lfp["sharp_wave_channel"],
        noise_channel=cached_lfp["noise_channel"],
        save_mat=bool(final_values["save_mat"]),
        overwrite=bool(final_values["overwrite"]),
        **detector_params,
    )
    full_seconds = time.perf_counter() - _full_start_time
    save_status = "not requested"
    if bool(final_values["save_mat"]):
        save_status = (
            "exists" if os.path.exists(cached_lfp["event_file"]) else "not found"
        )

    final_summary = pd.DataFrame(
        [
            ("event_count", len(full_session_swr)),
            ("full_session_seconds", f"{full_seconds:.2f}"),
            ("event_file", cached_lfp["event_file"]),
            ("save_status", save_status),
        ],
        columns=["metric", "value"],
    )

    mo.vstack(
        [
            mo.md("### Final full-session result"),
            mo.ui.table(final_summary, selection=None),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
