# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair>=5.0.0",
#     "marimo>=0.23.0",
#     "scikit-learn>=1.2.2",
# ]
# ///

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    import os
    import time

    import marimo as mo
    import matplotlib.pyplot as plt
    import nelpy as nel
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    from neuro_py.detectors.sharp_wave_ripple import (
        _get_noise_channel as get_noise_channel,
        _get_ripple_channel as get_ripple_channel,
        _get_sharp_wave_channel as get_sharp_wave_channel,
        detect_sharp_wave_ripples,
        save_ripple_events,
    )
    from neuro_py.io import loading

    return (
        PCA,
        StandardScaler,
        TSNE,
        detect_sharp_wave_ripples,
        get_noise_channel,
        get_ripple_channel,
        get_sharp_wave_channel,
        json,
        loading,
        mo,
        nel,
        np,
        os,
        pd,
        plt,
        save_ripple_events,
        time,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # SWR embedding curation

    This local notebook detects a deliberately liberal set of candidate SWRs, embeds
    their event-level and waveform features into 2D, and lets you select a cluster
    of likely real ripples from an interactive scatter plot. Treat this as a
    curation/QC layer, not as a replacement for the detector.

    The first implementation uses Altair interval selection because Marimo exposes
    the selected dataframe through `chart.value`. True freeform lasso selection can
    be explored later with Plotly if box selection feels too limiting.

    If startup is slow because of `nelpy` or `numba`, launch with JIT disabled:

    ```powershell
    $env:NUMBA_DISABLE_JIT="1"
    marimo edit tutorials/marimo/swr_embedding_curation.py
    ```
    """)
    return


@app.cell
def _(mo):
    load_form = (
        mo.md(
            r"""
    ## 1. Load LFP for candidate curation

    Channels are zero-indexed. Leave a channel blank to infer it from CellExplorer
    channel tags where possible (`Ripple`, `SharpWave`, and `Bad`).

    {basepath}

    {ext}

    | Channel | Value |
    |---|---|
    | Ripple channel | {ripple_channel} |
    | Sharp-wave channel | {sharp_wave_channel} |
    | Noise / bad channel | {noise_channel} |

    | Curation sample | Control |
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
                    "Full session",
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
            submit_button_label="Load curation sample",
            bordered=True,
        )
    )
    load_form
    return (load_form,)


@app.cell
def _(mo, np, pd):
    def parse_optional_int(value):
        value = str(value).strip()
        if value == "":
            return None
        return int(value)

    def parse_optional_float(value):
        value = str(value).strip()
        if value == "":
            return None
        return float(value)

    def interval_intersections(first, second):
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

    def center_sample_interval(start, stop, max_duration):
        duration = min(float(max_duration), float(stop - start))
        center = (float(start) + float(stop)) / 2.0
        return center - duration / 2.0, center + duration / 2.0

    def build_manual_window(start, stop):
        return pd.DataFrame(
            [
                {
                    "window_id": 0,
                    "subsession": "manual",
                    "state": "manual",
                    "start": float(start),
                    "stop": float(stop),
                    "duration": float(stop - start),
                    "sample_start": -1,
                    "sample_stop": -1,
                }
            ]
        )

    def build_full_session_windows(epoch_df):
        records = []
        for epoch_idx, epoch in epoch_df.reset_index(drop=True).iterrows():
            epoch_name = epoch.get("name", f"epoch_{epoch_idx}")
            if pd.isna(epoch_name):
                epoch_name = f"epoch_{epoch_idx}"
            start = float(epoch["startTime"])
            stop = float(epoch["stopTime"])
            records.append(
                {
                    "window_id": epoch_idx,
                    "subsession": f"{epoch_idx}: {epoch_name}",
                    "state": "full_session",
                    "start": start,
                    "stop": stop,
                    "duration": stop - start,
                    "sample_start": -1,
                    "sample_stop": -1,
                }
            )
        return pd.DataFrame(records)

    def build_representative_windows(epoch_df, state_dict, selected_states, seconds):
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

                intersections = interval_intersections(epoch_interval, state_epoch.data)
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
                    sample_start, sample_stop = center_sample_interval(
                        source_start,
                        source_stop,
                        remaining,
                    )
                    records.append(
                        {
                            "window_id": window_id,
                            "subsession": epoch_label,
                            "state": state_name,
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

    def empty_events():
        return pd.DataFrame(
            columns=[
                "event_id",
                "start",
                "stop",
                "peaks",
                "center",
                "duration",
                "amplitude",
                "frequency",
                "peakNormedPower",
                "noise_peakNormedPower",
                "ripple_duration",
                "sharp_wave_peakNormedPower",
                "sharp_wave_duration",
                "window_id",
                "subsession",
                "state",
            ]
        )

    mo.md("Loaded helper functions for SWR embedding curation.")
    return (
        build_full_session_windows,
        build_manual_window,
        build_representative_windows,
        empty_events,
        parse_optional_float,
        parse_optional_int,
    )


@app.cell
def _(
    build_full_session_windows,
    build_manual_window,
    build_representative_windows,
    get_noise_channel,
    get_ripple_channel,
    get_sharp_wave_channel,
    load_form,
    loading,
    mo,
    nel,
    np,
    os,
    parse_optional_float,
    parse_optional_int,
    pd,
    time,
):
    mo.stop(
        load_form.value is None,
        mo.md("Submit the **Load curation sample** form to cache LFP windows."),
    )

    load_values = load_form.value
    basepath = os.path.normpath(load_values["basepath"])
    mo.stop(
        not os.path.isdir(basepath),
        mo.md(f"Basepath does not exist: `{basepath}`"),
    )

    ripple_channel = parse_optional_int(load_values["ripple_channel"])
    sharp_wave_channel = parse_optional_int(load_values["sharp_wave_channel"])
    noise_channel = parse_optional_int(load_values["noise_channel"])

    if ripple_channel is None:
        ripple_channel = get_ripple_channel(basepath)
    if sharp_wave_channel is None:
        sharp_wave_channel = get_sharp_wave_channel(basepath)
    if noise_channel is None:
        noise_channel = get_noise_channel(basepath)

    skipped_windows = pd.DataFrame(columns=["subsession", "state", "reason"])
    sampling_mode_name = load_values["sampling_mode"]
    sampling_mode = {
        "Representative subsession x state subset": "representative",
        "Full session": "full",
        "Manual interval": "manual",
    }[sampling_mode_name]
    if sampling_mode == "manual":
        manual_start = parse_optional_float(load_values["manual_start"])
        manual_stop = parse_optional_float(load_values["manual_stop"])
        mo.stop(
            manual_start is None or manual_stop is None,
            mo.md("Manual interval mode requires both start and stop times."),
        )
        mo.stop(
            manual_stop <= manual_start,
            mo.md("The manual stop time must be greater than the start time."),
        )
        window_table = build_manual_window(manual_start, manual_stop)
    else:
        epoch_df = loading.load_epoch(basepath)
        mo.stop(
            epoch_df.empty,
            mo.md("Could not load CellExplorer session epochs from this basepath."),
        )
        if sampling_mode == "full":
            window_table = build_full_session_windows(epoch_df)
        else:
            selected_states = list(load_values["sleep_states"])
            mo.stop(
                len(selected_states) == 0,
                mo.md("Select at least one sleep state for representative sampling."),
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
            window_table, skipped_windows = build_representative_windows(
                epoch_df=epoch_df,
                state_dict=state_dict,
                selected_states=selected_states,
                seconds=float(load_values["seconds_per_combo"]),
            )

    mo.stop(
        window_table.empty,
        mo.md("No curation windows were available for the selected settings."),
    )

    channels = [int(ripple_channel)]
    channel_roles = {"ripple": 0}
    if sharp_wave_channel is not None:
        channel_roles["sharp_wave"] = len(channels)
        channels.append(int(sharp_wave_channel))
    if noise_channel is not None:
        channel_roles["noise"] = len(channels)
        channels.append(int(noise_channel))

    curation_epoch = nel.EpochArray(window_table[["start", "stop"]].to_numpy(float))

    _load_start_time = time.perf_counter()
    lfp = loading.LFPLoader(
        basepath=basepath,
        channels=channels,
        ext=load_values["ext"],
        epoch=curation_epoch,
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
        mo.md("The selected windows did not contain any loaded LFP samples."),
    )

    cached_lfp = {
        "basepath": basepath,
        "basename": os.path.basename(basepath),
        "curated_csv": os.path.join(
            basepath,
            f"{os.path.basename(basepath)}.swr_embedding_curated.csv",
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
            mo.md("### Cached curation sample"),
            mo.ui.table(loaded_summary, selection=None),
            mo.md("### Included curation windows"),
            mo.ui.table(indexed_windows, selection=None, page_size=10),
            mo.md("### Skipped subsession x state combinations"),
            mo.ui.table(skipped_windows, selection=None, page_size=10),
        ]
    )
    return (cached_lfp,)


@app.cell
def _(cached_lfp, mo):
    candidate_form = (
        mo.md(
            r"""
    ## 2. Liberal candidate detection

    These defaults are intentionally liberal so that the embedding has both likely
    true ripples and likely false positives. Candidate detection never saves a
    CellExplorer event file.

    | Candidate parameter | Control |
    |---|---|
    | Ripple low threshold | {low_threshold} |
    | Ripple high threshold | {high_threshold} |
    | Sharp-wave low threshold | {sharp_wave_low_threshold} |
    | Sharp-wave high threshold | {sharp_wave_high_threshold} |
    | Noise threshold | {noise_threshold} |
    | Minimum duration (s) | {min_duration} |
    | Maximum duration (s) | {max_duration} |
    | Boundary mode | {boundary_mode} |
    | Require sharp wave | {require_sharp_wave} |
            """
        )
        .batch(
            low_threshold=mo.ui.slider(
                0.05,
                2.0,
                step=0.05,
                value=0.40,
                label="Ripple low threshold",
                include_input=True,
            ),
            high_threshold=mo.ui.slider(
                0.25,
                5.0,
                step=0.05,
                value=1.25,
                label="Ripple high threshold",
                include_input=True,
            ),
            sharp_wave_low_threshold=mo.ui.slider(
                0.05,
                2.0,
                step=0.05,
                value=0.20,
                label="Sharp-wave low threshold",
                include_input=True,
            ),
            sharp_wave_high_threshold=mo.ui.slider(
                0.25,
                5.0,
                step=0.05,
                value=1.25,
                label="Sharp-wave high threshold",
                include_input=True,
            ),
            noise_threshold=mo.ui.number(
                start=0.0,
                stop=10.0,
                step=0.1,
                value=6.0 if cached_lfp["noise_signal"] is not None else None,
                label="Noise threshold",
            ),
            min_duration=mo.ui.slider(
                0.005,
                0.080,
                step=0.005,
                value=0.010,
                label="Minimum duration",
                include_input=True,
            ),
            max_duration=mo.ui.slider(
                0.050,
                0.600,
                step=0.010,
                value=0.300,
                label="Maximum duration",
                include_input=True,
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
        )
        .form(
            submit_button_label="Run liberal candidate detection",
            bordered=True,
        )
    )
    candidate_form
    return (candidate_form,)


@app.cell
def _(candidate_form, mo):
    mo.stop(
        candidate_form.value is None,
        mo.md("Submit the **Run liberal candidate detection** form."),
    )

    _values = candidate_form.value
    candidate_params = {
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
        "boundary_mode": _values["boundary_mode"],
        "require_sharp_wave": bool(_values["require_sharp_wave"]),
        "merge_gap": 0.020,
        "min_inter_event_interval": 0.0,
        "reject_edge_events": True,
        "reject_artifacts": True,
        "save_mat": False,
    }
    return (candidate_params,)


@app.cell
def _(
    cached_lfp,
    candidate_params,
    detect_sharp_wave_ripples,
    empty_events,
    mo,
    pd,
    time,
):
    _candidate_start_time = time.perf_counter()
    candidate_tables = []

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
            **candidate_params,
        )
        if events.empty:
            continue
        events = events.copy()
        events["window_id"] = int(_window.window_id)
        events["subsession"] = _window.subsession
        events["state"] = _window.state
        candidate_tables.append(events)

    candidate_seconds = time.perf_counter() - _candidate_start_time
    if candidate_tables:
        candidate_events = (
            pd.concat(candidate_tables, ignore_index=True)
            .sort_values("start")
            .reset_index(drop=True)
        )
        candidate_events.insert(0, "event_id", range(len(candidate_events)))
    else:
        candidate_events = empty_events()

    candidate_summary = pd.DataFrame(
        [
            ("candidate_count", len(candidate_events)),
            ("candidate_seconds", f"{candidate_seconds:.2f}"),
            ("sampled_windows", len(cached_lfp["window_table"])),
            (
                "sampled_duration_s",
                f"{cached_lfp['window_table']['duration'].sum():.1f}",
            ),
            ("save_status", "disabled for liberal candidate detection"),
        ],
        columns=["metric", "value"],
    )

    mo.vstack(
        [
            mo.md("### Liberal candidate detection summary"),
            mo.ui.table(candidate_summary, selection=None),
        ]
    )
    return (candidate_events,)


@app.cell
def _(candidate_events, mo):
    mo.stop(
        candidate_events.empty,
        mo.md("No candidates were detected. Lower thresholds or check channel selection."),
    )

    embedding_form = (
        mo.md(
            r"""
    ## 3. Extract features and embed candidates

    Waveform snippets are centered on the detected ripple peak, z-scored per event,
    and downsampled before PCA. The 2D scatter is for curation, so always validate
    clusters by previewing waveforms.

    | Embedding setting | Control |
    |---|---|
    | Method | {method} |
    | Snippet window (s) | {snippet_window} |
    | Snippet samples | {snippet_samples} |
    | Waveform PCs | {waveform_pcs} |
    | Color by | {color_by} |
            """
        )
        .batch(
            method=mo.ui.dropdown(
                options=["PCA", "t-SNE"],
                value="PCA",
                label="Embedding method",
            ),
            snippet_window=mo.ui.slider(
                0.050,
                0.500,
                step=0.025,
                value=0.200,
                label="Snippet window",
                include_input=True,
            ),
            snippet_samples=mo.ui.slider(
                20,
                200,
                step=10,
                value=80,
                label="Snippet samples",
                include_input=True,
            ),
            waveform_pcs=mo.ui.slider(
                2,
                20,
                step=1,
                value=8,
                label="Waveform PCs",
                include_input=True,
            ),
            color_by=mo.ui.dropdown(
                options=[
                    "state",
                    "subsession",
                    "duration",
                    "peakNormedPower",
                    "sharp_wave_peakNormedPower",
                    "frequency",
                ],
                value="state",
                label="Color by",
            ),
        )
        .form(
            submit_button_label="Compute embedding",
            bordered=True,
        )
    )
    embedding_form
    return (embedding_form,)


@app.cell
def _(StandardScaler, cached_lfp, candidate_events, embedding_form, mo, np):
    def _zscore_vector(values):
        values = np.asarray(values, dtype=float)
        scale = np.nanstd(values)
        if not np.isfinite(scale) or scale == 0:
            return np.zeros_like(values, dtype=float)
        return (values - np.nanmean(values)) / scale

    def _resample(values, n_samples):
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            return np.zeros(int(n_samples), dtype=float)
        source_x = np.linspace(0.0, 1.0, values.size)
        target_x = np.linspace(0.0, 1.0, int(n_samples))
        return np.interp(target_x, source_x, values)

    def _extract_snippet(signal_values, timestamps, peak, window, n_samples):
        if signal_values is None:
            return np.zeros(int(n_samples), dtype=float)
        mask = (timestamps >= peak - window / 2.0) & (
            timestamps <= peak + window / 2.0
        )
        return _resample(_zscore_vector(signal_values[mask]), n_samples)

    def _safe_feature_matrix(frame, columns):
        available = [column for column in columns if column in frame.columns]
        values = frame[available].copy()
        values = values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return values, available

    mo.stop(
        embedding_form.value is None,
        mo.md("Submit the **Compute embedding** form."),
    )

    embedding_values = embedding_form.value
    snippet_window = float(embedding_values["snippet_window"])
    snippet_samples = int(embedding_values["snippet_samples"])
    waveforms = []

    for event in candidate_events.itertuples(index=False):
        peak = float(event.peaks)
        ripple_snippet = _extract_snippet(
            cached_lfp["ripple_signal"],
            cached_lfp["timestamps"],
            peak,
            snippet_window,
            snippet_samples,
        )
        sharp_wave_snippet = _extract_snippet(
            cached_lfp["sharp_wave_signal"],
            cached_lfp["timestamps"],
            peak,
            snippet_window,
            snippet_samples,
        )
        waveforms.append(np.concatenate([ripple_snippet, sharp_wave_snippet]))

    waveform_matrix = np.vstack(waveforms)

    scalar_columns = [
        "duration",
        "amplitude",
        "frequency",
        "peakNormedPower",
        "noise_peakNormedPower",
        "ripple_duration",
        "sharp_wave_peakNormedPower",
        "sharp_wave_duration",
    ]
    scalar_features, scalar_feature_names = _safe_feature_matrix(
        candidate_events, scalar_columns
    )
    scalar_matrix = StandardScaler().fit_transform(scalar_features)

    feature_context = {
        "waveform_matrix": waveform_matrix,
        "scalar_matrix": scalar_matrix,
        "scalar_feature_names": scalar_feature_names,
        "embedding_values": embedding_values,
    }
    return (feature_context,)


@app.cell
def _(PCA, TSNE, candidate_events, feature_context, mo, np):
    _embedding_values = feature_context["embedding_values"]
    _waveform_matrix = feature_context["waveform_matrix"]
    _scalar_matrix = feature_context["scalar_matrix"]

    n_candidates = _waveform_matrix.shape[0]
    n_waveform_pcs = min(
        int(_embedding_values["waveform_pcs"]),
        _waveform_matrix.shape[0],
        _waveform_matrix.shape[1],
    )

    waveform_scores = PCA(n_components=n_waveform_pcs).fit_transform(_waveform_matrix)
    combined_features = np.hstack([_scalar_matrix, waveform_scores])

    if _embedding_values["method"] == "t-SNE" and n_candidates >= 4:
        perplexity = min(30.0, max(2.0, (n_candidates - 1) / 3.0))
        embedding = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=0,
        ).fit_transform(combined_features)
    else:
        if combined_features.shape[0] == 1:
            embedding = np.zeros((1, 2), dtype=float)
        else:
            embedding = PCA(n_components=2).fit_transform(combined_features)

    embedding_df = candidate_events.copy()
    embedding_df["embed_x"] = embedding[:, 0]
    embedding_df["embed_y"] = embedding[:, 1]

    mo.md(
        f"Computed a 2D {_embedding_values['method']} embedding for "
        f"{len(embedding_df)} candidates."
    )
    return (embedding_df,)


@app.cell
def _(embedding_df, feature_context, mo):
    import altair as alt

    alt.data_transformers.disable_max_rows()

    _color_by = feature_context["embedding_values"]["color_by"]
    _color_type = "N" if _color_by in {"state", "subsession"} else "Q"

    chart = mo.ui.altair_chart(
        alt.Chart(embedding_df)
        .mark_circle(size=42, opacity=0.75)
        .encode(
            x=alt.X("embed_x:Q", title="Embedding 1"),
            y=alt.Y("embed_y:Q", title="Embedding 2"),
            color=alt.Color(f"{_color_by}:{_color_type}", title=_color_by),
            tooltip=[
                "event_id:Q",
                "peaks:Q",
                "duration:Q",
                "peakNormedPower:Q",
                "sharp_wave_peakNormedPower:Q",
                "state:N",
                "subsession:N",
            ],
        )
        .properties(width=760, height=560),
        chart_selection="interval",
    )
    mo.vstack(
        [
            mo.md(
                "## 4. Select likely real SWRs\n\n"
                "Drag a box over the candidate cluster you want to keep. "
                "The selected rows become available below as `chart.value`."
            ),
            chart,
        ]
    )
    return (chart,)


@app.cell
def _(chart, embedding_df, mo, pd):
    selected_events = chart.value.copy()
    if len(selected_events) == 0:
        selected_events = pd.DataFrame(columns=embedding_df.columns)
    selected_ids = set(selected_events["event_id"].astype(int).tolist())
    rejected_events = embedding_df[
        ~embedding_df["event_id"].astype(int).isin(selected_ids)
    ].copy()

    selection_summary = pd.DataFrame(
        [
            ("selected_count", len(selected_events)),
            ("rejected_count", len(rejected_events)),
            ("total_candidates", len(embedding_df)),
        ],
        columns=["metric", "value"],
    )

    mo.vstack(
        [
            mo.md("### Selection summary"),
            mo.ui.table(selection_summary, selection=None),
            mo.md("### Selected candidates"),
            mo.ui.table(selected_events, selection=None, page_size=10),
        ]
    )
    return rejected_events, selected_events


@app.cell
def _(mo, selected_events):
    mo.stop(
        len(selected_events) == 0,
        mo.md("Select candidate points in the embedding to preview and save them."),
    )

    preview_form = (
        mo.md(
            r"""
    ## 5. Preview selected and rejected candidates

    | Preview setting | Control |
    |---|---|
    | Examples per group | {examples_per_group} |
    | Window around peak (s) | {window} |
    | Random seed | {seed} |
            """
        )
        .batch(
            examples_per_group=mo.ui.slider(
                1,
                40,
                step=1,
                value=12,
                label="Examples per group",
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
            submit_button_label="Refresh previews",
            bordered=True,
        )
    )
    preview_form
    return (preview_form,)


@app.cell
def _(cached_lfp, mo, np, plt, preview_form, rejected_events, selected_events):
    def _plot_event_examples(events, title, n_examples, plot_window, seed):
        fig, axes = plt.subplots(
            1,
            max(1, n_examples),
            figsize=(max(6, 1.25 * max(1, n_examples)), 2.0),
            squeeze=False,
        )
        for axis in axes.flat:
            axis.set_axis_off()
        if len(events) == 0:
            fig.suptitle(f"{title}: no events")
            return fig

        rng = np.random.default_rng(seed)
        chosen = rng.choice(
            events.index.to_numpy(),
            size=min(n_examples, len(events)),
            replace=False,
        )
        timestamps = cached_lfp["timestamps"]
        ripple_signal = cached_lfp["ripple_signal"]
        sharp_wave_signal = cached_lfp["sharp_wave_signal"]

        for axis, event_index in zip(axes.flat, chosen):
            event = events.loc[event_index]
            peak = float(event["peaks"])
            start = float(event["start"])
            stop = float(event["stop"])
            mask = (timestamps >= peak - plot_window / 2.0) & (
                timestamps <= peak + plot_window / 2.0
            )
            if not np.any(mask):
                continue
            local_time = timestamps[mask] - peak
            ripple = ripple_signal[mask] - np.nanmedian(ripple_signal[mask])

            axis.set_axis_on()
            axis.axvspan(start - peak, stop - peak, color="#f59e0b", alpha=0.25)
            axis.axvline(0.0, color="#111827", linewidth=0.8, alpha=0.7)
            axis.plot(local_time, ripple, color="#111827", linewidth=0.7)

            if sharp_wave_signal is not None:
                sharp_wave = sharp_wave_signal[mask] - np.nanmedian(
                    sharp_wave_signal[mask]
                )
                scale = np.nanstd(ripple) / np.nanstd(sharp_wave)
                if not np.isfinite(scale) or scale == 0:
                    scale = 1.0
                axis.plot(
                    local_time,
                    sharp_wave * scale - 3.0 * np.nanstd(ripple),
                    color="#047857",
                    linewidth=0.7,
                )
            axis.set_title(f"{peak:.2f}s", fontsize=7)
            axis.tick_params(labelsize=6, length=2)

        fig.suptitle(title)
        fig.tight_layout()
        return fig

    preview_values = preview_form.value
    if preview_values is None:
        preview_values = {"examples_per_group": 12, "window": 0.200, "seed": 0}

    n_examples = int(preview_values["examples_per_group"])
    plot_window = float(preview_values["window"])
    seed = int(preview_values["seed"])

    selected_fig = _plot_event_examples(
        selected_events,
        "Selected candidate examples",
        n_examples,
        plot_window,
        seed,
    )
    rejected_fig = _plot_event_examples(
        rejected_events,
        "Rejected candidate examples",
        n_examples,
        plot_window,
        seed + 1,
    )

    mo.vstack(
        [
            mo.md(
                "Black: ripple channel. Green: sharp-wave channel, scaled and offset. "
                "Orange span: detected candidate interval. Vertical line: peak."
            ),
            selected_fig,
            rejected_fig,
        ]
    )
    return


@app.cell
def _(mo, selected_events):
    save_form = (
        mo.md(
            r"""
    ## 6. Save curated selection

    Saving writes the selected event IDs and feature/selection table to CSV. You can
    also write a CellExplorer event file with `save_ripple_events`.

    | Save option | Control |
    |---|---|
    | Save selected CSV | {save_csv} |
    | Save CellExplorer MAT | {save_mat} |
    | MAT event name | {event_name} |
            """
        )
        .batch(
            save_csv=mo.ui.checkbox(value=True, label="Save selected CSV"),
            save_mat=mo.ui.checkbox(value=False, label="Save CellExplorer MAT"),
            event_name=mo.ui.text(
                value="ripples_curated",
                label="MAT event name",
                placeholder="use ripples to overwrite default ripples file",
            ),
        )
        .form(
            submit_button_label="Save curated events",
            bordered=True,
        )
    )
    selected_events
    save_form
    return (save_form,)


@app.cell
def _(
    cached_lfp,
    candidate_params,
    feature_context,
    json,
    mo,
    save_form,
    save_ripple_events,
    selected_events,
):
    mo.stop(
        save_form.value is None,
        mo.md("Submit the **Save curated events** form when you are ready."),
    )
    mo.stop(
        len(selected_events) == 0,
        mo.md("No selected events are available to save."),
    )

    save_values = save_form.value
    saved_paths = []
    curation_metadata = {
        "candidate_params": candidate_params,
        "embedding_values": feature_context["embedding_values"],
        "scalar_feature_names": feature_context["scalar_feature_names"],
        "selected_event_ids": selected_events["event_id"].astype(int).tolist(),
        "sampling_mode": cached_lfp["sampling_mode"],
        "window_table": cached_lfp["window_table"].to_dict(orient="records"),
    }

    selected_to_save = selected_events.copy().sort_values("start").reset_index(drop=True)
    metadata_json = json.dumps(curation_metadata, default=str)
    selected_to_save["curation_metadata"] = metadata_json

    if bool(save_values["save_csv"]):
        selected_to_save.to_csv(cached_lfp["curated_csv"], index=False)
        saved_paths.append(cached_lfp["curated_csv"])

    if bool(save_values["save_mat"]):
        mat_path = save_ripple_events(
            selected_to_save,
            basepath=cached_lfp["basepath"],
            detection_name="detect_sharp_wave_ripples_embedding_curated",
            detection_params=curation_metadata,
            ripple_channel=cached_lfp["ripple_channel"],
            detection_epochs=cached_lfp["window_table"][["start", "stop"]].to_numpy(
                float
            ),
            event_name=str(save_values["event_name"]).strip() or "ripples_curated",
        )
        saved_paths.append(mat_path)

    mo.md("Saved curated outputs:\n\n" + "\n".join(f"- `{path}`" for path in saved_paths))
    return


if __name__ == "__main__":
    app.run()
