import os
import tempfile

import nelpy as nel
import numpy as np
import pandas as pd
import pytest
import scipy.io as sio

from neuro_py.detectors.sharp_wave_ripple import (
    _bound_containing_index,
    _bounds_to_array,
    _enforce_min_inter_event_interval,
    _filter_events_to_detection_epochs,
    _find_local_peaks,
    _find_true_bounds,
    _get_noise_channel,
    _get_sharp_wave_channel,
    _nearest_trough,
    _select_sharp_wave_partner,
    detect_sharp_wave_ripples,
    save_ripple_events,
)
from neuro_py.io.loading import load_ripples_events


def _make_ripple_burst(
    timestamps: np.ndarray,
    center: float,
    frequency: float,
    duration: float,
    amplitude: float,
) -> np.ndarray:
    mask = np.abs(timestamps - center) <= (duration / 2.0)
    burst = np.zeros_like(timestamps, dtype=float)
    n_samples = int(mask.sum())
    if n_samples == 0:
        return burst
    window = np.hanning(n_samples)
    burst[mask] = (
        amplitude * np.sin(2 * np.pi * frequency * (timestamps[mask] - center)) * window
    )
    return burst


def _make_synthetic_ripple_session(
    centers: list[float],
    fs: float = 1250.0,
    duration: float = 8.0,
    ripple_frequency: float = 150.0,
    ripple_duration: float = 0.04,
    ripple_amplitude: float = 120.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps = np.arange(0.0, duration, 1.0 / fs)
    rng = np.random.default_rng(0)
    ripple_signal = rng.normal(scale=5.0, size=timestamps.size)
    sharp_wave_signal = rng.normal(scale=3.0, size=timestamps.size)

    for center in centers:
        ripple_signal += _make_ripple_burst(
            timestamps=timestamps,
            center=center,
            frequency=ripple_frequency,
            duration=ripple_duration,
            amplitude=ripple_amplitude,
        )
        sharp_wave_signal += -60.0 * np.exp(
            -((timestamps - center) ** 2) / (2 * 0.012**2)
        )

    return timestamps, ripple_signal, sharp_wave_signal


def _assert_event_invariants(events: pd.DataFrame) -> None:
    if events.empty:
        return
    assert np.all(events["duration"] > 0)
    assert np.all(events["start"] <= events["peaks"])
    assert np.all(events["peaks"] <= events["stop"])
    assert events["start"].is_monotonic_increasing
    assert not events["peaks"].duplicated().any()


def _write_xml(
    basepath: str, n_channels: int = 3, fs_lfp: int = 1250, fs_dat: int = 20000
) -> None:
    basename = os.path.basename(basepath)
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<parameters>
  <acquisitionSystem>
    <nChannels>{n_channels}</nChannels>
    <samplingRate>{fs_dat}</samplingRate>
  </acquisitionSystem>
  <fieldPotentials>
    <lfpSamplingRate>{fs_lfp}</lfpSamplingRate>
  </fieldPotentials>
  <anatomicalDescription>
    <channelGroups>
      <group>
        <channel>0</channel>
        <channel>1</channel>
        <channel>2</channel>
      </group>
    </channelGroups>
  </anatomicalDescription>
</parameters>
"""
    with open(os.path.join(basepath, f"{basename}.xml"), "w", encoding="utf-8") as f:
        f.write(xml)


def _write_session_mat(basepath: str) -> None:
    basename = os.path.basename(basepath)
    session = {
        "channelTags": {
            "ripple": {"channels": np.array([2])},
            "SharpWave": {"channels": np.array([3])},
            "Bad": {"channels": np.array([1])},
        }
    }
    sio.savemat(os.path.join(basepath, f"{basename}.session.mat"), {"session": session})


def test_channel_tag_helpers_detect_sharp_wave_and_noise_channels() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_tags")
        os.makedirs(basepath, exist_ok=True)
        _write_session_mat(basepath)

        assert _get_sharp_wave_channel(basepath) == 2
        assert _get_noise_channel(basepath) == 0


def test_detect_sharp_wave_ripples_finds_synthetic_events() -> None:
    centers = [1.5, 3.5, 5.5]
    timestamps, ripple_signal, sharp_wave_signal = _make_synthetic_ripple_session(
        centers
    )

    events = detect_sharp_wave_ripples(
        ripple_signal=ripple_signal,
        sharp_wave_signal=sharp_wave_signal,
        fs=1250.0,
        timestamps=timestamps,
        ripple_channel=4,
        low_threshold=1.0,
        high_threshold=3.0,
        sharp_wave_low_threshold=0.25,
        sharp_wave_high_threshold=1.5,
        smooth_sigma=0.002,
        min_duration=0.02,
        max_duration=0.10,
        merge_gap=0.01,
    )

    assert len(events) == 3
    assert {
        "start",
        "stop",
        "peaks",
        "amplitude",
        "frequency",
        "peakNormedPower",
    }.issubset(events.columns)
    assert "sharp_wave_amplitude" in events.columns
    assert "sharp_wave_peakNormedPower" in events.columns
    np.testing.assert_allclose(
        events["peaks"].to_numpy(), np.asarray(centers), atol=0.015
    )
    assert np.all(events["duration"].between(0.02, 0.10))
    assert np.all(events["ripple_channel"] == 4)
    _assert_event_invariants(events)


def test_detect_sharp_wave_ripples_requires_joint_sharp_wave_signal() -> None:
    timestamps = np.arange(0.0, 6.0, 1.0 / 1250.0)
    rng = np.random.default_rng(4)
    ripple_signal = rng.normal(scale=5.0, size=timestamps.size)
    sharp_wave_signal = rng.normal(scale=3.0, size=timestamps.size)

    ripple_centers = [1.0, 3.0, 5.0]
    sharp_wave_centers = [1.0, 5.0]

    for center in ripple_centers:
        ripple_signal += _make_ripple_burst(
            timestamps=timestamps,
            center=center,
            frequency=150.0,
            duration=0.04,
            amplitude=120.0,
        )

    for center in sharp_wave_centers:
        sharp_wave_signal += -60.0 * np.exp(
            -((timestamps - center) ** 2) / (2 * 0.012**2)
        )

    events = detect_sharp_wave_ripples(
        ripple_signal=ripple_signal,
        sharp_wave_signal=sharp_wave_signal,
        fs=1250.0,
        timestamps=timestamps,
        low_threshold=1.0,
        high_threshold=3.0,
        sharp_wave_low_threshold=0.25,
        sharp_wave_high_threshold=1.5,
        smooth_sigma=0.002,
        min_duration=0.02,
        max_duration=0.10,
        merge_gap=0.01,
    )

    assert len(events) == 2
    np.testing.assert_allclose(
        events["peaks"].to_numpy(), np.asarray(sharp_wave_centers), atol=0.02
    )
    _assert_event_invariants(events)


def test_detect_sharp_wave_ripples_requires_sharp_wave_by_default() -> None:
    timestamps, ripple_signal, _ = _make_synthetic_ripple_session([1.5], duration=3.0)

    with pytest.raises(ValueError, match="requires a sharp-wave signal"):
        detect_sharp_wave_ripples(
            ripple_signal=ripple_signal,
            fs=1250.0,
            timestamps=timestamps,
            low_threshold=1.0,
            high_threshold=3.0,
            smooth_sigma=0.002,
            min_duration=0.02,
            max_duration=0.10,
        )


def test_detect_sharp_wave_ripples_allows_explicit_ripple_only_mode() -> None:
    timestamps, ripple_signal, _ = _make_synthetic_ripple_session([1.5], duration=3.0)

    events = detect_sharp_wave_ripples(
        ripple_signal=ripple_signal,
        fs=1250.0,
        timestamps=timestamps,
        low_threshold=1.0,
        high_threshold=3.0,
        smooth_sigma=0.002,
        min_duration=0.02,
        max_duration=0.10,
        require_sharp_wave=False,
    )

    assert len(events) == 1
    assert "sharp_wave_amplitude" not in events.columns
    _assert_event_invariants(events)


def test_detect_sharp_wave_ripples_validates_boundary_mode_without_events() -> None:
    timestamps = np.arange(0.0, 4.0, 1.0 / 1250.0)
    ripple_signal = np.random.default_rng(2).normal(scale=0.1, size=timestamps.size)

    with pytest.raises(ValueError, match="boundary_mode"):
        detect_sharp_wave_ripples(
            ripple_signal=ripple_signal,
            fs=1250.0,
            timestamps=timestamps,
            boundary_mode="ripple",
            require_sharp_wave=False,
        )


def test_detect_sharp_wave_ripples_merges_restricts_and_rejects_noise() -> None:
    centers = [1.00, 1.05, 3.00, 5.00]
    timestamps, ripple_signal, _ = _make_synthetic_ripple_session(centers, duration=6.0)
    noise_signal = np.random.default_rng(1).normal(scale=3.0, size=timestamps.size)
    noise_signal += _make_ripple_burst(
        timestamps=timestamps,
        center=3.00,
        frequency=150.0,
        duration=0.04,
        amplitude=160.0,
    )

    events = detect_sharp_wave_ripples(
        ripple_signal=ripple_signal,
        noise_signal=noise_signal,
        fs=1250.0,
        timestamps=timestamps,
        detection_epochs=np.array([[0.5, 4.0]]),
        low_threshold=1.0,
        high_threshold=3.0,
        noise_threshold=2.5,
        smooth_sigma=0.002,
        min_duration=0.02,
        max_duration=0.12,
        merge_gap=0.03,
        require_sharp_wave=False,
    )

    assert len(events) == 1
    assert abs(events["peaks"].iloc[0] - 1.025) < 0.025
    assert events["noise_peakNormedPower"].iloc[0] < 2.5
    _assert_event_invariants(events)


def test_detect_sharp_wave_ripples_returns_empty_outputs_when_no_events() -> None:
    timestamps = np.arange(0.0, 4.0, 1.0 / 1250.0)
    ripple_signal = np.random.default_rng(2).normal(scale=0.1, size=timestamps.size)

    events = detect_sharp_wave_ripples(
        ripple_signal=ripple_signal,
        fs=1250.0,
        timestamps=timestamps,
        low_threshold=4.0,
        high_threshold=8.0,
        require_sharp_wave=False,
    )
    assert events.empty

    epochs = detect_sharp_wave_ripples(
        ripple_signal=ripple_signal,
        fs=1250.0,
        timestamps=timestamps,
        low_threshold=4.0,
        high_threshold=8.0,
        return_epoch_array=True,
        require_sharp_wave=False,
    )
    assert isinstance(epochs, nel.EpochArray)
    assert epochs.isempty


def test_detection_epoch_filter_requires_same_containing_interval() -> None:
    events = pd.DataFrame(
        {
            "start": [0.75, 1.25, 2.75, 1.25],
            "stop": [1.25, 2.75, 3.25, 3.25],
            "peaks": [1.0, 2.0, 3.0, 2.2],
        }
    )

    filtered = _filter_events_to_detection_epochs(
        events, np.array([[0.5, 1.5], [2.5, 3.5]])
    )

    np.testing.assert_allclose(filtered["peaks"].to_numpy(), np.array([1.0, 3.0]))


def test_min_inter_event_interval_keeps_strongest_nearby_event() -> None:
    events = pd.DataFrame(
        {
            "start": [0.95, 0.99, 1.20],
            "stop": [1.02, 1.06, 1.27],
            "peaks": [1.00, 1.03, 1.24],
            "peakNormedPower": [3.0, 5.0, 4.0],
            "sharp_wave_peakNormedPower": [2.0, 1.0, 2.0],
        }
    )

    filtered = _enforce_min_inter_event_interval(events, min_interval=0.05)

    np.testing.assert_allclose(filtered["peaks"].to_numpy(), np.array([1.03, 1.24]))


def test_min_inter_event_interval_does_not_chain_indirect_conflicts() -> None:
    events = pd.DataFrame(
        {
            "start": [0.00, 0.035, 0.075],
            "stop": [0.02, 0.055, 0.095],
            "peaks": [0.00, 0.04, 0.08],
            "peakNormedPower": [6.0, 5.0, 6.0],
            "sharp_wave_peakNormedPower": [1.0, 1.0, 1.0],
        }
    )

    filtered = _enforce_min_inter_event_interval(events, min_interval=0.05)

    np.testing.assert_allclose(filtered["peaks"].to_numpy(), np.array([0.00, 0.08]))


def test_find_true_bounds_handles_empty_and_contiguous_segments() -> None:
    assert _find_true_bounds(np.array([], dtype=bool)) == []
    assert _find_true_bounds(np.array([False, False])) == []
    assert _find_true_bounds(np.array([True, True, False])) == [(0, 1)]
    assert _find_true_bounds(np.array([False, True, False, True, True])) == [
        (1, 1),
        (3, 4),
    ]
    assert _find_true_bounds(np.array([True, False, True])) == [(0, 0), (2, 2)]


def test_bound_containing_index_handles_disjoint_boundaries() -> None:
    bounds = _bounds_to_array([(2, 4), (8, 10)])

    assert _bound_containing_index(bounds, 1) is None
    assert _bound_containing_index(bounds, 2) == (2, 4)
    assert _bound_containing_index(bounds, 4) == (2, 4)
    assert _bound_containing_index(bounds, 6) is None
    assert _bound_containing_index(bounds, 8) == (8, 10)
    assert _bound_containing_index(bounds, 10) == (8, 10)
    assert _bound_containing_index(bounds, 11) is None


def test_nearest_trough_clips_search_to_event_boundaries() -> None:
    filtered = np.ones(21)
    filtered[4] = -10.0
    filtered[12] = -3.0

    trough = _nearest_trough(
        filtered_signal=filtered,
        center_idx=4,
        fs=1250.0,
        min_idx=10,
        max_idx=14,
    )

    assert trough == 12
    assert 10 <= trough <= 14


def test_joint_detection_rejects_events_without_ripple_peak_in_final_boundary() -> None:
    timestamps = np.arange(0.0, 3.0, 1.0 / 1250.0)
    rng = np.random.default_rng(5)
    ripple_signal = rng.normal(scale=0.5, size=timestamps.size)
    sharp_wave_signal = rng.normal(scale=0.2, size=timestamps.size)

    ripple_signal += _make_ripple_burst(
        timestamps=timestamps,
        center=1.0,
        frequency=150.0,
        duration=0.04,
        amplitude=120.0,
    )
    sharp_wave_signal += -60.0 * np.exp(
        -((timestamps - 1.04) ** 2) / (2 * 0.006**2)
    )

    events = detect_sharp_wave_ripples(
        ripple_signal=ripple_signal,
        sharp_wave_signal=sharp_wave_signal,
        fs=1250.0,
        timestamps=timestamps,
        low_threshold=1.0,
        high_threshold=3.0,
        sharp_wave_low_threshold=0.25,
        sharp_wave_high_threshold=1.5,
        smooth_sigma=0.002,
        min_duration=0.02,
        max_duration=0.10,
        sharp_wave_min_duration=0.005,
        sharp_wave_max_duration=0.050,
        merge_gap=0.01,
        peak_window=0.060,
        boundary_mode="sharp_wave",
    )

    assert events.empty


def test_sharp_wave_partner_selection_splits_broad_intervals() -> None:
    sharp_wave_power = np.full(90, 0.6)
    sharp_wave_power[20] = 5.0
    sharp_wave_power[60] = 3.0
    sharp_wave_bounds = np.asarray([[10, 70]])

    first_partner = _select_sharp_wave_partner(
        sharp_wave_power=sharp_wave_power,
        sharp_wave_bounds=sharp_wave_bounds,
        ripple_start=18,
        ripple_stop=22,
        ripple_peak_idx=20,
        search_radius=20,
        low_threshold=0.5,
    )
    second_partner = _select_sharp_wave_partner(
        sharp_wave_power=sharp_wave_power,
        sharp_wave_bounds=sharp_wave_bounds,
        ripple_start=58,
        ripple_stop=62,
        ripple_peak_idx=60,
        search_radius=20,
        low_threshold=0.5,
    )

    assert first_partner is not None
    assert second_partner is not None
    assert first_partner[2] == 20
    assert second_partner[2] == 60
    assert first_partner[1] < second_partner[0]


def test_sharp_wave_partner_selection_falls_back_to_window_maximum() -> None:
    sharp_wave_power = np.linspace(0.0, 1.0, 30)
    sharp_wave_bounds = np.asarray([[10, 20]])

    assert _find_local_peaks(sharp_wave_power, 10, 20).tolist() == [20]
    partner = _select_sharp_wave_partner(
        sharp_wave_power=sharp_wave_power,
        sharp_wave_bounds=sharp_wave_bounds,
        ripple_start=12,
        ripple_stop=15,
        ripple_peak_idx=15,
        search_radius=10,
        low_threshold=0.25,
    )

    assert partner == (10, 20, 20, sharp_wave_power[20])


def test_sharp_wave_partner_selection_includes_left_endpoint_peak() -> None:
    sharp_wave_power = np.zeros(40)
    sharp_wave_power[10] = 5.0
    sharp_wave_power[15] = 2.0
    sharp_wave_bounds = np.asarray([[10, 20]])

    assert _find_local_peaks(sharp_wave_power, 10, 20).tolist() == [10, 15]
    partner = _select_sharp_wave_partner(
        sharp_wave_power=sharp_wave_power,
        sharp_wave_bounds=sharp_wave_bounds,
        ripple_start=8,
        ripple_stop=11,
        ripple_peak_idx=10,
        search_radius=10,
        low_threshold=0.25,
    )

    assert partner is not None
    assert partner[2] == 10


def test_sharp_wave_partner_selection_includes_right_endpoint_peak() -> None:
    sharp_wave_power = np.zeros(40)
    sharp_wave_power[15] = 2.0
    sharp_wave_power[20] = 5.0
    sharp_wave_bounds = np.asarray([[10, 20]])

    assert _find_local_peaks(sharp_wave_power, 10, 20).tolist() == [15, 20]
    partner = _select_sharp_wave_partner(
        sharp_wave_power=sharp_wave_power,
        sharp_wave_bounds=sharp_wave_bounds,
        ripple_start=19,
        ripple_stop=22,
        ripple_peak_idx=20,
        search_radius=10,
        low_threshold=0.25,
    )

    assert partner is not None
    assert partner[2] == 20


def test_joint_detection_keeps_nearby_ripples_with_distinct_sharp_waves() -> None:
    timestamps = np.arange(0.0, 2.0, 1.0 / 1250.0)
    rng = np.random.default_rng(8)
    ripple_signal = rng.normal(scale=0.4, size=timestamps.size)
    sharp_wave_signal = rng.normal(scale=0.2, size=timestamps.size)

    ripple_centers = np.asarray([0.80, 0.958])
    for center in ripple_centers:
        ripple_signal += _make_ripple_burst(
            timestamps=timestamps,
            center=float(center),
            frequency=150.0,
            duration=0.045,
            amplitude=150.0,
        )

    sharp_wave_signal += -90.0 * np.exp(
        -((timestamps - 0.85) ** 2) / (2 * 0.035**2)
    )
    sharp_wave_signal += -55.0 * np.exp(
        -((timestamps - 0.958) ** 2) / (2 * 0.025**2)
    )

    events = detect_sharp_wave_ripples(
        ripple_signal=ripple_signal,
        sharp_wave_signal=sharp_wave_signal,
        fs=1250.0,
        timestamps=timestamps,
        low_threshold=0.5,
        high_threshold=2.0,
        sharp_wave_low_threshold=0.25,
        sharp_wave_high_threshold=1.0,
        smooth_sigma=0.002,
        min_duration=0.02,
        max_duration=0.12,
        sharp_wave_min_duration=0.005,
        sharp_wave_max_duration=0.12,
        merge_gap=0.001,
        peak_window=0.150,
        min_inter_event_interval=0.025,
        reject_edge_events=False,
        reject_artifacts=False,
    )

    assert len(events) == 2
    np.testing.assert_allclose(events["peaks"].to_numpy(), ripple_centers, atol=0.02)
    _assert_event_invariants(events)


def test_local_threshold_mode_accepts_strong_local_events() -> None:
    timestamps, ripple_signal, sharp_wave_signal = _make_synthetic_ripple_session(
        [1.5, 3.5], duration=5.0
    )
    ripple_signal += np.linspace(0.0, 150.0, timestamps.size)

    events = detect_sharp_wave_ripples(
        ripple_signal=ripple_signal,
        sharp_wave_signal=sharp_wave_signal,
        fs=1250.0,
        timestamps=timestamps,
        low_threshold=0.75,
        high_threshold=2.0,
        sharp_wave_low_threshold=0.25,
        sharp_wave_high_threshold=1.0,
        smooth_sigma=0.002,
        min_duration=0.02,
        max_duration=0.10,
        threshold_mode="local",
        local_window=0.5,
        reject_edge_events=False,
    )

    assert len(events) == 2
    _assert_event_invariants(events)


def test_sharp_wave_polarity_defaults_to_negative_deflections() -> None:
    timestamps, ripple_signal, sharp_wave_signal = _make_synthetic_ripple_session(
        [1.5], duration=3.0
    )

    default_events = detect_sharp_wave_ripples(
        ripple_signal=ripple_signal,
        sharp_wave_signal=sharp_wave_signal,
        fs=1250.0,
        timestamps=timestamps,
        low_threshold=1.0,
        high_threshold=3.0,
        sharp_wave_low_threshold=0.25,
        sharp_wave_high_threshold=1.5,
        smooth_sigma=0.002,
        min_duration=0.02,
        max_duration=0.10,
        boundary_mode="sharp_wave",
    )
    positive_events_default = detect_sharp_wave_ripples(
        ripple_signal=ripple_signal,
        sharp_wave_signal=-sharp_wave_signal,
        fs=1250.0,
        timestamps=timestamps,
        low_threshold=1.0,
        high_threshold=3.0,
        sharp_wave_low_threshold=0.25,
        sharp_wave_high_threshold=1.5,
        smooth_sigma=0.002,
        min_duration=0.02,
        max_duration=0.10,
        boundary_mode="sharp_wave",
    )
    positive_events = detect_sharp_wave_ripples(
        ripple_signal=ripple_signal,
        sharp_wave_signal=-sharp_wave_signal,
        fs=1250.0,
        timestamps=timestamps,
        low_threshold=1.0,
        high_threshold=3.0,
        sharp_wave_low_threshold=0.25,
        sharp_wave_high_threshold=1.5,
        smooth_sigma=0.002,
        min_duration=0.02,
        max_duration=0.10,
        sharp_wave_polarity="positive",
        boundary_mode="sharp_wave",
    )

    assert len(default_events) == 1
    assert positive_events_default.empty
    assert len(positive_events) == 1


def test_detector_rejects_nonfinite_and_saturated_event_windows() -> None:
    timestamps, ripple_signal, sharp_wave_signal = _make_synthetic_ripple_session(
        [1.5], duration=3.0
    )
    nan_ripple = ripple_signal.copy()
    nan_ripple[np.abs(timestamps - 1.5) < 0.003] = np.nan
    saturated_ripple = ripple_signal.copy()
    saturated_ripple[np.abs(timestamps - 1.5) < 0.006] = 5000.0

    nan_events = detect_sharp_wave_ripples(
        ripple_signal=nan_ripple,
        sharp_wave_signal=sharp_wave_signal,
        fs=1250.0,
        timestamps=timestamps,
        low_threshold=1.0,
        high_threshold=3.0,
        sharp_wave_low_threshold=0.25,
        sharp_wave_high_threshold=1.5,
        smooth_sigma=0.002,
        min_duration=0.02,
        max_duration=0.10,
    )
    saturated_events = detect_sharp_wave_ripples(
        ripple_signal=saturated_ripple,
        sharp_wave_signal=sharp_wave_signal,
        fs=1250.0,
        timestamps=timestamps,
        low_threshold=1.0,
        high_threshold=3.0,
        sharp_wave_low_threshold=0.25,
        sharp_wave_high_threshold=1.5,
        smooth_sigma=0.002,
        min_duration=0.02,
        max_duration=0.10,
    )

    assert nan_events.empty
    assert saturated_events.empty


def test_edge_events_are_rejected_by_default_and_can_be_allowed() -> None:
    timestamps, ripple_signal, _ = _make_synthetic_ripple_session(
        [0.08], duration=1.0
    )

    rejected = detect_sharp_wave_ripples(
        ripple_signal=ripple_signal,
        fs=1250.0,
        timestamps=timestamps,
        low_threshold=1.0,
        high_threshold=3.0,
        smooth_sigma=0.002,
        min_duration=0.02,
        max_duration=0.10,
        edge_buffer=0.10,
        require_sharp_wave=False,
    )
    allowed = detect_sharp_wave_ripples(
        ripple_signal=ripple_signal,
        fs=1250.0,
        timestamps=timestamps,
        low_threshold=1.0,
        high_threshold=3.0,
        smooth_sigma=0.002,
        min_duration=0.02,
        max_duration=0.10,
        edge_buffer=0.10,
        reject_edge_events=False,
        require_sharp_wave=False,
    )

    assert rejected.empty
    assert len(allowed) == 1
    _assert_event_invariants(allowed)


def test_detect_sharp_wave_ripples_loads_from_basepath_and_round_trips_event_file() -> (
    None
):
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_ripples")
        os.makedirs(basepath, exist_ok=True)
        basename = os.path.basename(basepath)

        _write_xml(basepath)
        _write_session_mat(basepath)

        timestamps, ripple_signal, sharp_wave_signal = _make_synthetic_ripple_session(
            [1.25, 2.75, 4.25], duration=6.0
        )
        noise_signal = np.random.default_rng(3).normal(scale=2.0, size=timestamps.size)
        lfp = np.column_stack([noise_signal, ripple_signal, sharp_wave_signal]).astype(
            np.int16
        )
        lfp.tofile(os.path.join(basepath, f"{basename}.lfp"))

        events = detect_sharp_wave_ripples(
            basepath=basepath,
            low_threshold=1.0,
            high_threshold=3.0,
            sharp_wave_low_threshold=0.25,
            sharp_wave_high_threshold=1.5,
            smooth_sigma=0.002,
            min_duration=0.02,
            max_duration=0.10,
            merge_gap=0.01,
            save_mat=True,
            overwrite=True,
        )

        event_path = os.path.join(basepath, f"{basename}.ripples.events.mat")
        assert os.path.exists(event_path)
        assert len(events) == 3

        loaded = load_ripples_events(basepath)
        assert len(loaded) == 3
        assert np.all(loaded["ripple_channel"] == 1)

        reused = detect_sharp_wave_ripples(
            basepath=basepath, save_mat=True, overwrite=False
        )
        assert len(reused) == len(loaded)


def test_load_ripples_events_reads_file_saved_by_detect_sharp_wave_ripples() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_saved_events")
        os.makedirs(basepath, exist_ok=True)

        centers = [1.2, 2.8, 4.4]
        timestamps, ripple_signal, sharp_wave_signal = _make_synthetic_ripple_session(
            centers,
            duration=6.0,
        )

        detected = detect_sharp_wave_ripples(
            basepath=basepath,
            ripple_signal=ripple_signal,
            sharp_wave_signal=sharp_wave_signal,
            fs=1250.0,
            timestamps=timestamps,
            ripple_channel=4,
            low_threshold=1.0,
            high_threshold=3.0,
            sharp_wave_low_threshold=0.25,
            sharp_wave_high_threshold=1.5,
            smooth_sigma=0.002,
            min_duration=0.02,
            max_duration=0.10,
            merge_gap=0.01,
            save_mat=True,
            overwrite=True,
        )

        loaded = load_ripples_events(basepath)

        assert len(loaded) == len(detected)
        np.testing.assert_allclose(
            loaded["peaks"].to_numpy(dtype=float),
            detected["peaks"].to_numpy(dtype=float),
            atol=1e-6,
        )
        assert np.all(loaded["detectorName"] == "detect_sharp_wave_ripples")
        assert np.all(loaded["ripple_channel"] == 4)
        assert np.all(loaded["event_spk_thres"] == 0)
        assert os.path.normpath(loaded["basepath"].iloc[0]) == os.path.normpath(
            basepath
        )


def test_save_ripple_events_writes_empty_event_file() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_empty_ripples")
        os.makedirs(basepath, exist_ok=True)
        basename = os.path.basename(basepath)

        save_ripple_events(
            events=pd.DataFrame(
                columns=[
                    "start",
                    "stop",
                    "peaks",
                    "center",
                    "duration",
                    "amplitude",
                    "frequency",
                    "peakNormedPower",
                ]
            ),
            basepath=basepath,
            ripple_channel=0,
            detection_epochs=np.array([[0.0, 1.0]]),
        )

        assert os.path.exists(os.path.join(basepath, f"{basename}.ripples.events.mat"))
