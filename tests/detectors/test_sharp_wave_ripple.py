import os
import tempfile

import nelpy as nel
import numpy as np
import pandas as pd
import scipy.io as sio

from neuro_py.detectors.sharp_wave_ripple import (
    _get_noise_channel,
    _get_sharp_wave_channel,
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
    burst[mask] = amplitude * np.sin(2 * np.pi * frequency * (timestamps[mask] - center)) * window
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
        sharp_wave_signal += -60.0 * np.exp(-((timestamps - center) ** 2) / (2 * 0.012**2))

    return timestamps, ripple_signal, sharp_wave_signal


def _write_xml(basepath: str, n_channels: int = 3, fs_lfp: int = 1250, fs_dat: int = 20000) -> None:
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
    timestamps, ripple_signal, sharp_wave_signal = _make_synthetic_ripple_session(centers)

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
    assert {"start", "stop", "peaks", "amplitude", "frequency", "peakNormedPower"}.issubset(
        events.columns
    )
    assert "sharp_wave_amplitude" in events.columns
    assert "sharp_wave_peakNormedPower" in events.columns
    np.testing.assert_allclose(events["peaks"].to_numpy(), np.asarray(centers), atol=0.015)
    assert np.all(events["duration"].between(0.02, 0.10))
    assert np.all(events["ripple_channel"] == 4)


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
        sharp_wave_signal += -60.0 * np.exp(-((timestamps - center) ** 2) / (2 * 0.012**2))

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
    np.testing.assert_allclose(events["peaks"].to_numpy(), np.asarray(sharp_wave_centers), atol=0.02)


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
    )

    assert len(events) == 1
    assert abs(events["peaks"].iloc[0] - 1.025) < 0.025
    assert events["noise_peakNormedPower"].iloc[0] < 2.5


def test_detect_sharp_wave_ripples_returns_empty_outputs_when_no_events() -> None:
    timestamps = np.arange(0.0, 4.0, 1.0 / 1250.0)
    ripple_signal = np.random.default_rng(2).normal(scale=0.1, size=timestamps.size)

    events = detect_sharp_wave_ripples(
        ripple_signal=ripple_signal,
        fs=1250.0,
        timestamps=timestamps,
        low_threshold=4.0,
        high_threshold=8.0,
    )
    assert events.empty

    epochs = detect_sharp_wave_ripples(
        ripple_signal=ripple_signal,
        fs=1250.0,
        timestamps=timestamps,
        low_threshold=4.0,
        high_threshold=8.0,
        return_epoch_array=True,
    )
    assert isinstance(epochs, nel.EpochArray)
    assert epochs.isempty


def test_detect_sharp_wave_ripples_loads_from_basepath_and_round_trips_event_file() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_ripples")
        os.makedirs(basepath, exist_ok=True)
        basename = os.path.basename(basepath)

        _write_xml(basepath)
        _write_session_mat(basepath)

        timestamps, ripple_signal, sharp_wave_signal = _make_synthetic_ripple_session([1.25, 2.75, 4.25], duration=6.0)
        noise_signal = np.random.default_rng(3).normal(scale=2.0, size=timestamps.size)
        lfp = np.column_stack([noise_signal, ripple_signal, sharp_wave_signal]).astype(np.int16)
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

        reused = detect_sharp_wave_ripples(basepath=basepath, save_mat=True, overwrite=False)
        assert len(reused) == len(loaded)


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
