"""High-gamma confirmation of delta morphology, using independent voltage traces."""

import numpy as np
import pandas as pd
import pytest
from scipy import signal
from scipy.io import loadmat

from neuro_py.detectors import detect_delta_waves, save_delta_wave_events
from neuro_py.detectors import delta_wave
from tests.detectors.test_delta_wave import _detect, _trace, _write_session


@pytest.fixture
def gamma_example():
    fs = 1250
    t, delta = _trace(fs)
    rng = np.random.default_rng(818)
    sos = signal.butter(4, (100, 400), fs=fs, btype="bandpass", output="sos")
    gamma = signal.sosfiltfilt(sos, rng.normal(size=t.size))
    envelope = np.ones(t.size)
    for peak in (5, 25):
        envelope *= 1 - 0.99 * np.exp(-(((t - peak) / 0.22) ** 8))
    envelope += 4 * np.exp(-(((t - 15) / 0.22) ** 8))
    return t, delta, gamma * envelope


def test_gamma_keeps_suppressed_waves_and_rejects_gamma_burst(gamma_example):
    t, delta, gamma = gamma_example
    original = _detect(delta, t, fs=1250)
    confirmed = _detect(delta, t, fs=1250, gamma_lfp=gamma)
    np.testing.assert_allclose(confirmed.peaks, [5, 25], atol=0.02)
    # This is a veto, not a new boundary or amplitude estimator.
    pd.testing.assert_frame_equal(
        confirmed, original.iloc[[0, 2]].reset_index(drop=True)
    )
    params = confirmed.attrs["detectorinfo"]["detectionparms"]
    assert params["gamma_enabled"]
    assert params["gamma_rejected"] == 1
    assert params["gamma_band"] == (100, 400)


def test_gamma_is_opt_in(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("No gamma processing should run without a gamma source")

    monkeypatch.setattr(delta_wave, "_restrict_gamma", unexpected)
    t, delta = _trace()  # 250 Hz remains supported with gamma disabled.
    events = _detect(delta, t)
    assert len(events) == 3
    assert not events.attrs["detectorinfo"]["detectionparms"]["gamma_enabled"]


def test_gamma_threshold_and_duration_control_acceptance(gamma_example):
    t, delta, gamma = gamma_example
    options = dict(fs=1250, gamma_lfp=gamma)
    assert len(_detect(delta, t, **options, gamma_threshold=100)) == 3
    assert _detect(delta, t, **options, gamma_threshold=-100).empty
    assert _detect(delta, t, **options, gamma_min_duration=0.65).empty


def test_low_gamma_must_contain_peak_and_last_long_enough(gamma_example, monkeypatch):
    t, delta, gamma = gamma_example
    candidates = _detect(delta, t, fs=1250)
    # Isolate the coincidence rule from envelope estimation: a long dip before
    # the peak fails, a 20 ms dip at the peak fails, and a 100 ms dip passes.
    z = np.ones(t.size)
    first = candidates.iloc[0]
    z[(t >= first.start) & (t <= first.peaks - 0.01)] = -1
    z[np.abs(t - candidates.peaks.iloc[1]) < 0.01] = -1
    z[np.abs(t - candidates.peaks.iloc[2]) < 0.05] = -1
    monkeypatch.setattr(delta_wave, "_gamma_power_z", lambda *args: z)
    events = _detect(delta, t, fs=1250, gamma_lfp=gamma)
    np.testing.assert_allclose(events.peaks, [25], atol=0.02)


def test_custom_gamma_band_supports_lower_sampling_rate():
    t, delta = _trace(250)
    rng = np.random.default_rng(818)
    gamma = rng.normal(size=t.size)
    for peak in (5, 15, 25):
        gamma *= 1 - 0.99 * np.exp(-(((t - peak) / 0.22) ** 8))
    events = _detect(delta, t, gamma_lfp=gamma, gamma_band=(40, 100))
    assert len(events) == 3
    assert events.attrs["detectorinfo"]["detectionparms"]["gamma_band"] == (40, 100)


def test_gamma_scale_offset_and_absolute_timestamps(gamma_example):
    t, delta, gamma = gamma_example
    events = _detect(delta, t + 123, fs=1250, gamma_lfp=gamma * 17 + 1000)
    np.testing.assert_allclose(events.peaks, [128, 148], atol=0.02)


@pytest.mark.parametrize("value", [0.0, 7.0, np.nan])
def test_uninformative_gamma_cannot_confirm_an_event(gamma_example, value):
    t, delta, gamma = gamma_example
    events = _detect(delta, t, fs=1250, gamma_lfp=np.full_like(gamma, value))
    assert events.empty
    assert events.attrs["detectorinfo"]["detectionparms"]["gamma_rejected"] == 3
    assert "timestamps" in events


def test_missing_gamma_within_an_event_rejects_only_that_event(gamma_example):
    t, delta, gamma = gamma_example
    gamma[np.abs(t - 25) < 0.05] = np.nan
    events = _detect(delta, t, fs=1250, gamma_lfp=gamma)
    np.testing.assert_allclose(events.peaks, [5], atol=0.02)
    usable = events.attrs["detectorinfo"]["detectionintervals"]
    assert not np.any((usable[:, 0] < 25) & (usable[:, 1] > 25))


def test_gamma_respects_gaps_and_excluded_intervals(gamma_example):
    t, delta, gamma = gamma_example
    shifted = t.copy()
    shifted[t >= 10] += 100
    events = _detect(delta, shifted, fs=1250, gamma_lfp=gamma)
    np.testing.assert_allclose(events.peaks, [5, 125], atol=0.02)
    restricted = _detect(
        delta,
        t,
        fs=1250,
        gamma_lfp=gamma,
        detection_epochs=[[2, 8], [22, 28]],
        ignore_intervals=[[24.9, 25.1]],
    )
    np.testing.assert_allclose(restricted.peaks, [5], atol=0.02)


def test_independent_firing_check_sees_only_gamma_confirmed_events(
    gamma_example, monkeypatch
):
    t, delta, gamma = gamma_example
    calls = []

    def verify(events, spikes, threshold):
        calls.append(events.peaks.to_numpy())
        return events.iloc[:1], threshold, True

    monkeypatch.setattr(delta_wave, "_verify_firing", verify)
    events = _detect(delta, t, fs=1250, gamma_lfp=gamma, verify_firing=True)
    np.testing.assert_allclose(calls[0], [5, 25], atol=0.02)
    assert len(events) == 1
    assert events.attrs["detectorinfo"]["detectionparms"]["firing_verified"]


def test_file_backed_gamma_channel_and_saved_metadata(tmp_path, gamma_example):
    t, delta, gamma = gamma_example
    _write_session(tmp_path, np.column_stack([100 * delta, 100 * gamma]), fs=1250)
    events = detect_delta_waves(
        tmp_path,
        channel=0,
        gamma_channel=1,
        nrem_restrict=False,
        emg_threshold=np.inf,
        reject_artifacts=False,
        verify_firing=False,
        event_name="deltaWavesGamma",
    )
    np.testing.assert_allclose(events.peaks, [5, 25], atol=0.02)
    info = events.attrs["detectorinfo"]
    assert info["gammachannel"] == 1
    assert info["gammachannel1"] == 2
    saved = loadmat(events.attrs["event_file"], simplify_cells=True)["deltaWavesGamma"]
    assert saved["detectorinfo"]["detectionparms"]["gamma_rejected"] == 1
    assert saved["detectorinfo"]["gammachannel1"] == 2
    np.testing.assert_allclose(saved["timestamps"], events[["start", "stop"]])
    with pytest.raises(FileExistsError):
        save_delta_wave_events(events, tmp_path, "deltaWavesGamma")


@pytest.mark.parametrize("channel,tags", [(2, {}), (1, {"Bad": {"channels": [2]}})])
def test_gamma_channel_must_be_available_and_not_bad(tmp_path, channel, tags):
    _write_session(tmp_path, np.zeros((12500, 2)), fs=1250, tags=tags)
    with pytest.raises(ValueError, match="gamma_channel"):
        detect_delta_waves(
            tmp_path,
            channel=0,
            gamma_channel=channel,
            nrem_restrict=False,
            emg_threshold=np.inf,
            save_mat=False,
        )


@pytest.mark.parametrize("channel", [-1, 1.5, True, [1]])
def test_gamma_channel_requires_integer(channel):
    with pytest.raises(ValueError, match="gamma_channel"):
        _detect(np.zeros(2500), gamma_channel=channel)


def test_gamma_array_requires_matching_samples_and_explicit_data():
    with pytest.raises(ValueError, match="requires gamma_lfp"):
        _detect(np.zeros(2500), gamma_channel=0)
    for gamma in (np.zeros(2499), np.zeros((2500, 1))):
        with pytest.raises(ValueError, match="match lfp length"):
            _detect(np.zeros(2500), gamma_lfp=gamma)
    with pytest.raises(ValueError, match="requires array input"):
        detect_delta_waves("unused", gamma_lfp=np.zeros(2500), save_mat=False)


@pytest.mark.parametrize("band", [(100, 400), (0, 100), (100, 100), (100, np.nan)])
def test_gamma_band_must_be_supported_by_sampling_rate(band):
    with pytest.raises(ValueError, match="gamma_band"):
        _detect(np.zeros(2500), gamma_lfp=np.zeros(2500), gamma_band=band)


@pytest.mark.parametrize(
    "name", ["gamma_smoothing", "gamma_baseline", "gamma_min_duration"]
)
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_gamma_windows_must_be_finite_positive(name, value):
    with pytest.raises(ValueError, match=name):
        _detect(np.zeros(2500), fs=1250, gamma_lfp=np.zeros(2500), **{name: value})


def test_gamma_baseline_must_exceed_smoothing():
    with pytest.raises(ValueError, match="gamma_baseline"):
        _detect(np.zeros(2500), fs=1250, gamma_lfp=np.zeros(2500), gamma_baseline=0.08)


@pytest.mark.parametrize("threshold", [np.nan, np.inf, -np.inf])
def test_gamma_threshold_must_be_finite(threshold):
    with pytest.raises(ValueError, match="gamma_threshold"):
        _detect(
            np.zeros(2500), fs=1250, gamma_lfp=np.zeros(2500), gamma_threshold=threshold
        )


def test_empty_restriction_retains_gamma_metadata(gamma_example):
    t, delta, gamma = gamma_example
    events = _detect(delta, t, fs=1250, gamma_lfp=gamma, detection_epochs=[])
    assert events.empty
    assert events.attrs["detectorinfo"]["detectionparms"]["gamma_enabled"]
    assert events.attrs["detectorinfo"]["detectionintervals"].shape == (0, 2)
