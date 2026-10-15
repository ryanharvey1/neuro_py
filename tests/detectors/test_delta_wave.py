from pathlib import Path

import nelpy as nel
import numpy as np
import pandas as pd
import pytest
from scipy.io import loadmat, savemat

from neuro_py.detectors import detect_delta_waves, save_delta_wave_events
from neuro_py.detectors.delta_wave import _channel_candidates, _verify_firing


def _trace(fs=250.0):
    t = np.arange(0, 30, 1 / fs)
    rng = np.random.default_rng(42)
    x = rng.normal(0, 0.1, t.size)
    for peak in (5, 15, 25):
        x += (
            8
            * np.cos(2 * np.pi * 2.5 * (t - peak))
            * np.exp(-(((t - peak) / 0.15) ** 2))
        )
    return t, x


def _detect(x, t=None, **kwargs):
    options = dict(
        lfp=x,
        fs=250,
        timestamps=t,
        nrem_restrict=False,
        emg_threshold=np.inf,
        save_mat=False,
        reject_artifacts=False,
        verify_firing=False,
    )
    options.update(kwargs)
    return detect_delta_waves(**options)


def _write_session(path, data, fs=250, tags=None, regions=None):
    session = {
        "extracellular": {"sr": 20000, "srLfp": fs, "nChannels": data.shape[1]},
        "channelTags": tags or {},
        "brainRegions": regions or {},
    }
    savemat(path / f"{path.name}.session.mat", {"session": session})
    np.asarray(data, dtype=np.int16).tofile(path / f"{path.name}.lfp")


def test_detects_sparse_waves_and_normal_event_fields():
    t, x = _trace()
    events = _detect(x, t, channel=4)
    assert isinstance(events, pd.DataFrame)
    assert len(events) == 3
    np.testing.assert_allclose(events.peaks, [5, 15, 25], atol=0.02)
    assert (events.start < events.peaks).all()
    assert (events.peaks < events.stop).all()
    assert events.duration.between(0.15, 0.65).all()
    np.testing.assert_allclose(events.amplitude, events.peak_value - events.stop_value)
    np.testing.assert_allclose(events.center, (events.start + events.stop) / 2)
    np.testing.assert_allclose(
        np.asarray(events.timestamps.tolist()), events[["start", "stop"]]
    )
    assert set(
        ("amplitudeUnits", "eventID", "eventIDlabels", "eventIDbinary", "values")
    ) <= set(events.columns)
    assert set(events.amplitudeUnits) == {"zscore"}
    info = events.attrs["detectorinfo"]
    assert info["detectionchannel"] == 4
    assert info["detectionchannel1"] == 5
    assert info["detectionparms"]["fs"] == 250


@pytest.mark.parametrize("fs", [250, 1250])
def test_native_sampling_rate_and_absolute_timestamps(fs):
    t, x = _trace(fs)
    events = _detect(x, t + 123.0, fs=fs)
    np.testing.assert_allclose(events.peaks, [128, 138, 148], atol=0.02)


def test_threshold_and_duration_reject_candidates():
    t, x = _trace()
    assert _detect(x, t, threshold=100).empty
    assert _detect(x, t, low_peak_threshold=100, high_peak_threshold=100).empty
    assert _detect(x, t, min_duration=0.6, max_duration=0.7).empty


@pytest.mark.parametrize(
    "parameters, expected_count",
    [
        ({"low_peak_threshold": 10, "high_peak_threshold": 100}, 0),
        ({"low_peak_threshold": 1, "high_peak_threshold": 100}, 3),
        ({"high_peak_threshold": 3, "high_trough_threshold": 4}, 3),
        ({"high_peak_threshold": 100, "high_trough_threshold": 4}, 0),
        ({"low_trough_threshold": 4, "high_trough_threshold": 4}, 0),
        ({"min_duration": 0.4}, 0),
        ({"max_duration": 0.2}, 0),
    ],
)
def test_named_thresholds_and_duration_bounds_control_acceptance(
    parameters, expected_count
):
    t, x = _trace()
    assert len(_detect(x, t, **parameters)) == expected_count


@pytest.mark.parametrize(
    "parameter",
    [
        "low_peak_threshold",
        "high_peak_threshold",
        "low_trough_threshold",
        "high_trough_threshold",
        "min_duration",
        "max_duration",
    ],
)
@pytest.mark.parametrize("value", [-1, np.nan, np.inf])
def test_named_thresholds_and_durations_reject_invalid_values(parameter, value):
    with pytest.raises(ValueError, match=parameter):
        _detect(np.ones(2500), **{parameter: value})


@pytest.mark.parametrize("parameter", ["min_duration", "max_duration"])
def test_duration_bounds_must_be_positive(parameter):
    with pytest.raises(ValueError, match=parameter):
        _detect(np.ones(2500), **{parameter: 0})


def test_restrictions_and_exclusions_are_not_bridged():
    t, x = _trace()
    events = _detect(
        x,
        t,
        detection_epochs=nel.EpochArray([[2, 8], [12, 18], [22, 28]]),
        ignore_intervals=[[14.99, 15.01]],
        edge_buffer=0.5,
    )
    np.testing.assert_allclose(events.peaks, [5, 25], atol=0.02)
    intervals = events.attrs["detectorinfo"]["detectionintervals"]
    assert not np.any((intervals[:, 0] < 15) & (intervals[:, 1] > 15))
    assert _detect(x, t, detection_epochs=[]).empty


def test_nan_and_timestamp_gaps_do_not_poison_other_events():
    t, x = _trace()
    x[(t > 14.9) & (t < 15.1)] = np.nan
    events = _detect(x, t)
    np.testing.assert_allclose(events.peaks, [5, 25], atol=0.02)
    keep = ~((t > 14.9) & (t < 15.1))
    events = _detect(x[keep], t[keep])
    np.testing.assert_allclose(events.peaks, [5, 25], atol=0.02)


def test_large_artifact_is_excluded_before_filtering():
    t, x = _trace()
    x[np.argmin(abs(t - 15))] = 10000
    events = _detect(x, t, reject_artifacts=True)
    assert not events.peaks.between(14, 16).any()
    np.testing.assert_allclose(events.peaks, [5, 25], atol=0.02)


@pytest.mark.parametrize(
    "x", [[], [1], np.ones(2500), np.full(2500, np.nan), np.zeros(10)]
)
def test_empty_short_constant_and_nonfinite_signals(x):
    events = _detect(x)
    assert events.empty
    assert "duration" in events
    assert events.attrs["detectorinfo"]["detectionintervals"].shape[1] == 2


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"fs": 0}, "fs"),
        ({"fs": np.nan}, "fs"),
        ({"filter_band": (9, 1)}, "filter_band"),
        ({"low_peak_threshold": 4}, "low_peak_threshold"),
        ({"low_trough_threshold": 2}, "low_trough_threshold"),
        ({"min_duration": 0.6, "max_duration": 0.1}, "min_duration"),
        ({"threshold": -1}, "threshold"),
        ({"edge_buffer": -1}, "edge_buffer"),
        ({"emg_threshold": np.nan}, "emg_threshold"),
        ({"timestamps": [0, 0]}, "timestamps"),
        ({"detection_epochs": [[5, 1]]}, "Intervals"),
        ({"lfp": np.ones((3, 2))}, "one-dimensional"),
        ({"channel": -1}, "channel"),
    ],
)
def test_invalid_inputs(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _detect(np.ones(2500), **kwargs)


def test_session_loading_nrem_emg_and_mat_roundtrip(tmp_path):
    t, x = _trace()
    _write_session(tmp_path, (x[:, None] * 100), tags={"Delta": {"channels": 1}})
    savemat(
        tmp_path / f"{tmp_path.name}.SleepState.states.mat",
        {"SleepState": {"ints": {"NREMstate": [[2, 8], [12, 18], [22, 28]]}}},
    )
    emg_t = np.arange(0, 30, 0.1)
    emg = np.zeros_like(emg_t)
    emg[(emg_t > 12) & (emg_t < 18)] = 1
    savemat(
        tmp_path / f"{tmp_path.name}.EMGFromLFP.LFP.mat",
        {"EMGFromLFP": {"timestamps": emg_t, "data": emg}},
    )
    parameters = dict(
        low_peak_threshold=0.5,
        high_peak_threshold=2.0,
        low_trough_threshold=0.1,
        high_trough_threshold=1.0,
        min_duration=0.2,
        max_duration=0.5,
    )
    events = detect_delta_waves(
        tmp_path, reject_artifacts=False, verify_firing=False, **parameters
    )
    np.testing.assert_allclose(events.peaks, [5, 25], atol=0.02)
    saved = loadmat(
        events.attrs["event_file"], struct_as_record=False, squeeze_me=False
    )["deltaWaves"][0, 0]
    assert saved.timestamps.shape == (2, 2)
    assert saved.peaks.shape == (2, 1)
    assert saved.values.shape == (2, 3)
    np.testing.assert_allclose(saved.peaks.ravel(), events.peaks)
    assert saved.detectorinfo[0, 0].detectionchannel1.item() == 1
    assert saved.detectorinfo[0, 0].detectionchannel.item() == 0
    assert saved.eventIDlabels.shape == (1, 1)
    saved_parameters = saved.detectorinfo[0, 0].detectionparms[0, 0]
    for name, value in parameters.items():
        assert events.attrs["detectorinfo"]["detectionparms"][name] == value
        assert getattr(saved_parameters, name).item() == value
    assert not hasattr(saved_parameters, "thresholds")
    assert not hasattr(saved_parameters, "durations")


def test_existing_file_is_protected_and_another_name_can_be_saved(tmp_path):
    t, x = _trace()
    old = tmp_path / f"{tmp_path.name}.deltaWaves.events.mat"
    old.write_bytes(b"irreplaceable existing events")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        detect_delta_waves(tmp_path)
    events = _detect(
        x, t, basepath=tmp_path, save_mat=True, event_name="deltaWavesPython"
    )
    assert Path(events.attrs["event_file"]).is_file()
    with pytest.raises(FileExistsError):
        save_delta_wave_events(events, tmp_path)
    assert old.read_bytes() == b"irreplaceable existing events"


@pytest.mark.parametrize("trace", [False, True])
def test_empty_and_single_event_mat_shapes(tmp_path, trace):
    t, x = _trace()
    events = _detect(
        x if trace else np.zeros_like(x),
        t,
        detection_epochs=[[2, 8]],
        basepath=tmp_path,
        save_mat=True,
    )
    saved = loadmat(events.attrs["event_file"], struct_as_record=False)["deltaWaves"][
        0, 0
    ]
    assert saved.timestamps.shape == (int(trace), 2)
    assert saved.peaks.shape == (int(trace), 1)
    assert saved.values.shape == (int(trace), 3)


def test_channel_metadata_numbering_bad_channels_and_regions():
    session = {
        "channelTags": {"Bad": {"channels": 2}, "Delta": {"channels": 3}},
        "brainRegions": {"PFC": {"channels": [1, 2, 3]}},
    }
    assert _channel_candidates(session, None, ["PFC"], 3) == [2]
    assert _channel_candidates(session, [0, 1], ["PFC"], 3) == [0]
    session["channelTags"].pop("Delta")
    assert _channel_candidates(session, None, ["PFC"], 3) == [0, 2]
    with pytest.raises(ValueError, match="No usable"):
        _channel_candidates(session, None, ["Absent"], 3)
    with pytest.raises(ValueError, match="integer"):
        _channel_candidates(session, [0.5], [], 3)


def test_relative_power_channel_selection_and_eeg_fallback(tmp_path):
    t, x = _trace()
    data = np.column_stack((100 * np.sin(2 * np.pi * 40 * t), x * 100))
    _write_session(tmp_path, data, regions={"PFC": {"channels": [1, 2]}})
    (tmp_path / f"{tmp_path.name}.lfp").rename(tmp_path / f"{tmp_path.name}.eeg")
    events = detect_delta_waves(
        tmp_path,
        nrem_restrict=False,
        emg_threshold=np.inf,
        reject_artifacts=False,
        save_mat=False,
        verify_firing=False,
    )
    assert events.attrs["detectorinfo"]["detectionchannel"] == 1
    np.testing.assert_allclose(events.peaks, [5, 15, 25], atol=0.02)


@pytest.mark.parametrize("spike_times", [None, [], [np.array([])]])
def test_default_firing_verification_warns_when_unavailable(spike_times):
    t, x = _trace()
    with pytest.warns(UserWarning, match="Cannot verify firing: no cortical cells"):
        events = detect_delta_waves(
            lfp=x,
            fs=250,
            timestamps=t,
            nrem_restrict=False,
            emg_threshold=np.inf,
            reject_artifacts=False,
            save_mat=False,
            spike_times=spike_times,
        )
    assert len(events) == 3
    params = events.attrs["detectorinfo"]["detectionparms"]
    assert params["verify_firing"]
    assert not params["firing_verified"]
    assert events.attrs["detectorinfo"]["threshold"] == 3.0


def test_default_firing_verification_uses_available_spikes():
    t, x = _trace()
    spikes = np.arange(0, 30, 0.005)
    keep = np.all(abs(spikes[:, None] - np.array([5, 15, 25])) > 0.08, axis=1)
    events = detect_delta_waves(
        lfp=x,
        fs=250,
        timestamps=t,
        nrem_restrict=False,
        emg_threshold=np.inf,
        reject_artifacts=False,
        save_mat=False,
        spike_times=[spikes[keep]],
    )
    params = events.attrs["detectorinfo"]["detectionparms"]
    assert params["verify_firing"]
    assert params["firing_verified"]
    assert 0 < len(events) < 3
    assert events.attrs["detectorinfo"]["threshold"] > 3.0


def test_session_without_cortical_cells_warns_and_retains_lfp_events(tmp_path):
    _, x = _trace()
    _write_session(tmp_path, x[:, None] * 100, tags={"Delta": {"channels": 1}})
    trains = np.empty(2, dtype=object)
    trains[0] = np.arange(0, 30, 0.1)
    trains[1] = np.arange(0, 30, 0.15)
    savemat(
        tmp_path / f"{tmp_path.name}.cell_metrics.cellinfo.mat",
        {
            "cell_metrics": {
                "UID": [1, 2],
                "brainRegion": np.array(["CA1", "CA1"], dtype=object),
                "spikes": {"times": trains},
            },
        },
    )
    with pytest.warns(UserWarning, match="Cannot verify firing: no cortical cells"):
        events = detect_delta_waves(
            tmp_path,
            nrem_restrict=False,
            emg_threshold=np.inf,
            reject_artifacts=False,
            save_mat=False,
        )
    np.testing.assert_allclose(events.peaks, [5, 15, 25], atol=0.02)
    params = events.attrs["detectorinfo"]["detectionparms"]
    assert params["verify_firing"]
    assert not params["firing_verified"]


def test_disabling_firing_verification_skips_spike_loading(tmp_path, monkeypatch):
    t, x = _trace()

    def unexpected_spike_load(*args, **kwargs):
        pytest.fail("Spike loading must be skipped when verify_firing=False.")

    monkeypatch.setattr(
        "neuro_py.detectors.delta_wave.loading.load_spikes", unexpected_spike_load
    )
    events = _detect(x, t, basepath=tmp_path)
    assert len(events) == 3
    assert not events.attrs["detectorinfo"]["detectionparms"]["verify_firing"]


def test_firing_verification_raises_cutoff_for_population_suppression():
    peaks = np.arange(2, 42, 2)
    events = pd.DataFrame({"peaks": peaks, "amplitude": np.linspace(4, 8, peaks.size)})
    spikes = np.arange(0, 44, 0.005)
    keep = np.all(abs(spikes[:, None] - peaks) > 0.08, axis=1)
    selected, cutoff, verified = _verify_firing(events, [spikes[keep]], 3)
    assert verified
    assert cutoff >= 4
    assert (selected.amplitude > cutoff).all()


def test_save_validates_event_name(tmp_path):
    with pytest.raises(ValueError, match="MATLAB identifier"):
        detect_delta_waves(tmp_path, event_name="../unsafe")


def test_wave_amplitudes_are_invariant_to_voltage_scale_and_offset():
    t, x = _trace()
    original = _detect(x, t)
    rescaled = _detect(5 * x + 200, t)
    np.testing.assert_allclose(rescaled.peaks, original.peaks)
    np.testing.assert_allclose(rescaled.amplitude, original.amplitude, rtol=1e-6)


def test_missing_session_restrictions_require_explicit_opt_out():
    with pytest.raises(ValueError, match="basepath"):
        detect_delta_waves(lfp=np.ones(2500), fs=250, save_mat=False)
    with pytest.raises(ValueError, match="basepath"):
        detect_delta_waves(lfp=np.ones(2500), fs=250)


def test_firing_without_suppression_retains_lfp_events():
    t, x = _trace()
    with pytest.warns(UserWarning, match="Unable to verify"):
        events = _detect(x, t, verify_firing=True, spike_times=[np.arange(0, 30, 0.01)])
    assert len(events) == 3
    assert not events.attrs["detectorinfo"]["detectionparms"]["firing_verified"]


def test_empty_nrem_returns_empty_events_with_metadata(tmp_path):
    _, x = _trace()
    _write_session(tmp_path, x[:, None], tags={"Delta": {"channels": 1}})
    savemat(
        tmp_path / f"{tmp_path.name}.SleepState.states.mat",
        {"SleepState": {"ints": {"NREMstate": np.empty((0, 2))}}},
    )
    events = detect_delta_waves(
        tmp_path, emg_threshold=np.inf, save_mat=False, verify_firing=False
    )
    assert events.empty
    assert events.attrs["detectorinfo"]["detectionintervals"].shape == (0, 2)
