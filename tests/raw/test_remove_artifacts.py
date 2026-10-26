import os
import tempfile
import warnings

import numpy as np
import pytest

from neuro_py.raw.preprocessing import remove_artifacts


def test_remove_artifacts():
    n_channels = 4
    n_samples = 100
    precision = "int16"
    zero_intervals = np.array([[20, 30], [50, 60]])
    channels_to_remove = [1, 3]  # Only remove artifacts from channels 1 and 3

    with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as tmpfile:
        filepath = tmpfile.name

    try:
        # Generate synthetic data and save it to the file
        original_data = np.random.randint(
            -1000, 1000, size=(n_samples, n_channels), dtype=precision
        )
        with open(filepath, "wb") as f:
            original_data.tofile(f)

        # Test mode "zeros"
        remove_artifacts(
            filepath,
            n_channels,
            zero_intervals,
            precision,
            mode="zeros",
            channels_to_remove=channels_to_remove,
        )
        data = np.memmap(
            filepath, dtype=precision, mode="r", shape=(n_samples, n_channels)
        )
        for start, end in zero_intervals:
            # Ensure only the selected channels are zeroed
            for ch in channels_to_remove:
                assert np.all(data[start:end, ch] == 0)
        del data
        os.remove(filepath)

        # Test mode "linear"
        with open(filepath, "wb") as f:
            original_data.tofile(f)

        remove_artifacts(
            filepath,
            n_channels,
            zero_intervals,
            precision,
            mode="linear",
            channels_to_remove=channels_to_remove,
        )

        data = np.memmap(
            filepath, dtype=precision, mode="r", shape=(n_samples, n_channels)
        )
        for start, end in zero_intervals:
            for ch in channels_to_remove:
                # Interpolate from clean samples outside the half-open interval.
                expected = np.linspace(
                    original_data[start - 1, ch],
                    original_data[end, ch],
                    end - start + 2,
                )[1:-1].astype(data.dtype)
                np.testing.assert_array_equal(data[start:end, ch], expected)
        del data
        os.remove(filepath)

        # # Test mode "gaussian"
        # with open(filepath, "wb") as f:
        #     original_data.tofile(f)

        # remove_artifacts(
        #     filepath, n_channels, zero_intervals, precision, mode="gaussian", channels_to_remove=channels_to_remove
        # )
        # data = np.memmap(
        #     filepath, dtype=precision, mode="r", shape=(n_samples, n_channels)
        # )
        # for start, end in zero_intervals:
        #     for ch in channels_to_remove:
        #         segment = data[start:end, ch]
        #         channel_mean = np.mean(original_data[:, ch])
        #         channel_std = np.std(original_data[:, ch])
        #         assert abs(np.mean(segment) - channel_mean) < channel_std
        #         assert abs(np.std(segment) - channel_std) < channel_std
        # del data
    finally:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception as e:
                warnings.warn(f"Failed to delete temporary file: {e}")


@pytest.mark.parametrize("precision", ["int16", "float32"])
@pytest.mark.parametrize(
    "intervals, expected",
    [
        ([(1, 4)], [0, 0, 0, 0, 0]),
        ([(0, 4)], [0, 0, 0, 0, 0]),
        ([(1, 5)], [0, 0, 0, 0, 0]),
        ([(3, 4), (1, 3)], [0, 0, 0, 0, 0]),
        ([(2, 4), (1, 3)], [0, 0, 0, 0, 0]),
    ],
)
def test_linear_removal_uses_clean_anchors(tmp_path, precision, intervals, expected):
    path = tmp_path / "recording.dat"
    data = np.column_stack(([0, 1000, 1000, 1000, 0], np.arange(5))).astype(precision)
    data.tofile(path)

    remove_artifacts(str(path), 2, intervals, precision, channels_to_remove=[0])

    actual = np.fromfile(path, dtype=precision).reshape(-1, 2)
    np.testing.assert_array_equal(actual[:, 0], expected)
    np.testing.assert_array_equal(actual[:, 1], data[:, 1])
    assert path.stat().st_size == data.nbytes


def test_single_sample_artifact_is_interpolated(tmp_path):
    path = tmp_path / "recording.dat"
    np.array([2, 1000, 6], dtype=np.int16).tofile(path)
    remove_artifacts(str(path), 1, [(1, 2)])
    np.testing.assert_array_equal(np.fromfile(path, dtype=np.int16), [2, 4, 6])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"zero_intervals": [(1, 2), (4, 3)]},
        {"zero_intervals": [(1, 2), (3.5, 4)]},
        {"zero_intervals": [(1, 2), (3, np.nan)]},
        {"zero_intervals": [1, 2, 3]},
        {"zero_intervals": [(0, 3), (3, 5)]},
        {"channels_to_remove": [0, 1]},
        {"channels_to_remove": [0, -1]},
        {"channels_to_remove": [0.5]},
        {"n_channels": 0},
        {"n_channels": 2},
        {"mode": "unknown"},
    ],
)
def test_invalid_requests_leave_recording_unchanged(tmp_path, kwargs):
    path = tmp_path / "recording.dat"
    original = np.arange(5, dtype=np.int16).tobytes()
    path.write_bytes(original)
    options = dict(n_channels=1, zero_intervals=[(1, 3)])
    options.update(kwargs)

    with pytest.raises(ValueError):
        remove_artifacts(str(path), **options)

    assert path.read_bytes() == original


def test_boundary_interval_does_not_fail_after_earlier_edits(tmp_path):
    path = tmp_path / "recording.dat"
    np.array([0, 100, 100, 3, 100, 100], dtype=np.int16).tofile(path)
    remove_artifacts(str(path), 1, [(1, 3), (4, 6)])
    np.testing.assert_array_equal(np.fromfile(path, dtype=np.int16), [0, 1, 2, 3, 3, 3])


def test_out_of_bounds_regions_are_skipped_and_logged_accurately(tmp_path):
    path = tmp_path / "recording.dat"
    np.arange(5, dtype=np.int16).tofile(path)
    with pytest.warns(UserWarning, match="out of bounds"):
        remove_artifacts(str(path), 1, [(-1, 2), (1, 3), (4, 6)], mode="zeros")
    np.testing.assert_array_equal(np.fromfile(path, dtype=np.int16), [0, 0, 0, 3, 4])
    assert "[(1, 3)]" in path.with_name("recording_zeroed_intervals.log").read_text()


def test_zero_mode_can_remove_all_samples(tmp_path):
    path = tmp_path / "recording.dat"
    np.arange(5, dtype=np.int16).tofile(path)
    remove_artifacts(str(path), 1, [(0, 5)], mode="zeros")
    np.testing.assert_array_equal(np.fromfile(path, dtype=np.int16), np.zeros(5))


@pytest.mark.parametrize("intervals", [[], np.empty((0, 2), dtype=int)])
def test_empty_regions_are_noop(tmp_path, intervals):
    path = tmp_path / "recording.dat"
    original = np.arange(5, dtype=np.int16).tobytes()
    path.write_bytes(original)
    remove_artifacts(str(path), 1, intervals)
    assert path.read_bytes() == original
