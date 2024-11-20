import os
import tempfile
import warnings

import numpy as np
from neuro_py.raw.preprocessing import remove_artifacts


def test_remove_artifacts():
    n_channels = 4
    n_samples = 100
    precision = "int16"
    zero_intervals = np.array([[20, 30], [50, 60]])

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
        remove_artifacts(filepath, n_channels, zero_intervals, precision, mode="zeros")
        data = np.memmap(
            filepath, dtype=precision, mode="r", shape=(n_samples, n_channels)
        )
        for start, end in zero_intervals:
            assert np.all(data[start:end, :] == 0)
        del data
        os.remove(filepath)

        # Test mode "linear"
        with open(filepath, "wb") as f:
            original_data.tofile(f)

        remove_artifacts(filepath, n_channels, zero_intervals, precision, mode="linear")

        data = np.memmap(
            filepath, dtype=precision, mode="r", shape=(n_samples, n_channels)
        )
        for start, end in zero_intervals:
            for ch in range(n_channels):
                # Perform float interpolation with rounding before casting
                expected = np.linspace(
                    original_data[start, ch],
                    original_data[end, ch],
                    end - start,
                ).astype(
                    data.dtype
                )  # Explicit dtype match
                np.testing.assert_array_equal(data[start:end, ch], expected)
        del data
        os.remove(filepath)

        # # Test mode "gaussian"
        # with open(filepath, "wb") as f:
        #     original_data.tofile(f)

        # remove_artifacts(
        #     filepath, n_channels, zero_intervals, precision, mode="gaussian"
        # )
        # data = np.memmap(
        #     filepath, dtype=precision, mode="r", shape=(n_samples, n_channels)
        # )
        # for start, end in zero_intervals:
        #     for ch in range(n_channels):
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
