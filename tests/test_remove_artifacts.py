import numpy as np
import os
import tempfile

from neuro_py.raw.preprocessing import remove_artifacts

def test_remove_artifacts():
    # Parameters
    n_channels = 4
    n_samples = 100
    precision = "int16"
    zero_intervals = [(20, 30), (50, 60)]

    # Create a temporary binary file
    with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
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
        data = np.memmap(filepath, dtype=precision, mode="r", shape=(n_samples, n_channels))
        for start, end in zero_intervals:
            assert np.all(data[start:end, :] == 0)
        del data  # Close memmap

        # Test mode "linear"
        remove_artifacts(filepath, n_channels, zero_intervals, precision, mode="linear")
        data = np.memmap(filepath, dtype=precision, mode="r", shape=(n_samples, n_channels))
        for start, end in zero_intervals:
            for ch in range(n_channels):
                expected = np.linspace(original_data[start, ch], original_data[end, ch], end - start)
                np.testing.assert_allclose(data[start:end, ch], expected, rtol=1e-5)
        del data  # Close memmap

        # Test mode "gaussian"
        remove_artifacts(filepath, n_channels, zero_intervals, precision, mode="gaussian")
        data = np.memmap(filepath, dtype=precision, mode="r", shape=(n_samples, n_channels))
        for start, end in zero_intervals:
            for ch in range(n_channels):
                segment = data[start:end, ch]
                # Check that the segment follows the approximate statistics of the channel
                channel_mean = np.mean(original_data[:, ch])
                channel_std = np.std(original_data[:, ch])
                assert abs(np.mean(segment) - channel_mean) < channel_std
                assert abs(np.std(segment) - channel_std) < channel_std
        del data  # Close memmap
    finally:
        # Clean up temporary file
        os.remove(filepath)