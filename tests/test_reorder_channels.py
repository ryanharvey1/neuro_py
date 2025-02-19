import os

import numpy as np
import pytest

import neuro_py as npy


# Test function
def test_reorder_channels(tmp_path):
    # Create a temporary binary file with known data
    n_channels = 4
    n_samples = 10
    data = np.arange(n_samples * n_channels, dtype=np.int16).reshape(
        n_samples, n_channels
    )
    file_path = os.path.join(tmp_path, "test_file.dat")
    data.tofile(file_path)

    # Define the new channel order
    channel_order = [2, 0, 3, 1]  # Reorder channels

    # Call the function to reorder channels
    new_file_path = npy.raw.reorder_channels(file_path, n_channels, channel_order)

    # Verify the new file exists
    assert os.path.exists(new_file_path)

    # Read the reordered data from the new file
    reordered_data = np.fromfile(new_file_path, dtype=np.int16).reshape(
        n_samples, n_channels
    )

    # Expected reordered data
    expected_data = data[:, channel_order]

    # Compare the reordered data with the expected data
    np.testing.assert_array_equal(reordered_data, expected_data)

    # Clean up (optional, as tmp_path is automatically cleaned up by pytest)
    os.remove(file_path)
    os.remove(new_file_path)
