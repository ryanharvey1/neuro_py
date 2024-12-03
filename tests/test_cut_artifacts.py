import pytest
import numpy as np
import os
import tempfile
from typing import List, Tuple

from neuro_py.raw.preprocessing import cut_artifacts


@pytest.fixture
def create_test_file():
    """Fixture to create a temporary binary file with test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, "test_data.dat")

        # Test parameters
        n_channels = 4
        precision = "int16"
        original_data = np.arange(100).reshape(-1, n_channels).astype(precision)

        # Write test data to the file
        with open(filepath, "wb") as f:
            f.write(original_data.tobytes())

        yield filepath, n_channels, precision, original_data


def test_cut_artifacts(create_test_file):
    # Get the temporary file, parameters, and data
    filepath, n_channels, precision, original_data = create_test_file

    # Define intervals to cut
    cut_intervals: List[Tuple[int, int]] = [(5, 10), (15, 20)]  # In sample indices

    # Expected output after cutting
    keep_mask = np.ones(len(original_data), dtype=bool)
    for start, end in cut_intervals:
        keep_mask[start:end] = False
    expected_data = original_data[keep_mask]

    # Run the function
    output_filepath = os.path.splitext(filepath)[0] + "_cut.dat"
    cut_artifacts(
        filepath=filepath,
        n_channels=n_channels,
        cut_intervals=cut_intervals,
        precision=precision,
        output_filepath=output_filepath,
    )

    # Verify the output file
    with open(output_filepath, "rb") as f:
        cut_data = np.frombuffer(f.read(), dtype=precision).reshape(-1, n_channels)

    # Assertions
    assert len(cut_data) == len(expected_data), "The output file length does not match the expected length."
    np.testing.assert_array_equal(
        cut_data, expected_data, "The output data does not match the expected data."
    )

    # Check if the file exists and is smaller than the original
    assert os.path.exists(output_filepath), "The output file does not exist."
    assert os.path.getsize(output_filepath) < os.path.getsize(filepath), (
        "The output file size is not smaller than the original."
    )
