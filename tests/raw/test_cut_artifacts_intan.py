import os

import numpy as np
import pytest

from neuro_py.raw.preprocessing import cut_artifacts_intan


@pytest.fixture
def setup_test_folder(tmp_path):
    """
    Create a temporary folder with test Intan data files.
    """
    folder = tmp_path / "test_data"
    folder.mkdir()

    # Parameters for the test files
    n_samples = 1000
    n_channels_amplifier = 4
    precision = np.int16

    # Create amplifier.dat file
    amplifier_path = folder / "amplifier.dat"
    amplifier_data = np.random.randint(
        np.iinfo(precision).min,
        np.iinfo(precision).max,
        size=(n_samples, n_channels_amplifier),
        dtype=precision,
    )
    amplifier_data.tofile(amplifier_path)

    # Create auxiliary.dat file
    auxiliary_path = folder / "auxiliary.dat"
    auxiliary_data = np.random.randint(0, 65535, size=(n_samples, 2), dtype=np.uint16)
    auxiliary_data.tofile(auxiliary_path)

    # Create digitalin.dat file
    digitalin_path = folder / "digitalin.dat"
    digitalin_data = np.random.randint(0, 65535, size=(n_samples, 1), dtype=np.uint16)
    digitalin_data.tofile(digitalin_path)

    # Create time.dat file
    time_path = folder / "time.dat"
    time_data = np.arange(n_samples, dtype=np.int32)
    time_data.tofile(time_path)

    return folder, n_samples, n_channels_amplifier


def test_cut_artifacts_intan(setup_test_folder):
    """
    Test the `cut_artifacts_intan` function.
    """
    folder, n_samples, n_channels_amplifier = setup_test_folder

    # Define cut intervals (e.g., remove 100-199 and 300-399 samples)
    cut_intervals = [(100, 200), (300, 400)]

    # Run the function
    cut_artifacts_intan(
        folder_name=str(folder),
        n_channels_amplifier=n_channels_amplifier,
        cut_intervals=cut_intervals,
    )

    # Check if files are processed correctly
    for file_name in ["amplifier", "auxiliary", "digitalin", "time"]:
        file_path = folder / f"{file_name}_cut.dat"
        assert file_path.exists(), f"{file_name}_cut.dat file does not exist."

        # Verify that the file size matches the expected size after cutting
        original_size = n_samples
        for start, end in cut_intervals:
            original_size -= end - start

        # Amplifier and auxiliary have multiple channels, adjust size calculation
        if file_name in ["amplifier", "auxiliary"]:
            n_channels = n_channels_amplifier if file_name == "amplifier" else 2
            expected_size = original_size * n_channels * np.dtype(np.int16).itemsize
        elif file_name == "time":
            expected_size = original_size * np.dtype(np.int32).itemsize
        else:
            expected_size = original_size * np.dtype(np.uint16).itemsize

        assert os.path.getsize(file_path) == expected_size, (
            f"{file_name}_cut.dat size is incorrect."
        )

    # Check for ValueError when video files are present
    video_file_path = folder / "video.avi"
    video_file_path.touch()  # Create a dummy video file

    with pytest.raises(ValueError, match="Video files found in folder"):
        cut_artifacts_intan(
            folder_name=str(folder),
            n_channels_amplifier=n_channels_amplifier,
            cut_intervals=cut_intervals,
        )
