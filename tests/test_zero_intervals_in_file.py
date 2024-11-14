import numpy as np
import os
import tempfile
from neuro_py.raw.preprocessing import zero_intervals_in_file

def test_zero_intervals_in_file():
    # Set up test parameters
    n_channels = 4
    n_samples = 100
    precision = "int16"
    zero_intervals = [(10, 20), (40, 50), (90, 100)]
    
    # Create a temporary file to act as our binary data file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as tmpfile:
        filepath = tmpfile.name

    try:
        # Generate some sample data and write it to the temporary file
        original_data = (np.arange(n_samples * n_channels, dtype=precision)
                         .reshape(n_samples, n_channels))
        original_data.tofile(filepath)

        # Run the function
        zero_intervals_in_file(filepath, n_channels, zero_intervals, precision)

        # Load the file and check intervals are zeroed out
        data = np.fromfile(filepath, dtype=precision).reshape(n_samples, n_channels)

        for start, end in zero_intervals:
            # Check that the specified intervals are zeroed
            assert np.all(data[start:end, :] == 0), f"Interval ({start}, {end}) was not zeroed."
        
        # Check that the other intervals are unchanged
        for i in range(n_samples):
            if not any(start <= i < end for start, end in zero_intervals):
                expected_values = original_data[i, :]
                assert np.array_equal(data[i, :], expected_values), f"Data outside intervals was altered at index {i}."

        # Check if the log file was created and contains the correct intervals
        log_filepath = filepath.replace(".dat", "_zeroed_intervals.log")
        assert os.path.exists(log_filepath), "Log file was not created."

        with open(log_filepath, "r") as log_file:
            log_content = log_file.readlines()
            for i, (start, end) in enumerate(zero_intervals):
                assert log_content[i].strip() == f"{start} {end}", "Log file content is incorrect."

    finally:
        # Clean up the temporary file
        os.remove(filepath)
        os.remove(log_filepath)