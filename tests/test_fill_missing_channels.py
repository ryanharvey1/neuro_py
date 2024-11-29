import os
import tempfile
import unittest

import numpy as np

from neuro_py.raw.preprocessing import fill_missing_channels


class TestFillMissingChannels(unittest.TestCase):
    def test_fill_missing_channels(self):
        # Test parameters
        n_channels = 8
        missing_channels = [0, 5]
        n_samples = 100
        precision = "int16"
        dtype = np.dtype(precision)

        # Create temporary directory and file
        with tempfile.TemporaryDirectory() as tmpdir:
            input_file = os.path.join(tmpdir, "test_input.dat")
            output_file = os.path.join(tmpdir, "corrected_test_input.dat")

            # Generate test data: random data for present channels
            present_channels = [
                ch for ch in range(n_channels) if ch not in missing_channels
            ]
            original_data = np.random.randint(
                np.iinfo(dtype).min,
                np.iinfo(dtype).max,
                size=(n_samples, len(present_channels)),
                dtype=dtype,
            )
            original_data.tofile(input_file)

            # Call the function
            fill_missing_channels(
                basepath=tmpdir,
                n_channels=n_channels,
                filename="test_input.dat",
                missing_channels=missing_channels,
                precision=precision,
                chunk_size=10,
            )

            # Validate the output file
            self.assertTrue(os.path.exists(output_file), "Output file not created.")

            # Load and check the corrected data
            corrected_data = np.fromfile(output_file, dtype=dtype).reshape(
                n_samples, n_channels
            )
            for idx in range(n_channels):
                if idx in missing_channels:
                    # Missing channels should be filled with zeros
                    self.assertTrue(
                        np.all(corrected_data[:, idx] == 0),
                        f"Channel {idx} not properly filled with zeros.",
                    )
                else:
                    # Present channels should match the original data
                    self.assertTrue(
                        np.array_equal(
                            corrected_data[:, idx],
                            original_data[:, present_channels.index(idx)],
                        ),
                        f"Channel {idx} data mismatch.",
                    )


if __name__ == "__main__":
    unittest.main()
