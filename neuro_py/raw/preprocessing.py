import gc
import os
import warnings
from typing import List, Tuple

import numpy as np


def remove_artifacts(
    filepath: str,
    n_channels: int,
    zero_intervals: List[Tuple[int, int]],
    precision: str = "int16",
    mode: str = "linear",
) -> None:
    """
    Silence user-defined periods from recordings in a binary file.

    Parameters
    ----------
    filepath : str
        Path to the binary file.
    n_channels : int
        Number of channels in the file.
    zero_intervals : List[Tuple[int, int]]
        List of intervals (start, end) in sample indices to zero out.
    precision : str, optional
        Data precision, by default "int16".
    mode : str, optional
        Mode of interpolation,"zeros", "linear", "gaussian", default: "linear"

        - "zeros": zero out the interval

        - "linear": interpolate linearly between the start and end of the interval

        - "gaussian": interpolate using a gaussian function that has the same
            variance that the one in the recordings, on a per channel basis

    Returns
    -------
    None

    Examples
    --------
    >>> fs = 20_000
    >>> remove_artifacts(
    >>>     "U:\data\hpc_ctx_project\HP13\HP13_day12_20241112\HP13_day12_20241112.dat",
    >>>     n_channels=128,
    >>>     zero_intervals = (bad_intervals.data * fs).astype(int)
    >>>     )
    """
    # Check if file exists
    if not os.path.exists(filepath):
        warnings.warn("File does not exist.")
        return

    # Open the file in memory-mapped mode for read/write
    bytes_size = np.dtype(precision).itemsize
    with open(filepath, "rb") as f:
        startoffile = f.seek(0, 0)
        endoffile = f.seek(0, 2)
        n_samples = int((endoffile - startoffile) / n_channels / bytes_size)

    # Map the file to memory in read-write mode
    data = np.memmap(
        filepath, dtype=precision, mode="r+", shape=(n_samples, n_channels)
    )
    try:
        # if shape is (2,) then it is a single interval, then add dimension
        if zero_intervals.shape == (2,):
            zero_intervals = zero_intervals[np.newaxis, :]

        # Zero out the specified intervals
        if mode == "zeros":
            zero_value = np.zeros((1, n_channels), dtype=precision)
            for start, end in zero_intervals:
                if 0 <= start < n_samples and 0 < end <= n_samples:
                    data[start:end, :] = zero_value
                else:
                    warnings.warn(
                        f"Interval ({start}, {end}) is out of bounds and was skipped."
                    )
        elif mode == "linear":
            for start, end in zero_intervals:
                if 0 <= start < n_samples and 0 < end <= n_samples:
                    for ch in range(n_channels):
                        # Compute float interpolation and round before casting
                        interpolated = np.linspace(
                            data[start, ch],
                            data[end, ch],
                            end - start,
                        ).astype(
                            data.dtype
                        )  # Ensure consistent dtype
                        data[start:end, ch] = interpolated
                else:
                    warnings.warn(
                        f"Interval ({start}, {end}) is out of bounds and was skipped."
                    )
        elif mode == "gaussian":
            # Initialize arrays to store mean and std for each channel
            means = np.zeros(n_channels, dtype=np.float64)
            stds = np.zeros(n_channels, dtype=np.float64)

            # Calculate mean and variance for each channel outside the intervals
            for ch in range(n_channels):
                channel_data = data[:, ch]
                valid_samples = []

                for start, end in zero_intervals:
                    # Append valid data chunks before and after each interval
                    if start > 0:
                        valid_samples.append(channel_data[:start])
                    if end < n_samples:
                        valid_samples.append(channel_data[end:])

                # Concatenate all valid samples for this channel
                valid_data = (
                    np.concatenate(valid_samples) if valid_samples else channel_data
                )

                # Calculate mean and std for the channel
                means[ch] = np.mean(valid_data)
                stds[ch] = np.std(valid_data)

            # Fill intervals with Gaussian random values
            for start, end in zero_intervals:
                if 0 <= start < n_samples and 0 < end <= n_samples:
                    for ch in range(n_channels):
                        gaussian_values = np.random.normal(
                            means[ch], stds[ch], end - start
                        ).astype(precision)
                        data[start:end, ch] = gaussian_values
                else:
                    warnings.warn(
                        f"Interval ({start}, {end}) is out of bounds and was skipped."
                    )
    finally:
        # Explicitly flush and release the memory-mapped object
        data.flush()
        del data
        gc.collect()

    # Save a log file with intervals zeroed out
    log_file = os.path.splitext(filepath)[0] + "_zeroed_intervals.log"
    try:
        with open(log_file, "w") as f:
            f.write(f"Zeroed intervals: {zero_intervals.tolist()}\n")
    except Exception as e:
        warnings.warn(f"Failed to create log file: {e}")
