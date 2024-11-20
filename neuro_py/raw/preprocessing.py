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

        - "gaussian": [Not implemented, TBD] interpolate using a gaussian function that has the same
            variance that the one in the recordings, on a per channel basis

    Returns
    -------
    None

    Examples
    --------
    >>> fs = 20_000
    >>> remove_artifacts(
    >>>     r"U:\data\hpc_ctx_project\HP13\HP13_day12_20241112\HP13_day12_20241112.dat",
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
            # not implemented error message
            raise NotImplementedError("Gaussian mode not implemented.")

            # max_samples = 10_000
            # rng = np.random.default_rng()

            # # Compute valid regions and sample
            # valid_mask = np.ones(n_samples, dtype=bool)
            # for start, end in zero_intervals:
            #     valid_mask[start:end] = False

            # valid_indices = np.flatnonzero(valid_mask)
            # sampled_indices = rng.choice(
            #     valid_indices, size=min(max_samples, len(valid_indices)), replace=False
            # )
            # sampled_data = data[sampled_indices, :]

            # # Compute mean and std for each channel
            # means = np.mean(sampled_data, axis=0)
            # stds = np.std(sampled_data, axis=0)

            # from scipy.signal import butter, filtfilt

            # def bandpass_filter(signal, lowcut, highcut, fs, order=4):
            #     nyquist = 0.5 * fs
            #     low = lowcut / nyquist
            #     high = highcut / nyquist
            #     b, a = butter(order, [low, high], btype="band")
            #     return filtfilt(b, a, signal, axis=0)

            # # Parameters for bandpass filter
            # lowcut = 0.5
            # highcut = 100

            # for start, end in zero_intervals:
            #     if 0 <= start < n_samples and 0 < end <= n_samples:
            #         interval_length = end - start
            #         raw_noise = rng.normal(
            #             loc=means, scale=stds, size=(interval_length, n_channels)
            #         ).astype(precision)

            #         # Apply bandpass filter with handling for potential issues
            #         try:
            #             filtered_noise = bandpass_filter(raw_noise, lowcut, highcut, fs)
            #             filtered_noise = np.nan_to_num(filtered_noise, nan=0.0)
            #         except ValueError:
            #             warnings.warn(f"Filtering failed for interval ({start}, {end}), skipping.")
            #             continue

            #         # Prevent overwriting with unexpected data types
            #         data[start:end, :] = filtered_noise.astype(data.dtype)
            #     else:
            #         warnings.warn(f"Interval ({start}, {end}) is out of bounds and was skipped.")

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
