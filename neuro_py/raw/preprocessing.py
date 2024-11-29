import gc
import os
import warnings
from typing import List, Optional, Tuple

import numpy as np


def remove_artifacts(
    filepath: str,
    n_channels: int,
    zero_intervals: List[Tuple[int, int]],
    precision: str = "int16",
    mode: str = "linear",
    channels_to_remove: Optional[List[int]] = None,
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
        Mode of interpolation. Options are:

        - **"zeros"**: Zero out the interval.
        - **"linear"**: Interpolate linearly between the start and end of the interval (default).
        - **"gaussian"**: *(Not implemented, TBD)* Interpolate using a Gaussian function with the same variance as in the recordings, on a per-channel basis.

    channels_to_remove : List[int], optional
        List of channels (0-based indices) to remove artifacts from. If None, remove artifacts from all channels.

    Returns
    -------
    None

    Examples
    --------
    >>> fs = 20_000
    >>> remove_artifacts(
    ...     r"U:\\data\\hpc_ctx_project\\HP13\\HP13_day12_20241112\\HP13_day12_20241112.dat",
    ...     n_channels=128,
    ...     zero_intervals=(bad_intervals.data * fs).astype(int),
    ...     channels_to_remove=[0, 1, 2]  # Only remove artifacts from channels 0, 1, and 2
    ... )
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
        if np.shape(zero_intervals) == (2,):
            zero_intervals = np.expand_dims(zero_intervals, axis=0)

        # If no specific channels are provided, process all channels
        channels_to_remove = channels_to_remove or list(range(n_channels))

        # Zero out the specified intervals
        if mode == "zeros":
            zero_value = np.zeros((1, n_channels), dtype=precision)
            for start, end in zero_intervals:
                if 0 <= start < n_samples and 0 < end <= n_samples:
                    data[start:end, channels_to_remove] = zero_value[
                        0, channels_to_remove
                    ]
                else:
                    warnings.warn(
                        f"Interval ({start}, {end}) is out of bounds and was skipped."
                    )
        elif mode == "linear":
            for start, end in zero_intervals:
                if 0 <= start < n_samples and 0 < end <= n_samples:
                    for ch in channels_to_remove:
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


def fill_missing_channels(
    basepath: str,
    n_channels: int,
    filename: str,
    missing_channels: List[int],
    precision: str = "int16",
    chunk_size: int = 10_000,
) -> str:
    """
    Fill missing channels in a large binary file with zeros, processing in chunks.
    This function is useful when some channels were accidently deactivated during recording.

    Parameters
    ----------
    basepath : str
        Path to the folder containing the binary file.
    n_channels : int
        Total number of channels in the binary file (including the missing ones).
    filename : str
        Name of the binary file to modify.
    missing_channels : List[int]
        List of missing channel indices to be filled with zeros.
    precision : str, optional
        Data precision, by default "int16".
    chunk_size : int, optional
        Number of samples per chunk, by default 10,000.

    Returns
    -------
    str
        Path to the modified binary file.

    Examples
    --------
    >>> fill_missing_channels(
    ...    r"U:\\data\\hpc_ctx_project\\HP13\\HP13_day1_20241030\\HP13_cheeseboard_241030_153710",
    ...    128,
    ...    'amplifier.dat',
    ...    missing_channels = [0]
    ... )
    """
    file_path = os.path.join(basepath, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Binary file '{file_path}' does not exist.")

    dtype = np.dtype(precision)
    bytes_per_sample = dtype.itemsize
    present_channels = [ch for ch in range(n_channels) if ch not in missing_channels]

    # Calculate total number of samples
    file_size = os.path.getsize(file_path)
    n_samples = file_size // (bytes_per_sample * (n_channels - len(missing_channels)))
    if file_size % (bytes_per_sample * (n_channels - len(missing_channels))) != 0:
        raise ValueError("Data size is not consistent with expected shape.")

    # Prepare output file path
    new_file_path = os.path.join(basepath, f"corrected_{filename}")

    # Process file in chunks
    with open(file_path, "rb") as f_in, open(new_file_path, "wb") as f_out:
        for start in range(0, n_samples, chunk_size):
            # Read a chunk of data
            chunk = np.fromfile(
                f_in,
                dtype=dtype,
                count=chunk_size * (n_channels - len(missing_channels)),
            )
            chunk = chunk.reshape(-1, n_channels - len(missing_channels))

            # Create a new array with missing channels filled with zeros
            chunk_full = np.zeros((chunk.shape[0], n_channels), dtype=dtype)
            chunk_full[:, present_channels] = chunk

            # Write the chunk with missing channels added to the new file
            chunk_full.tofile(f_out)

    return new_file_path
