import gc
import os
import warnings
from typing import List, Optional, Tuple

import numba as nb
import numpy as np
from scipy.signal import butter, firwin, sosfiltfilt


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
    ...     r"U:\data\hpc_ctx_project\HP13\HP13_day12_20241112\HP13_day12_20241112.dat",
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
                        ).astype(data.dtype)  # Ensure consistent dtype
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


def downsample_binary(
    filepath: str,
    n_channels: int,
    original_fs: int = 20000,
    target_fs: int = 1250,
    precision: str = "int16",
    filter_order: int = 4,
) -> str:
    """
    Optimized function to downsample raw binary data.
    """
    if original_fs % target_fs != 0:
        raise ValueError(
            "Original sampling frequency must be an integer multiple of the target frequency."
        )

    downsample_factor = original_fs // target_fs
    nyquist = target_fs / 2

    # Design a stable low-pass filter
    sos = butter(filter_order, nyquist / (original_fs / 2), btype="low", output="sos")

    downsampled_filepath = (
        os.path.splitext(filepath)[0] + ".lfp"
    )

    bytes_size = np.dtype(precision).itemsize
    chunk_size = 10_000  # Adjust for optimal performance
    with open(filepath, "rb") as infile, open(downsampled_filepath, "wb") as outfile:
        infile.seek(0, 2)
        n_samples = infile.tell() // (n_channels * bytes_size)
        infile.seek(0, 0)

        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            n_chunk_samples = end_idx - start_idx

            # Load chunk
            data = np.fromfile(
                infile, dtype=precision, count=n_chunk_samples * n_channels
            )
            data = data.reshape((n_chunk_samples, n_channels))

            # Filter and downsample
            filtered_data = sosfiltfilt(sos, data, axis=0)
            downsampled_data = filtered_data[::downsample_factor, :]

            # Write to output file
            downsampled_data.astype(precision).tofile(outfile)

            del data, filtered_data, downsampled_data
            gc.collect()

    return downsampled_filepath





@nb.jit(nopython=True, parallel=True, fastmath=True)
def filter_and_downsample(data, fir_coeffs, downsample_factor):
    """
    JIT-compiled function to filter and downsample data.
    """
    n_samples, n_channels = data.shape
    n_output_samples = n_samples // downsample_factor
    output = np.zeros((n_output_samples, n_channels), dtype=data.dtype)

    for ch in nb.prange(n_channels):
        # Convolve with FIR filter (linear phase, symmetric)
        filtered = np.convolve(data[:, ch], fir_coeffs, mode="valid")
        # Downsample
        output[:, ch] = filtered[::downsample_factor]

    return output


def downsample_binary_ultrafast(
    filepath: str,
    n_channels: int,
    original_fs: int = 20000,
    target_fs: int = 1250,
    precision: str = "int16",
    filter_order: int = 64,
) -> str:
    """
    Ultrafast function to downsample raw binary data.
    """
    if original_fs % target_fs != 0:
        raise ValueError("Original sampling frequency must be an integer multiple of the target frequency.")

    downsample_factor = original_fs // target_fs
    nyquist = target_fs / 2

    # Design FIR filter
    fir_coeffs = firwin(filter_order + 1, nyquist / (original_fs / 2), pass_zero="lowpass")

    # Output file
    downsampled_filepath = os.path.splitext(filepath)[0] + ".lfp"

    # Memory-mapped I/O setup
    bytes_size = np.dtype(precision).itemsize
    chunk_size = 10_000_000  # Process 10M samples at a time for I/O efficiency
    with open(filepath, "rb") as infile, open(downsampled_filepath, "wb") as outfile:
        infile.seek(0, 2)
        n_samples = infile.tell() // (n_channels * bytes_size)
        infile.seek(0, 0)

        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            n_chunk_samples = end_idx - start_idx

            # Load chunk
            data = np.fromfile(infile, dtype=precision, count=n_chunk_samples * n_channels)
            data = data.reshape((n_chunk_samples, n_channels))

            # Filter and downsample
            downsampled_data = filter_and_downsample(data, fir_coeffs, downsample_factor)

            # Write to output file
            downsampled_data.astype(precision).tofile(outfile)

            del data, downsampled_data
            gc.collect()

    return downsampled_filepath


if __name__ == "__main__":
    # time function
    import time

    start = time.time()
    downsample_binary_ultrafast(
        filepath=r"U:\data\hpc_ctx_project\HP13\HP13_day1_20241030\HP13_probe_241030_111814\amplifier - Copy.dat",
        n_channels=128,
        original_fs=20000,
        target_fs=1250,
        precision="int16",
        filter_order=4,
    )
    print(f"Elapsed time: {time.time() - start:.2f} s")
