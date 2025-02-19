import gc
import os
import warnings
from multiprocessing import Pool
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
        if channels_to_remove is None:
            channels_to_remove = list(range(n_channels))

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


def cut_artifacts(
    filepath: str,
    n_channels: int,
    cut_intervals: List[Tuple[int, int]],
    precision: str = "int16",
    output_filepath: Optional[str] = None,
) -> None:
    """
    Remove user-defined periods from recordings in a binary file, resulting in a shorter file.

    Parameters
    ----------
    filepath : str
        Path to the original binary file.
    n_channels : int
        Number of channels in the file.
    cut_intervals : List[Tuple[int, int]]
        List of intervals (start, end) in sample indices to remove. Assumes sorted and non-overlapping.
    precision : str, optional
        Data precision, by default "int16".
    output_filepath : str, optional
        Path to save the modified binary file. If None, appends "_cut" to the original filename.

    Returns
    -------
    None
    """
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filepath}' does not exist.")

    # Set default output filepath
    if output_filepath is None:
        output_filepath = os.path.splitext(filepath)[0] + "_cut.dat"

    # Check for valid intervals
    for start, end in cut_intervals:
        if start >= end:
            raise ValueError(
                f"Invalid interval: ({start}, {end}). Start must be less than end."
            )

    # Map the original file and calculate parameters
    bytes_size = np.dtype(precision).itemsize
    with open(filepath, "rb") as f:
        startoffile = f.seek(0, 0)
        endoffile = f.seek(0, 2)
        n_samples = int((endoffile - startoffile) / n_channels / bytes_size)

    data = np.memmap(filepath, dtype=precision, mode="r", shape=(n_samples, n_channels))

    # Identify the indices to keep
    keep_mask = np.ones(n_samples, dtype=bool)
    for start, end in cut_intervals:
        if 0 <= start < n_samples and 0 < end <= n_samples:
            keep_mask[start:end] = False
        else:
            warnings.warn(
                f"Interval ({start}, {end}) is out of bounds and was skipped."
            )

    keep_indices = np.flatnonzero(keep_mask)

    # Create a new binary file with only the retained data
    with open(output_filepath, "wb") as output_file:
        for start_idx in range(0, len(keep_indices), 10_000):  # Process in chunks
            chunk_indices = keep_indices[start_idx : start_idx + 10_000]
            output_file.write(data[chunk_indices].tobytes())

    del data  # Release memory-mapped file


def cut_artifacts_intan(
    folder_name: str,
    n_channels_amplifier: int,
    cut_intervals: List[Tuple[int, int]],
    verbose: bool = True,
) -> None:
    """
    Cut specified artifact intervals from Intan data files.

    This function iterates through a set of Intan data files (amplifier, auxiliary,
    digitalin, digitalout, analogin, time, and supply), and for each file, it removes
    artifacts within the specified intervals by invoking the `cut_artifacts` function.

    Parameters
    ----------
    folder_name : str
        The folder where the Intan data files are located.
    n_channels_amplifier : int
        The number of amplifier channels used in the amplifier data file.
    cut_intervals : List[Tuple[int, int]]
        A list of intervals (start, end) in sample indices to remove artifacts.
        Each tuple represents the start and end sample index for an artifact to be cut.
        Assumes sorted and non-overlapping intervals.

    Returns
    -------
    None
        This function modifies the files in place, so there is no return value.

    Raises
    ------
    FileNotFoundError
        If the amplifier data file does not exist in the provided folder.
    ValueError
        If video files are found in the folder, as this function does not support video files.

    Examples
    --------
    >>> fs = 20_000
    >>> cut_artifacts_intan(
    ...     folder_name = r"path/to/data",
    ...     n_channels_amplifier = 128,
    ...     cut_intervals = (np.array([[394.4, 394.836], [400, 401], [404, 405]]) * fs).astype(int)
    ... )
    """

    # refuse to cut artifacts if any video file exist in folder
    video_files = [f for f in os.listdir(folder_name) if f.endswith(".avi")]
    if video_files:
        raise ValueError(f"Video files found in folder, refusing to cut: {video_files}")

    # Define data types for each file (from Intan documentation)
    files_table = {
        "amplifier": "int16",
        "auxiliary": "uint16",
        "digitalin": "uint16",
        "digitalout": "uint16",
        "analogin": "uint16",
        "time": "int32",
        "supply": "uint16",
    }

    # determine number of samples from amplifier file
    amplifier_file_path = os.path.join(folder_name, "amplifier.dat")
    if not os.path.exists(amplifier_file_path):
        raise FileNotFoundError(f"File '{amplifier_file_path}' does not exist.")

    # get number of bytes per sample
    bytes_size = np.dtype(files_table["amplifier"]).itemsize

    # each file should have the same number of samples
    n_samples = os.path.getsize(amplifier_file_path) // (
        n_channels_amplifier * bytes_size
    )

    for file_name, precision in files_table.items():
        file_path = os.path.join(folder_name, f"{file_name}.dat")

        if os.path.exists(file_path):
            if verbose:
                print(f"Processing {file_name}.dat file...")

            # get number of bytes per sample
            bytes_size = np.dtype(precision).itemsize

            # determine number of channels from n_samples
            n_channels = int(os.path.getsize(file_path) / n_samples / bytes_size)

            # for time file, cut and offset timestamps
            if file_name == "time":
                output_filepath = os.path.splitext(file_path)[0] + "_cut.dat"

                with open(output_filepath, "wb") as output_file:
                    # time indices as continuous array
                    filtered_time = np.arange(
                        n_samples - sum(end - start for start, end in cut_intervals),
                        dtype=np.int32,
                    )

                    # write to file
                    output_file.write(filtered_time.tobytes())
            else:
                # cut artifacts
                cut_artifacts(file_path, n_channels, cut_intervals, precision)

    # Calculate the expected number of samples after cutting
    total_samples_cut = sum(end - start for start, end in cut_intervals)
    expected_n_samples = n_samples - total_samples_cut

    # === Validation Section ===
    # Verify all `_cut.dat` files have the correct number of samples
    for file_name, precision in files_table.items():
        output_file_path = os.path.join(folder_name, f"{file_name}_cut.dat")
        original_file_path = os.path.join(folder_name, f"{file_name}.dat")

        if os.path.exists(output_file_path) and os.path.exists(original_file_path):
            # Dynamically calculate the number of channels
            bytes_size = np.dtype(precision).itemsize
            n_channels = os.path.getsize(original_file_path) // (n_samples * bytes_size)

            # Calculate the expected file size
            expected_size = expected_n_samples * n_channels * bytes_size
            actual_size = os.path.getsize(output_file_path)

            if actual_size != expected_size:
                raise RuntimeError(
                    f"{file_name}_cut.dat has an incorrect size. "
                    f"Expected {expected_size} bytes but found {actual_size} bytes."
                )


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


def process_chunk(args):
    """
    Process a chunk of the file in parallel.
    """
    start, end, file_path, n_channels, channel_order, dtype, new_file_path = args
    # Memory-map the input file for this chunk
    data = np.memmap(
        file_path,
        dtype=dtype,
        mode="r",
        shape=(end - start, n_channels),
        offset=start * n_channels * dtype.itemsize,
    )
    # Memory-map the output file for this chunk
    reordered_data = np.memmap(
        new_file_path,
        dtype=dtype,
        mode="r+",
        shape=(end - start, n_channels),
        offset=start * n_channels * dtype.itemsize,
    )
    # Reorder the channels
    reordered_data[:] = data[:, channel_order]
    # Flush changes to disk
    reordered_data.flush()


def reorder_channels(
    file_path: str,
    n_channels: int,
    channel_order: List[int],
    precision: str = "int16",
    num_processes: int = 8,  # Adjust based on your CPU cores
) -> str:
    """
    Reorder channels in a large binary file, processing in chunks.
    This function is useful when you want to reorder the channels in a binary file.

    Parameters
    ----------
    file_path : str
        Path to the file of the binary file to modify.
    n_channels : int
        Total number of channels in the binary file.
    channel_order : List[int]
        List of channel indices specifying the new order of channels.
    precision : str, optional
        Data precision, by default "int16".
    chunk_size : int, optional
        Number of samples per chunk, by default 10,000.

    Examples
    --------
    >>> reorder_channels(
    ...    r"U:\\data\\hpc_ctx_project\\HP13\\HP13_day1_20241030\\HP13_cheeseboard_241030_153710\\amplifier.dat",
    ...    128,
    ...    channel_order = [1, 0, 3, 2, ...]
    ... )
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Binary file '{file_path}' does not exist.")

    dtype = np.dtype(precision)
    bytes_per_sample = dtype.itemsize

    # Calculate total number of samples
    file_size = os.path.getsize(file_path)
    n_samples = file_size // (bytes_per_sample * n_channels)
    if file_size % (bytes_per_sample * n_channels) != 0:
        raise ValueError("Data size is not consistent with expected shape.")

    # Prepare output file path
    filename = os.path.basename(file_path)
    basepath = os.path.dirname(file_path)
    new_file_path = os.path.join(basepath, f"reordered_{filename}")

    # Create an empty output file of the correct size
    with open(new_file_path, "wb") as f:
        f.write(np.zeros(n_samples * n_channels, dtype=dtype).tobytes())

    # Split the work into chunks for parallel processing
    chunk_size = n_samples // num_processes
    chunks = [
        (
            i * chunk_size,
            (i + 1) * chunk_size,
            file_path,
            n_channels,
            channel_order,
            dtype,
            new_file_path,
        )
        for i in range(num_processes)
    ]

    # Handle the last chunk if n_samples is not divisible by num_processes
    if n_samples % num_processes != 0:
        chunks.append(
            (
                num_processes * chunk_size,
                n_samples,
                file_path,
                n_channels,
                channel_order,
                dtype,
                new_file_path,
            )
        )

    # Process chunks in parallel
    with Pool(num_processes) as pool:
        pool.map(process_chunk, chunks)
