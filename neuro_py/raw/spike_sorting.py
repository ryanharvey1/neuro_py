import re
import time
import warnings

import matplotlib.pyplot as plt
import nelpy as nel
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import clear_output, display


def spike_sorting_progress(file: str, wait_time: float = 300, hue: str = "amp"):
    """
    Monitor the progress of a spike sorting process by checking the number of unsorted clusters in a tsv file.
    Will plot the progress in real-time and estimate the remaining time.

    Parameters
    ----------
    file : str
        The path to the cluster_info.tsv file containing the spike sorting results.
    wait_time : float, optional
        The time to wait between checks, in seconds. Default is 300 seconds (5 minutes).
    hue : str, optional
        The column to use for the hue in the plot. Default is "amp". ("id","amp","ch","depth","fr","group","n_spikes","sh")

    Returns
    -------
    None

    Examples
    --------
    >>> import neuro_py as npy
    >>> npy.raw.spike_sorting_progress("D:/KiloSort/hp18_day12_20250416/Kilosort_2025-04-17_161532/cluster_info.tsv")

    Notes
    ------
    This function assumes a specific way of spike sorting in phy:
    - Once cleaning/merging is done for a unit, the unit is marked good
    - Needs at least 1 unit marked good to start the process


    """

    # dark mode plotting
    plt.style.use("dark_background")

    def safe_read_csv(file, retries=3):
        for _ in range(retries):
            try:
                return pd.read_csv(file, sep="\t")
            except (pd.errors.EmptyDataError, PermissionError):
                time.sleep(1)
        raise IOError(f"Failed to read {file} after {retries} attempts.")

    # Function to count unsorted clusters
    def count_unsorted_clusters(file):
        df = safe_read_csv(file)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            df["group"].replace(np.nan, "unsorted", inplace=True)

        first_good = df.query('group=="good"').index[0]
        df_before_good = df.loc[: first_good - 1].copy()

        n_unsorted = df_before_good.query('group=="unsorted"').shape[0]
        n_sorted = df.query('group=="good"').shape[0]
        return n_unsorted, n_sorted

    # Initial count of unsorted clusters
    initial_unsorted, _ = count_unsorted_clusters(file)
    print(f"Initial unsorted clusters: {initial_unsorted}")

    # Set a flag to indicate whether the process is complete
    completed = False
    # List to store the time and unsorted count for rate calculation
    time_unsorted_data = [(0, initial_unsorted)]

    # Create a figure and axes for the plot
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [1, 4]}
    )

    while not completed:
        time.sleep(wait_time)  # Wait for x minutes

        current_unsorted, n_sorted = count_unsorted_clusters(file)
        current_time = (
            time_unsorted_data[-1][0] + wait_time / 60
        )  # Increment time by x minutes
        time_unsorted_data.append((current_time, current_unsorted))

        sorted_clusters = (
            time_unsorted_data[0][1] - current_unsorted
        )  # Number of clusters sorted so far

        if sorted_clusters > 0:
            # Use all previous data points to calculate an average rate of sorting
            total_time_elapsed = sum(
                [
                    time_unsorted_data[i + 1][0] - time_unsorted_data[i][0]
                    for i in range(len(time_unsorted_data) - 1)
                ]
            )
            total_clusters_sorted = sum(
                [
                    time_unsorted_data[i][1] - time_unsorted_data[i + 1][1]
                    for i in range(len(time_unsorted_data) - 1)
                ]
            )

            average_rate_of_sorting = (
                total_clusters_sorted / total_time_elapsed
            )  # Average clusters sorted per minute
            estimated_time_remaining = (
                current_unsorted / average_rate_of_sorting
            )  # Estimate the time remaining

            progress_text = (
                f"Time elapsed: {current_time} minutes\n"
                f"Current unsorted clusters: {current_unsorted}\n"
                f"Sorted clusters: {n_sorted}\n"
                f"Average rate of sorting: {average_rate_of_sorting:.2f} clusters/minute\n"
                f"Estimated time to completion: {estimated_time_remaining:.2f} minutes"
            )

            # Check if sorting is complete
            if current_unsorted == 0:
                completed = True
                progress_text += "\nSorting complete!"
        else:
            progress_text = f"No progress made in the last {current_time} minutes. Check if the process is working correctly."

        # Update the plot
        df = safe_read_csv(file)
        df["group"] = df["group"].replace(np.nan, "unsorted")

        # Clear only the plot axes (not the printed output)
        for a in ax:
            a.clear()

        # Set plot limits
        ax[0].set_xlim(df.sh.min(), df.sh.max())
        ax[1].set_xlim(df.sh.min(), df.sh.max())

        # Create the stripplot
        sns.stripplot(
            data=df[["depth", "sh"]]
            .value_counts()
            .reset_index()
            .sort_values(by=["sh"]),
            x="sh",
            y="depth",
            ax=ax[1],
            jitter=False,
            legend=False,
            color="white",
            alpha=0.5,
        )

        sns.stripplot(
            data=df.query('group=="good"'),
            x="sh",
            y="depth",
            hue=hue,
            ax=ax[1],
            jitter=True,
            legend=False,
            palette="rainbow",
        )

        count_df = (
            df.query('group=="good"')
            .groupby("sh", observed=True)
            .size()
            .reset_index(name="counts")
        )

        sns.barplot(data=count_df, x="sh", y="counts", ax=ax[0], palette="winter")
        for p in ax[0].patches:
            ax[0].annotate(
                f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
                color="white",
            )
        ax[0].set_ylabel("Number of \n clusters")
        ax[1].set_xlabel("Shank")
        ax[1].set_ylabel("Depth (um)")

        sns.despine()
        # Draw the plot
        plt.draw()
        plt.pause(0.01)  # Pause to allow the plot to updateg

        # Display progress text below the plot
        clear_output(wait=True)  # Clear only the previous plot and progress text
        display(fig)  # Display the updated plot
        print(progress_text)  # Display the progress text below the plot

    # Close the interactive plot after completion
    plt.ioff()
    plt.show()


def phy_log_to_epocharray(filename: str, merge_gap: float = 30):
    """
    Extract timestamps from a Phy log file and convert them to a nel.EpochArray.
    Will estimate the amount of time it took to spikesort a session.

    Parameters
    ----------
    filename : str
        The path to the Phy log file.
    merge_gap : float, optional
        The number of seconds to merge timestamps, by default 30

    Returns
    -------
    nel.EpochArray
        A nel.EpochArray containing the timestamps.

    Examples
    --------
    >>> import neuro_py as npy
    >>> filename = "D:/KiloSort/HP18/hp18_day11_20250415/Kilosort_2025-04-16_224949/phy.log"
    >>> timestamps = npy.raw.phy_log_to_epocharray(filename)
    >>> timestamps
    <EpochArray at 0x1f6c7da5710: 80 epochs> of length 4:02:01:591 hours

    """

    # Read the log file
    try:
        with open(filename, "r") as file:
            log_lines = file.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Log file not found: {filename}")

    # Define the regex pattern to extract timestamps
    timestamp_pattern = re.compile(r"\x1b\[\d+m(\d{2}:\d{2}:\d{2}\.\d{3})")

    # Extract timestamps using the regex pattern
    timestamps = []
    for line in log_lines:
        match = timestamp_pattern.search(line)
        if match:
            timestamps.append(match.group(1))

    # Create a Pandas DataFrame
    df = pd.DataFrame(timestamps, columns=["Timestamp"])

    # Convert the 'Timestamp' column to datetime format
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%H:%M:%S.%f")

    # Convert timestamps to total seconds (including milliseconds)
    df["Seconds"] = (
        df["Timestamp"].dt.hour * 3600
        + df["Timestamp"].dt.minute * 60
        + df["Timestamp"].dt.second
        + df["Timestamp"].dt.microsecond / 1e6
    )
    df["continous"] = df.Seconds.diff().abs().cumsum()

    intervals = np.array([df.continous[1:], df.continous[1:]]).T

    return nel.EpochArray(intervals).merge(gap=merge_gap)
