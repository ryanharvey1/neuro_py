import os
import sys
from typing import Optional, Tuple

import nelpy as nel
import numpy as np
from scipy.signal import find_peaks

import neuro_py as npy
from neuro_py.io import loading
from neuro_py.io.saving import epoch_to_mat
from neuro_py.process.intervals import find_interval


def detect_up_down_states(
    basepath: Optional[str] = None,
    st: Optional[nel.SpikeTrainArray] = None,
    nrem_epochs: Optional[nel.EpochArray] = None,
    region: str = "ILA|PFC|PL|EC1|EC2|EC3|EC4|EC5|MEC|CTX",
    min_dur: float = 0.03,
    max_dur: float = 0.5,
    percentile: float = 20,
    bin_size: float = 0.01,
    smooth_sigma: float = 0.02,
    min_cells: int = 10,
    save_mat: bool = True,
    epoch_by_epoch: bool = False,
    beh_epochs: Optional[nel.EpochArray] = None,
    show_figure: bool = False,
    overwrite: bool = False,
) -> Tuple[Optional[nel.EpochArray], Optional[nel.EpochArray]]:
    """
    Detect UP and DOWN states in neural data.

    UP and DOWN states are identified by computing the total firing rate of all
    simultaneously recorded neurons in bins of 10 ms, smoothed with a Gaussian kernel
    of 20 ms s.d. Epochs with a firing rate below the specified percentile threshold
    are considered DOWN states, while the intervals between DOWN states are classified
    as UP states. Epochs shorter than `min_dur` or longer than `max_dur` are discarded.

    Parameters
    ----------
    basepath : str
        Base directory path where event files and neural data are stored.
    st : Optional[nel.SpikeTrain], default=None
        Spike train data. If None, spike data will be loaded based on specified regions.
    nrem_epochs : Optional[nel.EpochArray], default=None
        NREM epochs. If None, epochs will be loaded from the basepath.
    region : str, default="ILA|PFC|PL|EC1|EC2|EC3|EC4|EC5|MEC"
        Brain regions for loading spikes. The first region is prioritized.
    min_dur : float, default=0.03
        Minimum duration for DOWN states, in seconds.
    max_dur : float, default=0.5
        Maximum duration for DOWN states, in seconds.
    percentile : float, default=20
        Percentile threshold for determining DOWN states based on firing rate.
    bin_size : float, default=0.01
        Bin size for computing firing rates, in seconds.
    smooth_sigma : float, default=0.02
        Standard deviation for Gaussian kernel smoothing, in seconds.
    min_cells : int, default=10
        Minimum number of neurons required for analysis.
    save_mat : bool, default=True
        Whether to save the detected UP and DOWN states to .mat files.
    epoch_by_epoch : bool, default=False
        Whether to perform detection epoch by epoch. If True, detection will be performed separately for each sleep epoch.
    beh_epochs : Optional[nel.EpochArray], default=None
        Optional behavioral epochs to use for epoch-by-epoch detection. If None, sleep epochs will be loaded and used.
    show_figure : bool, default=False
        Whether to display a figure showing firing rates during detected UP and DOWN states.
    overwrite : bool, default=False
        Whether to overwrite existing .mat files when saving detected states.

    Returns
    -------
    Tuple[Optional[nel.EpochArray], Optional[nel.EpochArray]]
        A tuple containing the detected DOWN state epochs and UP state epochs.
        Returns (None, None) if no suitable states are found or insufficient data is available.

    Examples
    --------
    >>> down_state, up_state = detect_up_down_states(basepath="/path/to/data", show_figure=True)

    From command line:
    $ python up_down_state.py /path/to/data

    Notes
    -----
    Detection method based on https://doi.org/10.1038/s41467-020-15842-4
    """

    def _detect_states(bst_segment: nel.AnalogSignalArray, domain: nel.EpochArray):
        """Detect down/up states within a given domain using shared logic."""

        down_state_epochs = bst_segment.bin_centers[
            find_interval(
                bst_segment.data.flatten()
                < np.percentile(bst_segment.data.T, percentile)
            )
        ]
        if down_state_epochs.shape[0] == 0:
            return None, None

        durations = down_state_epochs[:, 1] - down_state_epochs[:, 0]
        down_state_epochs = down_state_epochs[durations > bin_size]

        down_state_epochs = (
            nel.EpochArray(data=down_state_epochs).merge(gap=bin_size * 2).data
        )
        durations = down_state_epochs[:, 1] - down_state_epochs[:, 0]
        down_state_epochs = down_state_epochs[
            ~((durations < min_dur) | (durations > max_dur)), :
        ]
        if down_state_epochs.shape[0] == 0:
            return None, None

        down_state_epochs = nel.EpochArray(data=down_state_epochs, domain=domain)

        up_state_epochs = ~down_state_epochs
        up_state_epochs = up_state_epochs.data
        # make sure up states are longer than bin size
        durations = up_state_epochs[:, 1] - up_state_epochs[:, 0]
        up_state_epochs = up_state_epochs[durations > bin_size]
        # merge nearby up states that are closer than 2*bin_size
        up_state_epochs = nel.EpochArray(data=up_state_epochs, domain=domain).merge(
            gap=bin_size * 2
        )

        return down_state_epochs, up_state_epochs

    # check for existence of event files
    if save_mat and not overwrite:
        filename_downstate = os.path.join(
            basepath, os.path.basename(basepath) + "." + "down_state" + ".events.mat"
        )
        filename_upstate = os.path.join(
            basepath, os.path.basename(basepath) + "." + "up_state" + ".events.mat"
        )
        if os.path.exists(filename_downstate) & os.path.exists(filename_upstate):
            down_state = loading.load_events(basepath=basepath, epoch_name="down_state")
            up_state = loading.load_events(basepath=basepath, epoch_name="up_state")
            return down_state, up_state

    # load brain states
    if nrem_epochs is None:
        state_dict = loading.load_SleepState_states(basepath)
        nrem_epochs = nel.EpochArray(state_dict["NREMstate"])

    if nrem_epochs.isempty:
        print(f"No NREM epochs found for {basepath}")
        return None, None

    # load spikes
    if st is None:
        st, _ = loading.load_spikes(basepath, brainRegion=region)

    # check if there are enough cells
    if st is None or st.isempty or st.data.shape[0] < min_cells:
        print(f"No spikes found for {basepath} {region}")
        return None, None

    # flatten spikes
    st = st[nrem_epochs].flatten()

    # bin and smooth
    bst = st.bin(ds=bin_size).smooth(sigma=smooth_sigma)

    if epoch_by_epoch:
        if beh_epochs is None:
            epoch_df = npy.io.load_epoch(basepath)
            epoch_df = npy.session.compress_repeated_epochs(epoch_df)
            epoch_df = epoch_df.query("environment == 'sleep'")
            beh_epochs = nel.EpochArray(epoch_df[["startTime", "stopTime"]].values)

        down_state_epochs = []
        up_state_epochs = []
        for ep in beh_epochs:
            domain = nrem_epochs & ep
            if domain.isempty:
                continue

            down_state_epochs_, up_state_epochs_ = _detect_states(bst[ep], domain)
            if down_state_epochs_ is None or up_state_epochs_ is None:
                continue

            down_state_epochs.append(down_state_epochs_.data)
            up_state_epochs.append(up_state_epochs_.data)

        if len(down_state_epochs) == 0 or len(up_state_epochs) == 0:
            print(f"No down states found for {basepath}")
            return None, None

        down_state_epochs = nel.EpochArray(
            data=np.concatenate(down_state_epochs), domain=nrem_epochs
        )
        up_state_epochs = nel.EpochArray(
            data=np.concatenate(up_state_epochs), domain=nrem_epochs
        )
    else:
        down_state_epochs, up_state_epochs = _detect_states(bst, nrem_epochs)
        if down_state_epochs is None or up_state_epochs is None:
            print(f"No down states found for {basepath}")
            return None, None

    # save to cell explorer mat file
    if save_mat:
        epoch_to_mat(down_state_epochs, basepath, "down_state", "detect_up_down_states")
        epoch_to_mat(up_state_epochs, basepath, "up_state", "detect_up_down_states")

    # optional figure to show firing rate during up and down states
    if show_figure:
        from matplotlib import pyplot as plt

        plt.figure()
        ax = plt.gca()
        psth = npy.process.compute_psth(st.data, down_state_epochs.starts, n_bins=500)
        psth.columns = ["Down states"]
        psth.plot(ax=ax)

        psth = npy.process.compute_psth(st.data, up_state_epochs.starts, n_bins=500)
        psth.columns = ["Up states"]

        psth.plot(ax=ax)
        ax.legend(loc="upper right", frameon=False)
        ax.axvline(0, color="k", linestyle="--")

        ax.set_xlabel("Time from state transition (s)")
        ax.set_ylabel("Firing rate (Hz)")

    return down_state_epochs, up_state_epochs


def detect_up_down_states_bimodal_thresh(
    basepath: Optional[str] = None,
    st: Optional[nel.SpikeTrainArray] = None,
    nrem_epochs: Optional[nel.EpochArray] = None,
    region: str = "ILA|PFC|PL|EC1|EC2|EC3|EC4|EC5|MEC|CTX",
    bin_size: float = 0.01,
    smooth_sigma: float = 0.02,
    min_cells: int = 10,
    save_mat: bool = True,
    epoch_by_epoch: bool = False,
    beh_epochs: Optional[nel.EpochArray] = None,
    show_figure: bool = False,
    overwrite: bool = False,
    schmidt: bool = False,
    nboot: int = 500,
) -> Tuple[Optional[nel.EpochArray], Optional[nel.EpochArray]]:
    """
    Detect UP and DOWN states using bimodal_thresh on firing rate distribution.

    Uses the same data loading and epoch-by-epoch logic as `detect_up_down_states`,
    but applies Hartigan's dip test and bimodal threshold detection instead of a
    fixed percentile. This is useful when UP/DOWN states form a clear bimodal
    distribution in the firing rate histogram.

    Parameters
    ----------
    basepath : str
        Base directory path where event files and neural data are stored.
    st : Optional[nel.SpikeTrainArray], default=None
        Spike train data. If None, spike data will be loaded based on specified regions.
    nrem_epochs : Optional[nel.EpochArray], default=None
        NREM epochs. If None, epochs will be loaded from the basepath.
    region : str, default="ILA|PFC|PL|EC1|EC2|EC3|EC4|EC5|MEC|CTX"
        Brain regions for loading spikes. The first region is prioritized.
    bin_size : float, default=0.01
        Bin size for computing firing rates, in seconds.
    smooth_sigma : float, default=0.02
        Standard deviation for Gaussian kernel smoothing, in seconds.
    min_cells : int, default=10
        Minimum number of neurons required for analysis.
    save_mat : bool, default=True
        Whether to save the detected UP and DOWN states to .mat files.
    epoch_by_epoch : bool, default=False
        Whether to perform detection epoch by epoch.
    beh_epochs : Optional[nel.EpochArray], default=None
        Optional behavioral epochs to use for epoch-by-epoch detection.
    show_figure : bool, default=False
        Whether to display a figure showing firing rates during detected UP and DOWN states.
    overwrite : bool, default=False
        Whether to overwrite existing .mat files when saving detected states.
    schmidt : bool, default=False
        Use Schmidt trigger (hysteresis) for state transitions in bimodal_thresh.
    nboot : int, default=500
        Number of bootstrap iterations for Hartigan's dip test.

    Returns
    -------
    Tuple[Optional[nel.EpochArray], Optional[nel.EpochArray]]
        A tuple containing the detected DOWN state epochs and UP state epochs.
        Returns (None, None) if no suitable states are found or insufficient data is available.
    """

    # check for existence of event files
    if save_mat and not overwrite:
        filename_downstate = os.path.join(
            basepath,
            os.path.basename(basepath) + "." + "down_state" + ".events.mat",
        )
        filename_upstate = os.path.join(
            basepath,
            os.path.basename(basepath) + "." + "up_state" + ".events.mat",
        )
        if os.path.exists(filename_downstate) & os.path.exists(filename_upstate):
            down_state = loading.load_events(
                basepath=basepath, epoch_name="down_state"
            )
            up_state = loading.load_events(
                basepath=basepath, epoch_name="up_state"
            )
            return down_state, up_state

    # load brain states
    if nrem_epochs is None:
        state_dict = loading.load_SleepState_states(basepath)
        nrem_epochs = nel.EpochArray(state_dict["NREMstate"])

    if nrem_epochs.isempty:
        print(f"No NREM epochs found for {basepath}")
        return None, None

    # load spikes
    if st is None:
        st, _ = loading.load_spikes(basepath, brainRegion=region)

    # check if there are enough cells
    if st is None or st.isempty or st.data.shape[0] < min_cells:
        print(f"No spikes found for {basepath} {region}")
        return None, None

    # flatten spikes
    st = st[nrem_epochs].flatten()

    # bin and smooth
    bst = st.bin(ds=bin_size).smooth(sigma=smooth_sigma)

    def _detect_states_bimodal(
        bst_segment: nel.AnalogSignalArray, domain: nel.EpochArray
    ):
        """Detect down/up states using bimodal_thresh within a given domain."""

        # Get firing rate time series
        firing_rates = bst_segment.data.flatten()
        if firing_rates.size == 0:
            return None, None

        # Apply bimodal_thresh to the firing rates
        thresh, cross, bihist, diptest_result = bimodal_thresh(
            firing_rates, schmidt=schmidt, nboot=nboot
        )

        # If not bimodal or no threshold found
        if np.isnan(thresh):
            return None, None

        # Get bin centers (times)
        bin_centers = bst_segment.bin_centers

        # Extract downints and upints from cross
        downints = cross["downints"]  # indices into firing_rates array
        upints = cross["upints"]

        # Convert indices to time intervals using bin_centers
        if downints.size == 0:
            return None, None

        down_state_times = bin_centers[downints.astype(int)]
        down_state_epochs = nel.EpochArray(data=down_state_times, domain=domain)

        if upints.size == 0:
            # Generate up states as complement
            up_state_epochs = ~down_state_epochs
        else:
            up_state_times = bin_centers[upints.astype(int)]
            up_state_epochs = nel.EpochArray(data=up_state_times, domain=domain)

        return down_state_epochs, up_state_epochs

    if epoch_by_epoch:
        if beh_epochs is None:
            epoch_df = npy.io.load_epoch(basepath)
            epoch_df = npy.session.compress_repeated_epochs(epoch_df)
            epoch_df = epoch_df.query("environment == 'sleep'")
            beh_epochs = nel.EpochArray(epoch_df[["startTime", "stopTime"]].values)

        down_state_epochs = []
        up_state_epochs = []
        for ep in beh_epochs:
            domain = nrem_epochs & ep
            if domain.isempty:
                continue

            down_state_epochs_, up_state_epochs_ = _detect_states_bimodal(
                bst[ep], domain
            )
            if down_state_epochs_ is None or up_state_epochs_ is None:
                continue

            down_state_epochs.append(down_state_epochs_.data)
            up_state_epochs.append(up_state_epochs_.data)

        if len(down_state_epochs) == 0 or len(up_state_epochs) == 0:
            print(f"No down states found for {basepath}")
            return None, None

        down_state_epochs = nel.EpochArray(
            data=np.concatenate(down_state_epochs), domain=nrem_epochs
        )
        up_state_epochs = nel.EpochArray(
            data=np.concatenate(up_state_epochs), domain=nrem_epochs
        )
    else:
        down_state_epochs, up_state_epochs = _detect_states_bimodal(bst, nrem_epochs)
        if down_state_epochs is None or up_state_epochs is None:
            print(f"No down states found for {basepath}")
            return None, None

    # save to cell explorer mat file
    if save_mat:
        epoch_to_mat(
            down_state_epochs,
            basepath,
            "down_state",
            "detect_up_down_states_bimodal_thresh",
        )
        epoch_to_mat(
            up_state_epochs,
            basepath,
            "up_state",
            "detect_up_down_states_bimodal_thresh",
        )

    # optional figure to show firing rate during up and down states
    if show_figure:
        from matplotlib import pyplot as plt

        plt.figure()
        ax = plt.gca()
        psth = npy.process.compute_psth(st.data, down_state_epochs.starts, n_bins=500)
        psth.columns = ["Down states"]
        psth.plot(ax=ax)

        psth = npy.process.compute_psth(st.data, up_state_epochs.starts, n_bins=500)
        psth.columns = ["Up states"]

        psth.plot(ax=ax)
        ax.legend(loc="upper right", frameon=False)
        ax.axvline(0, color="k", linestyle="--")

        ax.set_xlabel("Time from state transition (s)")
        ax.set_ylabel("Firing rate (Hz)")

    return down_state_epochs, up_state_epochs


if __name__ == "__main__":
    basepath = sys.argv[1]

    detect_up_down_states(basepath)


def hartigan_diptest(
    data: np.ndarray, n_boot: int = 500, seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Dependency-free approximation of Hartigan's dip test with bootstrap p-value.

    This implementation uses a simple piecewise-linear unimodal fit to approximate
    the dip statistic and estimates the p-value via bootstrap draws from a
    unimodal Gaussian null. It avoids the external ``diptest`` package while
    preserving the API footprint needed by ``bimodal_thresh``.
    """

    rng = np.random.default_rng(seed)
    x = np.asarray(data, dtype=float).ravel()
    x = x[~np.isnan(x)]
    n = x.size
    if n < 3:
        return 0.0, 1.0

    x = np.sort(x)
    ecdf = (np.arange(1, n + 1)) / n

    def _candidate_dip(mode_idx: int) -> float:
        # Build a unimodal CDF that rises linearly to the candidate mode
        # and then linearly to 1.0; dip is the sup norm against the ECDF.
        left_span = max(x[mode_idx] - x[0], 1e-12)
        right_span = max(x[-1] - x[mode_idx], 1e-12)

        left_mass = mode_idx / n
        right_mass = 1.0 - left_mass

        left_mask = x <= x[mode_idx]
        u = np.empty_like(x, dtype=float)
        u[left_mask] = ((x[left_mask] - x[0]) / left_span) * left_mass
        u[~left_mask] = (
            left_mass + ((x[~left_mask] - x[mode_idx]) / right_span) * right_mass
        )
        u = np.clip(u, 0.0, 1.0)

        return float(np.max(np.abs(ecdf - u)))

    # Scan candidate modes (skip endpoints so both sides have support)
    dip_stat = float(min(_candidate_dip(m) for m in range(1, n - 1)))

    # Bootstrap p-value under a unimodal Gaussian null
    boot_dips = np.empty(max(int(n_boot), 1), dtype=float)
    for i in range(boot_dips.size):
        boot_sample = rng.standard_normal(n)
        boot_sample.sort()
        boot_ecdf = (np.arange(1, n + 1)) / n

        def _boot_candidate(mode_idx: int) -> float:
            left_span = max(boot_sample[mode_idx] - boot_sample[0], 1e-12)
            right_span = max(boot_sample[-1] - boot_sample[mode_idx], 1e-12)
            left_mass = mode_idx / n
            right_mass = 1.0 - left_mass
            left_mask = boot_sample <= boot_sample[mode_idx]
            u = np.empty_like(boot_sample, dtype=float)
            u[left_mask] = (
                (boot_sample[left_mask] - boot_sample[0]) / left_span
            ) * left_mass
            u[~left_mask] = (
                left_mass
                + ((boot_sample[~left_mask] - boot_sample[mode_idx]) / right_span)
                * right_mass
            )
            u = np.clip(u, 0.0, 1.0)
            return float(np.max(np.abs(boot_ecdf - u)))

        boot_dips[i] = min(_boot_candidate(m) for m in range(1, n - 1))

    p_value = float(np.mean(boot_dips >= dip_stat))

    return dip_stat, p_value


def bimodal_thresh(
    bimodal_data,
    max_thresh=np.inf,
    schmidt=False,
    max_hist_bins=25,
    start_bins=10,
    set_thresh=None,
    nboot=500,
):
    """
    BimodalThresh: Find threshold between bimodal data modes (e.g., UP vs DOWN states)
    and return crossing times (UP/DOWN onset/offset times).

    Parameters
    ----------
    bimodal_data : array-like
        Vector of bimodal data
    max_thresh : float, optional
        Maximum threshold value (default: inf)
    schmidt : bool, optional
        Use Schmidt trigger with halfway points between trough and peaks (default: False)
    max_hist_bins : int, optional
        Maximum number of histogram bins to try before giving up (default: 25)
    start_bins : int, optional
        Minimum number of histogram bins for initial histogram (default: 10)
    set_thresh : float, optional
        Manually set your own threshold (default: None)
    nboot : int, optional
        Number of bootstrap iterations for dip test (default: 500)

    Returns
    -------
    thresh : float
        Threshold value between modes
    cross : dict
        Dictionary with keys:
        - 'upints': array of UP state intervals [onsets, offsets]
        - 'downints': array of DOWN state intervals [onsets, offsets]
    bihist : dict
        Dictionary with keys:
        - 'bins': bin centers
        - 'hist': counts
    diptest_result : dict
        Dictionary with keys:
        - 'dip': Hartigan's dip test statistic
        - 'p': p-value for bimodal distribution

    Example
    -------
    >>> data = np.concatenate([np.random.normal(0, 1, 1000),
    ...                        np.random.normal(5, 1, 1000)])
    >>> thresh, cross, bihist, diptest_result = bimodal_thresh(data)

    Notes
    -----
    Python translation of BimodalThresh.m from MehrotraLevenstein_2023

    """

    # Initialize
    bimodal_data = np.array(bimodal_data).flatten()
    bimodal_data = bimodal_data[~np.isnan(bimodal_data)]

    # Run Hartigan's dip test for bimodality using diptest package
    dip_stat, p_value = hartigan_diptest(bimodal_data, n_boot=nboot)
    diptest_result = {"dip": dip_stat, "p": p_value}

    # If not bimodal, return empty
    if p_value > 0.05:
        cross = {"upints": np.array([]), "downints": np.array([])}
        hist_counts, bin_edges = np.histogram(bimodal_data, bins=start_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bihist = {"hist": hist_counts, "bins": bin_centers}
        return np.nan, cross, bihist, diptest_result

    # Remove data over max threshold
    bimodal_data = bimodal_data[bimodal_data < max_thresh]

    # Find histogram with exactly 2 peaks
    num_peaks = 1
    num_bins = start_bins

    while num_peaks != 2:
        hist_counts, bin_edges = np.histogram(bimodal_data, bins=num_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Find peaks (add zeros at edges for edge detection)
        padded_hist = np.concatenate([[0], hist_counts, [0]])
        peaks, _ = find_peaks(padded_hist, distance=1)
        peaks = np.sort(peaks) - 1  # Adjust for padding

        # Keep only top 2 peaks
        if len(peaks) > 2:
            peak_heights = hist_counts[peaks]
            top_2_idx = np.argsort(peak_heights)[-2:]
            peaks = np.sort(peaks[top_2_idx])

        num_peaks = len(peaks)
        num_bins += 1

        if num_bins >= max_hist_bins and set_thresh is None:
            print("Unable to find trough")
            cross = {"upints": np.array([]), "downints": np.array([])}
            bihist = {"hist": hist_counts, "bins": bin_centers}
            return np.nan, cross, bihist, diptest_result

    bihist = {"hist": hist_counts, "bins": bin_centers}

    # Find trough between peaks
    between_peaks = bin_centers[peaks[0] : peaks[1] + 1]
    between_hist = hist_counts[peaks[0] : peaks[1] + 1]

    # Find minimum (trough)
    trough_idx = np.argmin(between_hist)

    if set_thresh is not None:
        thresh = set_thresh
    else:
        thresh = between_peaks[trough_idx]

    # Schmidt trigger: use halfway points between trough and peaks
    if schmidt:
        thresh_up = thresh + 0.5 * (between_peaks[-1] - thresh)
        thresh_down = thresh + 0.5 * (between_peaks[0] - thresh)

        over_up = bimodal_data > thresh_up
        over_down = bimodal_data > thresh_down

        cross_up = np.where(np.diff(over_up.astype(int)) == 1)[0]
        cross_down = np.where(np.diff(over_down.astype(int)) == -1)[0]

        # Delete incomplete (repeat) crossings
        all_crossings = np.vstack(
            [
                np.column_stack([cross_up, np.ones(len(cross_up))]),
                np.column_stack([cross_down, np.zeros(len(cross_down))]),
            ]
        )

        sort_order = np.argsort(all_crossings[:, 0])
        all_crossings = all_crossings[sort_order]

        up_down_switch = np.diff(all_crossings[:, 1])
        same_state = np.where(up_down_switch == 0)[0] + 1
        all_crossings = np.delete(all_crossings, same_state, axis=0)

        cross_up = all_crossings[all_crossings[:, 1] == 1, 0].astype(int)
        cross_down = all_crossings[all_crossings[:, 1] == 0, 0].astype(int)
    else:
        over_ind = bimodal_data > thresh
        cross_up = np.where(np.diff(over_ind.astype(int)) == 1)[0]
        cross_down = np.where(np.diff(over_ind.astype(int)) == -1)[0]

    # If only one crossing, return empty
    if len(cross_up) == 0 or len(cross_down) == 0:
        cross = {"upints": np.array([]), "downints": np.array([])}
        return thresh, cross, bihist, diptest_result

    # Create interval arrays
    up_for_up = cross_up.copy()
    up_for_down = cross_up.copy()
    down_for_up = cross_down.copy()
    down_for_down = cross_down.copy()

    # Adjust for proper pairing
    if cross_up[0] < cross_down[0]:
        up_for_down = up_for_down[1:]
    if cross_down[-1] > cross_up[-1]:
        down_for_down = down_for_down[:-1]
    if cross_down[0] < cross_up[0]:
        down_for_up = down_for_up[1:]
    if cross_up[-1] > cross_down[-1]:
        up_for_up = up_for_up[:-1]

    # Ensure equal length for pairing
    min_len_up = min(len(up_for_up), len(down_for_up))
    min_len_down = min(len(down_for_down), len(up_for_down))

    upints = np.column_stack([up_for_up[:min_len_up], down_for_up[:min_len_up]])
    downints = np.column_stack(
        [down_for_down[:min_len_down], up_for_down[:min_len_down]]
    )

    cross = {"upints": upints, "downints": downints}

    return thresh, cross, bihist, diptest_result
