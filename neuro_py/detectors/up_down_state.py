import os
import sys
from typing import Optional, Tuple

import nelpy as nel
import numpy as np

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
            # find down states, based on percentile
            down_state_epochs_ = bst[ep].bin_centers[
                find_interval(
                    bst[ep].data.flatten() < np.percentile(bst[ep].data.T, percentile)
                )
            ]
            if down_state_epochs_.shape[0] == 0:
                continue

            # remove short and long epochs
            durations = down_state_epochs_[:, 1] - down_state_epochs_[:, 0]
            down_state_epochs_ = down_state_epochs_[
                ~((durations < min_dur) | (durations > max_dur)), :
            ]

            if down_state_epochs_.shape[0] == 0:
                continue

            # convert to epoch array with same domain as nrem epochs (this is so complement will also be in nrem epochs)
            down_state_epochs_ = nel.EpochArray(
                data=down_state_epochs_, domain=nrem_epochs & ep
            )

            # store down states
            down_state_epochs.append(down_state_epochs_.data)

            # complement to get up states
            up_state_epochs_ = ~down_state_epochs_

            # store up states
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
        # find down states, based on percentile
        down_state_epochs = bst.bin_centers[
            find_interval(bst.data.flatten() < np.percentile(bst.data.T, percentile))
        ]
        if down_state_epochs.shape[0] == 0:
            print(f"No down states found for {basepath}")
            return None, None

        # remove short and long epochs
        durations = down_state_epochs[:, 1] - down_state_epochs[:, 0]
        down_state_epochs = down_state_epochs[
            ~((durations < min_dur) | (durations > max_dur)), :
        ]
        if down_state_epochs.shape[0] == 0:
            print(f"No down states found for {basepath}")
            return None, None

        # convert to epoch array with same domain as nrem epochs (this is so complement will also be in nrem epochs)
        down_state_epochs = nel.EpochArray(data=down_state_epochs, domain=nrem_epochs)

        # complement to get up states
        up_state_epochs = ~down_state_epochs

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


if __name__ == "__main__":
    basepath = sys.argv[1]

    detect_up_down_states(basepath)
