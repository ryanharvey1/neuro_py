import os
import sys

import nelpy as nel
import numpy as np
from typing import Optional, Tuple

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

    Returns
    -------
    Tuple[Optional[nel.EpochArray], Optional[nel.EpochArray]]
        A tuple containing the detected DOWN state epochs and UP state epochs.
        Returns (None, None) if no suitable states are found or insufficient data is available.

    Notes
    -----
    Detection method based on https://doi.org/10.1038/s41467-020-15842-4
    """

    # check for existance of event files
    if save_mat:
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

        # load spikes
        st, _ = loading.load_spikes(basepath, brainRegion=region[1])
        # check if there are enough cells
        if st is None or st.isempty or st.data.shape[0] < min_cells:
            print(f"No spikes found for {basepath} {region}")
            return None, None

    # flatten spikes
    st = st[nrem_epochs].flatten()

    # bin and smooth
    bst = st.bin(ds=bin_size).smooth(sigma=smooth_sigma)

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
    # convert to epoch array with same domain as nrem epochs (this is so compliment will also be in nrem epochs)
    down_state_epochs = nel.EpochArray(data=down_state_epochs, domain=nrem_epochs)

    # compliment to get up states
    up_state_epochs = ~down_state_epochs

    # save to cell explorer mat file
    if save_mat:
        epoch_to_mat(down_state_epochs, basepath, "down_state", "detect_up_down_states")
        epoch_to_mat(up_state_epochs, basepath, "up_state", "detect_up_down_states")

    return down_state_epochs, up_state_epochs


if __name__ == "__main__":
    basepath = sys.argv[1]

    detect_up_down_states(basepath)
