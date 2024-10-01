
from typing import Union

import numpy as np
import pandas as pd


def get_spindices(data: np.ndarray) -> pd.DataFrame:
    """
    Get spike timestamps and spike id for each spike train in a
        sorted dataframe of spike trains
    Parameters
    ----------
    data : np.ndarray
        spike times for each spike train, in a list of arrays
    Returns
    -------
    spikes : pd.DataFrame
        sorted dataframe of spike times and spike id

    """
    spikes_id = []
    for spk_i, spk in enumerate(data):
        spikes_id.append(spk_i * np.ones_like(spk))

    spikes = pd.DataFrame()
    spikes["spike_times"] = np.hstack(data)
    spikes["spike_id"] = np.hstack(spikes_id)
    spikes.sort_values("spike_times", inplace=True)
    return spikes


def spindices_to_ndarray(
    spikes: pd.DataFrame, spike_id: Union[list, np.ndarray, None] = None
) -> np.ndarray:
    """
    Convert spike times and spike id to a list of arrays
    Parameters
    ----------
    spikes : pd.DataFrame
        sorted dataframe of spike times and spike id
    spike_id: list or np.ndarray
        spike ids search for in the dataframe (important if spikes were restricted)
    Returns
    -------
    data : np.ndarray
        spike times for each spike train, in a list of arrays
    """
    if spike_id is None:
        spike_id = np.unique(spikes["spike_id"])
    data = []
    for spk_i in spike_id:
        data.append(spikes[spikes["spike_id"] == spk_i]["spike_times"].values)
    return data


def BurstIndex_Royer_2012(autocorrs):
    # calc burst index from royer 2012
    # burst_idx will range from -1 to 1
    # -1 being non-bursty and 1 being bursty

    # peak range 2 - 9 ms
    peak = autocorrs.loc[0.002:0.009].max()
    # baseline idx 40 - 50 ms
    baseline = autocorrs.loc[0.04:0.05].mean()

    burst_idx = []
    for p, b in zip(peak, baseline):

        if (p is None) | (b is None):
            burst_idx.append(np.nan)
            continue
        if p > b:
            burst_idx.append((p - b) / p)
        elif p < b:
            burst_idx.append((p - b) / b)
        else:
            burst_idx.append(np.nan)
    return burst_idx


def select_burst_spikes(spikes, mode="bursts", isiBursts=0.006, isiSpikes=0.020):
    """
    select_burst_spikes - Discriminate bursts vs single spikes.
    adpated from: http://fmatoolbox.sourceforge.net/Contents/FMAToolbox/Analyses/SelectSpikes.html

    Input:
        spikes: list of spike times
        mode: either 'bursts' (default) or 'single'
        isiBursts: max inter-spike interval for bursts (default = 0.006)
        isiSpikes: min for single spikes (default = 0.020)
    Output:
        selected: a logical vector indicating for each spike whether it
                    matches the criterion
    """

    dt = np.diff(spikes)

    if mode == "bursts":
        b = dt < isiBursts
        # either next or previous isi < threshold
        selected = np.insert(b, 0, False, axis=0) | np.append(b, False)
    else:
        s = dt > isiSpikes
        # either next or previous isi > threshold
        selected = np.insert(s, 0, False, axis=0) & np.append(s, False)

    return selected
