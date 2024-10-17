from typing import List, Union

import numpy as np
import pandas as pd


def get_spindices(data: np.ndarray) -> pd.DataFrame:
    """
    Get spike timestamps and spike IDs from each spike train in a sorted DataFrame.

    Parameters
    ----------
    data : np.ndarray
        Spike times for each spike train, where each element is an array of spike times.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing sorted spike times and the corresponding spike IDs.

    Examples
    -------
    >>> spike_trains = [np.array([0.1, 0.2, 0.4]), np.array([0.15, 0.35])]
    >>> spikes = get_spindices(spike_trains)
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
    spikes: pd.DataFrame, spike_id: Union[List[int], np.ndarray, None] = None
) -> List[np.ndarray]:
    """
    Convert spike times and spike IDs from a DataFrame into a list of arrays,
    where each array contains the spike times for a given spike train.

    Parameters
    ----------
    spikes : pd.DataFrame
        DataFrame containing 'spike_times' and 'spike_id' columns, sorted by 'spike_times'.
    spike_id : list or np.ndarray, optional
        List or array of spike IDs to search for in the DataFrame. If None, all spike IDs are used.

    Returns
    -------
    List[np.ndarray]
        A list of arrays, each containing the spike times for a corresponding spike train.

    Examples
    -------
    >>> spike_trains = spindices_to_ndarray(spikes_df, spike_id=[0, 1, 2])
    """
    if spike_id is None:
        spike_id = np.unique(spikes["spike_id"])
    data = []
    for spk_i in spike_id:
        data.append(spikes[spikes["spike_id"] == spk_i]["spike_times"].values)
    return data


def BurstIndex_Royer_2012(autocorrs: pd.DataFrame) -> list:
    """
    Calculate the burst index from Royer et al. (2012).
    The burst index ranges from -1 to 1, where:
    -1 indicates non-bursty behavior, and 1 indicates bursty behavior.

    Parameters
    ----------
    autocorrs : pd.DataFrame
        Autocorrelograms of spike trains, with time (in seconds) as index and correlation values as columns.

    Returns
    -------
    list
        List of burst indices for each autocorrelogram column.

    Notes
    -----
    The burst index is calculated as:
        burst_idx = (peak - baseline) / max(peak, baseline)

    - Peak is calculated as the maximum value of the autocorrelogram between 2-9 ms.
    - Baseline is calculated as the mean value of the autocorrelogram between 40-50 ms.

    Examples
    -------
    >>> burst_idx = BurstIndex_Royer_2012(autocorr_df)
    """
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


def select_burst_spikes(
    spikes: np.ndarray,
    mode: str = "bursts",
    isiBursts: float = 0.006,
    isiSpikes: float = 0.020,
) -> np.ndarray:
    """
    Discriminate bursts versus single spikes based on inter-spike intervals.

    Parameters
    ----------
    spikes : np.ndarray
        Array of spike times.
    mode : str, optional
        Either 'bursts' (default) or 'single'.
    isiBursts : float, optional
        Maximum inter-spike interval for bursts (default = 0.006 seconds).
    isiSpikes : float, optional
        Minimum inter-spike interval for single spikes (default = 0.020 seconds).

    Returns
    -------
    np.ndarray
        A boolean array indicating for each spike whether it matches the criterion.

    Notes
    -----
    Adapted from: http://fmatoolbox.sourceforge.net/Contents/FMAToolbox/Analyses/SelectSpikes.html
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
