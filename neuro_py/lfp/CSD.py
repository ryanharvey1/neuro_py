# CSD 
import neo
from elephant.current_source_density import estimate_csd
import quantities as pq
from neuro_py.io import loading

import numpy as np


def get_coords(basepath,shank = 0):
    """
    get the coordinates of the channels from the probe layout

    Parameters
    ----------
    basepath : str
        path to the basepath
    shank : int, optional
        shank to get the coordinates from, by default 0

    Returns
    -------
    np.array
        coordinates of the channels

    Dependencies
    ------------
    loading.load_probe_layout, numpy, quantities

    Laura Berkowitz, 2024
    """
    # load the probe layout
    probe_layout = loading.load_probe_layout(basepath)

    # get the coordinates of the channels
    coords = probe_layout.loc[shank == probe_layout.shank,"y"].values * pq.mm  
    coords = coords[:, np.newaxis]

    # compute slope of coords, if negative, flip
    if np.diff(coords[:, 0])[0] < 0:
        coords = coords[::-1]

    return coords

def get_csd(basepath, data, fs = 1250, diam = 0.015, method = 'DeltaiCSD'):
    """
    compute the CSD for a given basepath and data

    Parameters
    ----------
    basepath : str
        path to the basepath
    data : np.array         
        data to compute the CSD on [channels x time]
    fs : int, optional
        sampling rate of the data, by default 1250 Hz
    diam : float, optional  
        diameter of the electrode, by default 0.015 mm
    method : str, optional
        method to compute the CSD, by default 'DeltaiCSD' 

    Returns
    -------
    neo.AnalogSignal
        CSD signal
        
    Dependencies
    ------------
    get_coords, estimate_csd (Elephant), neo, quantities

    Laura Berkowitz, 2024

    """
    coords = get_coords(basepath)

    signal = neo.AnalogSignal(
        data,
        units="mV",
        t_start=0 * pq.s,
        sampling_rate=fs * pq.Hz,
        dtype=float,
    )
    csd = estimate_csd(signal, coordinates=coords, diam=diam * pq.mm, method=method)

    return csd