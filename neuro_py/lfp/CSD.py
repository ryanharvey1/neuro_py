from importlib import import_module
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from neuro_py.io import loading
from neuro_py.util._dependencies import _check_dependency

_check_dependency("neo", "csd")
_check_dependency("elephant", "csd")
_check_dependency("quantities", "csd")


def get_coords(basepath: str, shank: int = 0) -> NDArray[Any]:
    """
    Get the coordinates of the channels from the probe layout.

    Parameters
    ----------
    basepath : str
        Path to the basepath.
    shank : int, optional
        Shank to get the coordinates from, by default 0.

    Returns
    -------
    np.ndarray
        Coordinates of the channels.
    """
    pq = import_module("quantities")

    # load the probe layout
    probe_layout = cast(Any, loading.load_probe_layout(basepath))

    # get the coordinates of the channels
    coords = probe_layout.loc[shank == probe_layout.shank, "y"].values

    # rescale the coordinates so none are negative and in mm
    rescaled_coords = (coords - coords.min()) * pq.mm

    # add dimension to coords to make it (nchannels,1)
    rescaled_coords = rescaled_coords[:, np.newaxis]

    return rescaled_coords


def get_csd(
    basepath: str,
    data: NDArray[Any],
    shank: int,
    fs: float = 1250,
    diam: float = 0.015,
    method: Literal["DeltaiCSD", "StandardCSD", "KD1CSD"] = "DeltaiCSD",
    channel_offset: float = 0.046,
) -> Any:
    """
    compute the CSD for a given basepath and data using elephant estimate_csd.

    Klas H. Pettersen, Anna Devor, Istvan Ulbert, Anders M. Dale, Gaute T. Einevoll,
    Current-source density estimation based on inversion of electrostatic forward
    solution: Effects of finite extent of neuronal activity and conductivity
    discontinuities, Journal of Neuroscience Methods, Volume 154, Issues 1-2,
    30 June 2006, Pages 116-133, ISSN 0165-0270,
    http://dx.doi.org/10.1016/j.jneumeth.2005.12.005.

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

    """
    pq = import_module("quantities")
    estimate_csd = import_module("elephant.current_source_density").estimate_csd
    AnalogSignal = import_module("neo").AnalogSignal

    coords = get_coords(basepath, shank=shank)

    signal = AnalogSignal(
        data,
        units="mV",
        t_start=0 * pq.s,
        sampling_rate=fs * pq.Hz,
        dtype=float,
    )

    if method == "DeltaiCSD":
        csd = estimate_csd(signal, coordinates=coords, diam=diam * pq.mm, method=method)

    elif method == "StandardCSD":
        # create coordinates for the CSD
        coords = np.zeros(data.shape[1])
        for idx, i in enumerate(coords):
            if idx == 0:
                coords[idx] = 0
            else:
                coords[idx] = coords[idx - 1] + channel_offset

        coords = coords * pq.mm

        # add dimension to coords to make it (64,1)
        coords = coords[:, np.newaxis]

        csd = estimate_csd(signal, coordinates=coords, method=method)

    elif method == "KD1CSD":
        # create coordinates for the CSD
        coords = np.zeros(data.shape[1])
        for idx, i in enumerate(coords):
            if idx == 0:
                coords[idx] = 0
            else:
                coords[idx] = coords[idx - 1] + channel_offset

        coords = coords * pq.mm

        # add dimension to coords to make it (64,1)
        coords = coords[:, np.newaxis]
        csd = estimate_csd(signal, coordinates=coords, method=method)

    return csd
