import sys
from typing import Tuple

import nelpy as nel
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def linearize_position(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use PCA (a dimensionality reduction technique) to find the direction of maximal variance
    in our position data, and use this as the new 1D linear track axis.

    Parameters
    ----------
    x : numpy.ndarray
        x-coordinates of shape (n, 1)
    y : numpy.ndarray
        y-coordinates of shape (n, 1)

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Linearized x and y coordinates, both of shape (n, 1).
    """
    # locate and remove nans (sklearn pca does not like nans)
    badidx = (np.isnan(x)) | (np.isnan(y))
    badidx_pos = np.where(badidx)
    goodidx_pos = np.where(~badidx)
    n = len(x)

    x = x[~badidx]
    y = y[~badidx]

    # perform pca and return the first 2 components
    pca = PCA(n_components=2)
    # transform our coords
    linear = pca.fit_transform(np.array([x, y]).T)

    # add back nans
    x = np.zeros([n])
    x[badidx_pos] = np.nan
    x[goodidx_pos] = linear[:, 0]

    y = np.zeros([n])
    y[badidx_pos] = np.nan
    y[goodidx_pos] = linear[:, 1]

    # pca will center data at 0,0... adjust for this here
    x = x + np.abs(np.nanmin(x))
    y = y + np.abs(np.nanmin(y))

    return x, y


def find_laps(
    Vts: np.ndarray,
    Vdata: np.ndarray,
    newLapThreshold: float = 15,
    good_laps: bool = True,
    edgethresh: float = 0.1,
    completeprop: float = 0.2,
    posbins: int = 50,
) -> pd.DataFrame:
    """
    Find laps in a linear track.

    Parameters
    ----------
    Vts : numpy.ndarray
        Timestamps.
    Vdata : numpy.ndarray
        X coordinates representing position.
    newLapThreshold : float, optional
        Endpoint proximity threshold in percent of track length (default is 15%).
    good_laps : bool, optional
        If True, run find_good_laps to remove laps with excess NaNs and parts where the rat
        turns around in the middle of the track (default is True).
    edgethresh : float, optional
        Threshold for detecting turn-around points as a proportion of track length (default is 0.1).
    completeprop : float, optional
        Proportion of lap allowed to be missing (NaNs) and still be considered complete (default is 0.2).
    posbins : int, optional
        Number of bins to divide the track into for position coverage (default is 50).

    Returns
    -------
    pd.DataFrame
        DataFrame containing lap information with fields such as start timestamp, position,
        start index, and direction.
    """

    TL = np.abs(np.nanmax(Vdata) - np.nanmin(Vdata))  # % track length
    th1 = (
        np.nanmin(Vdata) + TL * newLapThreshold / 100
    )  # % lower threshold for lower end
    th2 = (
        np.nanmax(Vdata) - TL * newLapThreshold / 100
    )  # % upper threshold for upper end

    # % loop over all frames
    laps = pd.DataFrame()
    laps.loc[0, "start_ts"] = Vts[0]
    laps.loc[0, "pos"] = Vdata[0]
    laps.loc[0, "start_idx"] = 1
    laps.loc[0, "direction"] = 0
    iLap = 0

    newUpThCross = 1  # % flag for new lap top search
    newDownThCross = 1  # % flag for new lap top search
    for i in range(len(Vdata)):
        if Vdata[i] < th1:  # % search for min
            if newUpThCross == 1:  # % start a new lap
                newUpThCross = 0
                newDownThCross = 1
                iLap = iLap + 1
                laps.loc[iLap, "start_ts"] = Vts[i]
                laps.loc[iLap, "pos"] = Vdata[i]
                laps.loc[iLap, "start_idx"] = i
                laps.loc[iLap, "direction"] = 1

            if Vdata[i] < laps.iloc[iLap].pos:  # % record new min if any
                laps.loc[iLap, "start_ts"] = Vts[i]
                laps.loc[iLap, "pos"] = Vdata[i]
                laps.loc[iLap, "start_idx"] = i

        if Vdata[i] > th2:  # % search for max
            if newDownThCross:  # % start a new lap
                newUpThCross = 1
                newDownThCross = 0
                iLap = iLap + 1
                laps.loc[iLap, "start_ts"] = Vts[i]
                laps.loc[iLap, "pos"] = Vdata[i]
                laps.loc[iLap, "start_idx"] = i
                laps.loc[iLap, "direction"] = -1

            if Vdata[i] > laps.iloc[iLap].pos:  # % record new min if any
                laps.loc[iLap, "start_ts"] = Vts[i]
                laps.loc[iLap, "pos"] = Vdata[i]
                laps.loc[iLap, "start_idx"] = i

    # % fix direction of first lap which was unknown above
    # % make first lap direction opposite of second lap's direction (laps alternate!)
    laps.loc[0, "direction"] = -laps.iloc[1].direction

    # % make sure laps cross the halfway point
    middle = np.nanmedian(np.arange(np.nanmin(Vdata), np.nanmax(Vdata)))
    i = 0
    while True:
        try:
            positions = np.arange(laps.iloc[i].pos, laps.iloc[i + 1].pos)
        except Exception:
            positions = [np.nan, np.nan]
        if (np.any(positions > middle) is True) & (np.any(positions < middle) is False):
            laps = laps.drop(laps.index[i + 1])
        i = i + 1
        if i + 1 >= len(laps.pos):
            if len(laps.pos) < iLap:
                laps.iloc[0].direction = -laps.iloc[1].direction
            break

    if good_laps:
        laps = find_good_laps(
            Vts,
            Vdata,
            laps,
            edgethresh=edgethresh,
            completeprop=completeprop,
            posbins=posbins,
        )

    return laps


def peakdetz(
    v: np.ndarray, 
    delta: float, 
    lookformax: int = 1, 
    backwards: int = 0
) -> Tuple[list[Tuple[int, float]], list[Tuple[int, float]]]:
    """
    Detect peaks in a vector.

    Parameters
    ----------
    v : numpy.ndarray
        Input vector in which peaks are detected.
    delta : float
        Threshold value for detecting peaks.
    lookformax : int, optional
        If 1, will look for peaks first. If 0, will look for troughs (default is 1).
    backwards : int, optional
        If 1, search is conducted backwards in the vector (default is 0).

    Returns
    -------
    tuple[list[tuple[int, float]], list[tuple[int, float]]]
        A tuple containing the maxima and minima found in the input vector. Each list contains tuples of 
        the form (index, value).
    """

    maxtab = []
    mintab = []

    v = np.asarray(v)

    if not np.isscalar(delta):
        sys.exit("Input argument delta must be a scalar")

    if delta <= 0:
        sys.exit("Input argument delta must be positive")

    if backwards == 0:
        inc = 1
        first = 0
        last = len(v)
        iter_ = np.arange(first, last, inc)
    elif backwards:
        inc = -1
        first = len(v)
        last = 0
        iter_ = np.arange(first, last, inc)

    mn = np.inf
    mx = -np.inf
    mnpos = np.nan
    mxpos = np.nan

    for ii in iter_:
        this = v[ii]
        if this > mx:
            mx = this
            mxpos = ii
        if this < mn:
            mn = this
            mnpos = ii

        if lookformax:
            try:
                idx = mx - delta > mintab[-1]
            except Exception:
                idx = mx - delta > mintab

            if (this < mx - delta) | ((ii == last - 1) & (len(mintab) > 0) & idx):
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = ii
                lookformax = 0
        else:
            try:
                idx = mx - delta < maxtab[-1]
            except Exception:
                idx = mx - delta < maxtab
            if (this > mn + delta) | ((ii == last - 1) & (len(maxtab) > 0) & idx):
                mintab.append((mnpos, mn))
                mx = this
                mxpos = ii
                lookformax = 1

    if (len(maxtab) == 0) & (len(mintab) == 0):
        if lookformax:
            if mx - mn > delta:
                maxtab = [mxpos, mx]
        else:
            if mx - mn > delta:
                mintab = [mnpos, mn]
    return maxtab, mintab


def find_good_laps(
    ts: np.ndarray, 
    V_rest: np.ndarray, 
    laps: pd.DataFrame, 
    edgethresh: float = 0.1, 
    completeprop: float = 0.2, 
    posbins: int = 50
) -> pd.DataFrame:
    """
    Find and eliminate laps that have too many NaNs or laps where the rat turns around in the middle.

    Parameters
    ----------
    ts : numpy.ndarray
        Timestamps.
    V_rest : numpy.ndarray
        X coordinates of the rat with off-track periods masked out as NaNs.
    laps : pd.DataFrame
        DataFrame containing lap information.
    edgethresh : float, optional
        Threshold for detection of a turn-around point (default is 0.1).
    completeprop : float, optional
        The proportion of a lap that can be missing (NaNs) to still be considered valid (default is 0.2).
    posbins : int, optional
        Number of bins to divide the track into to determine position coverage percentage (default is 50).

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with bad laps removed.
    """

    if (
        edgethresh > 1
    ):  # % in case edgethresh is input as a percentage instead of a proportion
        edgethresh = edgethresh / 100

    if (
        completeprop > 1
    ):  # % in case completeprop is input as a percentage instead of a proportion
        completeprop = completeprop / 100

    bottomend = np.nanmin(V_rest)
    topend = np.nanmax(V_rest)
    bins = np.arange(
        bottomend,
        topend + (topend - bottomend) / posbins,
        (topend - bottomend) / posbins,
    )
    # % threshold for peak/trough detection
    delta = (topend - bottomend) * edgethresh
    startgoodlaps = []
    stopgoodlaps = []

    lap = 0
    lastlapend = np.nan
    while lap < len(laps) - 1:
        # % select out just this lap
        if lap == len(laps):
            endoflap = ts[-1]
        else:
            endoflap = laps.iloc[lap + 1].start_ts

        v = V_rest[
            np.where(ts == laps.iloc[lap].start_ts)[0][0] : np.where(ts == endoflap)[0][
                0
            ]
        ]
        t = ts[
            np.where(ts == laps.iloc[lap].start_ts)[0][0] : np.where(ts == endoflap)[0][
                0
            ]
        ]

        # % find turn around points during this lap
        lookformax = laps.iloc[lap].direction == 1
        peak, trough = peakdetz(v, delta, lookformax, 0)

        if lookformax:
            # % find the direct path from bottomend to topend (or mark lap for
            # % deleting if the turn around points are not in those ranges)
            if len(trough) > 0:
                # % find the last trough in range of bottomend (start of lap)
                gt = len(trough)
                while (gt > 0) & (trough(gt, 2) >= 2 * delta + bottomend):
                    gt = gt - 1

                # % assign the next peak after that trough as the end of the lap
                # % (or mark lap for deleting, if that peak is not at topend)
                if gt == 0:
                    if peak[1, 2] > topend - 2 * delta:
                        t = t[0 : peak[0]]
                        v = v[0 : peak[0]]
                    else:
                        # % this marks the lap for deleting
                        t = t[0:5]
                        v = v[0:5]
                else:
                    et = len(peak)
                    if gt + 1 > et:
                        gt = 0
                        t = t[0:2]
                        v = v[0:2]
                    else:
                        t = t[trough[gt, 1] : peak[gt + 1, 1]]
                        v = v[trough[gt, 1] : peak[gt + 1, 1]]

            else:
                # % make sure peak exists and is in range of topend
                if len(peak) == 0:
                    if len(t) > 2:
                        t = t[0:2]
                        v = v[0:2]
                elif peak[1] < topend - 2 * delta:
                    # % this marks the lap for deleting
                    if len(t) > 5:
                        t = t[0:5]
                        v = v[0:5]
        else:  # % if lookformax
            # % find the direct path from topend to bottomend (or mark lap for
            # % deleting if the turn around points are not in those ranges)
            if len(peak) > 0:
                # % find the last peak in range of topend (start of lap)
                gt = len(peak)
                while (gt > 0) & (peak[gt, 2] <= topend - 2 * delta):
                    gt = gt - 1
                # % assign the next trough after that peak as the end of the lap
                # % (or mark lap for deleting, if that trough is not at bottomend)
                if gt == 0:
                    if trough(1, 2) < bottomend + 2 * delta:
                        t = t[1 : trough[0]]
                        v = v[1 : trough[0]]
                    else:
                        # % this marks the lap for deleting
                        t = t[0:5]
                        v = v[0:5]
                else:
                    et = len(trough)
                    if gt + 1 > et:
                        t = t[0:2]
                        v = v[0:2]
                        gt = 0
                    else:
                        t = t[peak[gt, 1] : trough[gt + 1, 1]]
                        v = v[peak[gt, 1] : trough[gt + 1, 1]]
            else:  # % if ~isempty(peak)
                # % make sure trough exists and is in range of bottomend
                if len(trough) == 0:
                    if len(t) > 2:
                        t = t[0:2]
                        v = v[0:2]

                elif trough[1] > bottomend + 2 * delta:
                    # % this marks the lap for deleting
                    if len(t) > 5:
                        t = t[0:5]
                        v = v[0:5]
        vcovered, _ = np.histogram(v, bins=bins)

        if len(v) < 3:
            # % eliminate the lap if it is non-existent (as is sometimes the case for lap 1)
            laps = laps.drop(laps.index[lap])
        # % eliminate the lap if >completeprop of it is NaNs or if it has been marked for
        # % deleting above
        elif (len(v) < 6) | (sum(vcovered == 0) > completeprop * posbins):
            laps.drop(laps.index[lap])
            # % remove the other lap from the lap pair
            if lap % 2 == 0:
                # % delete previous lap from laps
                laps = laps.drop(laps.index[lap - 1])
                # % change goodlaps markers to delete previous lap from laps
                if len(stopgoodlaps) > 0:
                    if np.isnan(lastlapend).all() | (startgoodlaps[-1] > lastlapend):
                        startgoodlaps[-1] = []
                        stopgoodlaps[-1] = []
                    else:
                        stopgoodlaps[-1] = lastlapend

                lap = lap - 1
            elif lap <= len(laps) & lap > 1:
                # % delete next lap from laps
                laps = laps.drop(laps.index[lap])
        else:  # % if lap is good
            # % store last lap end just in case have to delete this lap with next lap
            if len(stopgoodlaps) > 0:
                lastlapend = stopgoodlaps[-1]

            # % add this lap to goodlaps
            try:
                idx = stopgoodlaps[-1] == t[0]
            except Exception:
                idx = stopgoodlaps == t[0]
            if (len(stopgoodlaps) > 0) & (idx):
                stopgoodlaps[-1] = t[-1]
            else:
                startgoodlaps.append(t[0])
                stopgoodlaps.append(t[-1])

            lap = lap + 1

    return laps


def get_linear_track_lap_epochs(
    ts: np.ndarray,
    x: np.ndarray,
    newLapThreshold: float = 15,
    good_laps: bool = False,
    edgethresh: float = 0.1,
    completeprop: float = 0.2,
    posbins: int = 50
) -> Tuple[nel.EpochArray, nel.EpochArray]:
    """
    Identifies lap epochs on a linear track and classifies them into outbound and inbound directions.
    
    Parameters:
    ----------
    ts : np.ndarray
        Array of timestamps corresponding to position data.
    x : np.ndarray
        Array of position data along the linear track.
    newLapThreshold : float, optional
        Minimum distance between laps to define a new lap, by default 15.
    good_laps : bool, optional
        If True, filter out laps that do not meet certain quality criteria, by default False.
    edgethresh : float, optional
        Threshold proportion of the track edge to identify potential boundary errors, by default 0.1.
    completeprop : float, optional
        Minimum proportion of the track that must be traversed for a lap to be considered complete, by default 0.2.
    posbins : int, optional
        Number of bins to divide the track into for analysis, by default 50.

    Returns:
    -------
    Tuple[nel.EpochArray, nel.EpochArray]
        A tuple containing two nelpy EpochArray objects:
        - outbound_epochs: Epochs representing outbound runs (towards the far end of the track).
        - inbound_epochs: Epochs representing inbound runs (back towards the start).

    Notes:
    ------
    - This function calls `find_laps` to determine the lap structure, then segregates epochs into outbound and inbound directions.
    - The EpochArray objects represent the start and stop timestamps for each identified lap.
    """
    laps = find_laps(
        np.array(ts),
        np.array(x),
        newLapThreshold=newLapThreshold,
        good_laps=good_laps,
        edgethresh=edgethresh,
        completeprop=completeprop,
        posbins=posbins,
    )

    outbound_start = []
    outbound_stop = []
    inbound_start = []
    inbound_stop = []

    for i in range(len(laps) - 1):
        if laps.iloc[i].direction == 1:
            outbound_start.append(laps.iloc[i].start_ts)
            outbound_stop.append(laps.iloc[i + 1].start_ts)

        if laps.iloc[i].direction == -1:
            inbound_start.append(laps.iloc[i].start_ts)
            inbound_stop.append(laps.iloc[i + 1].start_ts)

    outbound_epochs = nel.EpochArray([np.array([outbound_start, outbound_stop]).T])
    inbound_epochs = nel.EpochArray([np.array([inbound_start, inbound_stop]).T])

    return outbound_epochs, inbound_epochs


def find_good_lap_epochs(
    pos: nel.AnalogSignalArray, 
    dir_epoch: nel.EpochArray, 
    thres: float = 0.5, 
    binsize: int = 6, 
    min_laps: int = 10
) -> nel.EpochArray:
    """
    Find good laps in behavior data for replay analysis.
    
    Parameters
    ----------
    pos : nelpy.AnalogSignalArray
        A nelpy AnalogSignalArray containing the position data with a single dimension.
    dir_epoch : nelpy.EpochArray
        EpochArray defining the laps to analyze for good laps.
    thres : float, optional
        Occupancy threshold to determine good laps, by default 0.5.
    binsize : int, optional
        Size of the bins for calculating occupancy, by default 6.
    min_laps : int, optional
        Minimum number of laps required to consider laps as 'good', by default 10.
    
    Returns
    -------
    nelpy.EpochArray
        An EpochArray containing the good laps based on the occupancy threshold.
        Returns an empty EpochArray if no good laps are found or if the number 
        of laps is less than `min_laps`.
    
    Notes
    -----
    The function calculates the percent occupancy over position bins per lap, 
    and identifies laps that meet the occupancy threshold criteria. The laps 
    that meet this condition are returned as an EpochArray.
    """
    # make bin edges to calc occupancy
    x_edges = np.arange(np.nanmin(pos.data[0]), np.nanmax(pos.data[0]), binsize)
    # initialize occupancy matrix (position x time)
    occ = np.zeros([len(x_edges) - 1, dir_epoch.n_intervals])

    # much faster to not use nelpy objects here, so pull out needed data
    x_coord = pos.data[0]
    time = pos.abscissa_vals
    epochs = dir_epoch.data

    # iterate through laps
    for i, ep in enumerate(epochs):
        # bin position per lap
        occ[:, i], _ = np.histogram(
            x_coord[(time >= ep[0]) & (time <= ep[1])], bins=x_edges
        )

    # calc percent occupancy over position bins per lap and find good laps
    good_laps = np.where(~((np.sum(occ == 0, axis=0) / occ.shape[0]) > thres))[0]
    # if no good laps, return empty epoch
    if (len(good_laps) == 0) | (len(good_laps) < min_laps):
        dir_epoch = nel.EpochArray()
    else:
        dir_epoch = dir_epoch[good_laps]
    return dir_epoch
