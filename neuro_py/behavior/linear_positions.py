from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import sys
import nelpy as nel

def linearize_position(x, y):
    """
    use PCA (a dimensionality reduction technique) to find
    the direction of maximal variance in our position data,
    and we use this as our new 1D linear track axis.

    Input:
        x: numpy array of shape (n,1)
        y: numpy array of shape (n,1)
    Output:
        x_lin: numpy array of shape (n,1)
        y_lin: numpy array of shape (n,1)

    -Ryan H
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
    Vts,
    Vdata,
    newLapThreshold=15,
    good_laps=True,
    edgethresh=0.1,
    completeprop=0.2,
    posbins=50,
):
    """
    Find Laps in linear track

    INPUT:
    Vts: timestamps
    Vdata: x coords

    newLapThreshold: endpoint proximity threshold in percent of track length (default = 15%);
                    whenever rat enters the proximity zone of e.g. 15% of tracklength near a end, a new lap
                    is started and the maximum (or minimum) is searched
                    for a Lap-Top  or Lap-Bottom (around 0 end).

    good_laps: run find_good_laps to remove laps with excess nans and
                parts of laps where rat turns around in middle of track

    OUTPUT:
    laps  .... 1*nLaps struct array with fields
    laps(i).start_ts  ... start timestamp of i-th lap
    laps(i).pos       ... the value of input position V at lap start point
    laps(i).start_idx ... the index of the new lap start frame in input V
    laps(i).direction ... +1/-1 for up/down laps

    From NSMA toolbox
    Author: PL
    Version: 0.9  05/12/2005
    edited by Ryan Harvey to work with standard linear track
    edited for use in python by Ryan h 2022
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
    laps.iloc[0].direction = -laps.iloc[1].direction

    # % make sure laps cross the halfway point
    middle = np.nanmedian(np.arange(np.nanmin(Vdata), np.nanmax(Vdata)))
    i = 0
    while True:
        try:
            positions = np.arange(laps.iloc[i].pos, laps.iloc[i + 1].pos)
        except:
            positions = [np.nan, np.nan]
        if (np.any(positions > middle) == True) & (np.any(positions < middle) == False):
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


def peakdetz(v, delta, lookformax=1, backwards=0):
    """
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDETZ(V, DELTA, lookformax, backwards) finds
    %        the local maxima and minima ("peaks") in the vector V.
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA. MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    %
    % ZN edit 04/2010: added option to specify looking for troughs or peaks
    % first (lookformax variable: if 1, will look for peaks first, if 0 will
    % look for troughs; default is look for peaks); and option to go backwards
    % (so that find last instance of a peak/trough value instead of the first
    % instance: backwards variable: if 1 will go backwards, if 0 or absent,
    % will go forwards); and changed it so that last min/max value will be
    % assigned

    edited for use in python by Ryan H 2022
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
            except:
                idx = mx - delta > mintab

            if (this < mx - delta) | ((ii == last - 1) & (len(mintab) > 0) & idx):
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = ii
                lookformax = 0
        else:
            try:
                idx = mx - delta < maxtab[-1]
            except:
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


def find_good_laps(ts, V_rest, laps, edgethresh=0.1, completeprop=0.2, posbins=50):
    """
    % [startgoodlaps, stopgoodlaps, laps] =
    %        find_good_laps(V_rest,laps,edgethresh,completeprop,posbins)
    %
    % find and eliminate laps which have too many NaNs (because rat was off
    % track), and parts of laps where rat turns around in middle of track
    %
    % inputs: V_rest: V coordinates of rat with off track periods masked out
    %                 (as NaNs)
    %         laps: struct with lap start and end times (generated by
    %               find_laps)
    %         edgethresh: threshold for detection of a turn around point
    %                     (proportion of length of track) (default = 0.1)
    %         completeprop: the amount of lap that can be missing (NaNs) to
    %                       still be considered a lap (default = 0.2).
    %         plotlaps: flag for making plots of each lap, and pause for user
    %                   to hit key to continue (default = 1)
    %         posbins: number of bins to divide the track into to determine
    %                  position coverage percentage; at 60frames/s want at
    %                  least 2cm/bin (default = 50bins; this works for 100+ cm
    %                  track, as long as V_rest is in cm)
    % outputs:
    %          laps: a new laps struct, with the bad laps removed
    %
    % ZN 04/2011
    Edited for use in python by Ryan H 2022
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

    l = 0
    while l < len(laps) - 1:
        # % select out just this lap
        if l == len(laps):
            endoflap = ts[-1]
        else:
            endoflap = laps.iloc[l + 1].start_ts

        v = V_rest[
            np.where(ts == laps.iloc[l].start_ts)[0][0] : np.where(ts == endoflap)[0][0]
        ]
        t = ts[
            np.where(ts == laps.iloc[l].start_ts)[0][0] : np.where(ts == endoflap)[0][0]
        ]

        # % find turn around points during this lap
        lookformax = laps.iloc[l].direction == 1
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
            laps = laps.drop(laps.index[l])
        # % eliminate the lap if >completeprop of it is NaNs or if it has been marked for
        # % deleting above
        elif (len(v) < 6) | (sum(vcovered == 0) > completeprop * posbins):
            laps.drop(laps.index[l])
            # % remove the other lap from the lap pair
            if l % 2 == 0:
                # % delete previous lap from laps
                laps = laps.drop(laps.index[l - 1])
                # % change goodlaps markers to delete previous lap from laps
                if len(stopgoodlaps) > 0:
                    if "lastlapend" not in locals() | (startgoodlaps[-1] > lastlapend):
                        startgoodlaps[-1] = []
                        stopgoodlaps[-1] = []
                    else:
                        stopgoodlaps[-1] = lastlapend

                l = l - 1
            elif l <= len(laps) & l > 1:
                # % delete next lap from laps
                laps = laps.drop(laps.index[l])
        else:  # % if lap is good
            # % store last lap end just in case have to delete this lap with next lap
            if len(stopgoodlaps) > 0:
                lastlapend = stopgoodlaps[-1]

            # % add this lap to goodlaps
            try:
                idx = stopgoodlaps[-1] == t[0]
            except:
                idx = stopgoodlaps == t[0]
            if (len(stopgoodlaps) > 0) & (idx):
                stopgoodlaps[-1] = t[-1]
            else:
                startgoodlaps.append(t[0])
                stopgoodlaps.append(t[-1])

            l = l + 1

    return laps


def get_linear_track_lap_epochs(
    ts,
    x,
    newLapThreshold=15,
    good_laps=False,
    edgethresh=0.1,
    completeprop=0.2,
    posbins=50,
):
    """
    get_linear_track_lap_epochs: def that calls find_laps and outputs nelpy epochs
        for out and inbound running directions
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


def find_good_lap_epochs(pos, dir_epoch, thres=0.5, binsize=6, min_laps=10):
    """
    find_good_laps: finds good laps in behavior data
        Made to find good laps in nelpy array for replay analysis
    input:
        pos: nelpy analog array with single dim
        dir_epoch: EpochArray to find good lap
        thres: occupancy threshold for good lap
        binsize: size of bins to calculate occupancy
    output:
        good_laps: epoch array of good laps
    """
    # make bin edges to calc occupancy
    x_edges = np.arange(np.nanmin(pos.data[0]), np.nanmax(pos.data[0]), binsize)
    # initialize occupancy matrix (position x time)
    occ = np.zeros([len(x_edges) - 1, dir_epoch.n_intervals])
    # iterate through laps
    for i, ep in enumerate(dir_epoch):
        # bin position per lap
        occ[:, i], _ = np.histogram(pos[ep].data[0], bins=x_edges)
    # calc percent occupancy over position bins per lap and find good laps
    good_laps = np.where(~((np.sum(occ == 0, axis=0) / occ.shape[0]) > thres))[0]
    # if no good laps, return empty epoch
    if (len(good_laps) == 0) | (len(good_laps) < min_laps):
        dir_epoch = nel.EpochArray()
    else:
        dir_epoch = dir_epoch[good_laps]
    return dir_epoch
