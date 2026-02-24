from typing import List, Tuple, Union

import nelpy as nel
import numpy as np
import pandas as pd

from neuro_py.process.intervals import truncate_epoch


def circular_shift(m: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Circularly shift matrix rows or columns by specified amounts.

    Each matrix row (or column) is circularly shifted by a different amount.

    Parameters
    ----------
    m : np.ndarray
        Matrix to rotate. Should be a 2D array.
    s : np.ndarray
        Shift amounts for each row (horizontal vector) or column (vertical vector).
        Should be a 1D array.

    Returns
    -------
    shifted : np.ndarray
        Matrix `m` with rows (or columns) circularly shifted by the amounts in `s`.

    Raises
    ------
    ValueError
        If `s` is not a vector of integers or if `m` is not a 2D matrix.
        If the sizes of `m` and `s` are incompatible.

    Notes
    -----
    This function is adapted from CircularShift.m, Copyright (C) 2012 by Michaël Zugaro.
    """
    # Check number of parameters
    if len(s.shape) != 1:
        raise ValueError("Second parameter is not a vector of integers.")
    if len(m.shape) != 2:
        raise ValueError("First parameter is not a 2D matrix.")

    mm, nm = m.shape
    # if s is 1d array, add dimension
    if len(s.shape) == 1:
        s = s[np.newaxis, :]
    ms, ns = s.shape

    # Check parameter sizes
    if mm != ms and nm != ns:
        raise ValueError("Incompatible parameter sizes.")

    # The algorithm below works along columns; transpose if necessary
    s = -np.ravel(s)
    if ns == 1:
        m = m.T
        mm, nm = m.shape

    # Shift matrix S, where Sij is the vertical shift for element ij
    shift = np.tile(s, (mm, 1))

    # Before we start, each element Mij has a linear index Aij.
    # After circularly shifting the rows, it will have a linear index Bij.
    # We now construct Bij.

    # First, create matrix C where each item Cij = i (row number)
    lines = np.tile(np.arange(mm)[:, np.newaxis], (1, nm))
    # Next, update C so that Cij becomes the target row number (after circular shift)
    lines = np.mod(lines + shift, mm)
    # lines[lines == 0] = mm
    # Finally, transform Cij into a linear index, yielding Bij
    indices = lines + np.tile(np.arange(nm) * mm, (mm, 1))

    # Circular shift (reshape so that it is not transformed into a vector)
    shifted = m.ravel()[(indices.flatten() - 1).astype(int)].reshape(mm, nm)

    # flip matrix right to left
    # shifted = np.fliplr(shifted)

    shifted = np.flipud(shifted)

    # Transpose back if necessary
    if ns == 1:
        shifted = shifted.T

    return shifted


def average_diagonal(mat: np.ndarray) -> np.ndarray:
    """
    Average values over all offset diagonals of a 2D array.

    Parameters
    ----------
    mat : np.ndarray
        2D array from which to compute the average values over diagonals.

    Returns
    -------
    output : np.ndarray
        1D array containing the average values over all offset diagonals.

    Notes
    -----
    The method used for computing averages is based on the concept of
    accumulating values along each diagonal offset and then dividing by
    the number of elements in each diagonal.

    Reference
    ---------
    https://stackoverflow.com/questions/71362928/average-values-over-all-offset-diagonals
    """
    n = mat.shape[0]
    output = np.zeros(n * 2 - 1, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        output[i : i + n] += mat[n - 1 - i]
    output[0:n] /= np.arange(1, n + 1, 1, dtype=np.float64)
    output[n:] /= np.arange(n - 1, 0, -1, dtype=np.float64)
    return output


def remove_inactive_cells(
    st: nel.core._eventarray.SpikeTrainArray,
    cell_metrics: Union[pd.DataFrame, None] = None,
    epochs: Union[
        List[nel.core._intervalarray.EpochArray],
        nel.core._intervalarray.EpochArray,
        None,
    ] = None,
    min_spikes: int = 100,
) -> Tuple[nel.core._eventarray.SpikeTrainArray, Union[pd.DataFrame, None]]:
    """
    remove_inactive_cells: Remove cells with fewer than min_spikes spikes per sub-epoch

    Parameters
    ----------
    st : SpikeTrainArray
        SpikeTrainArray object containing spike times for multiple cells.

    cell_metrics : pd.DataFrame, optional
        DataFrame containing metrics for each cell (e.g., quality metrics).

    epochs : EpochArray or list of EpochArray, optional
        If a list of EpochArray objects is provided, each EpochArray object
        is treated as a sub-epoch. If a single EpochArray object is provided,
        each interval in the EpochArray object is treated as a sub-epoch.

    min_spikes : int, optional
        Minimum number of spikes required per sub-epoch to retain a cell.
        Default is 100.

    Returns
    -------
    Tuple[SpikeTrainArray, Union[pd.DataFrame, None]]
        A tuple containing:
        - SpikeTrainArray object with inactive cells removed.
        - DataFrame containing cell metrics with inactive cells removed (if provided).

    Examples
    -------
    >>> from neuro_py.process.intervals import truncate_epoch
    >>> from neuro_py.session.locate_epochs import (
    >>>     find_multitask_pre_post,
    >>>     compress_repeated_epochs,
    >>> )
    >>> from neuro_py.io import loading
    >>> import nelpy as nel
    >>> from neuro_py.process.utils import remove_inactive_cells

    >>> # load data from session
    >>> basepath = "Z:/Data/hpc_ctx_project/HP04/day_1_20240320"

    >>> # load spikes and cell metrics (cm)
    >>> st, cm = loading.load_spikes(basepath, brainRegion="CA1", putativeCellType="Pyr")

    >>> # load epochs and apply multitask epoch restrictions
    >>> epoch_df = loading.load_epoch(basepath)
    >>> epoch_df = compress_repeated_epochs(epoch_df)
    >>> pre_task_post = find_multitask_pre_post(
    >>>     epoch_df.environment, post_sleep_flank=True, pre_sleep_common=True
    >>> )

    >>> beh_epochs = nel.EpochArray(
    >>>     epoch_df.iloc[pre_task_post[0]][["startTime", "stopTime"]].values
    >>> )
    >>> # load sleep states to restrict to NREM and theta
    >>> state_dict = loading.load_SleepState_states(basepath)
    >>> nrem_epochs = nel.EpochArray(
    >>>     state_dict["NREMstate"],
    >>> )
    >>> theta_epochs = nel.EpochArray(
    >>>     state_dict["THETA"],
    >>> )
    >>> # create list of restricted epochs
    >>> restict_epochs = []
    >>> for epoch, epoch_label in zip(beh_epochs, ["pre", "task", "post"]):
    >>>     if epoch_label in "pre":
    >>>         # get cumulative hours of sleep
    >>>         epoch_restrict = truncate_epoch(epoch & nrem_epochs, time=3600)
    >>>     elif epoch_label in "post":
    >>>         # get cumulative hours of sleep
    >>>         epoch_restrict = truncate_epoch(epoch & nrem_epochs, time=3600)
    >>>     else:
    >>>         # get theta during task
    >>>         epoch_restrict = epoch & theta_epochs
    >>>     restict_epochs.append(epoch_restrict)

    >>> # remove inactive cells
    >>> st, cm = remove_inactive_cells(st, cm, restict_epochs)
    """

    def return_results(st, cell_metrics):
        if cell_metrics is None:
            return st
        else:
            return st, cell_metrics

    # check data types
    if not isinstance(st, nel.core._eventarray.SpikeTrainArray):
        raise ValueError("st must be a SpikeTrainArray object")

    if not isinstance(cell_metrics, (pd.core.frame.DataFrame, type(None))):
        raise ValueError("cell_metrics must be a DataFrame object")

    if not isinstance(epochs, (nel.core._intervalarray.EpochArray, list)):
        raise ValueError(
            "epochs must be an EpochArray object or a list of EpochArray objects"
        )

    if isinstance(epochs, list):
        for epoch in epochs:
            if not isinstance(epoch, nel.core._intervalarray.EpochArray):
                raise ValueError("list of epochs must contain EpochArray objects")

    # check if st is empty
    if st.isempty:
        return return_results(st, cell_metrics)

    # check if epochs is empty
    if isinstance(epochs, nel.core._intervalarray.EpochArray):
        if epochs.isempty:
            return return_results(st, cell_metrics)

    # check if cell_metrics is empty
    if cell_metrics is not None and cell_metrics.empty:
        return return_results(st, cell_metrics)

    # check if min_spikes is less than 1
    if min_spikes < 1:
        return return_results(st, cell_metrics)

    # check if st and cell_metrics have the same number of units
    if cell_metrics is not None and st.n_units != cell_metrics.shape[0]:
        # assert error message
        raise ValueError("st and cell_metrics must have the same number of units")

    spk_thres_met = []
    # check if each cell has at least min_spikes spikes in each epoch
    for epoch_restrict in epochs:
        if st[epoch_restrict].isempty:
            spk_thres_met.append([False] * st.n_units)
            continue
        spk_thres_met.append(st[epoch_restrict].n_events >= min_spikes)

    good_idx = np.vstack(spk_thres_met).all(axis=0)

    # remove inactive cells
    st = st.iloc[:, good_idx]
    if cell_metrics is not None:
        cell_metrics = cell_metrics[good_idx]

    return return_results(st, cell_metrics)


def remove_inactive_cells_pre_task_post(
    st: nel.core._eventarray.SpikeTrainArray,
    cell_metrics: Union[pd.core.frame.DataFrame, None] = None,
    beh_epochs: nel.core._intervalarray.EpochArray = None,
    nrem_epochs: nel.core._intervalarray.EpochArray = None,
    theta_epochs: nel.core._intervalarray.EpochArray = None,
    min_spikes: int = 100,
    nrem_time: Union[int, float] = 3600,
) -> tuple:
    """
    remove_inactive_cells_pre_task_post: Remove cells with fewer than min_spikes spikes per pre/task/post

    Parameters
    ----------
    st : SpikeTrainArray
        SpikeTrainArray object containing spike times for multiple cells.

    cell_metrics : pd.DataFrame, optional
        DataFrame containing metrics for each cell (e.g., quality metrics).

    beh_epochs : EpochArray
        EpochArray object containing pre/task/post epochs.

    nrem_epochs : EpochArray
        EpochArray object containing NREM epochs.

    theta_epochs : EpochArray
        EpochArray object containing theta epochs.

    min_spikes : int, optional
        Minimum number of spikes required per pre/task/post. Default is 100.

    nrem_time : int or float, optional
        Time in seconds to truncate NREM epochs. Default is 3600 seconds.

    Returns
    -------
    Tuple[nel.core._eventarray.SpikeTrainArray, Union[pd.DataFrame, None]]
        A tuple containing:
        - SpikeTrainArray object with inactive cells removed.
        - DataFrame containing cell metrics with inactive cells removed (if provided).

    Examples
    -------
    >>> from neuro_py.process.utils import remove_inactive_cells_pre_task_post
    >>> from neuro_py.io import loading
    >>> from neuro_py.session.locate_epochs import (
    >>>     find_multitask_pre_post,
    >>>     compress_repeated_epochs,
    >>> )
    >>> mport nelpy as nel

    >>> # load data from session
    >>> basepath = "Z:/Data/hpc_ctx_project/HP04/day_1_20240320"

    >>> # load spikes and cell metrics (cm)
    >>> st, cm = loading.load_spikes(basepath, brainRegion="CA1", putativeCellType="Pyr")

    >>> # load epochs and apply multitask epoch restrictions
    >>> epoch_df = loading.load_epoch(basepath)
    >>> epoch_df = compress_repeated_epochs(epoch_df)
    >>> pre_task_post = find_multitask_pre_post(
    >>>     epoch_df.environment, post_sleep_flank=True, pre_sleep_common=True
    >>> )

    >>> beh_epochs = nel.EpochArray(
    >>>     epoch_df.iloc[pre_task_post[0]][["startTime", "stopTime"]].values
    >>> )

    >>> # load sleep states to restrict to NREM and theta
    >>> state_dict = loading.load_SleepState_states(basepath)
    >>> nrem_epochs = nel.EpochArray(
    >>>     state_dict["NREMstate"],
    >>> )
    >>> theta_epochs = nel.EpochArray(
    >>>     state_dict["THETA"],
    >>> )

    >>> st,cm = remove_inactive_cells_pre_task_post(st,cm,beh_epochs,nrem_epochs,theta_epochs)
    """

    # check data types (further checks are done in remove_inactive_cells)
    if not isinstance(beh_epochs, nel.core._intervalarray.EpochArray):
        raise ValueError("beh_epochs must be an EpochArray object")

    if not isinstance(nrem_epochs, nel.core._intervalarray.EpochArray):
        raise ValueError("nrem_epochs must be an EpochArray object")

    if not isinstance(theta_epochs, nel.core._intervalarray.EpochArray):
        raise ValueError("theta_epochs must be an EpochArray object")

    # create list of restricted epochs
    restict_epochs = []
    for epoch, epoch_label in zip(beh_epochs, ["pre", "task", "post"]):
        if epoch_label in "pre":
            # get cumulative hours of sleep
            epoch_restrict = truncate_epoch(epoch & nrem_epochs, time=nrem_time)
        elif epoch_label in "post":
            # get cumulative hours of sleep
            epoch_restrict = truncate_epoch(epoch & nrem_epochs, time=nrem_time)
        else:
            # get theta during task
            epoch_restrict = epoch & theta_epochs
        restict_epochs.append(epoch_restrict)

    return remove_inactive_cells(
        st, cell_metrics, restict_epochs, min_spikes=min_spikes
    )


def compute_image_spread(
    X: np.ndarray, exponent: float = 2, normalize: bool = True
) -> Tuple[float, float]:
    """
    Compute the spread of an image using the square root of a weighted moment.

    The spread is calculated as the square root of a weighted moment of the image,
    where the weights are derived from the deviations of each pixel from the
    center of mass (COM) of the image.

    Parameters
    ----------
    X : np.ndarray
        A 2D numpy array of shape (numBinsY, numBinsX). If `normalize` is True,
        the input is assumed to represent a probability distribution.
    exponent : float, optional
        The exponent used in the moment calculation. Default is 2.
    normalize : bool, optional
        If True, normalize the input array so that its sum is 1. Default is True.

    Returns
    -------
    spread : float
        The computed spread, defined as the square root of the weighted moment.
    image_moment : float
        The raw weighted moment of the image.

    Examples
    --------
    >>> X = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    >>> spread, image_moment = compute_image_spread(X, exponent=2)
    >>> print(spread)
    0.5704157028642128
    >>> print(image_moment)
    0.325374074074074

    References
    ----------
    Widloski & Foster, 2022
    """
    if np.allclose(X, 0):
        return np.nan, np.nan  # Return NaN if the input is all zero

    if normalize:
        X = X / np.nansum(X)  # Normalize the input

    numBinsY, numBinsX = X.shape

    # Compute center of mass (COM) for the X (columns) direction.
    cols = np.arange(1, numBinsX + 1)
    sumX = np.nansum(X, axis=0)  # sum over rows, shape: (numBinsX,)
    totalX = np.nansum(sumX)
    # Add a small correction term
    comX = np.nansum(sumX * cols) / totalX + 0.5 / numBinsX

    # Compute center of mass for the Y (rows) direction.
    rows = np.arange(1, numBinsY + 1)
    sumY = np.nansum(X, axis=1)  # sum over columns, shape: (numBinsY,)
    totalY = np.nansum(sumY)
    comY = np.nansum(sumY * rows) / totalY + 0.5 / numBinsY

    # Create a meshgrid for the bin indices (using 1-indexing like MATLAB)
    XX, YY = np.meshgrid(np.arange(1, numBinsX + 1), np.arange(1, numBinsY + 1))

    # Compute the weighted moment using the product of the deviations raised to the given exponent.
    # For each bin, we compute:
    #     |XX - comX|^exponent * |YY - comY|^exponent * X(i,j)
    moment = np.nansum(
        (np.abs(XX - comX) ** exponent) * (np.abs(YY - comY) ** exponent) * X
    )

    # Normalize by the total probability.
    image_moment = moment / np.nansum(X)

    # The spread is the square root of the image moment.
    spread = np.sqrt(image_moment)

    return spread, image_moment
