from typing import List, Union

import nelpy as nel
import numpy as np
import pandas as pd

from neuro_py.process.intervals import truncate_epoch


def circular_shift(m: np.ndarray, s: np.ndarray):
    """
    CircularShift - Shift matrix rows or columns circularly.

    Shift each matrix row (or column) circularly by a different amount.

    USAGE

    shifted = circular_shift(m,s)

    Parameters
    ----------
    m: matrix to rotate
    s: shift amount for each row (horizontal vector) or column (vertical vector)

    Returns
    -------
    shifted: matrix m with rows (or columns) circularly shifted by the amounts in s


    adapted from  CircularShift.m  Copyright (C) 2012 by Michaël Zugaro

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


def avgerage_diagonal(mat):
    """
    Average values over all offset diagonals

    Parameters
    ----------
    mat: 2D array

    Returns
    -------
    output: 1D array
            Average values over all offset diagonals

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
    cell_metrics: Union[pd.core.frame.DataFrame, None] = None,
    epochs: Union[
        List[nel.core._intervalarray.EpochArray], nel.core._intervalarray.EpochArray
    ] = None,
    min_spikes: int = 100,
) -> tuple:
    """
    remove_inactive_cells: Remove cells with fewer than min_spikes spikes per sub-epoch

    Parameters
    ----------
    st: SpikeTrainArray
        SpikeTrainArray object

    cell_metrics: DataFrame
        DataFrame containing cell metrics
    epochs: EpochArray
        list of EpochArray objects or a single EpochArray object
        If a list of EpochArray objects is provided, each EpochArray object is treated as a sub-epoch
        If a single EpochArray object is provided, each interval in the EpochArray object is treated as a sub-epoch
    min_spikes: int
        Minimum number of spikes per sub-epoch

    Returns
    -------
    st: SpikeTrainArray
        SpikeTrainArray object with inactive cells removed
    cell_metrics: DataFrame
        DataFrame containing cell metrics with inactive cells removed

    Example
    -------
    from neuro_py.process.intervals import truncate_epoch
    from neuro_py.session.locate_epochs import (
        find_multitask_pre_post,
        compress_repeated_epochs,
    )
    from neuro_py.io import loading
    import nelpy as nel
    from neuro_py.process.utils import remove_inactive_cells

    # load data from session
    basepath = r"Z:\Data\hpc_ctx_project\HP04\day_1_20240320"

    # load spikes and cell metrics (cm)
    st, cm = loading.load_spikes(basepath, brainRegion="CA1", putativeCellType="Pyr")

    # load epochs and apply multitask epoch restrictions
    epoch_df = loading.load_epoch(basepath)
    epoch_df = compress_repeated_epochs(epoch_df)
    pre_task_post = find_multitask_pre_post(
        epoch_df.environment, post_sleep_flank=True, pre_sleep_common=True
    )

    beh_epochs = nel.EpochArray(
        epoch_df.iloc[pre_task_post[0]][["startTime", "stopTime"]].values
    )
    # load sleep states to restrict to NREM and theta
    state_dict = loading.load_SleepState_states(basepath)
    nrem_epochs = nel.EpochArray(
        state_dict["NREMstate"],
    )
    theta_epochs = nel.EpochArray(
        state_dict["THETA"],
    )
    # create list of restricted epochs
    restict_epochs = []
    for epoch, epoch_label in zip(beh_epochs, ["pre", "task", "post"]):
        if epoch_label in "pre":
            # get cumulative hours of sleep
            epoch_restrict = truncate_epoch(epoch & nrem_epochs, time=3600)
        elif epoch_label in "post":
            # get cumulative hours of sleep
            epoch_restrict = truncate_epoch(epoch & nrem_epochs, time=3600)
        else:
            # get theta during task
            epoch_restrict = epoch & theta_epochs
        restict_epochs.append(epoch_restrict)

    # remove inactive cells
    st, cm = remove_inactive_cells(st, cm, restict_epochs)

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
    st: SpikeTrainArray
        SpikeTrainArray object
    cell_metrics: DataFrame
        DataFrame containing cell metrics
    beh_epochs: EpochArray
        EpochArray object containing pre/task/post epochs
    nrem_epochs: EpochArray
        EpochArray object containing NREM epochs
    theta_epochs: EpochArray
        EpochArray object containing theta epochs
    min_spikes: int
        Minimum number of spikes per pre/task/post
    nrem_time: int or float
        Time in seconds to truncate NREM epochs

    Returns
    -------
    st: SpikeTrainArray
        SpikeTrainArray object with inactive cells removed
    cell_metrics: DataFrame
        DataFrame containing cell metrics with inactive cells removed

    Example
    -------
    from neuro_py.process.utils import remove_inactive_cells_pre_task_post
    from neuro_py.io import loading
    from neuro_py.session.locate_epochs import (
        find_multitask_pre_post,
        compress_repeated_epochs,
    )
    import nelpy as nel

    # load data from session
    basepath = r"Z:\Data\hpc_ctx_project\HP04\day_1_20240320"

    # load spikes and cell metrics (cm)
    st, cm = loading.load_spikes(basepath, brainRegion="CA1", putativeCellType="Pyr")

    # load epochs and apply multitask epoch restrictions
    epoch_df = loading.load_epoch(basepath)
    epoch_df = compress_repeated_epochs(epoch_df)
    pre_task_post = find_multitask_pre_post(
        epoch_df.environment, post_sleep_flank=True, pre_sleep_common=True
    )

    beh_epochs = nel.EpochArray(
        epoch_df.iloc[pre_task_post[0]][["startTime", "stopTime"]].values
    )

    # load sleep states to restrict to NREM and theta
    state_dict = loading.load_SleepState_states(basepath)
    nrem_epochs = nel.EpochArray(
        state_dict["NREMstate"],
    )
    theta_epochs = nel.EpochArray(
        state_dict["THETA"],
    )

    st,cm = remove_inactive_cells_pre_task_post(st,cm,beh_epochs,nrem_epochs,theta_epochs)
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
