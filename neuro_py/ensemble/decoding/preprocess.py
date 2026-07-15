from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import StratifiedKFold


def split_data(
    trial_nsvs: NDArray[Any],
    splitby: NDArray[Any],
    trainsize: float = 0.8,
    seed: int = 0,
) -> List[NDArray[Any]]:
    """
    Split data into stratified folds.

    Parameters
    ----------
    trial_nsvs : np.ndarray
        Neural state vectors for trials.
    splitby : np.ndarray
        Labels for stratification.
    trainsize : float, optional
        Proportion of data to use for training, by default 0.8
    seed : int, optional
        Random seed for reproducibility, by default 0

    Returns
    -------
    List[np.ndarray]
        List of indices for each fold.
    """
    n_splits = int(np.round(1 / ((1 - trainsize) / 2)))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [fold_indices for _, fold_indices in skf.split(trial_nsvs, splitby)]
    return folds


def partition_indices(
    folds: List[NDArray[Any]],
) -> List[Tuple[NDArray[Any], NDArray[Any], NDArray[Any]]]:
    """
    Partition indices into train, validation, and test sets.

    Parameters
    ----------
    folds : List[np.ndarray]
        Indices for each fold.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        Train, validation, and test indices.
    """
    partition_mask = np.zeros(len(folds), dtype=int)
    partition_mask[0:2] = (2, 1)
    folds_arr = np.asarray(folds, dtype=object)

    partitions_indices = []
    for i in range(len(folds)):
        curr_pmask = np.roll(partition_mask, i)
        train_indices = np.concatenate(folds_arr[curr_pmask == 0]).tolist()
        val_indices = np.concatenate(folds_arr[curr_pmask == 1]).tolist()
        test_indices = np.concatenate(folds_arr[curr_pmask == 2]).tolist()

        partitions_indices.append((train_indices, val_indices, test_indices))
    return partitions_indices


def partition_sets(
    partitions_indices: List[Tuple[NDArray[Any], NDArray[Any], NDArray[Any]]],
    nsv_trial_segs: Union[NDArray[Any], pd.DataFrame],
    bv_trial_segs: Union[NDArray[Any], pd.DataFrame],
) -> List[
    Tuple[
        NDArray[Any],
        NDArray[Any],
        NDArray[Any],
        NDArray[Any],
        NDArray[Any],
        NDArray[Any],
    ]
]:
    """
    Partition neural state vectors and behavioral variables into train,
    validation, and test sets.

    Parameters
    ----------
    partitions_indices : List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
        List of tuples containing indices of divided trials into train,
        validation, and test sets.
    nsv_trial_segs : Union[np.ndarray, pd.DataFrame]
        Neural state vectors for each trial.
        Shape: [n_trials, n_timepoints, n_neurons] or [n_timepoints, n_neurons]
    bv_trial_segs : Union[np.ndarray, pd.DataFrame]
        Behavioral variables for each trial.
        Shape: [n_trials, n_timepoints, n_bvars] or [n_timepoints, n_bvars]

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        List of tuples containing train, validation, and test sets for neural
        state vectors and behavioral variables.
    """
    partitions = []
    if isinstance(nsv_trial_segs, pd.DataFrame):
        is_2D = nsv_trial_segs.iloc[0].ndim == 1
    else:
        is_2D = nsv_trial_segs[0].ndim == 1
    for train_indices, val_indices, test_indices in partitions_indices:
        if is_2D:
            if isinstance(nsv_trial_segs, pd.DataFrame):
                assert isinstance(bv_trial_segs, pd.DataFrame)
                train = nsv_trial_segs.loc[train_indices]
                val = nsv_trial_segs.loc[val_indices]
                test = nsv_trial_segs.loc[test_indices]
                train_bv = bv_trial_segs.loc[train_indices]
                val_bv = bv_trial_segs.loc[val_indices]
                test_bv = bv_trial_segs.loc[test_indices]
            else:
                train = nsv_trial_segs[train_indices]
                val = nsv_trial_segs[val_indices]
                test = nsv_trial_segs[test_indices]
                train_bv = bv_trial_segs[train_indices]
                val_bv = bv_trial_segs[val_indices]
                test_bv = bv_trial_segs[test_indices]
        else:
            train = np.take(nsv_trial_segs, train_indices, axis=0)
            val = np.take(nsv_trial_segs, val_indices, axis=0)
            test = np.take(nsv_trial_segs, test_indices, axis=0)
            train_bv = np.take(bv_trial_segs, train_indices, axis=0)
            val_bv = np.take(bv_trial_segs, val_indices, axis=0)
            test_bv = np.take(bv_trial_segs, test_indices, axis=0)

        partitions.append((train, train_bv, val, val_bv, test, test_bv))
    return partitions
