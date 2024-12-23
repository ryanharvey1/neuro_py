import numpy as np
import sklearn.model_selection


def split_data(trial_nsvs, splitby, trainsize=.8, seed=0):
    n_splits = int(np.round(1 / ((1 - trainsize) / 2)))
    skf = sklearn.model_selection.StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [
        fold_indices for _, fold_indices in skf.split(trial_nsvs, splitby)]
    return folds

def partition_indices(folds):
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

def partition_sets(partitions_indices, nsv_trial_segs, bv_trial_segs):
    """_summary_

    Parameters
    ----------
    partitions_indices : _type_
        _description_
    nsv_trial_segs : np.ndarray
        
        Shape: [n_trials, n_timepoints, n_neurons] or [n_timepoints, n_neurons]
    bv_trial_segs : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    partitions = []
    is_2D = nsv_trial_segs[0].ndim == 1
    for (train_indices, val_indices, test_indices) in partitions_indices:
        train = nsv_trial_segs.loc[train_indices] if is_2D else \
            np.take(nsv_trial_segs, train_indices)
        val = nsv_trial_segs.loc[val_indices] if is_2D else \
            np.take(nsv_trial_segs, val_indices)
        test = nsv_trial_segs.loc[test_indices] if is_2D else \
            np.take(nsv_trial_segs, test_indices)

        train_bv, val_bv, test_bv = (
            bv_trial_segs.loc[train_indices] if is_2D else \
                np.take(bv_trial_segs, train_indices),
            bv_trial_segs.loc[val_indices] if is_2D else \
                np.take(bv_trial_segs, val_indices),
            bv_trial_segs.loc[test_indices] if is_2D else \
                np.take(bv_trial_segs, test_indices)
        )
        partitions.append((train, train_bv, val, val_bv, test, test_bv))
    return partitions
