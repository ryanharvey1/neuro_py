import itertools
import math
import multiprocessing
from typing import Any, Optional, Tuple

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy import stats

from neuro_py.stats.stats import get_significant_events


def similarity_index(
    patterns: NDArray[Any],
    n_shuffles: int = 1000,
    parallel: bool = True,
    groups: Optional[NDArray[Any]] = None,
    adjust_pvalue: bool = True,
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """
    Calculate the similarity index of a set of patterns.

    To use a quantitative criterion to compare assembly composition,
    a Similarity Index (SI) was defined as the absolute value of the
    inner product between the assembly patterns (unitary vectors) of
    two given assemblies, varying from 0 to 1. Thus, if two assemblies
    attribute large weights to the same neurons, SI will be large;
    if assemblies are orthogonal, SI will be zero.

    Parameters
    ----------
    patterns : np.ndarray
        List of patterns (n patterns x n neurons).
    n_shuffles : int, optional
        Number of shuffles to calculate the similarity index, by default 1000.
    parallel : bool, optional
        Whether to run in parallel, by default True.
    groups : np.ndarray, optional
        List of groups for each pattern (n patterns, ), will return cross-group comparisons by default None.
    adjust_pvalue : bool, optional
        Where to adjust p-values to control the false discovery rate.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        si: similarity index (n_combinations,)
        combos: list of all possible combinations of patterns (n_combinations, 2)
        pvalues: list of p-values for each pattern combination (n_combinations,)

    Examples
    --------
    >>> patterns = np.random.random_sample((60,20))
    >>> si, combos, pvalues = similarity_index(patterns)

    # with groups
    >>> patterns = np.random.random_sample((60,20))
    >>> groups = np.hstack([np.ones(20), np.ones(40)+1])
    >>> si, combos, pvalues = similarity_index(patterns, groups=groups)

    References
    ----------
    Based on Almeida-Filho et al., 2014 to detect similar assemblies.

    """
    # check to see if patterns are numpy arrays
    if not isinstance(patterns, np.ndarray):
        patterns = np.array(patterns)

    if patterns.shape[0] < 2:
        raise ValueError("At least 2 patterns are required to compute similarity.")

    # set seed to ensure exact results between runs
    np.random.seed(42)

    # maximum number of n_shuffles based on number of neurons
    n_shuffles = min(n_shuffles, int(math.factorial(patterns.shape[1])))

    # Normalize patterns
    patterns = patterns / np.linalg.norm(patterns, axis=1, keepdims=True)

    # shuffle patterns over neurons
    def shuffle_patterns(patterns):
        return np.array([np.random.permutation(pattern) for pattern in patterns])

    # Calculate absolute inner product between patterns
    combinations = np.array(list(itertools.combinations(range(patterns.shape[0]), 2)))

    def get_si(patterns: NDArray[Any]) -> NDArray[Any]:
        si = np.array(
            [np.abs(np.inner(patterns[i], patterns[j])) for i, j in combinations]
        )
        return si

    # calculate observed si
    si = get_si(patterns)

    # shuffle patterns and calculate si
    if parallel:
        num_cores = multiprocessing.cpu_count()
        si_shuffles = Parallel(n_jobs=num_cores)(
            delayed(get_si)(shuffle_patterns(patterns)) for _ in range(n_shuffles)
        )
    else:
        si_shuffles = [get_si(shuffle_patterns(patterns)) for _ in range(n_shuffles)]

    # calculate p-values for each pattern combination
    _, pvalues, _ = get_significant_events(si, np.array(si_shuffles))

    # Filter outputs to only include cross-group comparisons
    if groups is not None:
        # Ensure groups is a numpy array
        groups = np.asarray(groups)

        # Identify cross-group comparisons
        cross_group_mask = groups[combinations[:, 0]] != groups[combinations[:, 1]]
        si = si[cross_group_mask]
        combinations = combinations[cross_group_mask]
        pvalues = pvalues[cross_group_mask]

    if adjust_pvalue:
        pvalues = stats.false_discovery_control(pvalues)

    return si, combinations, pvalues
