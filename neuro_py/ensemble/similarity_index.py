import itertools
import multiprocessing
from typing import Tuple

import numpy as np
from joblib import Parallel, delayed

from neuro_py.stats.stats import get_significant_events


def similarity_index(
    patterns: np.ndarray, n_shuffles: int = 1000, parallel: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the similarity index of a set of patterns.

    Based on Almeida-Filho et al., 2014 to detect similar assemblies.

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

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        si: similarity index: float (0-1)
        combos: list of all possible combinations of patterns
        pvalues: list of p-values for each pattern combination
    """
    # check to see if patterns are numpy arrays
    if not isinstance(patterns, np.ndarray):
        patterns = np.array(patterns)

    # check if all values in matrix are less than 1
    if not all(i <= 1 for i in patterns.flatten()):
        raise ValueError("All values in matrix must be less than 1")

    # shuffle patterns over neurons
    def shuffle_patterns(patterns):
        return np.random.permutation(patterns.flatten()).reshape(patterns.shape)

    # calculate absolute inner product between patterns
    def get_si(patterns, return_combo=False):
        x = np.arange(0, patterns.shape[0])
        # use itertools to get all combinations of patterns
        combos = np.array(list(itertools.combinations(x, 2)))
        si = []
        for s in combos:
            si.append(np.abs(np.inner(patterns[s[0], :], patterns[s[1], :])))

        if return_combo:
            return np.array(si), combos
        else:
            return np.array(si)

    # calculate observed si
    si, combos = get_si(patterns, return_combo=True)

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

    return si, combos, pvalues
