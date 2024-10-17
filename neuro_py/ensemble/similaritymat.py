from typing import Optional, Tuple, Union

import numpy as np
import scipy.optimize as optimize
from sklearn.metrics.pairwise import cosine_similarity as getsim


def similaritymat(
    patternsX: np.ndarray,
    patternsY: Optional[np.ndarray] = None,
    method: str = "cosine",
    findpairs: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Calculate the similarity matrix of co-activation patterns (assemblies).

    Parameters
    ----------
    patternsX : np.ndarray
        Co-activation patterns (assemblies) - numpy array (assemblies, neurons).
    patternsY : Optional[np.ndarray], optional
        Co-activation patterns (assemblies) - numpy array (assemblies, neurons).
        If None, will compute similarity of patternsX to itself, by default None.
    method : str, optional
        Defines similarity measure method, by default 'cosine'.
        'cosine' - cosine similarity.
    findpairs : bool, optional
        Maximizes main diagonal of the similarity matrix to define pairs
        from patterns X and Y. Returns rowind, colind which can be used to reorder
        patterns X and Y to maximize the diagonal, by default False.

    Returns
    -------
    Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]
        Similarity matrix (assemblies from X, assemblies from Y).
        If findpairs is True, also returns rowind and colind.
    """

    if method != "cosine":
        print(method + " for similarity has not been implemented yet.")
        return

    inputs = {"X": patternsX, "Y": patternsY}
    simmat = getsim(**inputs)

    if findpairs:

        def fillmissingidxs(ind, n):
            missing = list(set(np.arange(n)) - set(ind))
            ind = np.array(list(ind) + missing)
            return ind

        rowind, colind = optimize.linear_sum_assignment(-simmat)

        rowind = fillmissingidxs(rowind, np.size(simmat, 0))
        colind = fillmissingidxs(colind, np.size(simmat, 1))

        return simmat, rowind, colind
    else:
        return simmat
