import numpy as np

from lazy_loader import attach as _attach
from sklearn.metrics.pairwise import cosine_similarity as getsim

__all__ = (
    "similaritymat",
)
__getattr__, __dir__, __all__ = _attach(f"{__name__}", submodules=__all__)
del _attach


def similaritymat(patternsX, patternsY=None, method="cosine", findpairs=False):

    """
    INPUTS

        patternsX:     co-activation patterns (assemblies)
                                    - numpy array (assemblies, neurons)
        patternsY:     co-activation patterns (assemblies)
                                    - numpy array (assemblies, neurons)
                                    - if None, will compute similarity
                                                of patternsX to itself

        method:        defines similarity measure method
                                    'cosine' - cosine similarity
        findpairs:     maximizes main diagonal of the sim matrix to define pairs
                                                from patterns X and Y
                                    returns rowind,colind which can be used to reorder
                                                patterns X and Y to maximize the diagonal
    OUTPUTS

        simmat:        similarity matrix
                                    - array (assemblies from X, assemblies from Y)
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

        import scipy.optimize as optimize

        rowind, colind = optimize.linear_sum_assignment(-simmat)

        rowind = fillmissingidxs(rowind, np.size(simmat, 0))
        colind = fillmissingidxs(colind, np.size(simmat, 1))

        return simmat, rowind, colind
    else:
        return simmat
