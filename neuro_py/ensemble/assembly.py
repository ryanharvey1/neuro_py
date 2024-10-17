"""
Codes for PCA/ICA methods described in Detecting cell assemblies in large neuronal populations, Lopes-dos-Santos et al (2013). 
https://doi.org/10.1016/j.jneumeth.2013.04.010
This implementation was written in Feb 2019.
Please e-mail me if you have comments, doubts, bug reports or criticism (Vítor, vtlsantos@gmail.com /  vitor.lopesdossantos@pharm.ox.ac.uk).
"""

import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA, FastICA


__author__ = "Vítor Lopes dos Santos"
__version__ = "2019.1"


def toyExample(
    assemblies: "ToyAssemblies",
    nneurons: int = 10,
    nbins: int = 1000,
    rate: float = 1.0,
) -> np.ndarray:
    """
    Generate a toy example activity matrix with assemblies.

    Parameters
    ----------
    assemblies : ToyAssemblies
        The toy assemblies.
    nneurons : int, optional
        Number of neurons, by default 10.
    nbins : int, optional
        Number of time bins, by default 1000.
    rate : float, optional
        Poisson rate, by default 1.0.

    Returns
    -------
    np.ndarray
        Activity matrix.
    """
    np.random.seed(42)

    actmat = np.random.poisson(rate, nneurons * nbins).reshape(nneurons, nbins)
    assemblies.actbins = [None] * len(assemblies.membership)
    for ai, members in enumerate(assemblies.membership):
        members = np.array(members)
        nact = int(nbins * assemblies.actrate[ai])
        actstrength_ = rate * assemblies.actstrength[ai]

        actbins = np.argsort(np.random.rand(nbins))[0:nact]

        actmat[members.reshape(-1, 1), actbins] = (
            np.ones((len(members), nact)) + actstrength_
        )

        assemblies.actbins[ai] = np.sort(actbins)

    return actmat


class ToyAssemblies:
    def __init__(
        self,
        membership: List[List[int]],
        actrate: List[float],
        actstrength: List[float],
    ):
        """
        Initialize ToyAssemblies.

        Parameters
        ----------
        membership : List[List[int]]
            List of lists containing neuron memberships for each assembly.
        actrate : List[float]
            List of activation rates for each assembly.
        actstrength : List[float]
            List of activation strengths for each assembly.
        """
        self.membership = membership
        self.actrate = actrate
        self.actstrength = actstrength


def marcenkopastur(significance: object) -> float:
    """
    Calculate statistical threshold from Marcenko-Pastur distribution.

    Parameters
    ----------
    significance : object
        Object containing significance parameters.

    Returns
    -------
    float
        Statistical threshold.
    """
    nbins = significance.nbins
    nneurons = significance.nneurons
    tracywidom = significance.tracywidom

    # calculates statistical threshold from Marcenko-Pastur distribution
    q = float(nbins) / float(nneurons)  # note that silent neurons are counted too
    lambdaMax = pow((1 + np.sqrt(1 / q)), 2)
    lambdaMax += tracywidom * pow(nneurons, -2.0 / 3)  # Tracy-Widom correction

    return lambdaMax


def getlambdacontrol(zactmat_: np.ndarray) -> float:
    """
    Get the maximum eigenvalue from PCA.

    Parameters
    ----------
    zactmat_ : np.ndarray
        Z-scored activity matrix.

    Returns
    -------
    float
        Maximum eigenvalue.
    """
    significance_ = PCA()
    significance_.fit(zactmat_.T)
    lambdamax_ = np.max(significance_.explained_variance_)

    return lambdamax_


def binshuffling(zactmat: np.ndarray, significance: object) -> float:
    """
    Perform bin shuffling to generate statistical threshold.

    Parameters
    ----------
    zactmat : np.ndarray
        Z-scored activity matrix.
    significance : object
        Object containing significance parameters.

    Returns
    -------
    float
        Statistical threshold.
    """
    np.random.seed()

    lambdamax_ = np.zeros(significance.nshu)
    for shui in range(significance.nshu):
        zactmat_ = np.copy(zactmat)
        for neuroni, activity in enumerate(zactmat_):
            randomorder = np.argsort(np.random.rand(significance.nbins))
            zactmat_[neuroni, :] = activity[randomorder]
        lambdamax_[shui] = getlambdacontrol(zactmat_)

    lambdaMax = np.percentile(lambdamax_, significance.percentile)

    return lambdaMax


def circshuffling(zactmat: np.ndarray, significance: object) -> float:
    """
    Perform circular shuffling to generate statistical threshold.

    Parameters
    ----------
    zactmat : np.ndarray
        Z-scored activity matrix.
    significance : object
        Object containing significance parameters.

    Returns
    -------
    float
        Statistical threshold.
    """
    np.random.seed()

    lambdamax_ = np.zeros(significance.nshu)
    for shui in range(significance.nshu):
        zactmat_ = np.copy(zactmat)
        for neuroni, activity in enumerate(zactmat_):
            cut = int(np.random.randint(significance.nbins * 2))
            zactmat_[neuroni, :] = np.roll(activity, cut)
        lambdamax_[shui] = getlambdacontrol(zactmat_)

    lambdaMax = np.percentile(lambdamax_, significance.percentile)

    return lambdaMax


def runSignificance(zactmat: np.ndarray, significance: object) -> object:
    """
    Run significance tests to estimate the number of assemblies.

    Parameters
    ----------
    zactmat : np.ndarray
        Z-scored activity matrix.
    significance : object
        Object containing significance parameters.

    Returns
    -------
    object
        Updated significance object with the number of assemblies.
    """
    if significance.nullhyp == "mp":
        lambdaMax = marcenkopastur(significance)
    elif significance.nullhyp == "bin":
        lambdaMax = binshuffling(zactmat, significance)
    elif significance.nullhyp == "circ":
        lambdaMax = circshuffling(zactmat, significance)
    else:
        raise ValueError(
            "nyll hypothesis method " + str(significance.nullhyp) + " not understood"
        )

    nassemblies = np.sum(significance.explained_variance_ > lambdaMax)
    significance.nassemblies = nassemblies

    return significance


def extractPatterns(
    actmat: np.ndarray, significance: object, method: str, whiten: str = "unit-variance"
) -> np.ndarray:
    """
    Extract co-activation patterns (assemblies).

    Parameters
    ----------
    actmat : np.ndarray
        Activity matrix.
    significance : object
        Object containing significance parameters.
    method : str
        Method to extract assembly patterns (ica, pca).
    whiten : str, optional
        Whitening method, by default "unit-variance".

    Returns
    -------
    np.ndarray
        Co-activation patterns (assemblies).
    """
    nassemblies = significance.nassemblies

    if method == "pca":
        idxs = np.argsort(-significance.explained_variance_)[0:nassemblies]
        patterns = significance.components_[idxs, :]
    elif method == "ica":
        ica = FastICA(n_components=nassemblies, random_state=0, whiten=whiten)
        ica.fit(actmat.T)
        patterns = ica.components_
    else:
        raise ValueError(
            "assembly extraction method " + str(method) + " not understood"
        )

    if patterns is not np.nan:
        patterns = patterns.reshape(nassemblies, -1)

        # sets norm of assembly vectors to 1
        norms = np.linalg.norm(patterns, axis=1)
        patterns /= np.tile(norms, [np.size(patterns, 1), 1]).T

    return patterns


def runPatterns(
    actmat: np.ndarray,
    method: str = "ica",
    nullhyp: str = "mp",
    nshu: int = 1000,
    percentile: int = 99,
    tracywidom: bool = False,
    whiten: str = "unit-variance",
    nassemblies: int = None,
) -> Union[Tuple[Union[np.ndarray, None], object, Union[np.ndarray, None]], None]:
    """
    Run pattern detection to identify cell assemblies.

    Parameters
    ----------
    actmat : np.ndarray
        Activity matrix (neurons, time bins).
    method : str, optional
        Method to extract assembly patterns (ica, pca), by default "ica".
    nullhyp : str, optional
        Null hypothesis method (bin, circ, mp), by default "mp".
    nshu : int, optional
        Number of shuffling controls, by default 1000.
    percentile : int, optional
        Percentile for shuffling methods, by default 99.
    tracywidom : bool, optional
        Use Tracy-Widom correction, by default False.
    whiten : str, optional
        Whitening method, by default "unit-variance".
    nassemblies : Optional[int], optional
        Number of assemblies, by default None.

    Returns
    -------
    Union[Tuple[Union[np.ndarray, None], object, Union[np.ndarray, None]], None]
        Patterns, significance object, and z-scored activity matrix.

    Notes
    -----
    nullhyp
        'bin' - bin shuffling, will shuffle time bins of each neuron independently
        'circ' - circular shuffling, will shift time bins of each neuron independently
        'mp' - Marcenko-Pastur distribution - analytical threshold
    """

    nneurons = np.size(actmat, 0)
    nbins = np.size(actmat, 1)

    silentneurons = np.var(actmat, axis=1) == 0
    actmat_ = actmat[~silentneurons, :]
    if actmat_.shape[0] == 0:
        warnings.warn("no active neurons")
        return None, None, None

    # z-scoring activity matrix
    zactmat_ = stats.zscore(actmat_, axis=1)

    # running significance (estimating number of assemblies)
    significance = PCA()
    significance.fit(zactmat_.T)
    significance.nneurons = nneurons
    significance.nbins = nbins
    significance.nshu = nshu
    significance.percentile = percentile
    significance.tracywidom = tracywidom
    significance.nullhyp = nullhyp
    significance = runSignificance(zactmat_, significance)

    if nassemblies is not None:
        significance.nassemblies = nassemblies

    if np.isnan(significance.nassemblies):
        return None, significance, None

    if significance.nassemblies < 1:
        warnings.warn("no assembly detected")

        patterns = None
        zactmat = None
    else:
        # extracting co-activation patterns
        patterns_ = extractPatterns(zactmat_, significance, method, whiten=whiten)
        if patterns_ is np.nan:
            return None

        # putting eventual silent neurons back (their assembly weights are defined as zero)
        patterns = np.zeros((np.size(patterns_, 0), nneurons))
        patterns[:, ~silentneurons] = patterns_
        zactmat = np.copy(actmat)
        zactmat[~silentneurons, :] = zactmat_

    return patterns, significance, zactmat


def computeAssemblyActivity(
    patterns: np.ndarray,
    zactmat: np.ndarray,
    zerodiag: bool = True,
) -> Optional[np.ndarray]:
    """
    Compute assembly activity.

    Parameters
    ----------
    patterns : np.ndarray
        Co-activation patterns (assemblies, neurons).
    zactmat : np.ndarray
        Z-scored activity matrix (neurons, time bins).
    zerodiag : bool, optional
        If True, diagonal of projection matrix is set to zero, by default True.

    Returns
    -------
    Optional[np.ndarray]
        Assembly activity matrix (assemblies, time bins).
    """
    # check if patterns is empty (no assembly detected) and return None if so
    if len(patterns) == 0:
        return None

    # number of assemblies and time bins
    nassemblies = len(patterns)
    nbins = np.size(zactmat, 1)

    # transpose for later matrix multiplication
    zactmat = zactmat.T

    # preallocate assembly activity matrix (nassemblies, nbins)
    assemblyAct = np.zeros((nassemblies, nbins))

    # loop over assemblies
    for assemblyi, pattern in enumerate(patterns):
        # compute projection matrix (neurons, neurons)
        projMat = np.outer(pattern, pattern)

        # set the diagonal to zero to not count coactivation of i and j when i=j
        if zerodiag:
            np.fill_diagonal(projMat, 0)

        # project assembly pattern onto z-scored activity matrix
        assemblyAct[assemblyi, :] = np.nansum(zactmat @ projMat * zactmat, axis=1)

    return assemblyAct
