"""
	Codes for PCA/ICA methods described in Detecting cell assemblies in large neuronal populations, Lopes-dos-Santos et al (2013).
											https://doi.org/10.1016/j.jneumeth.2013.04.010
	This implementation was written in Feb 2019.
	Please e-mail me if you have comments, doubts, bug reports or criticism (Vítor, vtlsantos@gmail.com /  vitor.lopesdossantos@pharm.ox.ac.uk).
"""
from typing import Tuple, Union
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from scipy import stats
import numpy as np
import numpy.matlib
from numba import jit
import warnings

__author__ = "Vítor Lopes dos Santos"
__version__ = "2019.1"


def toyExample(assemblies, nneurons=10, nbins=1000, rate=1.0):

    np.random.seed(42)

    actmat = np.random.poisson(rate, nneurons * nbins).reshape(nneurons, nbins)
    assemblies.actbins = [None] * len(assemblies.membership)
    for (ai, members) in enumerate(assemblies.membership):

        members = np.array(members)
        nact = int(nbins * assemblies.actrate[ai])
        actstrength_ = rate * assemblies.actstrength[ai]

        actbins = np.argsort(np.random.rand(nbins))[0:nact]

        actmat[members.reshape(-1, 1), actbins] = (
            np.ones((len(members), nact)) + actstrength_
        )

        assemblies.actbins[ai] = np.sort(actbins)

    return actmat


class toyassemblies:
    def __init__(self, membership, actrate, actstrength):

        self.membership = membership
        self.actrate = actrate
        self.actstrength = actstrength


def marcenkopastur(significance: object):

    nbins = significance.nbins
    nneurons = significance.nneurons
    tracywidom = significance.tracywidom

    # calculates statistical threshold from Marcenko-Pastur distribution
    q = float(nbins) / float(nneurons)  # note that silent neurons are counted too
    lambdaMax = pow((1 + np.sqrt(1 / q)), 2)
    lambdaMax += tracywidom * pow(nneurons, -2.0 / 3)  # Tracy-Widom correction

    return lambdaMax


def getlambdacontrol(zactmat_: np.ndarray):

    significance_ = PCA()
    significance_.fit(zactmat_.T)
    lambdamax_ = np.max(significance_.explained_variance_)

    return lambdamax_


def binshuffling(zactmat: np.ndarray, significance: object):

    np.random.seed()

    lambdamax_ = np.zeros(significance.nshu)
    for shui in range(significance.nshu):
        zactmat_ = np.copy(zactmat)
        for (neuroni, activity) in enumerate(zactmat_):
            randomorder = np.argsort(np.random.rand(significance.nbins))
            zactmat_[neuroni, :] = activity[randomorder]
        lambdamax_[shui] = getlambdacontrol(zactmat_)

    lambdaMax = np.percentile(lambdamax_, significance.percentile)

    return lambdaMax


def circshuffling(zactmat: np.ndarray, significance: object):

    np.random.seed()

    lambdamax_ = np.zeros(significance.nshu)
    for shui in range(significance.nshu):
        zactmat_ = np.copy(zactmat)
        for (neuroni, activity) in enumerate(zactmat_):
            cut = int(np.random.randint(significance.nbins * 2))
            zactmat_[neuroni, :] = np.roll(activity, cut)
        lambdamax_[shui] = getlambdacontrol(zactmat_)

    lambdaMax = np.percentile(lambdamax_, significance.percentile)

    return lambdaMax


def runSignificance(zactmat: np.ndarray, significance: object):

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
    actmat: np.ndarray, significance: object, method: str
) -> np.ndarray:
    nassemblies = significance.nassemblies

    if method == "pca":
        idxs = np.argsort(-significance.explained_variance_)[0:nassemblies]
        patterns = significance.components_[idxs, :]
    elif method == "ica":
        ica = FastICA(n_components=nassemblies, random_state=0, whiten="unit-variance")
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
        patterns /= np.matlib.repmat(norms, np.size(patterns, 1), 1).T

    return patterns


def runPatterns(
    actmat: np.ndarray,
    method: str = "ica",
    nullhyp: str = "mp",
    nshu: int = 1000,
    percentile: int = 99,
    tracywidom: bool = False,
) -> Union[Tuple[Union[np.ndarray,None], object, Union[np.ndarray,None]], None]:

    """
    INPUTS

        actmat:     activity matrix - numpy array (neurons, time bins)

        method:     defines how to extract assembly patterns (ica,pca).

        nullhyp:    defines how to generate statistical threshold for assembly detection.
                        'bin' - bin shuffling, will shuffle time bins of each neuron independently
                        'circ' - circular shuffling, will shift time bins of each neuron independently
                                                            obs: mantains (virtually) autocorrelations
                        'mp' - Marcenko-Pastur distribution - analytical threshold

        nshu:       defines how many shuffling controls will be done (n/a if nullhyp is 'mp')

        percentile: defines which percentile to be used use when shuffling methods are employed.
                                                                    (n/a if nullhyp is 'mp')

        tracywidow: determines if Tracy-Widom is used. See Peyrache et al 2010.
                                                (n/a if nullhyp is NOT 'mp')

    OUTPUTS

        patterns:     co-activation patterns (assemblies) - numpy array (assemblies, neurons)
        significance: object containing general information about significance tests
        zactmat:      returns z-scored actmat

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
    if np.isnan(significance.nassemblies):
        return None, significance, None

    if significance.nassemblies < 1:

        warnings.warn("no assembly detected")

        patterns = None
        zactmat = None
    else:
        # extracting co-activation patterns
        patterns_ = extractPatterns(zactmat_, significance, method)
        if patterns_ is np.nan:
            return None

        # putting eventual silent neurons back (their assembly weights are defined as zero)
        patterns = np.zeros((np.size(patterns_, 0), nneurons))
        patterns[:, ~silentneurons] = patterns_
        zactmat = np.copy(actmat)
        zactmat[~silentneurons, :] = zactmat_

    return patterns, significance, zactmat


@jit(nopython=True)
def computeAssemblyActivity(
    patterns: np.ndarray, zactmat: np.ndarray, zerodiag: bool = True
) -> np.ndarray:

    if len(patterns) == 0:
        return None

    nassemblies = len(patterns)
    nbins = zactmat.shape[1]

    assemblyAct = np.zeros((nassemblies, nbins))
    for (assemblyi, pattern) in enumerate(patterns):
        projMat = np.outer(pattern, pattern)
        projMat -= zerodiag * np.diag(np.diag(projMat))
        for bini in range(nbins):
            assemblyAct[assemblyi, bini] = (zactmat[:, bini] @ projMat) @ zactmat[
                :, bini
            ]

    return assemblyAct
