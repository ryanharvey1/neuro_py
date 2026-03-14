"""
Codes for PCA/ICA methods described in Detecting cell assemblies in large neuronal populations, Lopes-dos-Santos et al (2013).
https://doi.org/10.1016/j.jneumeth.2013.04.010
This implementation was written in Feb 2019.
Please e-mail me if you have comments, doubts, bug reports or criticism (Vítor, vtlsantos@gmail.com /  vitor.lopesdossantos@pharm.ox.ac.uk).
"""

import warnings
from concurrent.futures import ThreadPoolExecutor
from os import cpu_count
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


def getlambdacontrol(
    zactmat_: np.ndarray, cross_structural: Optional[np.ndarray] = None
) -> float:
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
    if cross_structural is None:
        significance_ = PCA()
        significance_.fit(zactmat_.T)
        lambdamax_ = np.max(significance_.explained_variance_)
    else:
        zactmat_norm = _normalize_by_group(zactmat_, cross_structural)
        correlations = _compute_cross_structural_correlation(
            zactmat_norm, cross_structural
        )
        lambdamax_ = np.max(np.linalg.eigvalsh(correlations))

    return lambdamax_


def _resolve_n_jobs(n_jobs: Optional[int]) -> int:
    """Resolve n_jobs into a valid positive worker count."""
    if n_jobs is None:
        return 1
    if n_jobs == -1:
        return max(1, cpu_count() or 1)
    if n_jobs < -1 or n_jobs == 0:
        raise ValueError("n_jobs must be -1 or a positive integer")
    return int(n_jobs)


def _bin_shuffle_lambdamax(
    zactmat: np.ndarray,
    nbins: int,
    cross_structural: Optional[np.ndarray],
    seed: int,
) -> float:
    """Compute one bin-shuffle control lambda max."""
    rng = np.random.default_rng(seed)
    randomorder = np.argsort(rng.random((zactmat.shape[0], nbins)), axis=1)
    zactmat_shuffled = np.take_along_axis(zactmat, randomorder, axis=1)
    return getlambdacontrol(zactmat_shuffled, cross_structural=cross_structural)


def _circ_shuffle_lambdamax(
    zactmat: np.ndarray,
    nbins: int,
    cross_structural: Optional[np.ndarray],
    seed: int,
) -> float:
    """Compute one circular-shuffle control lambda max."""
    rng = np.random.default_rng(seed)
    cuts = rng.integers(0, nbins * 2, size=zactmat.shape[0])
    base_indices = np.arange(nbins)[None, :]
    shift_indices = (base_indices - cuts[:, None]) % nbins
    zactmat_shuffled = np.take_along_axis(zactmat, shift_indices, axis=1)
    return getlambdacontrol(zactmat_shuffled, cross_structural=cross_structural)


def binshuffling(
    zactmat: np.ndarray,
    significance: object,
    cross_structural: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = 1,
) -> float:
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
    n_workers = _resolve_n_jobs(n_jobs)
    seed_seq = np.random.SeedSequence()
    child_seeds = seed_seq.spawn(significance.nshu)
    seeds = [int(seed.generate_state(1)[0]) for seed in child_seeds]

    if n_workers == 1:
        lambdamax_ = np.array(
            [
                _bin_shuffle_lambdamax(
                    zactmat,
                    significance.nbins,
                    cross_structural,
                    seed,
                )
                for seed in seeds
            ]
        )
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            lambdamax_ = np.array(
                list(
                    executor.map(
                        lambda seed: _bin_shuffle_lambdamax(
                            zactmat,
                            significance.nbins,
                            cross_structural,
                            seed,
                        ),
                        seeds,
                    )
                )
            )

    lambdaMax = np.percentile(lambdamax_, significance.percentile)

    return lambdaMax


def circshuffling(
    zactmat: np.ndarray,
    significance: object,
    cross_structural: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = 1,
) -> float:
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
    n_workers = _resolve_n_jobs(n_jobs)
    seed_seq = np.random.SeedSequence()
    child_seeds = seed_seq.spawn(significance.nshu)
    seeds = [int(seed.generate_state(1)[0]) for seed in child_seeds]

    if n_workers == 1:
        lambdamax_ = np.array(
            [
                _circ_shuffle_lambdamax(
                    zactmat,
                    significance.nbins,
                    cross_structural,
                    seed,
                )
                for seed in seeds
            ]
        )
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            lambdamax_ = np.array(
                list(
                    executor.map(
                        lambda seed: _circ_shuffle_lambdamax(
                            zactmat,
                            significance.nbins,
                            cross_structural,
                            seed,
                        ),
                        seeds,
                    )
                )
            )

    lambdaMax = np.percentile(lambdamax_, significance.percentile)

    return lambdaMax


def runSignificance(
    zactmat: np.ndarray,
    significance: object,
    cross_structural: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = 1,
) -> object:
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
        lambdaMax = binshuffling(
            zactmat,
            significance,
            cross_structural=cross_structural,
            n_jobs=n_jobs,
        )
    elif significance.nullhyp == "circ":
        lambdaMax = circshuffling(
            zactmat,
            significance,
            cross_structural=cross_structural,
            n_jobs=n_jobs,
        )
    else:
        raise ValueError(
            "nyll hypothesis method " + str(significance.nullhyp) + " not understood"
        )

    nassemblies = np.sum(significance.explained_variance_ > lambdaMax)
    significance.nassemblies = nassemblies

    return significance


def extractPatterns(
    zactmat: np.ndarray,
    significance: object,
    method: str,
    whiten: str = "unit-variance",
    cross_structural: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extract co-activation patterns (assemblies).

    Parameters
    ----------
    zactmat : np.ndarray
        Z-scored activity matrix.
    significance : object
        Object containing significance parameters.
    method : str
        Method to extract assembly patterns (ica, pca).
    whiten : str, optional
        Whitening method, by default "unit-variance".
    cross_structural : Optional[np.ndarray], optional
        Categorical vector indicating group membership for each neuron.
        If provided and method is 'ica', will run ICA on data with modified
        cross-structural correlation structure, by default None.

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
        if cross_structural is not None:
            zactmat_norm = _normalize_by_group(zactmat, cross_structural)
            # For cross-structural ICA, modify the input data to reflect the cross-structural correlation structure
            correlations = _compute_cross_structural_correlation(
                zactmat_norm, cross_structural
            )

            # Eigenvalue decomposition to get the cross-structural subspace
            eigenvalues, eigenvectors = np.linalg.eigh(correlations)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Use the top nassemblies components (already determined)
            eigenvectors_sig = eigenvectors[:, :nassemblies]
            eigenvalues_sig = eigenvalues[:nassemblies]

            # Project the data onto the cross-structural subspace
            projected_data = (
                eigenvectors_sig * np.sqrt(np.maximum(eigenvalues_sig, 0))
            ).T @ zactmat_norm

            # Run ICA on the projected data
            ica = FastICA(n_components=nassemblies, random_state=0, whiten=whiten)
            ica.fit(projected_data.T)
            # Transform ICA components back to original space
            patterns = (
                ica.components_
                @ (eigenvectors_sig * np.sqrt(np.maximum(eigenvalues_sig, 0))).T
            )
        else:
            # Standard ICA
            ica = FastICA(n_components=nassemblies, random_state=0, whiten=whiten)
            ica.fit(zactmat.T)
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


def _normalize_by_group(
    zactmat: np.ndarray, cross_structural: np.ndarray
) -> np.ndarray:
    """
    Normalize activity within each group by the square root of group size.

    Parameters
    ----------
    zactmat : np.ndarray
        Activity matrix (neurons, time bins).
    cross_structural : np.ndarray
        Categorical group label for each neuron.

    Returns
    -------
    np.ndarray
        Group-normalized activity matrix where each group's rows are scaled by
        :math:`1/\sqrt{n_g}`.
    """
    groups = np.asarray(cross_structural)
    zactmat_norm = np.array(zactmat, copy=True, dtype=float)
    for group in np.unique(groups):
        group_mask = groups == group
        group_size = np.sum(group_mask)
        if group_size > 0:
            zactmat_norm[group_mask, :] /= np.sqrt(group_size)
    return zactmat_norm


def _compute_cross_structural_correlation(
    zactmat: np.ndarray, cross_structural: np.ndarray
) -> np.ndarray:
    """
    Compute a block-structured cross-group correlation matrix.

    The matrix is explicitly built with zero within-group blocks and empirical
    cross-group blocks. For two groups A and B this corresponds to:

    .. math::

        C = \begin{bmatrix}0 & C_{AB} \\ C_{BA} & 0\end{bmatrix}

    This preserves symmetry without in-place masking of a full correlation
    matrix and generalizes to more than two groups.

    Parameters
    ----------
    zactmat : np.ndarray
        Z-scored activity matrix (neurons, time bins).
    cross_structural : np.ndarray
        Categorical vector indicating group membership for each neuron.

    Returns
    -------
    np.ndarray
        Symmetric matrix with non-zero entries only for cross-group pairs.
    """
    groups = np.asarray(cross_structural)
    n_neurons = zactmat.shape[0]
    correlations = np.zeros((n_neurons, n_neurons), dtype=float)

    unique_groups = np.unique(groups)
    for group_a_idx, group_a in enumerate(unique_groups):
        idx_a = np.where(groups == group_a)[0]
        data_a = zactmat[idx_a, :]
        for group_b in unique_groups[group_a_idx + 1 :]:
            idx_b = np.where(groups == group_b)[0]
            data_b = zactmat[idx_b, :]

            corr_ab_full = np.corrcoef(data_a, data_b)
            n_a = data_a.shape[0]
            corr_ab = corr_ab_full[:n_a, n_a:]

            correlations[np.ix_(idx_a, idx_b)] = corr_ab
            correlations[np.ix_(idx_b, idx_a)] = corr_ab.T

    return correlations


def _filter_cross_group_patterns(
    patterns: np.ndarray,
    cross_structural: np.ndarray,
    atol: float = 1e-12,
) -> np.ndarray:
    """Keep only patterns with non-zero weights in at least two groups."""
    groups = np.asarray(cross_structural)
    unique_groups = np.unique(groups)
    keep_pattern = []

    for pattern in patterns:
        active_groups = 0
        for group in unique_groups:
            group_mask = groups == group
            if np.any(~np.isclose(pattern[group_mask], 0.0, atol=atol)):
                active_groups += 1
        keep_pattern.append(active_groups >= 2)

    keep_pattern = np.array(keep_pattern, dtype=bool)
    return patterns[keep_pattern]


def _compute_cross_svd(
    zactmat: np.ndarray,
    cross_structural: np.ndarray,
    n_components: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute cross-area SVD on a two-group activity matrix.

    Parameters
    ----------
    zactmat : np.ndarray
        Z-scored activity matrix (neurons, time bins).
    cross_structural : np.ndarray
        Group labels (must contain exactly two groups).
    n_components : Optional[int], optional
        Number of singular vectors to retain, by default all.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Left vectors U, singular values S, right vectors Vt, and row indices for
        group 1 and group 2 within ``zactmat``.
    """
    groups = np.asarray(cross_structural)
    unique_groups = np.unique(groups)
    if len(unique_groups) != 2:
        raise ValueError(
            "cross_svd requires exactly two groups in cross_structural"
        )

    idx_group1 = np.where(groups == unique_groups[0])[0]
    idx_group2 = np.where(groups == unique_groups[1])[0]

    X1 = zactmat[idx_group1, :]
    X2 = zactmat[idx_group2, :]

    cross_cov = X1 @ X2.T / X1.shape[1]
    U, S, Vt = np.linalg.svd(cross_cov, full_matrices=False)

    if n_components is not None:
        n_keep = min(n_components, len(S))
        U = U[:, :n_keep]
        S = S[:n_keep]
        Vt = Vt[:n_keep, :]

    return U, S, Vt, idx_group1, idx_group2


def _cross_svd_significance(
    zactmat: np.ndarray,
    cross_structural: np.ndarray,
    nshu: int,
    percentile: int,
    n_components: Optional[int] = None,
    n_jobs: Optional[int] = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate significant cross-area SVD components by shuffling group-1 activity.

    Parameters
    ----------
    zactmat : np.ndarray
        Z-scored activity matrix (neurons, time bins).
    cross_structural : np.ndarray
        Group labels (must contain exactly two groups).
    nshu : int
        Number of shuffles.
    percentile : int
        Percentile threshold for singular values.
    n_components : Optional[int], optional
        Number of singular values/components to evaluate, by default all.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        U, S, Vt, significant-component mask, null thresholds, group-1 indices,
        group-2 indices.
    """
    U, S, Vt, idx_group1, idx_group2 = _compute_cross_svd(
        zactmat, cross_structural, n_components=n_components
    )
    n_components_eval = len(S)

    X1 = zactmat[idx_group1, :]
    X2 = zactmat[idx_group2, :]

    def _single_cross_svd_shuffle(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        randomorder = np.argsort(rng.random((X1.shape[0], X1.shape[1])), axis=1)
        X1_shuffled = np.take_along_axis(X1, randomorder, axis=1)
        cross_cov_shuffled = X1_shuffled @ X2.T / X1.shape[1]
        _, singular_values_shuffled, _ = np.linalg.svd(
            cross_cov_shuffled, full_matrices=False
        )
        return singular_values_shuffled[:n_components_eval]

    n_workers = _resolve_n_jobs(n_jobs)
    seed_seq = np.random.SeedSequence()
    child_seeds = seed_seq.spawn(nshu)
    seeds = [int(seed.generate_state(1)[0]) for seed in child_seeds]

    if n_workers == 1:
        null_singular_values = np.array([_single_cross_svd_shuffle(seed) for seed in seeds])
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            null_singular_values = np.array(list(executor.map(_single_cross_svd_shuffle, seeds)))

    null_thresholds = np.percentile(null_singular_values, percentile, axis=0)
    keep_components = S > null_thresholds

    return (
        U,
        S,
        Vt,
        keep_components,
        null_thresholds,
        idx_group1,
        idx_group2,
    )


def computeCrossAreaActivity(
    patterns: np.ndarray,
    zactmat: np.ndarray,
    cross_structural: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Compute time-resolved cross-area coactivation for cross-SVD assemblies.

    Parameters
    ----------
    patterns : np.ndarray
        Assembly patterns (assemblies, neurons).
    zactmat : np.ndarray
        Z-scored activity matrix (neurons, time bins).
    cross_structural : np.ndarray
        Group labels (must contain exactly two groups).

    Returns
    -------
    Optional[np.ndarray]
        Coactivation matrix (assemblies, time bins), or None if no patterns.
    """
    if patterns is None or len(patterns) == 0:
        return None

    groups = np.asarray(cross_structural)
    unique_groups = np.unique(groups)
    if len(unique_groups) != 2:
        raise ValueError(
            "computeCrossAreaActivity requires exactly two groups in cross_structural"
        )

    idx_group1 = groups == unique_groups[0]
    idx_group2 = groups == unique_groups[1]

    X1 = zactmat[idx_group1, :]
    X2 = zactmat[idx_group2, :]

    group1_proj = patterns[:, idx_group1] @ X1
    group2_proj = patterns[:, idx_group2] @ X2

    return group1_proj * group2_proj


def runPatterns(
    actmat: np.ndarray,
    method: str = "ica",
    nullhyp: str = "mp",
    nshu: int = 1000,
    percentile: int = 99,
    tracywidom: bool = False,
    whiten: str = "unit-variance",
    nassemblies: int = None,
    cross_structural: Optional[np.ndarray] = None,
    n_jobs: Optional[int] = 1,
) -> Union[Tuple[Union[np.ndarray, None], object, Union[np.ndarray, None]], None]:
    """
    Run pattern detection to identify cell assemblies.

    Parameters
    ----------
    actmat : np.ndarray
        Activity matrix (neurons, time bins).
    method : str, optional
        Method to extract assembly patterns (ica, pca, cross_svd),
        by default "ica".
    nullhyp : str, optional
        Null hypothesis method (bin, circ, mp), by default "mp".
        In cross-structural mode, ``"mp"`` is automatically replaced by
        ``"bin"`` because Marčenko–Pastur assumptions do not hold for the
        block-structured cross-group covariance matrix.
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
    cross_structural : Optional[np.ndarray], optional
        A categorical vector indicating group membership for each neuron.
        If provided, the function runs cross-structural detection by:

        1. removing silent neurons,
        2. z-scoring activity,
        3. scaling each group by :math:`1/\sqrt{n_g}`,
        4. building an explicit block cross-group correlation matrix,
        5. estimating significance in the same cross-structural space,
        6. filtering extracted patterns to keep only multi-group assemblies.

        Should have the same length as the number of neurons in ``actmat``.
        By default None.
    n_jobs : Optional[int], optional
        Number of workers for shuffle-based significance controls.
        Use ``1`` for serial execution, ``-1`` for all available cores,
        or any positive integer. By default 1.

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

    cross_structural
        When provided, this vector should have the same length as the number of neurons
        in actmat. Each element indicates the group membership (e.g., brain region,
        cell type) for the corresponding neuron.

        The cross-structural path keeps only cross-group covariance terms and
        requires each retained pattern to be active in at least two groups.

    warnings
        ``"no cross-structural assembly detected"`` can be emitted when candidate
        patterns are removed by the multi-group membership filter.
    """

    nneurons = np.size(actmat, 0)
    nbins = np.size(actmat, 1)

    # Validate cross_structural parameter if provided
    if cross_structural is not None:
        if len(cross_structural) != nneurons:
            raise ValueError(
                f"cross_structural length ({len(cross_structural)}) must match "
                f"number of neurons ({nneurons})"
            )

    silentneurons = np.var(actmat, axis=1) == 0
    actmat_ = actmat[~silentneurons, :]
    if actmat_.shape[0] == 0:
        warnings.warn("no active neurons")
        return None, None, None

    # Update cross_structural to match active neurons only
    cross_structural_ = None
    if cross_structural is not None:
        cross_structural_ = cross_structural[~silentneurons]

    # z-scoring activity matrix
    zactmat_ = stats.zscore(actmat_, axis=1)

    # running significance (estimating number of assemblies)
    significance = PCA()

    effective_nullhyp = nullhyp
    if cross_structural_ is not None and nullhyp == "mp":
        effective_nullhyp = "bin"

    if method == "cross_svd":
        if cross_structural_ is None:
            raise ValueError("cross_svd requires cross_structural labels")

        (
            U,
            singular_values,
            Vt,
            keep_components,
            null_thresholds,
            idx_group1,
            idx_group2,
        ) = _cross_svd_significance(
            zactmat_,
            cross_structural_,
            nshu=nshu,
            percentile=percentile,
            n_components=nassemblies,
            n_jobs=n_jobs,
        )

        selected_components = np.where(keep_components)[0]
        significance.nneurons = nneurons
        significance.nbins = nbins
        significance.nshu = nshu
        significance.percentile = percentile
        significance.tracywidom = tracywidom
        significance.nullhyp = "bin"
        significance.explained_variance_ = singular_values
        significance.cross_svd_null_thresholds_ = null_thresholds
        significance.cross_svd_keep_mask_ = keep_components
        significance.cross_svd_u_ = U
        significance.cross_svd_vt_ = Vt
        significance.nassemblies = len(selected_components)

        if significance.nassemblies < 1:
            warnings.warn("no cross-svd assembly detected")
            return None, significance, None

        patterns_active = np.zeros((significance.nassemblies, zactmat_.shape[0]))
        for out_i, comp_i in enumerate(selected_components):
            patterns_active[out_i, idx_group1] = U[:, comp_i]
            patterns_active[out_i, idx_group2] = Vt[comp_i, :]

        # sets norm of assembly vectors to 1
        norms = np.linalg.norm(patterns_active, axis=1)
        norms[norms == 0] = 1
        patterns_active /= np.tile(norms, [np.size(patterns_active, 1), 1]).T

        patterns = np.zeros((np.size(patterns_active, 0), nneurons))
        patterns[:, ~silentneurons] = patterns_active
        zactmat = np.copy(actmat)
        zactmat[~silentneurons, :] = zactmat_

        return patterns, significance, zactmat

    if cross_structural_ is not None:
        zactmat_cross = _normalize_by_group(zactmat_, cross_structural_)
        # Compute custom correlation matrix for cross-structural assemblies
        correlations = _compute_cross_structural_correlation(
            zactmat_cross, cross_structural_
        )
        # Perform eigenvalue decomposition on the custom correlation matrix
        eigenvalues, eigenvectors = np.linalg.eigh(correlations)
        # Sort in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # Store results in significance object to mimic PCA interface
        significance.explained_variance_ = eigenvalues
        significance.components_ = eigenvectors.T
    else:
        # Use standard PCA
        significance.fit(zactmat_.T)

    significance.nneurons = nneurons
    significance.nbins = nbins
    significance.nshu = nshu
    significance.percentile = percentile
    significance.tracywidom = tracywidom
    significance.nullhyp = effective_nullhyp
    significance = runSignificance(
        zactmat_,
        significance,
        cross_structural=cross_structural_,
        n_jobs=n_jobs,
    )

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
        patterns_ = extractPatterns(
            zactmat_,
            significance,
            method,
            whiten=whiten,
            cross_structural=cross_structural_,
        )
        if patterns_ is np.nan:
            return None

        if cross_structural_ is not None:
            patterns_ = _filter_cross_group_patterns(patterns_, cross_structural_)
            significance.nassemblies = patterns_.shape[0]
            if significance.nassemblies < 1:
                warnings.warn("no cross-structural assembly detected")
                return None, significance, None

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
