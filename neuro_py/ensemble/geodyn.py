import numpy as np

from scipy.stats import binned_statistic_dd

from ..util.array import (
    find_terminal_masked_indices,
    replace_border_zeros_with_nan,
)


def potential_landscape(X_dyn, projbins, domainbins=None):
    """Compute numerical approximation of potential energy landscape across
    1D state and domain (e.g. time, position, etc.).

    Potential landscape is defined as the integral of the flow vectors.

    Parameters
    ----------
    X_dyn : np.ndarray
        State vectors of shape: trials x bins
    projbins : int or array-like
        Number of bins for projection axis or bin edges
    domainbins : int or array-like, optional
        Number of bins for domain axis or bin edges, by default None

    Returns
    -------
    np.ndarray
        Potential energy landscape across state and domain
    np.ndarray
        Temporal gradient of potential energy landscape across state and domain
    np.ndarray
        Histogram of state vectors across state and domain
    np.ndarray
        Bin edges of state vectors
    np.ndarray
        Bin edges of domain

    References
    ----------
    .. [1] Wang, S., Falcone, R., Richmond, B. et al. Attractor dynamics reflect
           decision confidence in macaque prefrontal cortex. Nat Neurosci 26,
           1970–1980 (2023).
    """
    # _t suffix is following notation of paper but applicable across any domain
    nnrns = 1
    ntrials, nbins = X_dyn.shape
    delta_t = np.diff(X_dyn, axis=1)  # time derivatives: ntrials x nbins-1 x nnrns

    X_t_flat = np.reshape(X_dyn[:, :-1], (-1, nnrns), order='F').ravel()  # skip last bin as no displacement exists for last time point
    delta_t_flat = np.reshape(delta_t, (-1, nnrns), order='F').ravel()  # column-major order
    norm_tpts = np.repeat(np.arange(nbins-1), ntrials)

    nbins_domain = nbins - 1 if domainbins is None else domainbins  # downsample domain bins

    # 1D state space binning of time derivatives across domain
    # assumes landscape may morph across domain
    H, bin_edges, _ = binned_statistic_dd(  # posbins x time
        np.asarray((X_t_flat, norm_tpts)).T, delta_t_flat, statistic='count',
        bins=(projbins, nbins_domain))
    latentedges, domainedges = bin_edges

    grad_pos_t_svm = binned_statistic_dd(
        np.asarray((X_t_flat, norm_tpts)).T, delta_t_flat, statistic='sum',
        bins=(projbins, nbins_domain)).statistic
    # average derivative, a.k.a. flow/vector field for dynamics underlying
    # population activity
    grad_pos_t_svm = np.divide(grad_pos_t_svm, H, where=H != 0)
    grad_pos_t_svm[H == 0] = np.nan  # crucial to handle division by zero
    # spatial integration via nnancumsum treats nan as zero for cumulative sum
    potential_pos_t = -np.nancumsum(grad_pos_t_svm, axis=0)  # projbins x domainbins

    idx_zero_X_t = np.searchsorted(latentedges, 0)
    offset = potential_pos_t[idx_zero_X_t, :]  # use potential at X_t = 0 as reference
    potential_pos_t = potential_pos_t - offset  # potential difference

    nonzero_mask = H != 0
    idx_first_nonzero, idx_last_nonzero = \
        find_terminal_masked_indices(nonzero_mask, axis=0)  # each have shape: time
    # along axis 0 set all values from start to idx_first_nonzero to nan
    for t in range(H.shape[1]):
        potential_pos_t[:idx_first_nonzero[t], t] = np.nan
        potential_pos_t[idx_last_nonzero[t] + 1:, t] = np.nan

    return potential_pos_t, grad_pos_t_svm, H, latentedges, domainedges

def potential_landscape_nd(X_dyn, projbins, domainbins=None, nanborderempty=True):
    """Compute numerical approximation of potential energy landscape across
    n-dimensional state and domain (e.g. time, position, etc.).

    Potential landscape is defined as the integral of the flow vectors.

    Parameters
    ----------
    X_dyn : np.ndarray
        State vectors of shape: trials x bins x neurons
    projbins : int or array-like
        Number of bins for projection axis or bin edges for each neuron
    domainbins : int or array-like, optional
        Number of bins for domain axis or bin edges, by default None
    nanborderempty : bool, optional
        Whether to set border values to nan if they are empty, by default True

    Returns
    -------
    np.ndarray
        Potential energy landscape across state and domain for each neuron.
        Shape: projbins x domainbins x nnrns
    np.ndarray
        Temporal gradient of potential energy landscape across state and domain
        for each neuron. Shape: projbins x domainbins x nnrns
    np.ndarray
        Histogram of state vectors across state and domain for each neuron.
        Shape: projbins x domainbins x nnrns
    np.ndarray
        Bin edges of state vectors for each neuron
    np.ndarray
        Bin edges of domain for each neuron

    References
    ----------
    .. [1] Wang, S., Falcone, R., Richmond, B. et al. Attractor dynamics reflect
           decision confidence in macaque prefrontal cortex. Nat Neurosci 26,
           1970–1980 (2023).
    """
    # _t suffix is following notation of paper but applicable across any domain
    ntrials, nbins, nnrns = X_dyn.shape
    delta_t = np.diff(X_dyn, axis=1)  # time derivatives: ntrials x ndomainbins-1 x nnrns

    X_t_flat = np.reshape(X_dyn[:, :-1], (-1, nnrns), order='F')  # skip last bin as no displacement exists for last time point
    delta_t_flat = np.reshape(delta_t, (-1, nnrns), order='F')  # column-major order
    norm_tpts = np.repeat(np.arange(nbins-1), ntrials)

    nbins_domain = nbins - 1 if domainbins is None else domainbins  # downsample domain bins

    potential_pos_t_nrns = []
    grad_pos_t_svm_nrns = []
    hist_nrns = []
    latentedges_nrns = []
    domainedges_nrns = []
    for nnrn in range(nnrns):
        if isinstance(projbins, int):
            nprojbins = projbins
        elif isinstance(projbins[nnrn], int):
            nprojbins = projbins[nnrn]
        else:
            nprojbins = len(projbins[nnrn]) - 1  # bin edges
        # 1D state space binning of time derivatives across domain
        # assumes landscape may morph across domain
        H, bin_edges, _ = binned_statistic_dd(  # (nnrns times projbins) x time
            np.asarray((*X_t_flat.T, norm_tpts)).T, delta_t_flat[:, nnrn], statistic='count',
            bins=(*[projbins if isinstance(projbins, int) else projbins[idx] for idx in range(nnrns)], nbins_domain))
        latentedges= bin_edges[nnrn]
        domainedges = bin_edges[-1]

        grad_pos_t_svm = binned_statistic_dd(
            np.asarray((*X_t_flat.T, norm_tpts)).T, delta_t_flat[:, nnrn], statistic='sum',
            bins=(*[projbins if isinstance(projbins, int) else projbins[idx] for idx in range(nnrns)], nbins_domain)).statistic
        # average derivative, a.k.a. flow/vector field for dynamics underlying
        # population activity
        grad_pos_t_svm = np.divide(grad_pos_t_svm, H, where=H != 0)
        grad_pos_t_svm[H == 0] = np.nan  # crucial to handle division by zero
        # spatial integration via nnancumsum treats nan as zero for cumulative sum
        potential_pos_t = -np.nancumsum(grad_pos_t_svm, axis=nnrn)  # (nnrns times projbins) x domainbins

        # idx_zero_X_t = np.apply_along_axis(
        #     np.searchsorted, 1, np.asarray(bin_edges[:nnrns]), 0)
        # offset = potential_pos_t[tuple(idx_zero_X_t)]  # use potential at X_t = 0 as reference
        # potential_pos_t = potential_pos_t - offset  # potential difference

        if nanborderempty:
            nonzero_mask = H != 0

            # # shape: projbins x 2 x nbins_domain
            # idx_terminal_nonzero = np.asarray(list(zip(*find_terminal_masked_indices(
            #     nonzero_mask, axis=nnrn))))  # unzip the list

            # # along axis 0 set all values from start to idx_first_nonzero to nan
            # if nnrns == 1:
            #     for t in range(H.shape[1]):
            #         potential_pos_t[:idx_terminal_nonzero[t][0], t] = np.nan
            #         potential_pos_t[idx_terminal_nonzero[t][1] + 1:, t] = np.nan
            # else:
            #     for pb in range(nprojbins):
            #         nrndimslices = [pb] * nnrns
            #         nrndimslices.append(0)
            #         for t in range(nbins_domain):
            #             idx_first_nonzero = idx_terminal_nonzero[pb][0][t]
            #             idx_last_nonzero = idx_terminal_nonzero[pb][1][t]

            #             nrndimslices[-1] = t
            #             nrndimslices[nnrn] = slice(0, idx_first_nonzero)

            #             potential_pos_t[tuple(nrndimslices)] = np.nan
            #             nrndimslices[nnrn] = slice(idx_last_nonzero + 1, None)
            #             # print(nrndimslices, potential_pos_t.shape)
            #             potential_pos_t[tuple(nrndimslices)] = np.nan

            #             # if idx_first_nonzero and idx_last_nonzero take up whole axis, then set all values to nan
            #             if idx_first_nonzero == 0 and idx_last_nonzero == nprojbins - 1:
            #                 nrndimslices[nnrn] = slice(None)
            #                 potential_pos_t[tuple(nrndimslices)] = np.nan
            for t in range(nbins_domain):
                nrndimslices = [slice(None)] * nnrns
                nrndimslices.append(t)
                peripheral_zeros_nanmask = ~np.isnan(
                    replace_border_zeros_with_nan(nonzero_mask[tuple(nrndimslices)]))
                peripheral_zeros_nanmask = np.where(
                    peripheral_zeros_nanmask,
                    peripheral_zeros_nanmask,
                    np.nan
                )
                potential_pos_t[tuple(nrndimslices)] *= peripheral_zeros_nanmask

        potential_pos_t_nrns.append(potential_pos_t)
        grad_pos_t_svm_nrns.append(grad_pos_t_svm)
        hist_nrns.append(H)
        latentedges_nrns.append(latentedges)
        domainedges_nrns.append(domainedges)

    potential_pos_t_nrns = np.stack(potential_pos_t_nrns, axis=-1)  # projbins x domainbins x nnrns
    grad_pos_t_svm_nrns = np.stack(grad_pos_t_svm_nrns, axis=-1)  # projbins x domainbins x nnrns
    hist = np.stack(hist_nrns, axis=-1)  # projbins x domainbins x nnrns
    latentedges_nrns = np.stack(latentedges_nrns, axis=-1)  # projbins x nnrns
    domainedges_nrns = np.stack(domainedges_nrns, axis=-1)  # domainbins x nnrns
    potential_pos = np.nanmean(np.asarray([
        np.nanmean(potential_pos_t_nrns[:, :, :, nrn], axis=-1)
        for nrn in range(nnrns)
    ]), axis=0)

    return (
        potential_pos, potential_pos_t_nrns, grad_pos_t_svm_nrns, hist,
        latentedges_nrns,
        domainedges_nrns,
    )
