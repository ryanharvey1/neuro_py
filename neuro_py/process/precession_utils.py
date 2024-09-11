import multiprocessing

import numba
import numpy as np
import pyfftw
import scipy as sp
from lazy_loader import attach as _attach
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import neuro_py.stats.circ_stats as pcs

__all__ = (
    "corrcc",
    "corrcc_uniform",
    "spatial_phase_precession",
    "fast_acf",
    "acf_power",
    "nonspatial_phase_precession",
)
__getattr__, __dir__, __all__ = _attach(f"{__name__}", submodules=__all__)
del _attach


# These are the core functions used to identify both spatial and non-spatial phase precession
# https://github.com/seqasim/human_precession/blob/main/Precession_utils.py
# https://doi.org/10.1016/j.cell.2021.04.017


def corrcc(alpha1, alpha2, axis=None):
    """
    Circular correlation coefficient for two circular random variables.

    Parameters
    ----------
    alpha1: 1d array
        sample of angles in radians
    alpha2: 1d array
        sample of angles in radians
    axis: int
        correlation coefficient is computed along this dimension
                 (default axis=None, across all dimensions)
    Returns
    ----------
    rho: float
        Circular-circular correlation coefficient
    pval: float
        Circular-circular correlation p-value


    References: [Jammalamadaka2001]_
    Original: https://github.com/circstat/pycircstat
    modified by: Salman Qasim, 11/12/2018

    """
    assert alpha1.shape == alpha2.shape, "Input dimensions do not match."

    n = len(alpha1)

    # center data on circular mean
    alpha1_centered, alpha2_centered = pcs.center(alpha1, alpha2, axis=axis)

    num = np.sum(np.sin(alpha1_centered) * np.sin(alpha2_centered), axis=axis)
    den = np.sqrt(
        np.sum(np.sin(alpha1_centered) ** 2, axis=axis)
        * np.sum(np.sin(alpha2_centered) ** 2, axis=axis)
    )
    # compute correlation coefficient from p. 176
    rho = num / den

    # Modification:
    # significance of this correlation coefficient can be tested using the fact that Z is approx. normal

    l20 = np.mean(np.sin(alpha1_centered) ** 2)
    l02 = np.mean(np.sin(alpha2_centered) ** 2)
    l22 = np.mean((np.sin(alpha1_centered) ** 2) * (np.sin(alpha2_centered) ** 2))
    z = np.sqrt((n * l20 * l02) / l22) * rho
    pval = 2 * (1 - sp.stats.norm.cdf(np.abs(z)))  # two-sided test

    return rho, pval


def corrcc_uniform(alpha1, alpha2, axis=None):
    """
    Circular correlation coefficient for two circular random variables.
    Use if at least one of our variables may be a uniform distribution

    Parameters
    ----------
    alpha1: 1d array
        sample of angles in radians
    alpha2: 1d array
        sample of angles in radians
    axis: int
        correlation coefficient is computed along this dimension
                (default axis=None, across all dimensions)
    Returns
    ----------
    rho: float
        Circular-circular correlation coefficient
    pval: float
        Circular-circular correlation p-value


    References: [Jammalamadaka2001]_

    Original: https://github.com/circstat/pycircstat
    https://github.com/HoniSanders/measure_phaseprec/blob/master/cl_corr.m

    modified by: Salman Qasim, 11/12/2018

    """
    assert alpha1.shape == alpha2.shape, "Input dimensions do not match."

    n = len(alpha1)

    # center data on circular mean
    alpha1_centered, alpha2_centered = pcs.center(alpha1, alpha2, axis=axis)

    # One of the sample means is not well defined due to uniform distribution of data
    # so take the difference of the resultant vector length for the sum and difference
    # of the alphas
    num = pcs.resultant_vector_length(
        alpha1 - alpha2
    ) - pcs.resultant_vector_length(alpha1 + alpha2)
    den = 2 * np.sqrt(
        np.sum(np.sin(alpha1_centered) ** 2, axis=axis)
        * np.sum(np.sin(alpha2_centered) ** 2, axis=axis)
    )
    rho = n * num / den
    # significance of this correlation coefficient can be tested using the fact that Z
    # is approx. normal

    l20 = np.mean(np.sin(alpha1_centered) ** 2)
    l02 = np.mean(np.sin(alpha2_centered) ** 2)
    l22 = np.mean((np.sin(alpha1_centered) ** 2) * (np.sin(alpha2_centered) ** 2))
    z = np.sqrt((n * l20 * l02) / l22) * rho
    pval = 2 * (1 - sp.stats.norm.cdf(np.abs(z)))  # two-sided test

    return rho, pval


def spatial_phase_precession(circ, lin, slope_bounds=[-3 * np.pi, 3 * np.pi]):
    """
    Compute the circular-linear correlation as in: https://pubmed.ncbi.nlm.nih.gov/22487609/

    Parameters
    ----------
    circ : 1d array
        Circular data in radians (i.e. spike phases)
    lin : 1d array
        Linear data (i.e. spike positions)
    slope_bounds: 1d array, or tuple
        Slope range has to be restricted for optimization
    Returns
    ----------
    rho: float
        Circular-linear correlation coefficient
    pval: float
        Circular-linear correlation p-value
    sl: float
        Circular-linear correlation slope
    offs: float
        Circular-linear correlation offset

    Notes
    -----
    This is different from the linear-circular correlation used in:
        https://science.sciencemag.org/content/340/6138/1342

    I've modified the pcs.descriptive.corrcc function above to compute a p-value
        in two different scenarios

    """

    # Get rid of all the nans in this data
    nan_index = np.logical_or(np.isnan(circ), np.isnan(lin))
    circ = circ[~nan_index]
    lin = lin[~nan_index]

    # Make sure there are still valid data
    if np.size(lin) == 0:
        return np.nan, np.nan, np.nan, np.nan

    def myfun1(p):
        return -np.sqrt(
            (np.sum(np.cos(circ - (p * lin))) / len(circ)) ** 2
            + (np.sum(np.sin(circ - (p * lin))) / len(circ)) ** 2
        )

    # finding the optimal slope, note that we have to restrict the range of slopes

    sl = sp.optimize.fminbound(
        myfun1,
        slope_bounds[0] / (np.max(lin) - np.min(lin)),
        slope_bounds[1] / (np.max(lin) - np.min(lin)),
    )

    # calculate offset
    offs = np.arctan2(
        np.sum(np.sin(circ - (sl * lin))), np.sum(np.cos(circ - (sl * lin)))
    )
    # offs = (offs + np.pi) % (2 * np.pi) - np.pi
    offs = np.arctan2(np.sin(offs), np.cos(offs))

    # circular variable derived from the linearization
    linear_circ = np.mod(abs(sl) * lin, 2 * np.pi)

    # # marginal distributions:
    p1, z1 = pcs.rayleigh(circ)
    p2, z2 = pcs.rayleigh(linear_circ)

    # circular-linear correlation:
    if (p1 > 0.5) | (p2 > 0.5):
        # This means at least one of our variables may be a uniform distribution
        rho, pval = corrcc_uniform(circ, linear_circ)
    else:
        rho, pval = corrcc(circ, linear_circ)

    # Assign the correct sign to rho
    if sl < 0:
        rho = -np.abs(rho)
    else:
        rho = np.abs(rho)

    # if offs < 0:
    #     offs = offs + 2 * np.pi
    # if offs > np.pi:
    #     offs = offs - 2 * np.pi

    return rho, pval, sl, offs


@numba.jit(nopython=True)
def pcorrelate(t, u, bins):
    """
    From : https://github.com/OpenSMFS/pycorrelate

    Compute correlation of two arrays of discrete events (Point-process).

    The input arrays need to be values of a point process, such as
    photon arrival times or positions. The correlation is efficiently
    computed on an arbitrary array of lag-bins. As an example, bins can be
    uniformly spaced in log-space and span several orders of magnitudes.
    (you can use :func:`make_loglags` to creat log-spaced bins).
    This function implements the algorithm described in
    `(Laurence 2006) <https://doi.org/10.1364/OL.31.000829>`__.

    Arguments:
        t (array): first array of "points" to correlate. The array needs
            to be monothonically increasing.
        u (array): second array of "points" to correlate. The array needs
            to be monothonically increasing.
        bins (array): bin edges for lags where correlation is computed.
        normalize (bool): if True, normalize the correlation function
            as typically done in FCS using :func:`pnormalize`. If False,
            return the unnormalized correlation function.

    Returns:
        Array containing the correlation of `t` and `u`.
        The size is `len(bins) - 1`.
    """
    nbins = len(bins) - 1

    # Array of counts (histogram)
    counts = np.zeros(nbins, dtype=np.int64)

    # For each bins, imin is the index of first `u` >= of each left bin edge
    imin = np.zeros(nbins, dtype=np.int64)
    # For each bins, imax is the index of first `u` >= of each right bin edge
    imax = np.zeros(nbins, dtype=np.int64)

    # For each ti, perform binning of (u - ti) and accumulate counts in Y
    for ti in t:
        for k, (tau_min, tau_max) in enumerate(zip(bins[:-1], bins[1:])):
            if k == 0:
                j = imin[k]
                # We start by finding the index of the first `u` element
                # which is >= of the first bin edge `tau_min`
                while j < len(u):
                    if u[j] - ti >= tau_min:
                        break
                    j += 1

            imin[k] = j
            if imax[k] > j:
                j = imax[k]
            while j < len(u):
                if u[j] - ti >= tau_max:
                    break
                j += 1
            imax[k] = j
            # Now j is the index of the first `u` element >= of
            # the next bin left edge
        counts += imax - imin
    G = counts / np.diff(bins)
    return G


def fast_acf(counts, width, bin_width, cut_peak=True):
    """
    Super fast ACF function relying on numba (above).

    Parameters
    ----------
    cut_peak : bool
        Whether or not the largest central peak should be replaced for subsequent fitting
    counts : 1d array
        Variable of interest (i.e. spike times or spike phases)
    width: float
        Time window for ACF
    bin_width: float
        Width of bins

    Returns
    ----------
    acf: 1d array
        Counts for ACF
    bins: 1d array
        Lag bins for ACF

    Notes
    -----
    """

    n_b = int(np.ceil(width / bin_width))  # Num. edges per side
    # Define the edges of the bins (including rightmost bin)
    bins = np.linspace(-width, width, 2 * n_b, endpoint=True)
    temp = pcorrelate(counts, counts, np.split(bins, 2)[1])
    acf = np.ones(bins.shape[0] - 1)
    acf[0 : temp.shape[0]] = np.flip(temp)
    acf[temp.shape[0]] = temp[0]
    acf[temp.shape[0] + 1 :] = temp

    if cut_peak:
        acf[np.nanargmax(acf)] = np.sort(acf)[-2]

    return acf, bins


def acf_power(acf, norm=True):
    """
    Compute the power spectrum of the signal by computing the FFT of the autocorrelation.

    Parameters
    ----------
    acf: 1d array
        Counts for ACF

    norm: bool
        To normalize or not

    Returns
    ----------
    psd: 1d array
        Power spectrum

    Notes
    -----
    """

    # Take the FFT
    fft = pyfftw.interfaces.numpy_fft.fft(acf, threads=multiprocessing.cpu_count())

    # Compute the power from the real component squared
    pow = np.abs(fft) ** 2

    # Account for nyquist
    psd = pow[0 : round(pow.shape[0] / 2)]

    # normalize
    if norm:
        psd = psd / np.trapz(psd)

    return psd


def nonspatial_phase_precession(
    unwrapped_spike_phases: np.ndarray,
    width: float = 4 * 2 * np.pi,
    bin_width: float = np.pi / 3,
    cut_peak: bool = True,
    norm: bool = True,
    psd_lims: list = [0.65, 1.55],
    upsample: int = 4,
    smooth_sigma=1,
):
    """
    Compute the nonspatial spike-LFP relationship modulation index.

    Parameters
    ----------
    unwrapped_spike_phases : 1d array
        Spike phases that have been linearly unwrapped
    width: float
        Time window for ACF in cycles (default = 4 cycles)
    bin_width: float
        Width of bins in radians (default = 60 degrees)
    cut_peak : bool
        Whether or not the largest central peak should be replaced for
        subsequent fitting
    norm: bool
        To normalize the ACF or not
    psd_lims: list
        Limits of the PSD to consider for peak finding (default = [0.65, 1.55])
    Returns
    ----------
    max_freq: float
        Relative spike-LFP frequency of PSD peak

    MI: float
        Modulation index of non-spatial phase relationship


    Notes
    -----
    """

    frequencies = (
        (np.arange(2 * (width // bin_width) - 1))
        * (2 * np.pi)
        / (2 * width - bin_width)
    )

    frequencies = np.interp(
        np.arange(0, len(frequencies), 1 / upsample),
        np.arange(0, len(frequencies)),
        frequencies,
    )

    freqs_of_interest = np.intersect1d(
        np.where(frequencies > psd_lims[0]), np.where(frequencies < psd_lims[1])
    )

    acf, _ = fast_acf(unwrapped_spike_phases, width, bin_width, cut_peak=cut_peak)
    psd = acf_power(acf, norm=norm)

    # upsample 2x psd
    psd = np.interp(np.arange(0, len(psd), 1 / upsample), np.arange(0, len(psd)), psd)
    # smooth psd with gaussian filter
    psd = gaussian_filter1d(psd, smooth_sigma)

    # FIND ALL LOCAL MAXIMA IN WINDOW OF INTEREST
    all_peaks = find_peaks(psd[freqs_of_interest], None)[0]

    # make sure there is a peak
    if ~np.any(all_peaks):
        return (
            np.nan,
            np.nan,
            psd[freqs_of_interest],
            frequencies[freqs_of_interest],
            acf,
        )

    max_peak = np.max(psd[freqs_of_interest][all_peaks])
    max_idx = [all_peaks[np.argmax(psd[freqs_of_interest][all_peaks])]]
    max_freq = frequencies[freqs_of_interest][max_idx]
    MI = max_peak / np.trapz(psd[freqs_of_interest])

    return max_freq, MI, psd[freqs_of_interest], frequencies[freqs_of_interest], acf


# def nonspatial_phase_precession_v2(
#     spike_cycles: np.ndarray,
#     width: float = 4 * 2 * np.pi,
#     bin_width: float = np.pi / 3,
#     cut_peak: bool = True,
#     norm: bool = True,
#     psd_lims: list = [0.65, 1.55],
#     upsample: int = 4,
#     smooth_sigma=1
# ):

#     """
#     Compute the nonspatial spike-LFP relationship modulation index.

#     Parameters
#     ----------
#     spike_cycles : 1d array
#         Spike phases that have been linearly unwrapped (units=cycles)
#     width: float
#         Time window for ACF in cycles (default = 4 cycles)
#     bin_width: float
#         Width of bins in radians (default = 60 degrees)
#     cut_peak : bool
#         Whether or not the largest central peak should be replaced for
#         subsequent fitting
#     norm: bool
#         To normalize the ACF or not
#     psd_lims: list
#         Limits of the PSD to consider for peak finding (default = [0.65, 1.55])
#     Returns
#     ----------
#     max_freq: float
#         Relative spike-LFP frequency of PSD peak

#     MI: float
#         Modulation index of non-spatial phase relationship


#     Notes
#     -----
#     """

#     frequencies = (
#         (np.arange(2 * (width // bin_width) - 1))
#         * (2 * np.pi)
#         / (2 * width - bin_width)
#     )

#     frequencies = np.interp(
#         np.arange(0, len(frequencies), 1/upsample), np.arange(0, len(frequencies)), frequencies
#     )

#     freqs_of_interest = np.intersect1d(
#         np.where(frequencies > psd_lims[0]), np.where(frequencies < psd_lims[1])
#     )

#     acf, _ = fast_acf(unwrapped_spike_phases, width, bin_width, cut_peak=cut_peak)
#     psd = acf_power(acf, norm=norm)

#     # upsample 2x psd
#     psd = np.interp(np.arange(0, len(psd), 1/upsample), np.arange(0, len(psd)), psd)
#     # smooth psd with gaussian filter
#     psd = gaussian_filter1d(psd, smooth_sigma)

#     # FIND ALL LOCAL MAXIMA IN WINDOW OF INTEREST
#     all_peaks = find_peaks(psd[freqs_of_interest], None)[0]

#     # make sure there is a peak
#     if ~np.any(all_peaks):
#         return np.nan, np.nan, psd[freqs_of_interest], frequencies[freqs_of_interest]

#     max_peak = np.max(psd[freqs_of_interest][all_peaks])
#     max_idx = [all_peaks[np.argmax(psd[freqs_of_interest][all_peaks])]]
#     max_freq = frequencies[freqs_of_interest][max_idx]
#     MI = max_peak / np.trapz(psd[freqs_of_interest])

#     return max_freq, MI, psd[freqs_of_interest], frequencies[freqs_of_interest], acf
