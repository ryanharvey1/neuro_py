import warnings
from collections import namedtuple

import numpy as np
from decorator import decorator
from scipy import stats

# minimal version of pycircstat from https://github.com/circstat/pycircstat/tree/master

CI = namedtuple("confidence_interval", ["lower", "upper"])


def nd_bootstrap(data, iterations, axis=None, strip_tuple_if_one=True):
    """
    Bootstrap iterator for several n-dimensional data arrays.

    :param data: Iterable containing the data arrays
    :param iterations: Number of bootstrap iterations.
    :param axis: Bootstrapping is performed along this axis.
    """
    shape0 = data[0].shape
    if axis is None:
        axis = 0
        data = [d.ravel() for d in data]

    n = len(data[0].shape)
    K = len(data)
    data0 = []

    if axis is not None:
        m = data[0].shape[axis]
        to = tuple([axis]) + tuple(range(axis)) + tuple(range(axis + 1, n))
        fro = tuple(range(1, axis + 1)) + (0,) + tuple(range(axis + 1, n))
        for i in range(K):
            data0.append(data[i].transpose(to))

        for i in range(iterations):
            idx = np.random.randint(m, size=(m,))
            if len(data) == 1 and strip_tuple_if_one:
                yield (
                    data0[0][np.ix_(idx), ...].squeeze().transpose(fro).reshape(shape0)
                )
            else:
                yield tuple(
                    a[np.ix_(idx), ...].squeeze().transpose(fro).reshape(shape0)
                    for a in data0
                )


def mod2pi(f):
    """
    Decorator to apply modulo 2*pi on the output of the function.

    The decorated function must either return a tuple of numpy.ndarrays or a
    numpy.ndarray itself.
    """

    def wrapper(f, *args, **kwargs):
        ret = f(*args, **kwargs)

        if isinstance(ret, tuple):
            ret2 = []
            for r in ret:
                if isinstance(r, np.ndarray) or np.isscalar(r):
                    ret2.append(r % (2 * np.pi))
                elif isinstance(r, CI):
                    ret2.append(CI(r.lower % (2 * np.pi), r.upper % (2 * np.pi)))
                else:
                    raise TypeError("Type not known!")
            return tuple(ret2)
        elif isinstance(ret, np.ndarray) or np.isscalar(ret):
            return ret % (2 * np.pi)
        else:
            raise TypeError("Type not known!")

    return decorator(wrapper, f)


class bootstrap:
    """
    Decorator to implement bootstrapping. It looks for the arguments ci, axis,
    and bootstrap_iter to determine the proper parameters for bootstrapping.
    The argument scale determines whether the percentile is taken on a circular
    scale or on a linear scale.

    :param no_bootstrap: the number of arguments that are bootstrapped
                        (e.g. for correlation it would be two, for median it
                        would be one)
    :param scale: linear or ciruclar scale (default is 'linear')
    """

    def __init__(self, no_bootstrap, scale="linear"):
        self.no_boostrap = no_bootstrap
        self.scale = scale

    def _get_var(self, f, what, default, args, kwargs, remove=False):
        varnames = f.__code__.co_varnames

        if what in varnames:
            what_idx = varnames.index(what)
        else:
            raise ValueError(
                "Function %s does not have variable %s." % (f.__name__, what)
            )

        if len(args) >= what_idx + 1:
            val = args[what_idx]
            if remove:
                args[what_idx] = default
        else:
            val = default

        return val

    def __call__(self, f):

        def wrapper(f, *args, **kwargs):
            args = list(args)
            ci = self._get_var(f, "ci", None, args, kwargs, remove=True)
            bootstrap_iter = self._get_var(
                f, "bootstrap_iter", None, args, kwargs, remove=True
            )
            axis = self._get_var(f, "axis", None, args, kwargs)

            alpha = args[: self.no_boostrap]
            args0 = args[self.no_boostrap :]

            if bootstrap_iter is None:
                bootstrap_iter = (
                    alpha[0].shape[axis] if axis is not None else alpha[0].size
                )

            r0 = f(*(alpha + args0), **kwargs)
            if ci is not None:
                r = np.asarray(
                    [
                        f(*(list(a) + args0), **kwargs)
                        for a in nd_bootstrap(
                            alpha, bootstrap_iter, axis=axis, strip_tuple_if_one=False
                        )
                    ]
                )

                if self.scale == "linear":
                    ci_low, ci_high = np.percentile(
                        r, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100], axis=0
                    )
                elif self.scale == "circular":
                    ci_low, ci_high = percentile(
                        r,
                        [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100],
                        q0=(r0 + np.pi) % (2 * np.pi),
                        axis=0,
                    )
                else:
                    raise ValueError("Scale %s not known!" % (self.scale,))
                return r0, CI(ci_low, ci_high)
            else:
                return r0

        return decorator(wrapper, f)


@mod2pi
@bootstrap(1, "circular")
def percentile(alpha, q, q0, axis=None, ci=None, bootstrap_iter=None):
    """
    Computes circular percentiles

    :param alpha: array with circular samples
    :param q: percentiles in [0,100] (single number or iterable)
    :param q0: value of the 0 percentile
    :param axis: percentiles will be computed along this axis.
                 If None percentiles will be computed over the entire array
    :param ci: if not None, confidence level is bootstrapped
    :param bootstrap_iter: number of bootstrap iterations
                           (number of samples if None)

    :return: percentiles

    """
    if axis is None:
        alpha = (alpha.ravel() - q0) % (2 * np.pi)
    else:
        if len(q0.shape) == len(alpha.shape) - 1:
            reshaper = tuple(
                slice(None, None) if i != axis else np.newaxis
                for i in range(len(alpha.shape))
            )
            q0 = q0[reshaper]
        elif not len(q0.shape) == len(alpha.shape):
            raise ValueError("Dimensions of start and alpha are inconsistent!")

        alpha = (alpha - q0) % (2 * np.pi)

    ret = []
    if axis is not None:
        selector = tuple(
            slice(None) if i != axis else 0 for i in range(len(alpha.shape))
        )
        q0 = q0[selector]

    for qq in np.atleast_1d(q):
        ret.append(np.percentile(alpha, qq, axis=axis) + q0)

    if not hasattr(q, "__iter__"):  # if q is not some sort of list, array, etc
        return np.asarray(ret).squeeze()
    else:
        return np.asarray(ret)


def _complex_mean(alpha, w=None, axis=None, axial_correction=1):
    if w is None:
        w = np.ones_like(alpha)
    alpha = np.asarray(alpha)

    assert w.shape == alpha.shape, (
        "Dimensions of data "
        + str(alpha.shape)
        + " and w "
        + str(w.shape)
        + " do not match!"
    )

    return (w * np.exp(1j * alpha * axial_correction)).sum(axis=axis) / np.sum(
        w, axis=axis
    )


@bootstrap(1, "linear")
def resultant_vector_length(
    alpha, w=None, d=None, axis=None, axial_correction=1, ci=None, bootstrap_iter=None
):
    """
    Computes mean resultant vector length for circular data.

    This statistic is sometimes also called vector strength.

    :param alpha: sample of angles in radians
    :param w: number of incidences in case of binned angle data
    :param ci: ci-confidence limits are computed via bootstrapping,
               default None.
    :param d: spacing of bin centers for binned data, if supplied
              correction factor is used to correct for bias in
              estimation of r, in radians (!)
    :param axis: compute along this dimension, default is None
                 (across all dimensions)
    :param axial_correction: axial correction (2,3,4,...), default is 1
    :param bootstrap_iter: number of bootstrap iterations
                          (number of samples if None)
    :return: mean resultant length

    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_
    """
    if axis is None:
        axis = 0
        alpha = alpha.ravel()
        if w is not None:
            w = w.ravel()

    cmean = _complex_mean(alpha, w=w, axis=axis, axial_correction=axial_correction)

    # obtain length
    r = np.abs(cmean)

    # for data with known spacing, apply correction factor to correct for bias
    # in the estimation of r (see Zar, p. 601, equ. 26.16)
    if d is not None:
        if axial_correction > 1:
            warnings.warn("Axial correction ignored for bias correction.")
        r *= d / 2 / np.sin(d / 2)
    return r


# defines synonym for resultant_vector_length
vector_strength = resultant_vector_length


def mean_ci_limits(alpha, ci=0.95, w=None, d=None, axis=None):
    """
    Computes the confidence limits on the mean for circular data.

    :param alpha: sample of angles in radians
    :param ci: ci-confidence limits are computed, default 0.95
    :param w: number of incidences in case of binned angle data
    :param d: spacing of bin centers for binned data, if supplied
              correction factor is used to correct for bias in
              estimation of r, in radians (!)
    :param axis: compute along this dimension, default is None
                 (across all dimensions)

    :return: confidence limit width d; mean +- d yields upper/lower
             (1-xi)% confidence limit

    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_
    """

    if w is None:
        w = np.ones_like(alpha)

    assert alpha.shape == w.shape, "Dimensions of data and w do not match!"

    r = np.atleast_1d(resultant_vector_length(alpha, w=w, d=d, axis=axis))
    n = np.atleast_1d(np.sum(w, axis=axis))

    R = n * r
    c2 = stats.chi2.ppf(ci, df=1)

    t = np.NaN * np.empty_like(r)

    idx = (r < 0.9) & (r > np.sqrt(c2 / 2 / n))
    t[idx] = np.sqrt(
        (2 * n[idx] * (2 * R[idx] ** 2 - n[idx] * c2)) / (4 * n[idx] - c2)
    )  # eq. 26.24

    idx2 = r >= 0.9
    t[idx2] = np.sqrt(
        n[idx2] ** 2 - (n[idx2] ** 2 - R[idx2] ** 2) * np.exp(c2 / n[idx2])
    )  # equ. 26.25

    if not np.all(idx | idx2):
        raise UserWarning(
            """Requirements for confidence levels not met:
                CI limits require a certain concentration of the data around the mean"""
        )

    return np.squeeze(np.arccos(t / R))


@mod2pi
def mean(alpha, w=None, ci=None, d=None, axis=None, axial_correction=1):
    """
    Compute mean direction of circular data.

    :param alpha: circular data
    :param w: 	 weightings in case of binned angle data
    :param ci: if not None, the upper and lower 100*ci% confidence
               interval is returned as well
    :param d: spacing of bin centers for binned data, if supplied
              correction factor is used to correct for bias in
              estimation of r, in radians (!)
    :param axis: compute along this dimension, default is None
                 (across all dimensions)
    :param axial_correction: axial correction (2,3,4,...), default is 1
    :return: circular mean if ci=None, or circular mean as well as lower and
             upper confidence interval limits

    Example
    >>> import numpy as np
    >>> data = 2*np.pi*np.random.rand(10)
    >>> mu, (ci_l, ci_u) = mean(data, ci=0.95)

    """

    cmean = _complex_mean(alpha, w=w, axis=axis, axial_correction=axial_correction)

    mu = np.angle(cmean) / axial_correction

    if ci is None:
        return mu
    else:
        if axial_correction > 1:  # TODO: implement CI for axial correction
            warnings.warn("Axial correction ignored for confidence intervals.")
        t = mean_ci_limits(alpha, ci=ci, w=w, d=d, axis=axis)
        return mu, CI(mu - t, mu + t)


@mod2pi
def center(*args, **kwargs):
    """
    Centers the data on its circular mean.

    Each non-keyword argument is another data array that is centered.

    :param axis: the mean is computed along this dimension (default axis=None).
                **Must be used as a keyword argument!**
    :return: tuple of centered data arrays

    """

    axis = kwargs.pop("axis", None)
    if axis is None:
        axis = 0
        args = [a.ravel() for a in args]

    reshaper = tuple(
        slice(None, None) if i != axis else np.newaxis
        for i in range(len(args[0].shape))
    )
    if len(args) == 1:
        return args[0] - mean(args[0], axis=axis)
    else:
        return tuple(
            [
                a - mean(a, axis=axis)[reshaper]
                for a in args
                if isinstance(a, np.ndarray)
            ]
        )


def get_var(f, varnames, args, kwargs):
    fvarnames = f.__code__.co_varnames

    var_idx = []
    kwar_keys = []
    for varname in varnames:
        if varname in fvarnames:
            var_pos = fvarnames.index(varname)
        else:
            raise ValueError(
                "Function %s does not have variable %s." % (f.__name__, varnames)
            )
        if len(args) >= var_pos + 1:
            var_idx.append(var_pos)
        elif varname in kwargs:
            kwar_keys.append(varname)
        else:
            raise ValueError("%s was not specified in  %s." % (varnames, f.__name__))

    return var_idx, kwar_keys


class swap2zeroaxis:
    """
    This decorator is best explained by an example::

        @swap2zeroaxis(['x','y'], [0, 1])
        def dummy(x,y,z, axis=None):
            return np.mean(x[::2,...], axis=0), np.mean(y[::2, ...], axis=0), z

    This creates a new function that

    - either swaps the axes axis to zero for the arguments x and y if axis
      is specified in dummy or ravels x and y
    - swaps back the axes from the output arguments 0 and 1. Here it is
      assumed that the outputs lost one dimension during the function
      (e.g. like numpy.mean(x, axis=1) looses one axis).
    """

    def __init__(self, inputs, out_idx):
        self.inputs = inputs
        self.out_idx = out_idx

    def __call__(self, f):

        def _deco(f, *args, **kwargs):

            to_swap_idx, to_swap_keys = get_var(f, self.inputs, args, kwargs)
            args = list(args)

            # extract axis parameter
            try:
                axis_idx, axis_kw = get_var(f, ["axis"], args, kwargs)
                if len(axis_idx) == 0 and len(axis_kw) == 0:
                    axis = None
                else:
                    if len(axis_idx) > 0:
                        axis, args[axis_idx[0]] = args[axis_idx[0]], 0
                    else:
                        axis, kwargs[axis_kw[0]] = kwargs[axis_kw[0]], 0
            except ValueError:
                axis = None

            # adjust axes or flatten
            if axis is not None:
                for i in to_swap_idx:
                    if args[i] is not None:
                        args[i] = args[i].swapaxes(0, axis)
                for k in to_swap_keys:
                    if kwargs[k] is not None:
                        kwargs[k] = kwargs[k].swapaxes(0, axis)
            else:
                for i in to_swap_idx:
                    if args[i] is not None:
                        args[i] = args[i].ravel()
                for k in to_swap_keys:
                    if kwargs[k] is not None:
                        kwargs[k] = kwargs[k].ravel()

            # compute function
            outputs = f(*args, **kwargs)

            # swap everything back into place
            if len(self.out_idx) > 0 and axis is not None:
                if isinstance(outputs, tuple):
                    outputs = list(outputs)
                    for i in self.out_idx:
                        outputs[i] = (
                            outputs[i][np.newaxis, ...].swapaxes(0, axis).squeeze()
                        )

                    return tuple(outputs)
                else:
                    if self.out_idx != [0]:
                        raise ValueError(
                            "Single output argument and out_idx \
                                         != [0] are inconsistent!"
                        )
                    return outputs[np.newaxis, ...].swapaxes(0, axis).squeeze()
            else:
                return outputs

        return decorator(_deco, f)


@swap2zeroaxis(["alpha", "w"], [0, 1])
def rayleigh(alpha, w=None, d=None, axis=None):
    """
    Computes Rayleigh test for non-uniformity of circular data.

    H0: the population is uniformly distributed around the circle
    HA: the populatoin is not distributed uniformly around the circle

    Assumption: the distribution has maximally one mode and the data is
    sampled from a von Mises distribution!

    :param alpha: sample of angles in radian
    :param w:       number of incidences in case of binned angle data
    :param d:     spacing of bin centers for binned data, if supplied
                  correction factor is used to correct for bias in
                  estimation of r
    :param axis:  compute along this dimension, default is None
                  if axis=None, array is raveled
    :return pval: two-tailed p-value
    :return z:    value of the z-statistic

    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_
    """

    if w is None:
        w = np.ones_like(alpha)

    assert w.shape == alpha.shape, "Dimensions of alpha and w must match"

    r = resultant_vector_length(alpha, w=w, d=d, axis=axis)
    n = np.sum(w, axis=axis)

    # compute Rayleigh's R (equ. 27.1)
    R = n * r

    # compute Rayleigh's z (equ. 27.2)
    z = R**2 / n

    # compute p value using approxation in Zar, p. 617
    pval = np.exp(np.sqrt(1 + 4 * n + 4 * (n**2 - R**2)) - (1 + 2 * n))

    return pval, z
