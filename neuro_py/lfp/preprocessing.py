import os
import warnings
from typing import List, Tuple, Union

import nelpy as nel
import numpy as np

from neuro_py.process import intervals


def clean_lfp(
    lfp: Union[nel.AnalogSignalArray, np.ndarray],
    t: np.ndarray = None,
    thresholds: Tuple[float, float] = (5, 10),
    artifact_time_expand: Tuple[float, float] = (0.25, 0.1),
    return_bad_intervals: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, nel.EpochArray]]:
    """
    Remove artefacts and noise from a local field potential (LFP) signal.

    Parameters
    ----------
    lfp : nel.AnalogSignalArray
        The LFP signal to be cleaned. Single signal only.
    thresholds : tuple of float, optional
        A tuple of two thresholds for detecting artefacts and noise. The first threshold is used to detect large global
        artefacts by finding values in the z-scored LFP signal that deviate by more than the threshold number of sigmas
        from the mean. The second threshold is used to detect noise by finding values in the derivative of the z-scored
        LFP signal that are greater than the threshold. Default is (5, 10).
    artifact_time_expand : tuple of float, optional
        A tuple of two time intervals around detected artefacts and noise. The first interval is used to expand the detected
        large global artefacts. The second interval is used to expand the detected noise. Default is (0.25, 0.1).
    return_bad_intervals : bool, optional
        If True, also returns intervals of artefacts and noise as an `nel.EpochArray`. Default is False.

    Returns
    -------
    Union[np.ndarray, Tuple[np.ndarray, nel.EpochArray]]
        The cleaned LFP signal. If `return_bad_intervals` is True, also returns an `nel.EpochArray`
        representing the intervals of artefacts and noise.

    Notes
    -----
    Based on https://github.com/ayalab1/neurocode/blob/master/lfp/CleanLFP.m by Ralitsa Todorova

    Examples
    --------
    >>> lfp = nel.AnalogSignalArray(data=np.random.randn(1250), timestamps=np.arange(1250)/1250)
    >>> clean_lfp(lfp)
    array([-1.73104885,  1.08192036,  1.40332741, ..., -2.78671212,
        -1.63661574, -1.10868426])
    """
    threshold1 = thresholds[0]  # in sigmas deviating from the mean
    aroundArtefact1 = artifact_time_expand[
        0
    ]  # interval to expand large global artefacts

    threshold2 = thresholds[1]  # for derivative of z-scored signal
    aroundArtefact2 = artifact_time_expand[1]  # interval to expand detected noise

    if isinstance(lfp, nel.AnalogSignalArray):
        t = lfp.time  # time points of LFP signal
        values = lfp.copy().data.flatten()  # values of LFP signal
        z = lfp.zscore().data.flatten()  # z-scored values of LFP signal
    elif isinstance(lfp, np.ndarray):
        if t is None:
            raise ValueError("t must be provided when lfp is np.ndarray")
        values = lfp.flatten()
        z = (values - np.mean(values)) / np.std(values)
    else:
        raise ValueError("lfp must be nel.AnalogSignalArray or np.ndarray")

    d = np.append(np.diff(z), 0)  # derivative of z-scored LFP signal

    # Detect large global artefacts [0]
    artefactInterval = t[
        np.array(intervals.find_interval(np.abs(z) > threshold1), dtype=int)
    ]
    artefactInterval = nel.EpochArray(artefactInterval)
    if not artefactInterval.isempty:
        artefactInterval = artefactInterval.expand(aroundArtefact1)

    # Find noise using the derivative of the z-scored signal [1]
    noisyInterval = t[
        np.array(intervals.find_interval(np.abs(d) > threshold2), dtype=int)
    ]
    noisyInterval = nel.EpochArray(noisyInterval)
    if not noisyInterval.isempty:
        noisyInterval = noisyInterval.expand(aroundArtefact2)

    # Combine intervals for artefacts and noise
    bad = (artefactInterval | noisyInterval).merge()

    if bad.isempty:
        return values

    # Find timestamps within intervals for artefacts and noise
    in_interval = intervals.in_intervals(t, bad.data)

    # Interpolate values for timestamps within intervals for artefacts and noise
    values[in_interval] = np.interp(
        t[in_interval], t[~in_interval], values[~in_interval]
    )

    return (values, bad) if return_bad_intervals else values



