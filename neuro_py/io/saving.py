import os
from typing import Union

import nelpy as nel
import numpy as np
from scipy.io import savemat


def epoch_to_mat(
    epoch: nel.EpochArray,
    basepath: str,
    epoch_name: str,
    detection_name: Union[None, str] = None,
) -> None:
    """
    Save an EpochArray to a .mat file in the Cell Explorer format.

    Parameters
    ----------
    epoch : nel.EpochArray
        EpochArray to save.
    basepath : str
        Basepath to save the file to.
    epoch_name : str
        Name of the epoch.
    detection_name : Union[None, str], optional
        Name of the detection, by default None.
    """
    filename = os.path.join(
        basepath, os.path.basename(basepath) + "." + epoch_name + ".events.mat"
    )
    data = {}
    data[epoch_name] = {}

    data[epoch_name]["timestamps"] = epoch.data

    # check if only single epoch
    if epoch.data.ndim == 1:
        data[epoch_name]["peaks"] = np.median(epoch.data, axis=0)
    else:
        data[epoch_name]["peaks"] = np.median(epoch.data, axis=1)

    data[epoch_name]["amplitudes"] = []
    data[epoch_name]["amplitudeUnits"] = []
    data[epoch_name]["eventID"] = []
    data[epoch_name]["eventIDlabels"] = []
    data[epoch_name]["eventIDbinary"] = []

    # check if only single epoch
    if epoch.data.ndim == 1:
        data[epoch_name]["duration"] = epoch.data[1] - epoch.data[0]
    else:
        data[epoch_name]["duration"] = epoch.durations

    data[epoch_name]["center"] = data[epoch_name]["peaks"]
    data[epoch_name]["detectorinfo"] = {}
    if detection_name is None:
        data[epoch_name]["detectorinfo"]["detectorname"] = []
    else:
        data[epoch_name]["detectorinfo"]["detectorname"] = detection_name
    data[epoch_name]["detectorinfo"]["detectionparms"] = []
    data[epoch_name]["detectorinfo"]["detectionintervals"] = []

    savemat(filename, data, long_field_names=True)
