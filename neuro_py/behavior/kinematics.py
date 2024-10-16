from typing import Union

import numpy as np


def get_velocity(
    position: np.ndarray, time: Union[np.ndarray, None] = None
) -> np.ndarray:
    """
    Computes the velocity from position data.

    If time is not provided, it assumes a constant time step between position
    samples. The velocity is calculated as the gradient of the position with
    respect to time along the first axis.

    Parameters
    ----------
    position : np.ndarray
        An array of position data. This can be 1D (for single-dimensional positions)
        or 2D (for multi-dimensional positions, e.g., x and y coordinates over time).
    time : Union[np.ndarray, None], optional
        An array of time values corresponding to the position data. If None,
        the function assumes a constant time step. Default is None.

    Returns
    -------
    np.ndarray
        An array of velocity values, where each velocity is the rate of change of
        position with respect to time.

    Examples
    --------
    >>> position = np.array([0, 1, 4, 9, 16])
    >>> get_velocity(position)
    array([1., 2., 4., 6., 7.])

    >>> position = np.array([[0, 0], [1, 1], [4, 4], [9, 9], [16, 16]])
    >>> time = np.array([0, 1, 2, 3, 4])
    >>> get_velocity(position, time)
    array([[1., 1.],
        [2., 2.],
        [4., 4.],
        [6., 6.],
        [7., 7.]])
    """
    if time is None:
        time = np.arange(position.shape[0])
    return np.gradient(position, time, axis=0)


def get_speed(position: np.ndarray, time: Union[np.ndarray, None] = None) -> np.ndarray:
    """
    Computes the speed from position data.

    Speed is the magnitude of the velocity vector at each time point. If time is
    not provided, it assumes a constant time step between position samples.

    Parameters
    ----------
    position : np.ndarray
        An array of position data. This can be 1D (for single-dimensional positions)
        or 2D (for multi-dimensional positions, e.g., x and y coordinates over time).
    time : Union[np.ndarray, None], optional
        An array of time values corresponding to the position data. If None,
        the function assumes a constant time step. Default is None.

    Returns
    -------
    np.ndarray
        An array of speed values, where each speed is the magnitude of the velocity
        at the corresponding time point.

    Examples
    --------
    >>> position = np.array([0, 1, 4, 9, 16])
    >>> get_speed(position)
    array([1.41421356, 2.82842712, 5.65685425, 8.48528137, 9.89949494])

    >>> position = np.array([[0, 0], [1, 1], [4, 4], [9, 9], [16, 16]])
    >>> time = np.array([0, 1, 2, 3, 4])
    >>> get_speed(position, time)
    array([1.41421356, 2.82842712, 5.65685425, 8.48528137, 9.89949494])
    """
    velocity = get_velocity(position, time=time)
    return np.sqrt(np.sum(velocity**2, axis=1))
