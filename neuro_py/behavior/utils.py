import numpy as np
from typing import Union

def get_velocity(position:np.ndarray, time:Union[np.ndarray,None]=None) -> np.ndarray:
    if time is None:
        time = np.arange(position.shape[0])
    return np.gradient(position, time, axis=0)


def get_speed(position:np.ndarray, time:Union[np.ndarray,None]=None) -> np.ndarray:
    velocity = get_velocity(position, time=time)
    return np.sqrt(np.sum(velocity**2, axis=1))