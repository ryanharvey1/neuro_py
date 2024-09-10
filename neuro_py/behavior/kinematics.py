import numpy as np

from typing import Union

from lazy_loader import attach as _attach

__all__ = (
    "get_velocity",
    "get_speed",
)
__getattr__, __dir__, __all__ = _attach(f"{__name__}", submodules=__all__)
del _attach


def get_velocity(position:np.ndarray, time:Union[np.ndarray,None]=None) -> np.ndarray:
    if time is None:
        time = np.arange(position.shape[0])
    return np.gradient(position, time, axis=0)


def get_speed(position:np.ndarray, time:Union[np.ndarray,None]=None) -> np.ndarray:
    velocity = get_velocity(position, time=time)
    return np.sqrt(np.sum(velocity**2, axis=1))
