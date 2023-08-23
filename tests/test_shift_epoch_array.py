import numpy as np
from neuro_py.process.intervals import shift_epoch_array
import nelpy as nel

def test_shift_epoch_array():
    # Create some example data for testing
    epoch_data = np.array([[1, 3], [5, 8], [10, 12]])
    epoch_shift_data = np.array([[2, 4], [7, 9]])

    # Create EpochArray instances
    epoch = nel.EpochArray(epoch_data)
    epoch_shift = nel.EpochArray(epoch_shift_data)

    # Shift the epoch array using the function
    epoch_shifted = shift_epoch_array(epoch, epoch_shift)

    # Test if the shifted epoch array has the expected intervals
    expected_shifted_data = np.array([[0, 1], [2, 3]])
    assert(np.all(epoch_shifted.data == expected_shifted_data))

    # Test if the shifted epoch array has the correct domain
    expected_domain = nel.EpochArray([-np.inf, np.inf])
    assert(np.all(epoch_shifted.domain.data == expected_domain.domain.data))

