import numpy as np


def circular_shift(m: np.ndarray, s: np.ndarray):
    """
    CircularShift - Shift matrix rows or columns circularly.

    Shift each matrix row (or column) circularly by a different amount.

    USAGE

    shifted = circular_shift(m,s)

    Parameters
    ----------
    m: matrix to rotate
    s: shift amount for each row (horizontal vector) or column (vertical vector)

    Returns
    -------
    shifted: matrix m with rows (or columns) circularly shifted by the amounts in s


    adapted from  CircularShift.m  Copyright (C) 2012 by Michaël Zugaro

    """
    # Check number of parameters
    if len(s.shape) != 1:
        raise ValueError("Second parameter is not a vector of integers.")
    if len(m.shape) != 2:
        raise ValueError("First parameter is not a 2D matrix.")

    mm, nm = m.shape
    # if s is 1d array, add dimension
    if len(s.shape) == 1:
        s = s[np.newaxis, :]
    ms, ns = s.shape

    # Check parameter sizes
    if mm != ms and nm != ns:
        raise ValueError("Incompatible parameter sizes.")

    # The algorithm below works along columns; transpose if necessary
    s = -np.ravel(s)
    if ns == 1:
        m = m.T
        mm, nm = m.shape

    # Shift matrix S, where Sij is the vertical shift for element ij
    shift = np.tile(s, (mm, 1))

    # Before we start, each element Mij has a linear index Aij.
    # After circularly shifting the rows, it will have a linear index Bij.
    # We now construct Bij.

    # First, create matrix C where each item Cij = i (row number)
    lines = np.tile(np.arange(mm)[:, np.newaxis], (1, nm))
    # Next, update C so that Cij becomes the target row number (after circular shift)
    lines = np.mod(lines + shift, mm)
    # lines[lines == 0] = mm
    # Finally, transform Cij into a linear index, yielding Bij
    indices = lines + np.tile(np.arange(nm) * mm, (mm, 1))

    # Circular shift (reshape so that it is not transformed into a vector)
    shifted = m.ravel()[(indices.flatten() - 1).astype(int)].reshape(mm, nm)

    # flip matrix right to left
    # shifted = np.fliplr(shifted)

    shifted = np.flipud(shifted)

    # Transpose back if necessary
    if ns == 1:
        shifted = shifted.T

    return shifted


def avgerage_diagonal(mat):
    """
    Average values over all offset diagonals

    Parameters
    ----------
    mat: 2D array

    Returns
    -------
    output: 1D array
            Average values over all offset diagonals

    https://stackoverflow.com/questions/71362928/average-values-over-all-offset-diagonals

    """
    n = mat.shape[0]
    output = np.zeros(n * 2 - 1, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        output[i : i + n] += mat[n - 1 - i]
    output[0:n] /= np.arange(1, n + 1, 1, dtype=np.float64)
    output[n:] /= np.arange(n - 1, 0, -1, dtype=np.float64)
    return output
