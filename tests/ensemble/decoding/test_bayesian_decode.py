import numpy as np
import pytest
from numba.core.errors import TypingError

from neuro_py.ensemble.decoding.bayesian import decode


def test_uniform_prior():
    ct = np.random.rand(10, 5)
    tc = np.random.rand(3, 3, 5)
    occupancy = np.ones((3, 3))
    bin_size_s = 0.1
    p = decode(ct, tc, occupancy, bin_size_s, uniform_prior=True)
    assert p.shape == (10, 3, 3)


def test_non_uniform_prior():
    ct = np.random.rand(10, 5)
    tc = np.random.rand(3, 3, 5)
    occupancy = np.random.rand(3, 3)
    bin_size_s = 0.1
    p = decode(ct, tc, occupancy, bin_size_s)
    assert p.shape == (10, 3, 3)


def test_zero_spike_counts():
    ct = np.zeros((10, 5))
    tc = np.random.rand(3, 3, 5)
    occupancy = np.random.rand(3, 3)
    bin_size_s = 0.1
    p = decode(ct, tc, occupancy, bin_size_s)
    assert p.shape == (10, 3, 3)


def test_zero_occupancy():
    ct = np.random.rand(10, 5)
    tc = np.random.rand(3, 3, 5)
    occupancy = np.zeros((3, 3))
    bin_size_s = 0.1
    p = decode(ct, tc, occupancy, bin_size_s)
    assert p.shape == (10, 3, 3)
    assert np.isnan(p).all()


def test_invalid_input_shapes():
    ct = np.random.rand(10, 5)
    tc = np.random.rand(3, 3)
    occupancy = np.random.rand(3, 3)
    bin_size_s = 0.1
    with pytest.raises(AssertionError):
        decode(ct, tc, occupancy, bin_size_s)


def test_non_numeric_input_values():
    ct = np.random.rand(10, 5)
    tc = np.random.rand(3, 3, 5)
    occupancy = np.random.rand(3, 3)
    bin_size_s = "a"
    with pytest.raises(TypingError):
        decode(ct, tc, occupancy, bin_size_s)


def test_1d_input():
    ct = np.random.rand(10, 5)
    tc = np.random.rand(3, 5)
    occupancy = np.random.rand(3)
    bin_size_s = 0.1
    p = decode(ct, tc, occupancy, bin_size_s)
    assert p.shape == (10, 3)


def test_3d_input():
    ct = np.random.rand(10, 5)
    tc = np.random.rand(3, 3, 3, 5)
    occupancy = np.random.rand(3, 3, 3)
    bin_size_s = 0.1
    p = decode(ct, tc, occupancy, bin_size_s)
    assert p.shape == (10, 3, 3, 3)


def test_4d_input():
    ct = np.random.rand(10, 5)
    tc = np.random.rand(3, 3, 3, 3, 5)
    occupancy = np.random.rand(3, 3, 3, 3)
    bin_size_s = 0.1
    p = decode(ct, tc, occupancy, bin_size_s)
    assert p.shape == (10, 3, 3, 3, 3)
