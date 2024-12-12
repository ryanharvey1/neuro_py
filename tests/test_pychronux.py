from neuro_py.process import pychronux as px
import numpy as np
import pytest
from scipy.signal.windows import dpss


def test_pychronux():
    # Test the function
    assert px is not None


def test_getfgrid():
    Fs = 1000
    nfft = 1024
    fpass = [0, 500]

    expected_df = Fs / nfft
    expected_f = np.arange(0, Fs + expected_df, expected_df)
    expected_f = expected_f[0:nfft]
    expected_findx = (expected_f >= fpass[0]) & (expected_f <= fpass[-1])
    expected_f = expected_f[expected_findx]

    f, findx = px.getfgrid(Fs, nfft, fpass)

    assert np.allclose(
        f, expected_f
    ), "Frequency vector does not match expected values."
    assert np.array_equal(
        findx, expected_findx
    ), "Index array does not match expected values."


def test_dpsschk():
    tapers = (2.5, 4)
    N = 256
    Fs = 1000.0

    # Generate expected DPSS tapers using scipy
    expected_tapers, _ = dpss(
        N, NW=tapers[0], Kmax=tapers[1], sym=False, return_ratios=True
    )
    expected_tapers = expected_tapers * np.sqrt(Fs)
    expected_tapers = expected_tapers.T

    # Call the function
    result_tapers = px.dpsschk(tapers, N, Fs)

    # Check if the result matches the expected tapers
    assert np.allclose(
        result_tapers, expected_tapers
    ), "DPSS tapers do not match expected values."


def test_get_tapers():
    N = 256
    bandwidth = 4.0
    fs = 1000.0
    min_lambda = 0.95

    # Generate expected DPSS tapers using scipy
    NW = bandwidth * N / fs
    K = int(np.ceil(2 * NW)) - 1
    expected_tapers, expected_lambdas = dpss(
        N, NW=NW, Kmax=K, sym=False, norm=2, return_ratios=True
    )
    mask = expected_lambdas > min_lambda
    expected_tapers = expected_tapers[mask]
    expected_lambdas = expected_lambdas[mask]

    # Ensure n_tapers does not exceed the number of tapers that satisfy the criteria
    n_tapers = min(5, expected_tapers.shape[0])

    if n_tapers is not None:
        expected_tapers = expected_tapers[:n_tapers]
        expected_lambdas = expected_lambdas[:n_tapers]

    # Call the function
    tapers, lambdas = px.get_tapers(
        N, bandwidth, fs=fs, min_lambda=min_lambda, n_tapers=n_tapers
    )

    # Check if the result matches the expected tapers and lambdas
    assert np.allclose(tapers, expected_tapers), "Tapers do not match expected values."
    assert np.allclose(
        lambdas, expected_lambdas
    ), "Lambdas do not match expected values."


def test_get_tapers_value_error():
    N = 256
    bandwidth = 0.1
    fs = 1000.0
    min_lambda = 0.95

    with pytest.raises(ValueError, match="Not enough tapers"):
        px.get_tapers(N, bandwidth, fs=fs, min_lambda=min_lambda)

    bandwidth = 4.0
    n_tapers = 10

    with pytest.raises(
        ValueError,
        match="None of the tapers satisfied the minimum energy concentration",
    ):
        px.get_tapers(N, bandwidth, fs=fs, min_lambda=1.0)

    with pytest.raises(ValueError, match="'n_tapers' of 10 is greater than the"):
        px.get_tapers(N, bandwidth, fs=fs, min_lambda=min_lambda, n_tapers=n_tapers)


if __name__ == "__main__":
    test_get_tapers()
#     pytest.main()
