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



def test_mtfftpt():
    # Test Case 1: Basic functionality
    data = np.array([0.1, 0.2, 0.4, 0.5])  # Spike times
    tapers = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
        [0.7, 0.8],
    ])  # DPSS tapers
    nfft = 8
    t = np.linspace(0, 1, 4)  # Time vector
    f = np.linspace(0, 10, 5)  # Frequency vector

    # Update findx to match the size of nfft
    findx = np.zeros(nfft, dtype=bool)
    findx[:len(f)] = True

    # Call the function
    J, Msp, Nsp = px.mtfftpt(data, tapers, nfft, t, f, findx)

    # Assertions
    assert J.shape == (5, 2)  # Ensure correct shape of J
    assert np.isclose(Msp, 4 / (t[-1] - t[0]))  # Verify mean spike rate
    assert Nsp == 4  # Verify total spike count

    # Test Case 2: Empty spike times
    data_empty = np.array([])
    J_empty, Msp_empty, Nsp_empty = px.mtfftpt(data_empty, tapers, nfft, t, f, findx)

    # Assertions
    assert J_empty.shape == (5, 2)
    assert np.allclose(J_empty, 0)  # J should be zeros
    assert Msp_empty == 0  # Mean spike rate should be zero
    assert Nsp_empty == 0  # Total spike count should be zero

    # Test Case 3: No valid spike times in range
    data_out_of_range = np.array([-0.5, 1.5])
    J_out, Msp_out, Nsp_out = px.mtfftpt(data_out_of_range, tapers, nfft, t, f, findx)

    # Assertions
    assert J_out.shape == (5, 2)
    assert np.allclose(J_out, 0)  # J should be zeros
    assert Msp_out == 0  # Mean spike rate should be zero
    assert Nsp_out == 0  # Total spike count should be zero

    # Test Case 4: Single spike time
    data_single = np.array([0.2])
    J_single, Msp_single, Nsp_single = px.mtfftpt(data_single, tapers, nfft, t, f, findx)

    # Assertions
    assert J_single.shape == (5, 2)  # Ensure correct shape of J
    assert Msp_single == 1 / (t[-1] - t[0])  # Verify mean spike rate
    assert Nsp_single == 1  # Verify total spike count

    # Test Case 5: Multiple tapers and frequencies
    data_multiple = np.array([0.1, 0.25, 0.5])
    tapers_multiple = np.random.rand(4, 3)  # Random tapers (4 time points, 3 tapers)
    nfft_multiple = 16
    t_multiple = np.linspace(0, 1, 4)
    f_multiple = np.linspace(0, 20, 8)

    # Update findx for this case
    findx_multiple = np.zeros(nfft_multiple, dtype=bool)
    findx_multiple[:len(f_multiple)] = True

    J_mult, Msp_mult, Nsp_mult = px.mtfftpt(
        data_multiple, tapers_multiple, nfft_multiple, t_multiple, f_multiple, findx_multiple
    )

    # Assertions
    assert J_mult.shape == (8, 3)  # Ensure correct shape of J
    assert Msp_mult == len(data_multiple) / (t_multiple[-1] - t_multiple[0])
    assert Nsp_mult == len(data_multiple)



# if __name__ == "__main__":
#     test_mtfftpt()
#     pytest.main()
