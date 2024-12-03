import numpy as np
import pytest

from neuro_py.lfp.spectral import filter_signal


@pytest.fixture
def sample_signal():
    """Generate a sample signal with multiple frequency components."""
    fs = 1000  # Sampling rate (Hz)
    t = np.linspace(0, 1, fs, endpoint=False)  # Time vector (1 second)
    # Create a signal with 1 Hz and 50 Hz components
    sig = np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)
    return sig, fs


def test_filter_signal_fir_lowpass(sample_signal):
    """Test FIR lowpass filter."""
    sig, fs = sample_signal
    pass_type = "lowpass"
    f_range = 10  # Lowpass filter at 10 Hz
    filt_sig = filter_signal(
        sig, fs, pass_type, f_range, filter_type="fir", remove_edges=False
    )

    # Assert that the output shape matches the input shape
    assert filt_sig.shape == sig.shape

    # Assert that high-frequency noise (50 Hz) is attenuated
    assert np.max(filt_sig) < np.max(sig)


def test_filter_signal_fir_highpass(sample_signal):
    """Test FIR highpass filter."""
    sig, fs = sample_signal
    pass_type = "highpass"
    f_range = 10  # Highpass filter at 10 Hz
    filt_sig = filter_signal(
        sig, fs, pass_type, f_range, filter_type="fir", remove_edges=False
    )

    # Assert that the output shape matches the input shape
    assert filt_sig.shape == sig.shape

    # Assert that low-frequency component (1 Hz) is attenuated
    assert np.mean(filt_sig[:100]) < np.mean(sig[:100])


def test_filter_signal_fir_with_nan_handling(sample_signal):
    """Test FIR filter with edge removal."""
    sig, fs = sample_signal
    pass_type = "lowpass"
    f_range = 10  # Lowpass filter at 10 Hz
    filt_sig = filter_signal(
        sig, fs, pass_type, f_range, filter_type="fir", remove_edges=True
    )

    # Assert that the output shape matches the input shape
    assert filt_sig.shape == sig.shape

    # Use np.nanmax to handle NaN values in assertions
    assert np.nanmax(filt_sig) < np.max(sig)

    # Assert that the first and last samples are NaN
    assert np.isnan(filt_sig[:10]).all()
    assert np.isnan(filt_sig[-10:]).all()


def test_filter_signal_invalid_pass_type(sample_signal):
    """Test invalid `pass_type` raises ValueError."""
    sig, fs = sample_signal
    with pytest.raises(ValueError, match="`pass_type` must be one of"):
        filter_signal(sig, fs, "invalid_type", (10, 20))
