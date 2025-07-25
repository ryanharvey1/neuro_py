import nelpy as nel
import numpy as np

from neuro_py.lfp.preprocessing import clean_lfp


def test_clean_lfp():
    # Create an LFP signal with some global artefacts and noise
    fs = 1250
    n_samples = fs * 20
    t = np.arange(n_samples) / fs
    values = np.random.randn(n_samples)
    values[250:275] += 6  # Add global artefact
    values[1250:1260] += 7  # Add global artefact
    values[5000:5010] += 11  # Add noise
    lfp = nel.AnalogSignalArray(data=values, timestamps=t)

    # Clean the LFP signal
    cleaned_values = clean_lfp(lfp)

    # Check that the global artefacts and noise have been removed
    assert np.abs(cleaned_values[250:275]).max() < 5, "Global artefact not removed"
    assert np.abs(cleaned_values[1250:1260]).max() < 5, "Global artefact not removed"
    assert np.abs(cleaned_values[5000:5010]).max() < 10, "Noise not removed"
