import warnings
from unittest.mock import patch

import pandas as pd

from neuro_py.io.loading import load_trials


def test_load_trials_file_not_found():
    """Test that the function returns an empty DataFrame and issues a warning when the file does not exist."""
    basepath = "/fake/path"

    with warnings.catch_warnings(record=True) as w:
        result = load_trials(basepath)

        # Check that the result is an empty DataFrame
        assert result.empty
        # Check that a warning was issued
        assert len(w) == 1
        assert "file does not exist" in str(w[0].message)


def test_load_trials_no_trials_key():
    """Test that the function returns an empty DataFrame and issues a warning when the 'trials' key is not found."""
    basepath = "/fake/path"

    # Mock the file existence and the loadmat function
    with (
        patch("os.path.exists", return_value=True),
        patch("scipy.io.loadmat", return_value={"behavior": {}}),
    ):
        with warnings.catch_warnings(record=True) as w:
            result = load_trials(basepath)

            # Check that the result is an empty DataFrame
            assert result.empty
            # Check that a warning was issued
            assert len(w) == 1
            assert "trials not found in file" in str(w[0].message)


def test_load_trials_current_standard():
    """Test that the function correctly loads trials following the current standard."""
    basepath = "/fake/path"

    # Mock data following the current standard
    mock_data = {
        "behavior": {
            "trials": {
                "starts": [1.0, 2.0, 3.0],
                "stops": [1.5, 2.5, 3.5],
                "stateName": ["A", "B", "C"],
            }
        }
    }

    # Mock the file existence and the loadmat function
    with (
        patch("os.path.exists", return_value=True),
        patch("scipy.io.loadmat", return_value=mock_data),
    ):
        result = load_trials(basepath)

        # Check that the result is a DataFrame with the correct columns and data
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["startTime", "stopTime", "trialsID"]
        assert result["startTime"].tolist() == [1.0, 2.0, 3.0]
        assert result["stopTime"].tolist() == [1.5, 2.5, 3.5]
        assert result["trialsID"].tolist() == ["A", "B", "C"]


def test_load_trials_old_standard():
    """Test that the function correctly loads trials following the old standard."""
    basepath = "/fake/path"

    # Mock data following the old standard
    mock_data = {
        "behavior": {
            "trials": [[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]],
            "trialsID": ["A", "B", "C"],
        }
    }

    # Mock the file existence and the loadmat function
    with (
        patch("os.path.exists", return_value=True),
        patch("scipy.io.loadmat", return_value=mock_data),
    ):
        result = load_trials(basepath)

        # Check that the result is a DataFrame with the correct columns and data
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["startTime", "stopTime", "trialsID"]
        assert result["startTime"].tolist() == [1.0, 2.0, 3.0]
        assert result["stopTime"].tolist() == [1.5, 2.5, 3.5]
        assert result["trialsID"].tolist() == ["A", "B", "C"]


def test_load_trials_old_standard_exception():
    """Test that the function handles exceptions when loading trials following the old standard."""
    basepath = "/fake/path"

    # Mock data that will cause an exception when trying to load the old standard
    mock_data = {
        "behavior": {
            "trials": [
                1.0,
                1.5,
            ],  # This will cause an exception when trying to convert to DataFrame
            "trialsID": ["A", "B"],
        }
    }

    # Mock the file existence and the loadmat function
    with (
        patch("os.path.exists", return_value=True),
        patch("scipy.io.loadmat", return_value=mock_data),
    ):
        result = load_trials(basepath)

        # Check that the result is a DataFrame with the correct columns and data
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["startTime", "stopTime"]
        assert result["startTime"].tolist() == [1.0]
        assert result["stopTime"].tolist() == [1.5]
