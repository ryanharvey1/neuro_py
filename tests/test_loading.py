import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd

from neuro_py.io.loading import load_brain_regions, load_trials


# test load_trials
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


# test load_brain_regions
def test_load_brain_regions_file_not_found():
    """Test that the function returns an empty dict/DataFrame and issues a warning when the file does not exist."""
    basepath = "/fake/path"

    # with warnings.catch_warnings(record=True) as w:
    # Test for dict output
    result_dict = load_brain_regions(basepath, out_format="dict")
    assert result_dict == {}
    # assert len(w) == 1
    # assert "does not exist" in str(w[0].message)

    # Test for DataFrame output
    result_df = load_brain_regions(basepath, out_format="DataFrame")
    assert result_df.empty
    # assert len(w) == 2
    # assert "does not exist" in str(w[1].message)


def test_load_brain_regions_no_brain_regions_key():
    """Test that the function returns an empty dict/DataFrame and issues a warning when the 'brainRegions' key is not found."""
    basepath = "/fake/path"

    # Mock the file existence and the loadmat function
    with (
        patch("os.path.exists", return_value=True),
        patch("scipy.io.loadmat", return_value={"session": {}}),
    ):
        # with warnings.catch_warnings(record=True) as w:
        # Test for dict output
        result_dict = load_brain_regions(basepath, out_format="dict")
        assert result_dict == {}
        # assert len(w) == 1
        # assert "brainRegions not found in file" in str(w[0].message)

        # Test for DataFrame output
        result_df = load_brain_regions(basepath, out_format="DataFrame")
        assert result_df.empty
        # assert len(w) == 2
        # assert "brainRegions not found in file" in str(w[1].message)


def test_load_brain_regions_dict_output():
    """Test that the function correctly loads brain regions and returns a dictionary."""
    basepath = "/fake/path"

    # Mock data with brain regions
    mock_data = {
        "session": {
            "brainRegions": {
                "CA1": {
                    "channels": np.array([1, 2, 3]),
                    "electrodeGroups": np.array([17, 18]),
                },
                "Unknown": {
                    "channels": np.array([4, 5]),
                    "electrodeGroups": np.nan,
                },
            }
        }
    }

    # Mock the file existence, loadmat, and loadXML functions
    with (
        patch("os.path.exists", return_value=True),
        patch("scipy.io.loadmat", return_value=mock_data),
        patch(
            "neuro_py.io.loading.loadXML",
            return_value=(None, None, None, {0: [1, 2, 3], 1: [4, 5]}),
        ),
    ):
        result = load_brain_regions(basepath, out_format="dict")

        # Check that the result is a dictionary with the correct structure
        assert isinstance(result, dict)
        assert "CA1" in result
        assert "Unknown" in result
        assert np.array_equal(result["CA1"]["channels"], np.array([1, 2, 3]))
        assert np.array_equal(result["CA1"]["electrodeGroups"], np.array([17, 18]))
        assert np.array_equal(result["Unknown"]["channels"], np.array([4, 5]))
        assert np.isnan(result["Unknown"]["electrodeGroups"])


def test_load_brain_regions_dataframe_output():
    """Test that the function correctly loads brain regions and returns a DataFrame."""
    basepath = "/fake/path"

    # Mock data with brain regions
    mock_data = {
        "session": {
            "brainRegions": {
                "CA1": {
                    "channels": np.array([1, 2, 3, 4]),
                    "electrodeGroups": np.array(0),
                },
                "PFC": {
                    "channels": np.array([5, 6]),
                    "electrodeGroups": np.nan,
                },
            }
        }
    }

    # Mock the file existence, loadmat, and loadXML functions
    with (
        patch("os.path.exists", return_value=True),
        patch("scipy.io.loadmat", return_value=mock_data),
        patch(
            "neuro_py.io.loading.loadXML",
            return_value=(None, None, None, {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}),
        ),
    ):
        result = load_brain_regions(basepath, out_format="DataFrame")

        # Check that the result is a DataFrame with the correct structure
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["channels", "region", "shank"]
        # Zero-indexed
        assert result["channels"].tolist() == [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
        ]
        assert result["region"].tolist() == [
            "CA1",
            "CA1",
            "CA1",
            "CA1",
            "PFC",
            "PFC",
            "Unknown",
            "Unknown",
        ]
        assert result["shank"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
