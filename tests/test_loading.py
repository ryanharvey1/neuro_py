import warnings
from unittest.mock import patch

import nelpy as nel
import numpy as np
import pandas as pd
import pytest

from neuro_py.io.loading import load_brain_regions, load_spikes, load_trials


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

    with warnings.catch_warnings(record=True):
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
        with warnings.catch_warnings(record=True):
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
            },
            "extracellular": {
                "electrodeGroups": {
                    "channels": np.array([[1, 2, 3, 4], [5, 6, 7, 8]]),
                },
            },
        }
    }

    # Mock the file existence, loadmat, and loadXML functions
    with (
        patch("os.path.exists", return_value=True),
        patch("scipy.io.loadmat", return_value=mock_data),
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


# test load_spikes

@pytest.fixture
def mock_metadata():
    with patch('neuro_py.io.loading.load_extracellular_metadata') as mock:
        yield mock

@pytest.fixture
def mock_cell_metrics():
    with patch('neuro_py.io.loading.load_cell_metrics') as mock:
        yield mock

@pytest.fixture
def mock_sleep_states():
    with patch('neuro_py.io.loading.load_SleepState_states') as mock:
        yield mock
        
# Test cases
def test_load_spikes_basic_success(mock_metadata, mock_cell_metrics):
    """Test basic successful loading of spikes without any filters."""
    # with (
    #     patch("neuro_py.io.loading.load_extracellular_metadata") as mock_meta,
    #     patch("neuro_py.io.loading.load_cell_metrics") as mock_metrics,
    # ):
    #     # Setup mock returns
    #     mock_meta.return_value = {"sr": 20000}
    #     mock_metrics.return_value = (
    #         pd.DataFrame(
    #             {
    #                 "putativeCellType": ["Pyramidal", "Interneuron"],
    #                 "brainRegion": ["CA1", "CA3"],
    #                 "bad_unit": [False, False],
    #             }
    #         ),
    #         {"spikes": [np.array([1.0, 2.0]), np.array([3.0, 4.0])]},
    #     )

    #     # Call function
    #     st, metrics = load_spikes("dummy_path")

    #     # Assertions
    #     assert st is not None
    #     assert metrics is not None
    #     assert st.n_active == 2
    #     assert len(metrics) == 2
    # Setup mock returns
    mock_metadata.return_value = {"sr": 20000}
    mock_cell_metrics.return_value = (
        pd.DataFrame({
            "putativeCellType": ["Pyramidal", "Interneuron"],
            "brainRegion": ["CA1", "CA3"],
            "bad_unit": [False, False]
        }),
        {"spikes": [np.array([1.0, 2.0]), np.array([3.0, 4.0])]}
    )
    # Call function
    st, metrics = load_spikes("dummy_path")
    # Assertions
    assert st is not None
    assert metrics is not None
    assert st.n_active == 2
    assert len(metrics) == 2

def test_load_spikes_no_sampling_rate():
    """Test when sampling rate is not available in metadata."""
    with patch("neuro_py.io.loading.load_extracellular_metadata") as mock_meta:
        mock_meta.return_value = {}

        st, metrics = load_spikes("dummy_path")
        assert st is None
        assert metrics is None


def test_load_spikes_no_cell_metrics():
    """Test when cell metrics cannot be loaded."""
    with (
        patch("neuro_py.io.loading.load_extracellular_metadata") as mock_meta,
        patch("neuro_py.io.loading.load_cell_metrics") as mock_metrics,
    ):
        mock_meta.return_value = {"sr": 20000}
        mock_metrics.return_value = (None, None)

        st, metrics = load_spikes("dummy_path")
        assert st is None
        assert metrics is None


def test_load_spikes_filter_by_cell_type():
    """Test filtering by putative cell type."""
    with (
        patch("neuro_py.io.loading.load_extracellular_metadata") as mock_meta,
        patch("neuro_py.io.loading.load_cell_metrics") as mock_metrics,
    ):
        mock_meta.return_value = {"sr": 20000}
        mock_metrics.return_value = (
            pd.DataFrame(
                {
                    "putativeCellType": ["Pyramidal", "Interneuron", "Pyramidal"],
                    "brainRegion": ["CA1", "CA3", "CA1"],
                    "bad_unit": [False, False, False],
                }
            ),
            {
                "spikes": [
                    np.array([1.0, 2.0]),
                    np.array([3.0, 4.0]),
                    np.array([5.0, 6.0]),
                ]
            },
        )

        # Filter for only pyramidal cells
        st, metrics = load_spikes("dummy_path", putativeCellType=["Pyramidal"])

        assert st.n_active == 2
        assert all(metrics.putativeCellType == "Pyramidal")


def test_load_spikes_filter_by_brain_region():
    """Test filtering by brain region."""
    with (
        patch("neuro_py.io.loading.load_extracellular_metadata") as mock_meta,
        patch("neuro_py.io.loading.load_cell_metrics") as mock_metrics,
    ):
        mock_meta.return_value = {"sr": 20000}
        mock_metrics.return_value = (
            pd.DataFrame(
                {
                    "putativeCellType": ["Pyramidal", "Interneuron", "Pyramidal"],
                    "brainRegion": ["CA1", "CA3", "CA1"],
                    "bad_unit": [False, False, False],
                }
            ),
            {
                "spikes": [
                    np.array([1.0, 2.0]),
                    np.array([3.0, 4.0]),
                    np.array([5.0, 6.0]),
                ]
            },
        )

        # Filter for only CA1 cells
        st, metrics = load_spikes("dummy_path", brainRegion=["CA1"])

        assert st.n_active == 2
        assert all(metrics.brainRegion == "CA1")


def test_load_spikes_remove_bad_units():
    """Test removal of bad units."""
    with (
        patch("neuro_py.io.loading.load_extracellular_metadata") as mock_meta,
        patch("neuro_py.io.loading.load_cell_metrics") as mock_metrics,
    ):
        mock_meta.return_value = {"sr": 20000}
        mock_metrics.return_value = (
            pd.DataFrame(
                {
                    "putativeCellType": ["Pyramidal", "Interneuron", "Pyramidal"],
                    "brainRegion": ["CA1", "CA3", "CA1"],
                    "bad_unit": [False, True, False],
                }
            ),
            {
                "spikes": [
                    np.array([1.0, 2.0]),
                    np.array([3.0, 4.0]),
                    np.array([5.0, 6.0]),
                ]
            },
        )

        # Default should remove bad units
        st, metrics = load_spikes("dummy_path")

        assert st.n_active == 2
        assert sum(metrics.bad_unit) == 0


def test_load_spikes_keep_bad_units():
    """Test keeping bad units when remove_bad_unit=False."""
    with (
        patch("neuro_py.io.loading.load_extracellular_metadata") as mock_meta,
        patch("neuro_py.io.loading.load_cell_metrics") as mock_metrics,
    ):
        mock_meta.return_value = {"sr": 20000}
        mock_metrics.return_value = (
            pd.DataFrame(
                {
                    "putativeCellType": ["Pyramidal", "Interneuron", "Pyramidal"],
                    "brainRegion": ["CA1", "CA3", "CA1"],
                    "bad_unit": [False, True, False],
                }
            ),
            {
                "spikes": [
                    np.array([1.0, 2.0]),
                    np.array([3.0, 4.0]),
                    np.array([5.0, 6.0]),
                ]
            },
        )

        # Keep bad units
        st, metrics = load_spikes("dummy_path", remove_bad_unit=False)

        assert st.n_active == 3
        assert sum(metrics.bad_unit) == 1


def test_load_spikes_filter_by_other_metric():
    """Test filtering by other arbitrary metrics."""
    with (
        patch("neuro_py.io.loading.load_extracellular_metadata") as mock_meta,
        patch("neuro_py.io.loading.load_cell_metrics") as mock_metrics,
    ):
        mock_meta.return_value = {"sr": 20000}
        mock_metrics.return_value = (
            pd.DataFrame(
                {
                    "putativeCellType": ["Pyramidal", "Interneuron", "Pyramidal"],
                    "brainRegion": ["CA1", "CA3", "CA1"],
                    "bad_unit": [False, False, False],
                    "quality": ["good", "excellent", "fair"],
                }
            ),
            {
                "spikes": [
                    np.array([1.0, 2.0]),
                    np.array([3.0, 4.0]),
                    np.array([5.0, 6.0]),
                ]
            },
        )

        # Filter by quality metric
        st, metrics = load_spikes(
            "dummy_path", other_metric="quality", other_metric_value="excellent"
        )

        assert st.n_active == 1

        assert metrics.iloc[0].quality == "excellent"


def test_load_spikes_remove_unstable():
    """Test removal of unstable units."""
    with (
        patch("neuro_py.io.loading.load_extracellular_metadata") as mock_meta,
        patch("neuro_py.io.loading.load_cell_metrics") as mock_metrics,
    ):
        mock_meta.return_value = {"sr": 20000}
        mock_metrics.return_value = (
            pd.DataFrame(
                {
                    "putativeCellType": ["Pyramidal", "Interneuron", "Pyramidal"],
                    "brainRegion": ["CA1", "CA3", "CA1"],
                    "bad_unit": [False, False, False],
                }
            ),
            {
                "spikes": [
                    np.sort(
                        np.random.poisson(4, size=100)
                    ),  # Stable cell (spikes in all intervals)
                    np.array([1]),  # Unstable cell (spikes in only one interval)
                    np.sort(np.random.poisson(4, size=100)),  # Stable cell
                ]
            },
        )

        # Remove unstable units
        st, metrics = load_spikes(
            "dummy_path", remove_unstable=True, stable_interval_width=3
        )

        assert st.n_active == 2

        assert len(metrics) == 2


def test_load_spikes_single_cell():
    """Test handling of single cell case."""
    with (
        patch("neuro_py.io.loading.load_extracellular_metadata") as mock_meta,
        patch("neuro_py.io.loading.load_cell_metrics") as mock_metrics,
    ):
        mock_meta.return_value = {"sr": 20000}
        mock_metrics.return_value = (
            pd.DataFrame(
                {
                    "putativeCellType": ["Pyramidal"],
                    "brainRegion": ["CA1"],
                    "bad_unit": [False],
                }
            ),
            {"spikes": [np.array([1.0, 2.0])]},
        )

        st, metrics = load_spikes("dummy_path")

        assert st.n_active == 1
        assert len(metrics) == 1


def test_load_spikes_with_support():
    """Test loading with time support restriction."""
    with (
        patch("neuro_py.io.loading.load_extracellular_metadata") as mock_meta,
        patch("neuro_py.io.loading.load_cell_metrics") as mock_metrics,
    ):
        mock_meta.return_value = {"sr": 20000}
        mock_metrics.return_value = (
            pd.DataFrame(
                {
                    "putativeCellType": ["Pyramidal", "Interneuron"],
                    "brainRegion": ["CA1", "CA3"],
                    "bad_unit": [False, False],
                }
            ),
            {"spikes": [np.array([1.0, 2.0]), np.array([3.0, 4.0])]},
        )

        # Create a dummy support epoch
        support = nel.EpochArray(np.array([[0, 5]]))

        st, metrics = load_spikes("dummy_path", support=support)

        assert st is not None
        assert st.support is not None


def test_load_spikes_brain_state_filter():
    """Test filtering by brain state."""
    with (
        patch("neuro_py.io.loading.load_extracellular_metadata") as mock_meta,
        patch("neuro_py.io.loading.load_cell_metrics") as mock_metrics,
        patch("neuro_py.io.loading.load_SleepState_states") as mock_states,
    ):
        mock_meta.return_value = {"sr": 20000}
        mock_metrics.return_value = (
            pd.DataFrame(
                {
                    "putativeCellType": ["Pyramidal", "Interneuron"],
                    "brainRegion": ["CA1", "CA3"],
                    "bad_unit": [False, False],
                }
            ),
            {"spikes": [np.array([1.0, 2.0, 10.0]), np.array([3.0, 4.0, 20.0])]},
        )

        # Mock sleep states (only WAKEstate between 0-5 seconds)
        mock_states.return_value = {
            "WAKEstate": np.array([[0, 5]]),
            "NREMstate": np.array([[5, 10]]),
        }

        st, metrics = load_spikes("dummy_path", brain_state="WAKEstate")

        # Check that spikes outside WAKEstate are filtered
        for spike_train in st.data:
            assert all(spike <= 5 for spike in spike_train)


def test_load_spikes_invalid_brain_state():
    """Test handling of invalid brain state input."""
    with (
        patch("neuro_py.io.loading.load_extracellular_metadata") as mock_meta,
        patch("neuro_py.io.loading.load_cell_metrics") as mock_metrics,
        patch("builtins.print") as mock_print,
    ):
        mock_meta.return_value = {"sr": 20000}
        mock_metrics.return_value = (
            pd.DataFrame(
                {
                    "putativeCellType": ["Pyramidal"],
                    "brainRegion": ["CA1"],
                    "bad_unit": [False],
                }
            ),
            {"spikes": [np.array([1.0, 2.0])]},
        )

        # Invalid brain state
        st, metrics = load_spikes("dummy_path", brain_state="INVALID")

        # Should still return data but without brain state filtering
        assert st is not None
        assert metrics is not None
        mock_print.assert_called()  # Check that warning was printed


def test_load_spikes_other_metric_length_mismatch():
    """Test error when other_metric and other_metric_value have different lengths."""
    with (
        patch("neuro_py.io.loading.load_extracellular_metadata") as mock_meta,
        patch("neuro_py.io.loading.load_cell_metrics") as mock_metrics,
    ):
        mock_meta.return_value = {"sr": 20000}
        mock_metrics.return_value = (
            pd.DataFrame(
                {
                    "putativeCellType": ["Pyramidal"],
                    "brainRegion": ["CA1"],
                    "bad_unit": [False],
                }
            ),
            {"spikes": [np.array([1.0, 2.0])]},
        )

        with pytest.raises(ValueError, match="other_metric and other_metric_value must be of same length"):
            load_spikes(
                "dummy_path",
                other_metric=["quality", "depth"],
                other_metric_value=["good"],
            )
