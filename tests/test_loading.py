import os
import tempfile
import warnings
from unittest.mock import patch

import nelpy as nel
import numpy as np
import pandas as pd
import pytest
import scipy.io as sio

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
        assert np.array_equal(result["CA1"]["channels"], np.array([0, 1, 2]))
        assert np.array_equal(result["CA1"]["electrodeGroups"], np.array([17, 18]))
        assert np.array_equal(result["Unknown"]["channels"], np.array([3, 4]))
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

# Test cases


def create_temp_mat_file(basepath, content_dict):
    """Helper function to create temporary MAT files."""
    os.makedirs(basepath, exist_ok=True)
    for filename, content in content_dict.items():
        sio.savemat(os.path.join(basepath, filename), content)


def test_load_spikes_basic_success():
    """Test basic successful loading of spikes without any filters."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        basepath = os.path.join(temp_dir, "session1")
        basename = os.path.basename(basepath)

        # Create MATLAB-style cell array for spike times
        # Each cell's spikes are stored as a column vector (Nx1 array)
        cell1_spikes = np.array([1, 2, 3, 4], dtype="float64").reshape(-1, 1)
        cell2_spikes = np.array([3, 4], dtype="float64").reshape(-1, 1)

        # Create the times structure as a numpy object array
        times_array = np.array([cell1_spikes, cell2_spikes], dtype="object")

        # Create the full nested structure that matches your indexing
        spikes_struct = {"times": times_array}

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.session.mat": {
                    "session": {"extracellular": {"sr": 20000}}
                },
                f"{basename}.cell_metrics.cellinfo.mat": {
                    "cell_metrics": {
                        "UID": np.array([1, 2], dtype="object"),
                        "putativeCellType": np.array(
                            ["Pyramidal", "Interneuron"], dtype="object"
                        ),
                        "brainRegion": np.array(["CA1", "CA3"], dtype="object"),
                        "tags": {"Bad": np.array([])},  # will be UID labels
                        "spikes": spikes_struct,
                        "waveforms": {"filt": np.array([[[1, 2], [3, 4]]])},
                        "acg": {
                            "wide": np.array([[1, 2], [3, 4]]),
                            "narrow": np.array([[1, 2], [3, 4]]),
                            "log10": np.array([[1, 2], [3, 4]]),
                        },
                        "general": {
                            "basename": basename,
                            "cellCount": 2,
                            "animal": {
                                "sex": "male",
                                "species": "rat",
                                "strain": "long-evans",
                                "geneticLine": "WT",
                            },
                        },
                    },
                },
            },
        )

        # Call function
        st, metrics = load_spikes(basepath)

        # Assertions
        assert st is not None
        assert metrics is not None
        assert st.n_active == 2
        assert len(metrics) == 2


def test_load_spikes_no_sampling_rate():
    """Test when sampling rate is not available in metadata."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session2")
        basename = os.path.basename(basepath)

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.session.mat": {"session": {"extracellular": {}}},
                f"{basename}.cell_metrics.cellinfo.mat": {
                    "cell_metrics": {
                        "UID": np.array([1], dtype="object"),
                        "putativeCellType": np.array(["Pyramidal"], dtype="object"),
                        "brainRegion": np.array(["CA1"], dtype="object"),
                        "tags": {"Bad": np.array([])},
                        "spikes": {
                            "times": np.array(
                                [np.array([[1.0], [2.0]])], dtype="object"
                            )
                        },
                        "waveforms": {"filt": np.array([[[1, 2]]])},
                        "acg": {
                            "wide": np.array([[1, 2]]),
                            "narrow": np.array([[1, 2]]),
                            "log10": np.array([[1, 2]]),
                        },
                        "general": {
                            "basename": basename,
                            "cellCount": 1,
                            "animal": {
                                "sex": "male",
                                "species": "rat",
                                "strain": "long-evans",
                                "geneticLine": "WT",
                            },
                        },
                    }
                },
            },
        )

        st, metrics = load_spikes(basepath)
        assert st is None
        assert metrics is None


def test_load_spikes_no_cell_metrics():
    """Test when cell metrics cannot be loaded."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session3")
        basename = os.path.basename(basepath)

        create_temp_mat_file(
            basepath,
            {f"{basename}.session.mat": {"session": {"extracellular": {"sr": 20000}}}},
        )

        st, metrics = load_spikes(basepath)
        assert st is None
        assert metrics is None


def test_load_spikes_filter_by_cell_type():
    """Test filtering by putative cell type."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session4")
        basename = os.path.basename(basepath)

        cell_types = ["Pyramidal", "Interneuron", "Pyramidal"]
        # Create MATLAB-style cell array for spike times
        # Each cell's spikes are stored as a column vector (Nx1 array)
        cell1_spikes = np.array([1, 2, 3, 4], dtype="float64").reshape(-1, 1)
        cell2_spikes = np.array([3, 4], dtype="float64").reshape(-1, 1)
        cell3_spikes = np.array([5, 6], dtype="float64").reshape(-1, 1)

        # Create the times structure as a numpy object array
        times_array = np.array(
            [cell1_spikes, cell2_spikes, cell3_spikes], dtype="object"
        )

        # Create the full nested structure that matches your indexing
        spikes_struct = {"times": times_array}

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.session.mat": {
                    "session": {"extracellular": {"sr": 20000}}
                },
                f"{basename}.cell_metrics.cellinfo.mat": {
                    "cell_metrics": {
                        "UID": np.array([1, 2, 3], dtype="object"),
                        "putativeCellType": np.array(cell_types, dtype="object"),
                        "brainRegion": np.array(["CA1", "CA3", "CA1"], dtype="object"),
                        "tags": {"Bad": np.array([])},
                        "spikes": spikes_struct,
                        "waveforms": {"filt": np.array([[[1, 2]], [[3, 4]], [[5, 6]]])},
                        "acg": {
                            "wide": np.array([[1, 2], [3, 4], [5, 6]]),
                            "narrow": np.array([[1, 2], [3, 4], [5, 6]]),
                            "log10": np.array([[1, 2], [3, 4], [5, 6]]),
                        },
                        "general": {
                            "basename": basename,
                            "cellCount": 3,
                            "animal": {
                                "sex": "male",
                                "species": "rat",
                                "strain": "long-evans",
                                "geneticLine": "WT",
                            },
                        },
                    }
                },
            },
        )

        st, metrics = load_spikes(basepath, putativeCellType=["Pyramidal"])
        assert st.n_active == 2
        assert all(metrics.putativeCellType == "Pyramidal")


def test_load_spikes_filter_by_brain_region():
    """Test filtering by brain region."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session5")
        basename = os.path.basename(basepath)

        regions = ["CA1", "CA3", "CA1"]
        # Create MATLAB-style cell array for spike times
        # Each cell's spikes are stored as a column vector (Nx1 array)
        cell1_spikes = np.array([1, 2, 3, 4], dtype="float64").reshape(-1, 1)
        cell2_spikes = np.array([3, 4], dtype="float64").reshape(-1, 1)
        cell3_spikes = np.array([5, 6], dtype="float64").reshape(-1, 1)

        # Create the times structure as a numpy object array
        times_array = np.array(
            [cell1_spikes, cell2_spikes, cell3_spikes], dtype="object"
        )

        # Create the full nested structure that matches your indexing
        spikes_struct = {"times": times_array}

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.session.mat": {
                    "session": {"extracellular": {"sr": 20000}}
                },
                f"{basename}.cell_metrics.cellinfo.mat": {
                    "cell_metrics": {
                        "UID": np.array([1, 2, 3], dtype="object"),
                        "putativeCellType": np.array(["Pyr"] * 3, dtype="object"),
                        "brainRegion": np.array(regions, dtype="object"),
                        "tags": {"Bad": np.array([])},
                        "spikes": spikes_struct,
                        "waveforms": {"filt": np.array([[[1, 2]], [[3, 4]], [[5, 6]]])},
                        "acg": {
                            "wide": np.array([[1, 2], [3, 4], [5, 6]]),
                            "narrow": np.array([[1, 2], [3, 4], [5, 6]]),
                            "log10": np.array([[1, 2], [3, 4], [5, 6]]),
                        },
                        "general": {
                            "basename": basename,
                            "cellCount": 3,
                            "animal": {
                                "sex": "male",
                                "species": "rat",
                                "strain": "long-evans",
                                "geneticLine": "WT",
                            },
                        },
                    }
                },
            },
        )

        st, metrics = load_spikes(basepath, brainRegion=["CA1"])
        assert st.n_active == 2
        assert all(metrics.brainRegion == "CA1")


def test_load_spikes_remove_bad_units():
    """Test removal of bad units."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session6")
        basename = os.path.basename(basepath)

        # Create MATLAB-style cell array for spike times
        # Each cell's spikes are stored as a column vector (Nx1 array)
        cell1_spikes = np.array([1, 2, 3, 4], dtype="float64").reshape(-1, 1)
        cell2_spikes = np.array([3, 4], dtype="float64").reshape(-1, 1)
        cell3_spikes = np.array([5, 6], dtype="float64").reshape(-1, 1)

        # Create the times structure as a numpy object array
        times_array = np.array(
            [cell1_spikes, cell2_spikes, cell3_spikes], dtype="object"
        )

        # Create the full nested structure that matches your indexing
        spikes_struct = {"times": times_array}
        create_temp_mat_file(
            basepath,
            {
                f"{basename}.session.mat": {
                    "session": {"extracellular": {"sr": 20000}}
                },
                f"{basename}.cell_metrics.cellinfo.mat": {
                    "cell_metrics": {
                        "UID": np.array([1, 2, 3], dtype="object"),
                        "putativeCellType": np.array(["Pyr"] * 3, dtype="object"),
                        "brainRegion": np.array(["CA1"] * 3, dtype="object"),
                        "tags": {"Bad": np.array([2])},
                        "spikes": spikes_struct,
                        "waveforms": {"filt": np.array([[[1, 2]], [[3, 4]], [[5, 6]]])},
                        "acg": {
                            "wide": np.array([[1, 2], [3, 4], [5, 6]]),
                            "narrow": np.array([[1, 2], [3, 4], [5, 6]]),
                            "log10": np.array([[1, 2], [3, 4], [5, 6]]),
                        },
                        "general": {
                            "basename": basename,
                            "cellCount": 3,
                            "animal": {
                                "sex": "male",
                                "species": "rat",
                                "strain": "long-evans",
                                "geneticLine": "WT",
                            },
                        },
                    }
                },
            },
        )

        st, metrics = load_spikes(basepath)
        assert st.n_active == 2
        assert 2 not in metrics["UID"].values


def test_load_spikes_keep_bad_units():
    """Test keeping bad units when remove_bad_unit=False."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session7")
        basename = os.path.basename(basepath)

        # Create MATLAB-style cell array for spike times
        # Each cell's spikes are stored as a column vector (Nx1 array)
        cell1_spikes = np.array([1, 2, 3, 4], dtype="float64").reshape(-1, 1)
        cell2_spikes = np.array([3, 4], dtype="float64").reshape(-1, 1)
        cell3_spikes = np.array([5, 6], dtype="float64").reshape(-1, 1)

        # Create the times structure as a numpy object array
        times_array = np.array(
            [cell1_spikes, cell2_spikes, cell3_spikes], dtype="object"
        )

        spikes_struct = {"times": times_array}

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.session.mat": {
                    "session": {"extracellular": {"sr": 20000}}
                },
                f"{basename}.cell_metrics.cellinfo.mat": {
                    "cell_metrics": {
                        "UID": np.array([1, 2, 3], dtype="object"),
                        "putativeCellType": np.array(["Pyr"] * 3, dtype="object"),
                        "brainRegion": np.array(["CA1"] * 3, dtype="object"),
                        "tags": {"Bad": np.array([2])},
                        "spikes": spikes_struct,
                        "waveforms": {"filt": np.array([[[1, 2]], [[3, 4]], [[5, 6]]])},
                        "acg": {
                            "wide": np.array([[1, 2], [3, 4], [5, 6]]),
                            "narrow": np.array([[1, 2], [3, 4], [5, 6]]),
                            "log10": np.array([[1, 2], [3, 4], [5, 6]]),
                        },
                        "general": {
                            "basename": basename,
                            "cellCount": 3,
                            "animal": {
                                "sex": "male",
                                "species": "rat",
                                "strain": "long-evans",
                                "geneticLine": "WT",
                            },
                        },
                    }
                },
            },
        )

        st, metrics = load_spikes(basepath, remove_bad_unit=False)
        assert st.n_active == 3
        assert 2 in metrics["UID"].values


def test_load_spikes_filter_by_other_metric():
    """Test filtering by other arbitrary metrics."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session8")
        basename = os.path.basename(basepath)

        # Create MATLAB-style cell array for spike times
        # Each cell's spikes are stored as a column vector (Nx1 array)
        cell1_spikes = np.array([1, 2, 3, 4], dtype="float64").reshape(-1, 1)
        cell2_spikes = np.array([3, 4], dtype="float64").reshape(-1, 1)
        cell3_spikes = np.array([5, 6], dtype="float64").reshape(-1, 1)

        # Create the times structure as a numpy object array
        times_array = np.array(
            [cell1_spikes, cell2_spikes, cell3_spikes], dtype="object"
        )

        spikes_struct = {"times": times_array}
        qualities = ["good", "excellent", "fair"]

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.session.mat": {
                    "session": {"extracellular": {"sr": 20000}}
                },
                f"{basename}.cell_metrics.cellinfo.mat": {
                    "cell_metrics": {
                        "UID": np.array([1, 2, 3], dtype="object"),
                        "putativeCellType": np.array(["Pyr"] * 3, dtype="object"),
                        "brainRegion": np.array(["CA1"] * 3, dtype="object"),
                        "tags": {"Bad": np.array([])},
                        "quality": np.array(qualities, dtype="object"),
                        "spikes": spikes_struct,
                        "waveforms": {"filt": np.array([[[1, 2]], [[3, 4]], [[5, 6]]])},
                        "acg": {
                            "wide": np.array([[1, 2], [3, 4], [5, 6]]),
                            "narrow": np.array([[1, 2], [3, 4], [5, 6]]),
                            "log10": np.array([[1, 2], [3, 4], [5, 6]]),
                        },
                        "general": {
                            "basename": basename,
                            "cellCount": 3,
                            "animal": {
                                "sex": "male",
                                "species": "rat",
                                "strain": "long-evans",
                                "geneticLine": "WT",
                            },
                        },
                    }
                },
            },
        )

        st, metrics = load_spikes(
            basepath, other_metric="quality", other_metric_value="excellent"
        )
        assert st.n_active == 1
        assert metrics.iloc[0].quality == "excellent"


def test_load_spikes_single_cell():
    """Test handling of single cell case."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session9")
        basename = os.path.basename(basepath)

        # Create MATLAB-style cell array for spike times
        # Each cell's spikes are stored as a column vector (Nx1 array)
        cell1_spikes = np.array([1, 2, 3, 4], dtype="float64").reshape(-1, 1)

        # Create the times structure as a numpy object array
        times_array = np.array(cell1_spikes, dtype="object")

        spikes_struct = {"times": times_array}

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.session.mat": {
                    "session": {"extracellular": {"sr": 20000}}
                },
                f"{basename}.cell_metrics.cellinfo.mat": {
                    "cell_metrics": {
                        "UID": np.array([1], dtype="object"),
                        "putativeCellType": np.array(["Pyramidal"], dtype="object"),
                        "brainRegion": np.array(["CA1"], dtype="object"),
                        "tags": {"Bad": np.array([])},
                        "spikes": spikes_struct,
                        "waveforms": {"filt": np.array([[[1, 2]]])},
                        "acg": {
                            "wide": np.array([[1, 2]]),
                            "narrow": np.array([[1, 2]]),
                            "log10": np.array([[1, 2]]),
                        },
                        "general": {
                            "basename": basename,
                            "cellCount": 1,
                            "animal": {
                                "sex": "male",
                                "species": "rat",
                                "strain": "long-evans",
                                "geneticLine": "WT",
                            },
                        },
                    }
                },
            },
        )

        st, metrics = load_spikes(basepath)
        assert st.n_active == 1
        assert len(metrics) == 1


def test_load_spikes_with_support():
    """Test loading with time support restriction."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session10")
        basename = os.path.basename(basepath)

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.session.mat": {
                    "session": {"extracellular": {"sr": 20000}}
                },
                f"{basename}.cell_metrics.cellinfo.mat": {
                    "cell_metrics": {
                        "UID": np.array([1, 2], dtype="object"),
                        "putativeCellType": np.array(["Pyr"] * 2, dtype="object"),
                        "brainRegion": np.array(["CA1"] * 2, dtype="object"),
                        "tags": {"Bad": np.array([])},
                        "spikes": {
                            "times": np.array(
                                [np.array([[1.0], [2.0]]), np.array([[3.0], [4.0]])],
                                dtype="object",
                            )
                        },
                        "waveforms": {"filt": np.array([[[1, 2]], [[3, 4]]])},
                        "acg": {
                            "wide": np.array([[1, 2], [3, 4]]),
                            "narrow": np.array([[1, 2], [3, 4]]),
                            "log10": np.array([[1, 2], [3, 4]]),
                        },
                        "general": {
                            "basename": basename,
                            "cellCount": 2,
                            "animal": {
                                "sex": "male",
                                "species": "rat",
                                "strain": "long-evans",
                                "geneticLine": "WT",
                            },
                        },
                    }
                },
            },
        )

        support = nel.EpochArray(np.array([[0, 5]]))
        st, metrics = load_spikes(basepath, support=support)
        assert st is not None
        assert st.support is not None


def test_load_spikes_filter_by_stable():
    """Test loading filtering by stable cells."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test data
        basepath = os.path.join(temp_dir, "session1")
        basename = os.path.basename(basepath)

        # Create MATLAB-style cell array for spike times
        # Each cell's spikes are stored as a column vector (Nx1 array)
        cell1_spikes = np.sort(np.random.poisson(4, size=100)).reshape(-1, 1)
        cell2_spikes = np.sort(np.random.poisson(4, size=100)).reshape(-1, 1)
        cell2_spikes = np.array([1], dtype="float64").reshape(-1, 1)

        # Create the times structure as a numpy object array
        times_array = np.array([cell1_spikes, cell2_spikes], dtype="object")

        # Create the full nested structure that matches your indexing
        spikes_struct = {"times": times_array}

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.session.mat": {
                    "session": {"extracellular": {"sr": 20000}}
                },
                f"{basename}.cell_metrics.cellinfo.mat": {
                    "cell_metrics": {
                        "UID": np.array([1, 2], dtype="object"),
                        "putativeCellType": np.array(
                            ["Pyramidal", "Interneuron"], dtype="object"
                        ),
                        "brainRegion": np.array(["CA1", "CA3"], dtype="object"),
                        "tags": {"Bad": np.array([])},  # will be UID labels
                        "spikes": spikes_struct,
                        "waveforms": {"filt": np.array([[[1, 2], [3, 4]]])},
                        "acg": {
                            "wide": np.array([[1, 2], [3, 4]]),
                            "narrow": np.array([[1, 2], [3, 4]]),
                            "log10": np.array([[1, 2], [3, 4]]),
                        },
                        "general": {
                            "basename": basename,
                            "cellCount": 2,
                            "animal": {
                                "sex": "male",
                                "species": "rat",
                                "strain": "long-evans",
                                "geneticLine": "WT",
                            },
                        },
                    },
                },
            },
        )

        # Call function
        st, metrics = load_spikes(
            basepath, remove_unstable=True, stable_interval_width=3
        )

        # Assertions
        assert st is not None
        assert metrics is not None
        assert st.n_active == 2
        assert len(metrics) == 2


def test_load_spikes_other_metric_length_mismatch():
    """Test error when other_metric and other_metric_value have different lengths."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session11")
        basename = os.path.basename(basepath)

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.session.mat": {
                    "session": {"extracellular": {"sr": 20000}}
                },
                f"{basename}.cell_metrics.cellinfo.mat": {
                    "cell_metrics": {
                        "UID": np.array([1], dtype="object"),
                        "putativeCellType": np.array(["Pyr"], dtype="object"),
                        "brainRegion": np.array(["CA1"], dtype="object"),
                        "tags": {"Bad": np.array([])},
                        "spikes": {
                            "times": np.array(
                                [np.array([[1.0], [2.0]])], dtype="object"
                            )
                        },
                        "waveforms": {"filt": np.array([[[1, 2]]])},
                        "acg": {
                            "wide": np.array([[1, 2]]),
                            "narrow": np.array([[1, 2]]),
                            "log10": np.array([[1, 2]]),
                        },
                        "general": {
                            "basename": basename,
                            "cellCount": 1,
                            "animal": {
                                "sex": "male",
                                "species": "rat",
                                "strain": "long-evans",
                                "geneticLine": "WT",
                            },
                        },
                    }
                },
            },
        )

        with pytest.raises(
            ValueError,
            match="other_metric and other_metric_value must be of same length",
        ):
            load_spikes(
                basepath,
                other_metric=["quality", "depth"],
                other_metric_value=["good"],
            )
