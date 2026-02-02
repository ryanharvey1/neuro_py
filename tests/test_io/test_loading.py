import os
import tempfile
import warnings
from unittest.mock import patch

import nelpy as nel
import numpy as np
import pandas as pd
import pytest
import scipy.io as sio

from neuro_py.io.loading import (
    load_brain_regions,
    load_SleepState_states,
    load_animal_behavior,
    load_cell_metrics,
    load_epoch,
    load_manipulation,
    load_ripples_events,
    load_spikes,
    load_theta_cycles,
    load_trials,
)


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


def test_load_animal_behavior_basic():
    """Test basic successful loading of animal behavior data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_behavior")
        basename = os.path.basename(basepath)

        behavior = {
            "timestamps": np.array([0.0, 1.0, 2.0, 3.0]),
            "position": {
                "x": np.array([0.0, 1.0, 2.0, 3.0]),
                "y": np.array([0.0, 1.0, 0.0, 1.0]),
                "z": np.array([0.0, 0.0, 0.0, 0.0]),
            },
            "linearized": np.array([0.0, 1.0, 2.0, 3.0]),
            "trials": np.array([[1.0, 2.0]]),
            "states": np.array([[0, 1], [1, 0], [0, 1], [1, 0]]),
        }

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.animal.behavior.mat": {"behavior": behavior},
                f"{basename}.session.mat": {
                    "session": {
                        "epochs": [
                            {
                                "name": "task",
                                "startTime": 0.0,
                                "stopTime": 3.0,
                                "environment": "box",
                            }
                        ]
                    }
                },
            },
        )

        df = load_animal_behavior(basepath)

        assert not df.empty
        for col in [
            "time",
            "x",
            "y",
            "z",
            "linearized",
            "speed",
            "acceleration",
            "epochs",
            "environment",
        ]:
            assert col in df.columns

        assert "states" not in df.columns
        assert (df["epochs"] == "task").all()
        assert (df["environment"] == "box").all()
        if "trials" in df.columns:
            assert df.loc[df["time"].between(1.0, 2.0), "trials"].eq(0).all()


def test_load_animal_behavior_filters_random_fields():
    """Test that random/invalid fields are filtered out from behavior payloads."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_behavior_extra")
        basename = os.path.basename(basepath)

        behavior = {
            "timestamps": np.array([0.0, 1.0, 2.0, 3.0]),
            "position": {
                "x": np.array([0.0, 1.0, 2.0, 3.0]),
                "y": np.array([0.0, 1.0, 0.0, 1.0]),
            },
            "linearized": np.array([0.0, 1.0, 2.0, 3.0]),
            "trials": np.array([[1.0, 2.0]]),
            "notes": "freeform text",
            "random_dict": {"foo": 1, "bar": 2},
            "empty_array": np.array([]),
            "wrong_len": np.array([1.0, 2.0]),
            "two_d": np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
        }

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.animal.behavior.mat": {"behavior": behavior},
                f"{basename}.session.mat": {
                    "session": {
                        "epochs": [
                            {
                                "name": "task",
                                "startTime": 0.0,
                                "stopTime": 3.0,
                                "environment": "box",
                            }
                        ]
                    }
                },
            },
        )

        df = load_animal_behavior(basepath)

        assert not df.empty
        assert "random_dict" not in df.columns
        assert "wrong_len" not in df.columns
        assert "two_d" not in df.columns
        assert "empty_array" not in df.columns
        assert "notes" in df.columns
        assert (df["epochs"] == "task").all()


def test_load_epoch_basic():
    """Test basic successful loading of epochs with missing columns filled."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_epoch")
        basename = os.path.basename(basepath)

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.session.mat": {
                    "session": {
                        "epochs": {
                            "name": "task",
                            "startTime": 0.0,
                            "stopTime": 3.0,
                        }
                    }
                }
            },
        )

        epoch_df = load_epoch(basepath)

        assert not epoch_df.empty
        for col in [
            "name",
            "startTime",
            "stopTime",
            "environment",
            "manipulation",
            "behavioralParadigm",
            "stimuli",
            "notes",
            "basepath",
        ]:
            assert col in epoch_df.columns
        assert epoch_df.name.iloc[0] == "task"
        assert epoch_df.basepath.iloc[0] == basepath


def test_load_epoch_with_distractor_fields():
    """Test that distractor fields are preserved when loading epochs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_epoch_extra")
        basename = os.path.basename(basepath)

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.session.mat": {
                    "session": {
                        "epochs": [
                            {
                                "name": "sleep",
                                "startTime": 0.0,
                                "stopTime": 10.0,
                                "environment": "home",
                                "extraField": "ignore_me",
                                "weirdMetric": 123,
                            }
                        ]
                    }
                }
            },
        )

        epoch_df = load_epoch(basepath)

        assert not epoch_df.empty
        assert "extraField" in epoch_df.columns
        assert "weirdMetric" in epoch_df.columns
        assert epoch_df.extraField.iloc[0] == "ignore_me"
        assert epoch_df.weirdMetric.iloc[0] == 123


def test_load_cell_metrics_only_metrics():
    """Test loading cell metrics with only_metrics=True."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_metrics")
        basename = os.path.basename(basepath)

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.cell_metrics.cellinfo.mat": {
                    "cell_metrics": {
                        "UID": np.array([1, 2, 3], dtype="object"),
                        "putativeCellType": np.array(
                            ["Pyramidal", "Interneuron", "Pyramidal"], dtype="object"
                        ),
                        "brainRegion": np.array(
                            ["CA1", "CA3", "CA1"], dtype="object"
                        ),
                        "tags": {},
                        "spikes": {
                            "times": np.array(
                                [
                                    np.array([[1.0], [2.0]]),
                                    np.array([[3.0], [4.0]]),
                                    np.array([[5.0], [6.0]]),
                                ],
                                dtype="object",
                            )
                        },
                        "waveforms": {
                            "filt": np.array([[[1, 2]], [[3, 4]], [[5, 6]]])
                        },
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

        df = load_cell_metrics(basepath, only_metrics=True)

        assert df is not None
        assert len(df) == 3
        assert "UID" in df.columns
        assert "putativeCellType" in df.columns
        assert "brainRegion" in df.columns
        assert "basename" in df.columns
        assert "basepath" in df.columns
        assert df.basepath.iloc[0] == basepath


def test_load_cell_metrics_with_data():
    """Test loading cell metrics with only_metrics=False returns data dict."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_metrics_full")
        basename = os.path.basename(basepath)

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.cell_metrics.cellinfo.mat": {
                    "cell_metrics": {
                        "UID": np.array([1, 2], dtype="object"),
                        "putativeCellType": np.array(
                            ["Pyramidal", "Interneuron"], dtype="object"
                        ),
                        "brainRegion": np.array(["CA1", "CA3"], dtype="object"),
                        "tags": {"Bad": np.array([])},
                        "spikes": {
                            "times": np.array(
                                [
                                    np.array([[1.0], [2.0]]),
                                    np.array([[3.0], [4.0]]),
                                ],
                                dtype="object",
                            )
                        },
                        "waveforms": {
                            "filt": np.array([[[1, 2]], [[3, 4]]])
                        },
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

        df, data = load_cell_metrics(basepath, only_metrics=False)

        assert df is not None
        assert data is not None
        assert len(df) == 2
        assert isinstance(data, dict)
        assert "spikes" in data
        assert "waveforms" in data
        assert "acg_wide" in data
        assert len(data["spikes"]) == 2


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
        # cell2_spikes = np.sort(np.random.poisson(4, size=100)).reshape(-1, 1)
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
        assert st.n_active == 1
        assert len(metrics) == 1


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


def test_load_SleepState_states():
    """Test basic successful loading of SleepState"""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session11")
        basename = os.path.basename(basepath)

        statenames = np.array(
            [
                "WAKE",
                np.array([], dtype="<U1"),
                "NREM",
                np.array([], dtype="<U1"),
                "REM",
            ],
            dtype=object,
        )

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.SleepState.states.mat": {
                    "SleepState": {
                        "idx": {
                            "statenames": statenames,
                            "states": np.array([1, 2, 3]),
                            "timestamps": np.array([[0, 1], [1, 2], [2, 3]]),
                        },
                        "ints": {
                            "WAKEstate": np.array([[0, 1], [2, 3]]),
                            "NREMstate": np.array([[1, 2], [3, 4]]),
                            "REMstate": np.array([[2, 3], [4, 5]]),
                            "THETA": np.array([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]]),
                            "nonTHETA": np.array([[1.5, 2.5], [3.5, 4.5]]),
                        },
                    }
                }
            },
        )

        # Call the function under test
        result = load_SleepState_states(basepath)

        # Assertions
        assert result is not None
        assert isinstance(result, dict)
        assert result["wake_id"] == 1
        assert result["rem_id"] == 5
        assert "NREMstate" in result
        assert isinstance(result["states"], np.ndarray)
        assert result["THETA"].shape == (3, 2)


def test_load_SleepState_states_missing_statenames():
    """Test successful loading of SleepState if keywords are missing in SleepState.idx.statenames"""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session11")
        basename = os.path.basename(basepath)

        statenames = np.array(["NREM", np.array([], dtype="<U1"), "REM"], dtype=object)

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.SleepState.states.mat": {
                    "SleepState": {
                        "idx": {
                            "statenames": statenames,
                            "states": np.array([1, 2, 3]),
                            "timestamps": np.array([[0, 1], [1, 2], [2, 3]]),
                        },
                        "ints": {
                            "WAKEstate": np.array([[0, 1], [2, 3]]),
                            "NREMstate": np.array([[1, 2], [3, 4]]),
                            "REMstate": np.array([[2, 3], [4, 5]]),
                            "THETA": np.array([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]]),
                            "nonTHETA": np.array([[1.5, 2.5], [3.5, 4.5]]),
                        },
                    }
                }
            },
        )

        # Call the function under test
        result = load_SleepState_states(basepath)

        # Assertions
        assert result["wake_id"] is None
        assert result["rem_id"] == 3


def test_load_ripples_events_basic():
    """Test basic successful loading of ripple events."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_ripples")
        basename = os.path.basename(basepath)

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.ripples.events.mat": {
                    "ripples": {
                        "timestamps": np.array([[0.1, 0.15], [0.5, 0.55], [1.0, 1.05]]),
                        "peaks": np.array([[0.125], [0.525], [1.025]]),
                        "amplitude": np.array([[100.0], [150.0], [120.0]]),
                        "duration": np.array([[0.05], [0.05], [0.05]]),
                        "frequency": np.array([[150.0], [160.0], [155.0]]),
                        "peakNormedPower": np.array([[2.5], [3.0], [2.8]]),
                        "detectorName": "auto_detector",
                        "detectorinfo": {
                            "detectorname": "auto_detector",
                            "detectionparms": {
                                "Channels": 10,
                            },
                        },
                    }
                },
            },
        )

        df = load_ripples_events(basepath)

        assert not df.empty
        assert len(df) == 3
        assert "start" in df.columns
        assert "stop" in df.columns
        assert "peaks" in df.columns
        assert "amplitude" in df.columns
        assert "duration" in df.columns
        assert "frequency" in df.columns
        assert "detectorName" in df.columns
        assert "ripple_channel" in df.columns
        assert "basepath" in df.columns
        assert df.start.iloc[0] == 0.1
        assert df.stop.iloc[0] == 0.15
        assert df.peaks.iloc[0] == 0.125


def test_load_ripples_events_epoch_array():
    """Test loading ripple events as EpochArray."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_ripples_epoch")
        basename = os.path.basename(basepath)

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.ripples.events.mat": {
                    "ripples": {
                        "timestamps": np.array([[0.1, 0.15], [0.5, 0.55]]),
                        "peaks": np.array([[0.125], [0.525]]),
                        "amplitude": np.array([[100.0], [150.0]]),
                        "duration": np.array([[0.05], [0.05]]),
                        "frequency": np.array([[150.0], [160.0]]),
                        "detectorName": "auto_detector",
                    }
                },
            },
        )

        epochs = load_ripples_events(basepath, return_epoch_array=True)

        assert epochs is not None
        assert isinstance(epochs, nel.EpochArray)
        assert epochs.n_intervals == 2


def test_load_theta_cycles_basic():
    """Test basic successful loading of theta cycles."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_theta")
        basename = os.path.basename(basepath)

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.thetacycles.events.mat": {
                    "thetacycles": {
                        "timestamps": np.array([[0.1, 0.25], [0.25, 0.4], [0.4, 0.55]]),
                        "duration": np.array([0.15, 0.15, 0.15]),
                        "center": np.array([0.175, 0.325, 0.475]),
                        "peaks": np.array([0.2, 0.35, 0.5]),
                        "detectorinfo": {"theta_channel": 15},
                    }
                },
            },
        )

        df = load_theta_cycles(basepath)

        assert not df.empty
        assert len(df) == 3
        assert "start" in df.columns
        assert "stop" in df.columns
        assert "duration" in df.columns
        assert "center" in df.columns
        assert "trough" in df.columns
        assert "theta_channel" in df.columns
        assert df.start.iloc[0] == 0.1
        assert df.stop.iloc[0] == 0.25
        assert df.theta_channel.iloc[0] == 15


def test_load_theta_cycles_epoch_array():
    """Test loading theta cycles as EpochArray."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_theta_epoch")
        basename = os.path.basename(basepath)

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.thetacycles.events.mat": {
                    "thetacycles": {
                        "timestamps": np.array([[0.1, 0.25], [0.25, 0.4]]),
                        "duration": np.array([0.15, 0.15]),
                        "center": np.array([0.175, 0.325]),
                        "peaks": np.array([0.2, 0.35]),
                        "detectorinfo": {"theta_channel": 15},
                    }
                },
            },
        )

        epochs = load_theta_cycles(basepath, return_epoch_array=True)

        assert epochs is not None
        assert isinstance(epochs, nel.EpochArray)
        assert epochs.n_intervals == 2


def test_load_manipulation_basic():
    """Test basic successful loading of manipulation events."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_manip")
        basename = os.path.basename(basepath)

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.optoStim.manipulation.mat": {
                    "optoStim": {
                        "timestamps": np.array([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]),
                        "peaks": np.array([[1.25], [2.25], [3.25]]),
                        "center": np.array([[1.25], [2.25], [3.25]]),
                        "duration": np.array([[0.5], [0.5], [0.5]]),
                        "amplitude": np.array([[100.0], [100.0], [100.0]]),
                        "amplitudeUnits": "mW",
                        "eventIDlabels": np.array(["stim_on", "stim_off"], dtype=object),
                        "eventID": np.array([[1], [1], [2]]),
                    }
                },
                f"{basename}.session.mat": {
                    "session": {
                        "epochs": {
                            "name": "task",
                            "startTime": 0.0,
                            "stopTime": 10.0,
                        }
                    }
                },
            },
        )

        result = load_manipulation(basepath, struct_name="optoStim")

        assert result is not None
        assert isinstance(result, dict)
        assert "stim_on" in result
        assert "stim_off" in result
        assert result["stim_on"].n_intervals == 2
        assert result["stim_off"].n_intervals == 1


def test_load_manipulation_dataframe():
    """Test loading manipulation as DataFrame."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_manip_df")
        basename = os.path.basename(basepath)

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.optoStim.manipulation.mat": {
                    "optoStim": {
                        "timestamps": np.array([[1.0, 1.5], [2.0, 2.5]]),
                        "peaks": np.array([[1.25], [2.25]]),
                        "center": np.array([[1.25], [2.25]]),
                        "duration": np.array([[0.5], [0.5]]),
                        "amplitude": np.array([[100.0], [100.0]]),
                        "amplitudeUnits": "mW",
                        "eventIDlabels": np.array(["stim_on"], dtype=object),
                        "eventID": np.array([[1], [1]]),
                    }
                },
            },
        )

        df = load_manipulation(
            basepath, struct_name="optoStim", return_epoch_array=False
        )

        assert df is not None
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "start" in df.columns
        assert "stop" in df.columns
        assert "peaks" in df.columns
        assert "amplitude" in df.columns
        assert "ev_label" in df.columns
        assert df.start.iloc[0] == 1.0
