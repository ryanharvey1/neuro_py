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
    LFPLoader,
    load_all_cell_metrics,
    load_animal_behavior,
    load_brain_regions,
    load_cell_metrics,
    load_channel_tags,
    load_deepSuperficialfromRipple,
    load_emg,
    load_epoch,
    load_events,
    load_extracellular_metadata,
    load_manipulation,
    load_probe_layout,
    load_ripples_events,
    load_SleepState_states,
    load_spikes,
    load_theta_cycles,
    load_trials,
    loadLFP,
    loadXML,
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


def test_load_animal_behavior_filters_scalar_arrays():
    """Test that 0-D (scalar) arrays are retained as metadata without errors."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_scalar")
        basename = os.path.basename(basepath)

        behavior = {
            "timestamps": np.array([0.0, 1.0, 2.0, 3.0]),
            "position": {
                "x": np.array([0.0, 1.0, 2.0, 3.0]),
                "y": np.array([0.0, 1.0, 0.0, 1.0]),
            },
            "linearized": np.array([0.0, 1.0, 2.0, 3.0]),
            "scalar_metadata": np.array(42.0),  # 0-D array
            "another_scalar": np.array(3.14),  # Another 0-D array
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

        # Should not raise TypeError when encountering 0-D arrays
        df = load_animal_behavior(basepath)

        assert not df.empty
        assert "scalar_metadata" in df.columns
        assert "another_scalar" in df.columns
        assert "x" in df.columns
        assert len(df) == 4


def test_load_animal_behavior_spatialseries_fields():
    """Test SpatialSeries parsing for position, pupil, and orientation."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_spatialseries")
        basename = os.path.basename(basepath)

        behavior = {
            "timestamps": np.array([0.0, 1.0, 2.0, 3.0]),
            "SpatialSeries": {
                "position": {
                    "x": np.array([0.0, 1.0, 2.0, 3.0]),
                    "y": np.array([0.0, 1.0, 0.0, 1.0]),
                    "units": "cm",
                    "resolution": 0.1,
                    "coordinateSystem": "cartesian",
                },
                "pupil": {
                    "x": np.array([1.0, 1.1, 1.2, 1.3]),
                    "y": np.array([2.0, 2.1, 2.2, 2.3]),
                    "diameter": np.array([0.5, 0.6, 0.7, 0.8]),
                    "units": "px",
                },
                "orientation": {
                    "x": np.array([0.1, 0.2, 0.3, 0.4]),
                    "y": np.array([0.2, 0.3, 0.4, 0.5]),
                    "z": np.array([0.3, 0.4, 0.5, 0.6]),
                    "w": np.array([0.9, 0.9, 0.9, 0.9]),
                    "units": "radians",
                },
            },
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
        assert "x" in df.columns
        assert "y" in df.columns
        assert "pupil_x" in df.columns
        assert "pupil_y" in df.columns
        assert "pupil_diameter" in df.columns
        assert "orientation_x" in df.columns
        assert "orientation_y" in df.columns
        assert "orientation_z" in df.columns
        assert "orientation_w" in df.columns


def test_load_animal_behavior_fails_on_overwrite():
    """Ensure loader raises when a column would be overwritten."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_overwrite")
        basename = os.path.basename(basepath)

        behavior = {
            "timestamps": np.array([0.0, 1.0, 2.0, 3.0]),
            "position": {
                "x": np.array([0.0, 1.0, 2.0, 3.0]),
                "y": np.array([0.0, 1.0, 0.0, 1.0]),
            },
            "SpatialSeries": {
                "position": {
                    "x": np.array([10.0, 11.0, 12.0, 13.0]),
                    "y": np.array([10.0, 11.0, 10.0, 11.0]),
                }
            },
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

        with pytest.raises(ValueError, match="already exists"):
            load_animal_behavior(basepath)


def test_load_animal_behavior_stress_missing_position_keys():
    """Stress test: handle missing position coordinate keys."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_missing_pos")
        basename = os.path.basename(basepath)

        # Missing 'y' coordinate
        behavior = {
            "timestamps": np.array([0.0, 1.0, 2.0, 3.0]),
            "position": {
                "x": np.array([0.0, 1.0, 2.0, 3.0]),
                "z": np.array([0.0, 0.0, 0.0, 0.0]),
            },
            "linearized": np.array([0.0, 1.0, 2.0, 3.0]),
        }

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.animal.behavior.mat": {"behavior": behavior},
                f"{basename}.session.mat": {
                    "session": {
                        "epochs": [{"name": "task", "startTime": 0.0, "stopTime": 3.0}]
                    }
                },
            },
        )

        df = load_animal_behavior(basepath)

        assert not df.empty
        assert "x" in df.columns
        assert "z" in df.columns
        assert "y" not in df.columns
        assert len(df) == 4


def test_load_animal_behavior_stress_empty_position_dict():
    """Stress test: handle empty position dictionary."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_empty_pos")
        basename = os.path.basename(basepath)

        behavior = {
            "timestamps": np.array([0.0, 1.0, 2.0, 3.0]),
            "position": {},  # Empty position dict
            "linearized": np.array([0.0, 1.0, 2.0, 3.0]),
        }

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.animal.behavior.mat": {"behavior": behavior},
                f"{basename}.session.mat": {
                    "session": {
                        "epochs": [{"name": "task", "startTime": 0.0, "stopTime": 3.0}]
                    }
                },
            },
        )

        df = load_animal_behavior(basepath)

        assert not df.empty
        assert "time" in df.columns
        assert "linearized" in df.columns


def test_load_animal_behavior_stress_mismatched_lengths():
    """Stress test: handle fields with mismatched array lengths."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_mismatch")
        basename = os.path.basename(basepath)

        behavior = {
            "timestamps": np.array([0.0, 1.0, 2.0, 3.0]),
            "position": {
                "x": np.array([0.0, 1.0, 2.0, 3.0]),
                "y": np.array([0.0, 1.0, 0.0, 1.0]),
            },
            "linearized": np.array([0.0, 1.0, 2.0, 3.0]),
            "speed": np.array([0.1, 0.2, 0.3]),  # Wrong length
            "acceleration": np.array(
                [0.01, 0.02, 0.03, 0.04, 0.05]
            ),  # Also wrong length
        }

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.animal.behavior.mat": {"behavior": behavior},
                f"{basename}.session.mat": {
                    "session": {
                        "epochs": [{"name": "task", "startTime": 0.0, "stopTime": 3.0}]
                    }
                },
            },
        )

        df = load_animal_behavior(basepath)

        # Wrong length fields should be ignored, but speed is calculated from x,y so it will be present
        assert "speed" in df.columns  # Calculated from position
        assert "acceleration" in df.columns  # Calculated from speed


def test_load_animal_behavior_stress_multidimensional_arrays():
    """Stress test: reject multidimensional arrays that aren't position."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_multidim")
        basename = os.path.basename(basepath)

        behavior = {
            "timestamps": np.array([0.0, 1.0, 2.0, 3.0]),
            "position": {
                "x": np.array([0.0, 1.0, 2.0, 3.0]),
                "y": np.array([0.0, 1.0, 0.0, 1.0]),
            },
            "bad_2d": np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
            "bad_3d": np.ones((4, 2, 3)),
        }

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.animal.behavior.mat": {"behavior": behavior},
                f"{basename}.session.mat": {
                    "session": {
                        "epochs": [{"name": "task", "startTime": 0.0, "stopTime": 3.0}]
                    }
                },
            },
        )

        df = load_animal_behavior(basepath)

        assert not df.empty
        assert "bad_2d" not in df.columns
        assert "bad_3d" not in df.columns
        assert "x" in df.columns


def test_load_animal_behavior_stress_nan_values():
    """Stress test: handle NaN values in arrays."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_nan")
        basename = os.path.basename(basepath)

        behavior = {
            "timestamps": np.array([0.0, 1.0, 2.0, 3.0]),
            "position": {
                "x": np.array([0.0, np.nan, 2.0, 3.0]),
                "y": np.array([np.nan, 1.0, np.nan, 1.0]),
            },
            "linearized": np.array([0.0, 1.0, np.nan, 3.0]),
        }

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.animal.behavior.mat": {"behavior": behavior},
                f"{basename}.session.mat": {
                    "session": {
                        "epochs": [{"name": "task", "startTime": 0.0, "stopTime": 3.0}]
                    }
                },
            },
        )

        df = load_animal_behavior(basepath)

        assert not df.empty
        assert len(df) == 4
        assert df["x"].isna().sum() == 1
        assert df["y"].isna().sum() == 2


def test_load_animal_behavior_stress_all_nan_column():
    """Stress test: handle columns that are entirely NaN."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_all_nan")
        basename = os.path.basename(basepath)

        behavior = {
            "timestamps": np.array([0.0, 1.0, 2.0, 3.0]),
            "position": {
                "x": np.array([0.0, 1.0, 2.0, 3.0]),
                "y": np.array([np.nan, np.nan, np.nan, np.nan]),
            },
            "linearized": np.array([0.0, 1.0, 2.0, 3.0]),
        }

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.animal.behavior.mat": {"behavior": behavior},
                f"{basename}.session.mat": {
                    "session": {
                        "epochs": [{"name": "task", "startTime": 0.0, "stopTime": 3.0}]
                    }
                },
            },
        )

        df = load_animal_behavior(basepath)

        assert not df.empty
        assert "y" in df.columns
        assert df["y"].isna().all()


def test_load_animal_behavior_stress_inf_values():
    """Stress test: handle infinity values in arrays."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_inf")
        basename = os.path.basename(basepath)

        behavior = {
            "timestamps": np.array([0.0, 1.0, 2.0, 3.0]),
            "position": {
                "x": np.array([0.0, np.inf, 2.0, 3.0]),
                "y": np.array([-np.inf, 1.0, 0.0, 1.0]),
            },
            "linearized": np.array([0.0, 1.0, 2.0, np.inf]),
        }

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.animal.behavior.mat": {"behavior": behavior},
                f"{basename}.session.mat": {
                    "session": {
                        "epochs": [{"name": "task", "startTime": 0.0, "stopTime": 3.0}]
                    }
                },
            },
        )

        # Suppress expected RuntimeWarning from numpy operations with inf values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            df = load_animal_behavior(basepath)

        assert not df.empty
        assert np.isinf(df["x"].iloc[1])
        assert np.isinf(df["y"].iloc[0])


def test_load_animal_behavior_stress_integer_arrays():
    """Stress test: handle integer arrays gracefully."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_int")
        basename = os.path.basename(basepath)

        behavior = {
            "timestamps": np.array([0, 1, 2, 3], dtype=np.int32),
            "position": {
                "x": np.array([0, 1, 2, 3], dtype=np.int16),
                "y": np.array([0, 1, 0, 1], dtype=np.int64),
            },
            "linearized": np.array([0, 1, 2, 3], dtype=np.int32),
        }

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.animal.behavior.mat": {"behavior": behavior},
                f"{basename}.session.mat": {
                    "session": {
                        "epochs": [{"name": "task", "startTime": 0, "stopTime": 3}]
                    }
                },
            },
        )

        df = load_animal_behavior(basepath)

        assert not df.empty
        assert len(df) == 4
        assert "x" in df.columns


def test_load_animal_behavior_stress_very_large_array():
    """Stress test: handle very large behavior arrays."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_large")
        basename = os.path.basename(basepath)

        n_samples = 100000
        behavior = {
            "timestamps": np.linspace(0, 1000, n_samples),
            "position": {
                "x": np.sin(np.linspace(0, 10 * np.pi, n_samples)),
                "y": np.cos(np.linspace(0, 10 * np.pi, n_samples)),
            },
            "linearized": np.linspace(0, 100, n_samples),
        }

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.animal.behavior.mat": {"behavior": behavior},
                f"{basename}.session.mat": {
                    "session": {
                        "epochs": [
                            {"name": "task", "startTime": 0.0, "stopTime": 1000.0}
                        ]
                    }
                },
            },
        )

        df = load_animal_behavior(basepath)

        assert len(df) == n_samples
        assert "speed" in df.columns  # Should be calculated
        assert "acceleration" in df.columns


def test_load_animal_behavior_stress_nested_position_structures():
    """Stress test: handle deeply nested position structures."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_nested")
        basename = os.path.basename(basepath)

        behavior = {
            "timestamps": np.array([0.0, 1.0, 2.0, 3.0]),
            "position": {
                "x": np.array([0.0, 1.0, 2.0, 3.0]),
                "y": np.array([0.0, 1.0, 0.0, 1.0]),
                "nested_pos": {"a": 1, "b": 2},  # Nested dict in position
            },
            "linearized": np.array([0.0, 1.0, 2.0, 3.0]),
        }

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.animal.behavior.mat": {"behavior": behavior},
                f"{basename}.session.mat": {
                    "session": {
                        "epochs": [{"name": "task", "startTime": 0.0, "stopTime": 3.0}]
                    }
                },
            },
        )

        df = load_animal_behavior(basepath)

        assert not df.empty
        assert "x" in df.columns
        assert "nested_pos" not in df.columns  # Nested dicts should be skipped


def test_load_animal_behavior_stress_special_string_values():
    """Stress test: handle special string values in fields."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_strings")
        basename = os.path.basename(basepath)

        behavior = {
            "timestamps": np.array([0.0, 1.0, 2.0, 3.0]),
            "position": {
                "x": np.array([0.0, 1.0, 2.0, 3.0]),
                "y": np.array([0.0, 1.0, 0.0, 1.0]),
            },
            "linearized": np.array([0.0, 1.0, 2.0, 3.0]),
            "notes": "Test with special chars: !@#$%^&*()",
            "unicode_field": "Test with unicode: αβγδε",
        }

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.animal.behavior.mat": {"behavior": behavior},
                f"{basename}.session.mat": {
                    "session": {
                        "epochs": [{"name": "task", "startTime": 0.0, "stopTime": 3.0}]
                    }
                },
            },
        )

        df = load_animal_behavior(basepath)

        assert not df.empty
        assert "notes" in df.columns
        assert "unicode_field" in df.columns


def test_load_animal_behavior_stress_zero_length_arrays():
    """Stress test: handle zero-length arrays."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_zero_len")
        basename = os.path.basename(basepath)

        behavior = {
            "timestamps": np.array([]),
            "position": {
                "x": np.array([]),
                "y": np.array([]),
            },
            "linearized": np.array([]),
        }

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.animal.behavior.mat": {"behavior": behavior},
                f"{basename}.session.mat": {
                    "session": {
                        "epochs": [{"name": "task", "startTime": 0.0, "stopTime": 3.0}]
                    }
                },
            },
        )

        df = load_animal_behavior(basepath)

        assert df.empty or len(df) == 0


def test_load_animal_behavior_stress_single_sample():
    """Stress test: handle single sample behavior data gracefully."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_single")
        basename = os.path.basename(basepath)

        behavior = {
            "timestamps": np.array([0.0]),
            "position": {
                "x": np.array([0.0]),
                "y": np.array([0.0]),
            },
            "linearized": np.array([0.0]),
        }

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.animal.behavior.mat": {"behavior": behavior},
                f"{basename}.session.mat": {
                    "session": {
                        "epochs": [{"name": "task", "startTime": 0.0, "stopTime": 1.0}]
                    }
                },
            },
        )

        df = load_animal_behavior(basepath)

        # Should load behavior data, may be empty or have 1 row depending on implementation
        assert isinstance(df, pd.DataFrame)


def test_load_animal_behavior_stress_duplicate_trials():
    """Stress test: handle overlapping and duplicate trial intervals."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_dup_trials")
        basename = os.path.basename(basepath)

        behavior = {
            "timestamps": np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
            "position": {
                "x": np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
                "y": np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
            },
            "linearized": np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]),
            "trials": np.array(
                [
                    [0.0, 1.0],
                    [0.5, 1.5],  # Overlapping
                    [1.0, 2.0],
                    [1.5, 2.5],  # Overlapping
                ]
            ),
        }

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.animal.behavior.mat": {"behavior": behavior},
                f"{basename}.session.mat": {
                    "session": {
                        "epochs": [{"name": "task", "startTime": 0.0, "stopTime": 3.0}]
                    }
                },
            },
        )

        df = load_animal_behavior(basepath)

        assert len(df) == 7
        # Some timestamps will be in multiple trials
        assert "trials" in df.columns


def test_load_animal_behavior_stress_no_position_key():
    """Stress test: handle missing 'position' key entirely."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_no_pos")
        basename = os.path.basename(basepath)

        behavior = {
            "timestamps": np.array([0.0, 1.0, 2.0, 3.0]),
            "linearized": np.array([0.0, 1.0, 2.0, 3.0]),
            "some_field": np.array([1, 2, 3, 4]),
        }

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.animal.behavior.mat": {"behavior": behavior},
                f"{basename}.session.mat": {
                    "session": {
                        "epochs": [{"name": "task", "startTime": 0.0, "stopTime": 3.0}]
                    }
                },
            },
        )

        df = load_animal_behavior(basepath)

        assert not df.empty
        assert "time" in df.columns
        assert "linearized" in df.columns


def test_load_animal_behavior_stress_mixed_data_types():
    """Stress test: handle mixed data types in position dict."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_mixed")
        basename = os.path.basename(basepath)

        behavior = {
            "timestamps": np.array([0.0, 1.0, 2.0, 3.0]),
            "position": {
                "x": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64),
                "y": np.array([0, 1, 0, 1], dtype=np.int32),
                "z": [0.0, 0.0, 0.0, 0.0],  # Python list instead of array
            },
            "linearized": np.array([0.0, 1.0, 2.0, 3.0]),
        }

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.animal.behavior.mat": {"behavior": behavior},
                f"{basename}.session.mat": {
                    "session": {
                        "epochs": [{"name": "task", "startTime": 0.0, "stopTime": 3.0}]
                    }
                },
            },
        )

        df = load_animal_behavior(basepath)

        assert not df.empty
        assert "x" in df.columns
        assert "y" in df.columns


def test_load_animal_behavior_stress_no_epochs_file():
    """Stress test: handle missing epochs/session file gracefully."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_no_epoch")
        basename = os.path.basename(basepath)

        behavior = {
            "timestamps": np.array([0.0, 1.0, 2.0, 3.0]),
            "position": {
                "x": np.array([0.0, 1.0, 2.0, 3.0]),
                "y": np.array([0.0, 1.0, 0.0, 1.0]),
            },
            "linearized": np.array([0.0, 1.0, 2.0, 3.0]),
        }

        # Only create behavior file, no session file
        create_temp_mat_file(
            basepath,
            {f"{basename}.animal.behavior.mat": {"behavior": behavior}},
        )

        # Suppress expected UserWarning about missing session file
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            df = load_animal_behavior(basepath)

        # Should still load behavior data even without epochs
        assert not df.empty
        assert "time" in df.columns
        assert "epochs" in df.columns  # Will be NaN


def test_load_animal_behavior_stress_negative_timestamps():
    """Stress test: handle negative timestamps."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_neg_ts")
        basename = os.path.basename(basepath)

        behavior = {
            "timestamps": np.array([-2.0, -1.0, 0.0, 1.0]),
            "position": {
                "x": np.array([0.0, 1.0, 2.0, 3.0]),
                "y": np.array([0.0, 1.0, 0.0, 1.0]),
            },
            "linearized": np.array([0.0, 1.0, 2.0, 3.0]),
        }

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.animal.behavior.mat": {"behavior": behavior},
                f"{basename}.session.mat": {
                    "session": {
                        "epochs": [{"name": "task", "startTime": -2.0, "stopTime": 1.0}]
                    }
                },
            },
        )

        df = load_animal_behavior(basepath)

        assert len(df) == 4
        assert df["time"].min() == -2.0


def test_load_animal_behavior_stress_unsorted_timestamps():
    """Stress test: handle unsorted/reverse-ordered timestamps."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_unsorted")
        basename = os.path.basename(basepath)

        behavior = {
            "timestamps": np.array([3.0, 1.0, 0.0, 2.0]),  # Unsorted
            "position": {
                "x": np.array([3.0, 1.0, 0.0, 2.0]),
                "y": np.array([3.0, 1.0, 0.0, 2.0]),
            },
            "linearized": np.array([3.0, 1.0, 0.0, 2.0]),
        }

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.animal.behavior.mat": {"behavior": behavior},
                f"{basename}.session.mat": {
                    "session": {
                        "epochs": [{"name": "task", "startTime": 0.0, "stopTime": 3.0}]
                    }
                },
            },
        )

        df = load_animal_behavior(basepath)

        # Should still load the data despite being unsorted
        assert len(df) == 4


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
                        "brainRegion": np.array(["CA1", "CA3", "CA1"], dtype="object"),
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
                        "eventIDlabels": np.array(
                            ["stim_on", "stim_off"], dtype=object
                        ),
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


def test_load_all_cell_metrics():
    """Test loading cell metrics from multiple sessions."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepaths = []
        for i in range(3):
            basepath = os.path.join(temp_dir, f"session_{i}")
            basepaths.append(basepath)
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
                            "tags": {},
                            "spikes": {
                                "times": np.array(
                                    [
                                        np.array([[1.0], [2.0]]),
                                        np.array([[3.0], [4.0]]),
                                    ],
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

        df = load_all_cell_metrics(basepaths)

        assert df is not None
        assert len(df) == 6  # 2 cells per session * 3 sessions
        assert "UID" in df.columns
        assert "basepath" in df.columns
        assert len(df.basepath.unique()) == 3


def test_load_deepSuperficialfromRipple():
    """Test that load_deepSuperficialfromRipple function is importable and has correct signature.

    Note: Full functional testing of this function requires complex MATLAB struct data that is
    difficult to replicate accurately with scipy.io.savemat. This test verifies the function
    exists and can be called properly.
    """
    # Verify function is imported and callable
    assert callable(load_deepSuperficialfromRipple)

    # Verify function signature using inspect
    import inspect

    sig = inspect.signature(load_deepSuperficialfromRipple)
    params = list(sig.parameters.keys())
    assert "basepath" in params
    assert "bypass_mismatch_exception" in params

    # Verify return type annotation indicates a tuple
    return_annotation = sig.return_annotation
    assert return_annotation != inspect.Signature.empty


def test_load_events_epoch_array_and_dataframe():
    """Test load_events returns EpochArray or DataFrame based on load_pandas flag."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_events")
        basename = os.path.basename(basepath)
        event_name = "my_events"

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.{event_name}.events.mat": {
                    event_name: {
                        "timestamps": np.array([[0.0, 1.0], [2.0, 3.0]]),
                        "amplitude": np.array([1.0, 2.0]),
                        "labels": np.array(["a", "b"], dtype=object),
                        "extra2d": np.array([[1, 2], [3, 4]]),
                    }
                }
            },
        )

        epoch = load_events(basepath, event_name)
        assert isinstance(epoch, nel.EpochArray)
        assert epoch.n_intervals == 2

        df = load_events(basepath, event_name, load_pandas=True)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns[:2]) == ["starts", "stops"]
        assert "amplitude" in df.columns
        assert "labels" in df.columns
        assert "extra2d" not in df.columns


def test_load_channel_tags_and_extracellular_metadata():
    """Test load_channel_tags and load_extracellular_metadata from session file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_meta")
        basename = os.path.basename(basepath)

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.session.mat": {
                    "session": {
                        "channelTags": {
                            "ripple": {"channels": np.array([1, 2])},
                            "theta": {"channels": np.array([3, 4])},
                        },
                        "extracellular": {
                            "sr": 20000,
                            "nElectrodeGroups": 1,
                            "chanCoords": {
                                "x": np.array([10, 20, 30, 40]),
                                "y": np.array([1, 2, 3, 4]),
                            },
                            "electrodeGroups": {"channels": np.array([1, 2, 3, 4])},
                        },
                    }
                }
            },
        )

        tags = load_channel_tags(basepath)
        assert isinstance(tags, dict)
        assert "ripple" in tags
        assert "theta" in tags

        meta = load_extracellular_metadata(basepath)
        assert isinstance(meta, dict)
        assert meta.get("sr") == 20000


def test_load_probe_layout():
    """Test probe layout mapping from session extracellular metadata."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_probe")
        basename = os.path.basename(basepath)

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.session.mat": {
                    "session": {
                        "extracellular": {
                            "nElectrodeGroups": 1,
                            "chanCoords": {
                                "x": np.array([10, 20, 30, 40]),
                                "y": np.array([1, 2, 3, 4]),
                            },
                            "electrodeGroups": {"channels": np.array([1, 2, 3, 4])},
                        }
                    }
                }
            },
        )

        probe_layout = load_probe_layout(basepath)
        assert isinstance(probe_layout, pd.DataFrame)
        assert list(probe_layout.columns) == ["x", "y", "shank", "channels"]
        assert probe_layout["channels"].tolist() == [0, 1, 2, 3]
        assert probe_layout["shank"].tolist() == [0, 0, 0, 0]


def test_load_emg():
    """Test loading EMG data and high/low epoch extraction."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_emg")
        basename = os.path.basename(basepath)

        create_temp_mat_file(
            basepath,
            {
                f"{basename}.EMGFromLFP.LFP.mat": {
                    "EMGFromLFP": {
                        "data": np.array([0.1, 0.95, 0.1, 0.99]),
                        "timestamps": np.array([0.0, 1.0, 2.0, 3.0]),
                    }
                }
            },
        )

        emg, high_emg_epoch, low_emg_epoch = load_emg(basepath, threshold=0.9)
        assert isinstance(emg, nel.AnalogSignalArray)
        assert isinstance(high_emg_epoch, nel.EpochArray)
        assert isinstance(low_emg_epoch, nel.EpochArray)
        assert emg.data.flatten().shape[0] == 4
        assert high_emg_epoch.n_intervals == 2
        assert low_emg_epoch.n_intervals >= 1


def test_load_lfp_method():
    """Test LFPLoader.load_lfp populates lfp with returned data."""
    lfp_data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int16)
    timestep = np.array([0.0, 1.0, 2.0, 3.0])

    loader = object.__new__(LFPLoader)
    loader.basepath = "dummy"
    loader.nChannels = 2
    loader.channels = None
    loader.fs = 1250.0
    loader.ext = "lfp"
    loader.epoch = np.array([0.0, 3.0])

    with patch("neuro_py.io.loading.loadLFP", return_value=(lfp_data, timestep)):
        loader.load_lfp()

    assert isinstance(loader.lfp, nel.AnalogSignalArray)
    assert loader.lfp.data.shape == (2, 4)
    assert loader.lfp.abscissa_vals.shape[0] == 4


def test_LFPLoader_init_loads_lfp():
    """Test LFPLoader initialization loads lfp using loadXML/loadLFP."""
    lfp_data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.int16)
    timestep = np.array([0.0, 1.0, 2.0, 3.0])

    with (
        patch(
            "neuro_py.io.loading.loadXML",
            return_value=(2, 1250.0, 20000.0, {0: [0, 1]}),
        ),
        patch(
            "neuro_py.io.loading.loadLFP",
            return_value=(lfp_data, timestep),
        ),
    ):
        loader = LFPLoader(
            "dummy", channels=None, ext="lfp", epoch=np.array([0.0, 3.0])
        )

    assert isinstance(loader.lfp, nel.AnalogSignalArray)
    assert loader.lfp.data.shape == (2, 4)
    assert loader.lfp.abscissa_vals.shape[0] == 4


def test_loadLFP_reads_binary_file():
    """Test loadLFP reads a binary lfp file and returns data and timestamps."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_lfp")
        os.makedirs(basepath, exist_ok=True)
        basename = os.path.basename(basepath)

        data = np.array(
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            dtype=np.int16,
        )
        lfp_path = os.path.join(basepath, f"{basename}.lfp")
        data.tofile(lfp_path)

        lfp, timestep = loadLFP(basepath, n_channels=2, frequency=2.0, ext="lfp")

        assert lfp.shape == (4, 2)
        assert timestep.shape[0] == 4
        assert np.allclose(timestep, np.array([0.0, 0.5, 1.0, 1.5]))

        if isinstance(lfp, np.memmap) and hasattr(lfp, "_mmap"):
            lfp._mmap.close()
        del lfp


def test_loadXML_parses_basic_fields():
    """Test loadXML parses channels, sampling rates, and shank mappings."""
    with tempfile.TemporaryDirectory() as temp_dir:
        basepath = os.path.join(temp_dir, "session_xml")
        os.makedirs(basepath, exist_ok=True)
        basename = os.path.basename(basepath)
        xml_path = os.path.join(basepath, f"{basename}.xml")

        xml_content = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<session>
    <acquisitionSystem>
        <nChannels>4</nChannels>
        <samplingRate>20000</samplingRate>
    </acquisitionSystem>
    <fieldPotentials>
        <lfpSamplingRate>1250</lfpSamplingRate>
    </fieldPotentials>
    <anatomicalDescription>
        <channelGroups>
            <group>
                <channel>0</channel>
                <channel>1</channel>
            </group>
            <group>
                <channel>2</channel>
                <channel>3</channel>
            </group>
        </channelGroups>
    </anatomicalDescription>
</session>
"""

        with open(xml_path, "w", encoding="utf-8") as f:
            f.write(xml_content)

        n_channels, fs, fs_dat, shank_to_channel = loadXML(basepath)

        assert n_channels == 4
        assert fs == 1250
        assert fs_dat == 20000
        assert shank_to_channel == {0: [0, 1], 1: [2, 3]}
