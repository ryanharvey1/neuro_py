import numpy as np
import pandas as pd

# Import the function to be tested
from neuro_py.behavior.preprocessing import filter_tracker_jumps


def test_filter_tracker_jumps():
    """
    Test the filter_tracker_jumps function.
    """
    # Create a sample DataFrame with tracker jumps
    data = {
        "x": [0, 1, 2, 100, 3, 4],  # Example x coordinates with a jump
        "y": [0, 1, 2, 100, 3, 4],  # Example y coordinates with a jump
        "ts": [0, 1, 2, 3, 4, 5],  # Example timestamps
    }
    beh_df = pd.DataFrame(data)

    # Expected output after filtering jumps
    expected_data = {
        "x": [0, 1, 2, np.nan, 3, 4],  # Jump replaced with NaN
        "y": [0, 1, 2, np.nan, 3, 4],  # Jump replaced with NaN
        "ts": [0, 1, 2, 3, 4, 5],  # Timestamps remain unchanged
    }
    expected_df = pd.DataFrame(expected_data)

    # Call the function to filter jumps
    filtered_df = filter_tracker_jumps(beh_df, max_speed=100)

    # Check if the output matches the expected DataFrame
    pd.testing.assert_frame_equal(filtered_df, expected_df)

def test_filter_tracker_jumps_multi_jumps():
    """
    Test the filter_tracker_jumps function.
    """
    # Create a sample DataFrame with tracker jumps
    data = {
        "x": [0, 1, 2, 100, 3, 4, 100, 5],  # Example x coordinates with a jump
        "y": [0, 1, 2, 100, 3, 4, 100, 5],  # Example y coordinates with a jump
        "ts": [0, 1, 2, 3, 4, 5, 6, 7],  # Example timestamps
    }
    beh_df = pd.DataFrame(data)

    # Expected output after filtering jumps
    expected_data = {
        "x": [0, 1, 2, np.nan, 3, 4, np.nan, 5],  # Jump replaced with NaN
        "y": [0, 1, 2, np.nan, 3, 4, np.nan, 5],  # Jump replaced with NaN
        "ts": [0, 1, 2, 3, 4, 5, 6, 7],  # Timestamps remain unchanged
    }
    expected_df = pd.DataFrame(expected_data)

    # Call the function to filter jumps
    filtered_df = filter_tracker_jumps(beh_df, max_speed=100)

    # Check if the output matches the expected DataFrame
    pd.testing.assert_frame_equal(filtered_df, expected_df)

def test_filter_tracker_jumps_no_jumps():
    """
    Test the filter_tracker_jumps function when there are no jumps.
    """
    # Create a sample DataFrame without jumps
    data = {
        "x": [0, 1, 2, 3, 4],  # Example x coordinates without jumps
        "y": [0, 1, 2, 3, 4],  # Example y coordinates without jumps
        "ts": [0, 1, 2, 3, 4],  # Example timestamps
    }
    beh_df = pd.DataFrame(data)

    # Expected output (no changes)
    expected_df = beh_df.copy()

    # Call the function to filter jumps
    filtered_df = filter_tracker_jumps(beh_df, max_speed=100)

    # Check if the output matches the expected DataFrame
    pd.testing.assert_frame_equal(filtered_df, expected_df, check_dtype=False)


def test_filter_tracker_jumps_empty_input():
    """
    Test the filter_tracker_jumps function with an empty DataFrame.
    """
    # Create an empty DataFrame
    beh_df = pd.DataFrame(columns=["x", "y", "ts"])

    # Expected output (empty DataFrame)
    expected_df = beh_df.copy()

    # Call the function to filter jumps
    filtered_df = filter_tracker_jumps(beh_df, max_speed=100)

    # Check if the output matches the expected DataFrame
    pd.testing.assert_frame_equal(filtered_df, expected_df)
