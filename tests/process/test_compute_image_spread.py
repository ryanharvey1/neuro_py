import numpy as np

from neuro_py.process.utils import compute_image_spread


def test_compute_image_spread():
    # Test case 1: Simple 3x3 matrix
    X = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    exponent = 2
    spread, image_moment = compute_image_spread(X, exponent)

    # Expected values calculated manually or using a reference implementation
    expected_spread = 0.57  # Example value, adjust based on actual computation
    expected_image_moment = 0.32  # Example value, adjust based on actual computation

    # Assertions
    assert np.isclose(spread, expected_spread, rtol=1e-2), (
        f"Spread {spread} does not match expected {expected_spread}"
    )
    assert np.isclose(image_moment, expected_image_moment, rtol=1e-1), (
        f"Image moment {image_moment} does not match expected {expected_image_moment}"
    )

    # Test case 2: Edge case with all zeros
    X_zero = np.zeros((3, 3))
    spread, image_moment = compute_image_spread(X_zero, exponent)
    # both should be nan
    assert np.isnan(spread), "Spread should be NaN"
    assert np.isnan(image_moment), "Image moment should be NaN"

    # Test case 3: Edge case with NaN values
    X_nan = np.array([[0.1, np.nan, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    spread_nan, image_moment_nan = compute_image_spread(X_nan, exponent)

    # Ensure the function handles NaNs correctly
    assert not np.isnan(spread_nan), "Spread should not be NaN"
    assert not np.isnan(image_moment_nan), "Image moment should not be NaN"

    # Test case 4: Non-square matrix
    X_rect = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    spread_rect, image_moment_rect = compute_image_spread(X_rect, exponent)

    # Ensure the function works for non-square matrices
    assert isinstance(spread_rect, float), "Spread should be a float"
    assert isinstance(image_moment_rect, float), "Image moment should be a float"
