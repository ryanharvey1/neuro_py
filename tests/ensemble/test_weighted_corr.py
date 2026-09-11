import warnings

import numpy as np
from numpy.testing import assert_allclose

from neuro_py.ensemble.replay import weighted_correlation


class TestWeightedCorrelation:
    """Test suite for weighted_correlation function."""

    def test_basic_functionality(self):
        """Test the basic functionality with simple inputs."""
        # Create a simple 2D posterior (space x time)
        posterior = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float64
        )
        time = np.array([0, 1, 2], dtype=np.float64)
        place_bin_centers = np.array([0, 1, 2], dtype=np.float64)

        result = weighted_correlation(posterior, time, place_bin_centers)

        # Result should be a float (correlation coefficient)
        assert isinstance(result, (float, np.floating))
        # Correlation should be between -1 and 1
        assert -1 <= result <= 1

    def test_full_output(self):
        """Test with return_full_output=True."""
        posterior = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float64
        )
        time = np.array([0, 1, 2], dtype=np.float64)
        place_bin_centers = np.array([0, 1, 2], dtype=np.float64)

        result = weighted_correlation(
            posterior, time, place_bin_centers, return_full_output=True
        )

        # Result should be a tuple with 6 elements
        assert isinstance(result, tuple)
        assert len(result) == 6

        (
            correlation,
            place_trajectory,
            slope_place,
            mean_time,
            mean_place,
            intercept_place,
        ) = result

        # Check types
        assert isinstance(correlation, (float, np.floating))
        assert isinstance(place_trajectory, np.ndarray)
        assert isinstance(slope_place, (float, np.floating))
        assert isinstance(mean_time, (float, np.floating))
        assert isinstance(mean_place, (float, np.floating))

        # Check shapes
        assert place_trajectory.shape == time.shape

        # Check correlation bounds
        assert -1 <= correlation <= 1

    def test_default_parameters(self):
        """Test that default parameters are created correctly."""
        posterior = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float64
        )

        # Call with no time or place_bin_centers
        result = weighted_correlation(posterior)

        # Should create default arrays: time=[0,1,2], place_bin_centers=[0,1,2]
        assert isinstance(result, (float, np.floating))
        assert -1 <= result <= 1

    def test_perfect_positive_correlation(self):
        """Test with data that should yield perfect positive correlation."""
        # Create posterior where weight increases with both time and place
        time = np.array([0, 1, 2], dtype=np.float64)
        place_bin_centers = np.array([0, 1, 2], dtype=np.float64)

        # Weight concentrated along diagonal (increasing place with time)
        posterior = np.array(
            [[1.0, 0.1, 0.1], [0.1, 1.0, 0.1], [0.1, 0.1, 1.0]], dtype=np.float64
        )

        result = weighted_correlation(posterior, time, place_bin_centers)

        # Should have strong positive correlation
        assert result > 0.7

    def test_perfect_negative_correlation(self):
        """Test with data that should yield perfect negative correlation."""
        # Create posterior where weight decreases in place as time increases
        time = np.array([0, 1, 2], dtype=np.float64)
        place_bin_centers = np.array([0, 1, 2], dtype=np.float64)

        # Weight concentrated along anti-diagonal
        posterior = np.array(
            [[0.1, 0.1, 1.0], [0.1, 1.0, 0.1], [1.0, 0.1, 0.1]], dtype=np.float64
        )

        result = weighted_correlation(posterior, time, place_bin_centers)

        # Should have strong negative correlation
        assert result < -0.7

    def test_zero_correlation(self):
        """Test with data that should yield near-zero correlation."""
        # Uniform posterior - no relationship between time and place
        posterior = np.ones((3, 3), dtype=np.float64)
        time = np.array([0, 1, 2], dtype=np.float64)
        place_bin_centers = np.array([0, 1, 2], dtype=np.float64)

        result = weighted_correlation(posterior, time, place_bin_centers)

        # Correlation should be close to zero (or NaN if degenerate)
        # With uniform weights, there's no covariance structure
        assert np.isnan(result) or abs(result) < 0.1

    def test_trajectory_calculation(self):
        """Test that trajectory is calculated correctly."""
        posterior = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float64
        )
        time = np.array([0, 1, 2], dtype=np.float64)
        place_bin_centers = np.array([0, 1, 2], dtype=np.float64)

        result = weighted_correlation(
            posterior, time, place_bin_centers, return_full_output=True
        )

        (
            correlation,
            place_trajectory,
            slope_place,
            mean_time,
            mean_place,
            intercept_place,
        ) = result

        # Trajectory should be: mean_place + slope_place * (time - mean_time)
        expected_trajectory = mean_place + slope_place * (time - mean_time)
        assert_allclose(place_trajectory, expected_trajectory)
        # Intercept should equal mean_place - slope_place * mean_time
        assert_allclose(intercept_place, mean_place - slope_place * mean_time)

    def test_nan_handling(self):
        """Test that NaN values in posterior are handled correctly."""
        posterior = np.array(
            [[0.1, np.nan, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float64
        )
        time = np.array([0, 1, 2], dtype=np.float64)
        place_bin_centers = np.array([0, 1, 2], dtype=np.float64)

        result = weighted_correlation(posterior, time, place_bin_centers)

        # Function should convert NaNs to 0, result should not be NaN
        # (or it could be NaN if the remaining data is degenerate)
        assert isinstance(result, (float, np.floating))

    def test_all_zeros(self):
        """Test behavior with all-zero posterior."""
        posterior = np.zeros((3, 3), dtype=np.float64)
        time = np.array([0, 1, 2], dtype=np.float64)
        place_bin_centers = np.array([0, 1, 2], dtype=np.float64)

        # Suppress expected warning for division by zero
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = weighted_correlation(posterior, time, place_bin_centers)

        # With zero weights, result should be NaN or 0
        assert np.isnan(result) or result == 0

    def test_single_time_bin(self):
        """Test with a single time bin."""
        posterior = np.array([[0.1], [0.4], [0.7]], dtype=np.float64)
        time = np.array([0], dtype=np.float64)
        place_bin_centers = np.array([0, 1, 2], dtype=np.float64)

        # Suppress expected warning for division by zero (no temporal variance)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = weighted_correlation(posterior, time, place_bin_centers)

        # With single time bin, no temporal covariance
        assert np.isnan(result) or result == 0

    def test_single_place_bin(self):
        """Test with a single place bin."""
        posterior = np.array([[0.1, 0.2, 0.3]], dtype=np.float64)
        time = np.array([0, 1, 2], dtype=np.float64)
        place_bin_centers = np.array([0], dtype=np.float64)

        # With single place bin (becomes scalar after squeeze), indexing fails
        # This is an edge case that should either raise an error or return NaN
        try:
            result = weighted_correlation(posterior, time, place_bin_centers)
            # If it doesn't raise an error, result should be NaN or 0 (no spatial variance)
            assert np.isnan(result) or result == 0
        except (IndexError, ValueError):
            # Expected behavior for this edge case
            pass

    def test_different_shapes(self):
        """Test with different posterior shapes."""
        # Tall posterior (more place bins than time bins)
        posterior = np.random.rand(10, 5)
        time = np.arange(5)
        place_bin_centers = np.arange(10)

        result = weighted_correlation(posterior, time, place_bin_centers)
        assert isinstance(result, (float, np.floating))
        assert -1 <= result <= 1 or np.isnan(result)

        # Wide posterior (more time bins than place bins)
        posterior = np.random.rand(5, 10)
        time = np.arange(10)
        place_bin_centers = np.arange(5)

        result = weighted_correlation(posterior, time, place_bin_centers)
        assert isinstance(result, (float, np.floating))
        assert -1 <= result <= 1 or np.isnan(result)

    def test_custom_coordinates(self):
        """Test with custom coordinate values."""
        posterior = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float64
        )
        time = np.array([10, 20, 30], dtype=np.float64)  # Non-standard time values
        place_bin_centers = np.array(
            [100, 200, 300], dtype=np.float64
        )  # Non-standard place values

        result = weighted_correlation(
            posterior, time, place_bin_centers, return_full_output=True
        )

        (
            correlation,
            place_trajectory,
            slope_place,
            mean_time,
            mean_place,
            intercept_place,
        ) = result

        # Check that means reflect the custom values
        assert 10 <= mean_time <= 30
        assert 100 <= mean_place <= 300

        # Trajectory should still be valid
        assert place_trajectory.shape == time.shape

    def test_slope_calculation(self):
        """Test that slope is calculated correctly."""
        # Create a clear linear relationship
        time = np.array([0, 1, 2, 3, 4], dtype=np.float64)
        place_bin_centers = np.array([0, 10, 20, 30, 40], dtype=np.float64)

        # Put all weight on diagonal to create perfect linear relationship
        posterior = np.eye(5, dtype=np.float64)

        result = weighted_correlation(
            posterior, time, place_bin_centers, return_full_output=True
        )

        (
            correlation,
            place_trajectory,
            slope_place,
            mean_time,
            mean_place,
            intercept_place,
        ) = result
        # Intercept should be approximately 0 for perfect diagonal
        assert_allclose(intercept_place, 0.0, rtol=1e-5, atol=1e-8)

        # Slope should be approximately 10 (place increases by 10 for each unit time)
        assert_allclose(slope_place, 10.0, rtol=1e-5)

        # Correlation should be perfect
        assert_allclose(correlation, 1.0, rtol=1e-5)

    def test_consistency_between_outputs(self):
        """Test that full output is consistent with simple output."""
        posterior = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float64
        )
        time = np.array([0, 1, 2], dtype=np.float64)
        place_bin_centers = np.array([0, 1, 2], dtype=np.float64)

        # Get simple output
        simple_result = weighted_correlation(posterior, time, place_bin_centers)

        # Get full output
        full_result = weighted_correlation(
            posterior, time, place_bin_centers, return_full_output=True
        )

        # Correlation from full output should match simple output
        assert_allclose(simple_result, full_result[0])

    def test_weighted_mean_calculation(self):
        """Test that weighted means are calculated correctly."""
        # Create a posterior where we can manually calculate expected means
        posterior = np.array(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=np.float64
        )
        time = np.array([0, 1, 2], dtype=np.float64)
        place_bin_centers = np.array([10, 20, 30], dtype=np.float64)

        result = weighted_correlation(
            posterior, time, place_bin_centers, return_full_output=True
        )

        (
            correlation,
            place_trajectory,
            slope_place,
            mean_time,
            mean_place,
            intercept_place,
        ) = result
        # Intercept formula consistency
        assert_allclose(
            intercept_place, mean_place - slope_place * mean_time, rtol=1e-5
        )

        # Manual calculation of weighted means
        # Weight at (place=10, time=0) = 1.0
        # Weight at (place=20, time=1) = 2.0
        # Weight at (place=30, time=2) = 3.0
        total_weight = 1.0 + 2.0 + 3.0  # = 6.0
        expected_mean_time = (
            1.0 * 0 + 2.0 * 1 + 3.0 * 2
        ) / total_weight  # = 8/6 = 1.333
        expected_mean_place = (
            1.0 * 10 + 2.0 * 20 + 3.0 * 30
        ) / total_weight  # = 140/6 = 23.333

        assert_allclose(mean_time, expected_mean_time, rtol=1e-5)
        assert_allclose(mean_place, expected_mean_place, rtol=1e-5)
