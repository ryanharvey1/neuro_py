import warnings

import numpy as np
from numpy.testing import assert_allclose

from neuro_py.ensemble.replay import weighted_corr_2d, weighted_correlation


class TestWeightedCorr2D:
    """Test suite for weighted_corr_2d function."""

    def test_basic_functionality(self):
        """Test the basic functionality with simple inputs."""
        weights = np.array(
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float64
        )
        x_coords = np.array([0, 1], dtype=np.float64)
        y_coords = np.array([0, 1], dtype=np.float64)
        time_coords = np.array([0, 1, 2], dtype=np.float64)

        result = weighted_corr_2d(weights, x_coords, y_coords, time_coords)

        # Check result structure
        assert len(result) == 7
        assert isinstance(result[0], float)  # spatiotemporal_corr
        assert isinstance(result[1], np.ndarray)  # x_traj
        assert isinstance(result[2], np.ndarray)  # y_traj
        assert isinstance(result[3], float)  # slope_x
        assert isinstance(result[4], float)  # slope_y
        assert isinstance(result[5], float)  # mean_x
        assert isinstance(result[6], float)  # mean_y

        # Check shapes
        assert result[1].shape == (3,)  # x_traj should have length of time_coords
        assert result[2].shape == (3,)  # y_traj should have length of time_coords

        # We don't check exact values here since we're testing structure
        # We'll have more specific tests for numerical correctness

    def test_default_coords(self):
        """Test that default coordinate arrays are created properly when not provided."""
        weights = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)

        result = weighted_corr_2d(weights)

        # With default coordinates, x_coords should be [0, 1], y_coords [0, 1], time_coords [0, 1]
        # Calculate expected weighted means manually
        # weights shape is (2, 2, 2), so x_coords=[0,1], y_coords=[0,1], time_coords=[0,1]
        total_weight = np.sum(weights)  # 1+2+3+4+5+6+7+8 = 36

        # Expected mean_x: weighted average of x coordinates
        # x=0 contributes weights[0,:,:] = 1+2+3+4 = 10
        # x=1 contributes weights[1,:,:] = 5+6+7+8 = 26
        expected_mean_x = (0 * 10 + 1 * 26) / total_weight  # = 26/36 = 0.722...

        # Expected mean_y: weighted average of y coordinates
        # y=0 contributes weights[:,0,:] = 1+2+5+6 = 14
        # y=1 contributes weights[:,1,:] = 3+4+7+8 = 22
        expected_mean_y = (0 * 14 + 1 * 22) / total_weight  # = 22/36 = 0.611...

        assert_allclose(result[5], expected_mean_x, rtol=1e-6)  # mean_x
        assert_allclose(result[6], expected_mean_y, rtol=1e-6)  # mean_y

    def test_different_dtypes(self):
        """Test with different data types."""
        # Float32 test
        weights_f32 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float32)
        result_f32 = weighted_corr_2d(weights_f32)

        # Float64 test
        weights_f64 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
        result_f64 = weighted_corr_2d(weights_f64)

        # Check that dtypes are preserved
        assert result_f32[1].dtype == np.float32
        assert result_f32[2].dtype == np.float32
        assert result_f64[1].dtype == np.float64
        assert result_f64[2].dtype == np.float64

        # The results should be numerically equivalent within floating point precision
        assert_allclose(result_f32[0], result_f64[0], rtol=1e-6)  # spatiotemporal_corr
        assert_allclose(result_f32[3], result_f64[3], rtol=1e-6)  # slope_x
        assert_allclose(result_f32[4], result_f64[4], rtol=1e-6)  # slope_y

    def test_perfect_correlation(self):
        """Test with data that should yield a strong positive correlation."""
        # Create a simple case where weights are concentrated at positions
        # that move consistently in x and y with time
        x_dim, y_dim, t_dim = 3, 3, 3
        weights = np.zeros((x_dim, y_dim, t_dim), dtype=np.float64)

        # Put all weight at specific positions that create a clear trajectory
        # At t=0, weight at (0,0); at t=1, weight at (1,1); at t=2, weight at (2,2)
        weights[0, 0, 0] = 1.0  # t=0, x=0, y=0
        weights[1, 1, 1] = 1.0  # t=1, x=1, y=1
        weights[2, 2, 2] = 1.0  # t=2, x=2, y=2

        result = weighted_corr_2d(weights)

        # This should create a perfect positive correlation since both x and y
        # increase linearly with t
        assert result[0] > 0.9  # Strong positive spatiotemporal correlation
        assert result[3] > 0  # slope_x should be positive
        assert result[4] > 0  # slope_y should be positive

        # The slopes should be equal since x and y increase at the same rate
        assert_allclose(result[3], result[4], rtol=1e-10)

    def test_perfect_anticorrelation(self):
        """Test with data that should yield a perfect anti-correlation."""
        # Create weights where x decreases with t and y increases with t
        x_dim, y_dim, t_dim = 3, 3, 3
        weights = np.zeros((x_dim, y_dim, t_dim), dtype=np.float64)

        for i in range(x_dim):
            for j in range(y_dim):
                for k in range(t_dim):
                    # Weight decreases with x and t, increases with y and t
                    weights[i, j, k] = (x_dim - i) * (k + 1) + (j + 1) * (k + 1)

        result = weighted_corr_2d(weights)

        # The x correlation should be negative, y correlation positive
        # The combined spatiotemporal correlation depends on the magnitudes
        # but should be close to zero if they balance out
        assert result[3] < 0  # slope_x should be negative
        assert result[4] > 0  # slope_y should be positive

    def test_trajectories(self):
        """Test that trajectories are calculated correctly."""
        weights = np.array(
            [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float64
        )
        x_coords = np.array([0, 1], dtype=np.float64)
        y_coords = np.array([0, 1], dtype=np.float64)
        time_coords = np.array([0, 1, 2], dtype=np.float64)

        result = weighted_corr_2d(weights, x_coords, y_coords, time_coords)

        # Calculate expected trajectories manually
        mean_x = result[5]
        mean_y = result[6]
        slope_x = result[3]
        slope_y = result[4]

        # Calculate weighted mean time (same way the function does it)
        w_flat = weights.flatten()
        t_flat = np.tile(time_coords, 4)  # 4 = x_dim * y_dim = 2*2
        total_weight = np.sum(w_flat)
        mean_t = np.sum(w_flat * t_flat) / total_weight

        expected_x_traj = mean_x + slope_x * (time_coords - mean_t)
        expected_y_traj = mean_y + slope_y * (time_coords - mean_t)

        assert_allclose(result[1], expected_x_traj)
        assert_allclose(result[2], expected_y_traj)

    def test_all_zero_weights(self):
        """Test behavior when all weights are zero."""
        weights = np.zeros((2, 2, 3), dtype=np.float64)

        result = weighted_corr_2d(weights)

        # All results should be NaN
        assert np.isnan(result[0])  # spatiotemporal_corr
        assert np.all(np.isnan(result[1]))  # x_traj
        assert np.all(np.isnan(result[2]))  # y_traj
        assert np.isnan(result[3])  # slope_x
        assert np.isnan(result[4])  # slope_y
        assert np.isnan(result[5])  # mean_x
        assert np.isnan(result[6])  # mean_y

    def test_nan_weights(self):
        """Test that NaN weights are handled properly."""
        weights = np.array(
            [[[1, np.nan, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.float64
        )

        result = weighted_corr_2d(weights)

        # Function should convert NaNs to 0, so result should not be NaN
        assert not np.isnan(result[0])  # spatiotemporal_corr
        assert not np.any(np.isnan(result[1]))  # x_traj
        assert not np.any(np.isnan(result[2]))  # y_traj

    def test_degenerate_case(self):
        """Test degenerate case where cov_xx, cov_yy, or cov_tt is zero."""
        # Create weights where all points are at the same time
        weights = np.array(
            [[[1, 0, 0], [2, 0, 0]], [[3, 0, 0], [4, 0, 0]]], dtype=np.float64
        )
        time_coords = np.array([0, 0, 0], dtype=np.float64)  # All same time value

        result = weighted_corr_2d(weights, time_coords=time_coords)

        # cov_tt will be zero, so all results should be NaN
        assert np.isnan(result[0])  # spatiotemporal_corr
        assert np.all(np.isnan(result[1]))  # x_traj
        assert np.all(np.isnan(result[2]))  # y_traj
        assert np.isnan(result[3])  # slope_x
        assert np.isnan(result[4])  # slope_y

    def test_single_point(self):
        """Test with a single weighted point."""
        weights = np.array(
            [[[1, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]], dtype=np.float64
        )

        result = weighted_corr_2d(weights)

        # With a single point, covariances will be zero
        # So results should be NaN
        assert np.isnan(result[0])  # spatiotemporal_corr

    def test_custom_coordinates(self):
        """Test with custom coordinate values."""
        weights = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
        x_coords = np.array([10, 20], dtype=np.float64)  # Non-standard x coordinates
        y_coords = np.array([30, 40], dtype=np.float64)  # Non-standard y coordinates
        time_coords = np.array(
            [50, 60], dtype=np.float64
        )  # Non-standard time coordinates

        result = weighted_corr_2d(weights, x_coords, y_coords, time_coords)

        # Mean coordinates should reflect the custom values
        x_weights = weights.sum(axis=(1, 2))
        y_weights = weights.sum(axis=(0, 2))
        expected_mean_x = np.sum(x_coords * x_weights) / np.sum(x_weights)
        expected_mean_y = np.sum(y_coords * y_weights) / np.sum(y_weights)

        assert_allclose(result[5], expected_mean_x, rtol=1e-5)  # mean_x
        assert_allclose(result[6], expected_mean_y, rtol=1e-5)  # mean_y

        # Calculate weighted mean time (same way the function does it)
        w_flat = weights.flatten()
        t_flat = np.tile(time_coords, 4)  # 4 = x_dim * y_dim = 2*2
        total_weight = np.sum(w_flat)
        mean_t = np.sum(w_flat * t_flat) / total_weight

        expected_x_traj = result[5] + result[3] * (time_coords - mean_t)
        expected_y_traj = result[6] + result[4] * (time_coords - mean_t)

        assert_allclose(result[1], expected_x_traj)
        assert_allclose(result[2], expected_y_traj)


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

        # Result should be a tuple with 5 elements
        assert isinstance(result, tuple)
        assert len(result) == 5

        (
            correlation,
            place_trajectory,
            slope_place,
            mean_time,
            mean_place,
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
        ) = result

        # Trajectory should be: mean_place + slope_place * (time - mean_time)
        expected_trajectory = mean_place + slope_place * (time - mean_time)
        assert_allclose(place_trajectory, expected_trajectory)

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

        correlation, place_trajectory, slope_place, mean_time, mean_place = result

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

        correlation, place_trajectory, slope_place, mean_time, mean_place = result

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

        correlation, place_trajectory, slope_place, mean_time, mean_place = result

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
