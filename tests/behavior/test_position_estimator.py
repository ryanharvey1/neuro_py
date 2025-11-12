import numpy as np
import pytest

from neuro_py.ensemble import position_estimator


class TestPositionEstimator:
    """Test suite for position_estimator function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.bin_centers_1d = np.array([0, 1, 2, 3, 4])
        self.y_centers = np.array([0, 1, 2])
        self.x_centers = np.array([0, 1])

    # 1D Tests
    def test_1d_center_of_mass_single_peak(self):
        """Test 1D center of mass with a single probability peak."""
        posterior = np.zeros((1, 5))
        posterior[0, 2] = 1.0  # Peak at position 2

        result = position_estimator(posterior, self.bin_centers_1d, method="com")

        assert result.shape == (1,)
        np.testing.assert_allclose(result[0], 2.0)

    def test_1d_center_of_mass_uniform(self):
        """Test 1D center of mass with uniform distribution."""
        posterior = np.ones((1, 5))  # Uniform distribution

        result = position_estimator(posterior, self.bin_centers_1d, method="com")

        # Should be at the center of the bins
        expected = np.mean(self.bin_centers_1d)  # 2.0
        np.testing.assert_allclose(result[0], expected)

    def test_1d_maximum_method(self):
        """Test 1D maximum a posteriori method."""
        posterior = np.zeros((1, 5))
        posterior[0, 3] = 0.8  # Maximum at position 3
        posterior[0, 1] = 0.2  # Smaller peak at position 1

        result = position_estimator(posterior, self.bin_centers_1d, method="max")

        np.testing.assert_allclose(result[0], 3.0)

    def test_1d_multiple_time_bins(self):
        """Test 1D with multiple time bins."""
        n_time = 3
        posterior = np.zeros((n_time, 5))

        # Different peak positions for each time bin
        posterior[0, 0] = 1.0  # Position 0
        posterior[1, 2] = 1.0  # Position 2
        posterior[2, 4] = 1.0  # Position 4

        result = position_estimator(posterior, self.bin_centers_1d, method="com")

        expected = np.array([0.0, 2.0, 4.0])
        np.testing.assert_allclose(result, expected)

    def test_1d_weighted_center_of_mass(self):
        """Test 1D center of mass calculation with specific weights."""
        posterior = np.zeros((1, 3))
        posterior[0, 0] = 1.0  # Position 0, weight 1
        posterior[0, 1] = 2.0  # Position 1, weight 2
        posterior[0, 2] = 3.0  # Position 2, weight 3

        bin_centers = np.array([0, 1, 2])
        result = position_estimator(posterior, bin_centers, method="com")

        # Manual calculation: (0*1 + 1*2 + 2*3) / (1+2+3) = 8/6 = 4/3
        expected = 8.0 / 6.0
        np.testing.assert_allclose(result[0], expected)

    # 2D Tests
    def test_2d_center_of_mass_single_peak(self):
        """Test 2D center of mass with a single probability peak."""
        # Create posterior with peak at (x=1, y=2)
        posterior = np.zeros((1, 3, 2))
        posterior[0, 2, 1] = 1.0  # y=2, x=1

        result = position_estimator(
            posterior, self.y_centers, self.x_centers, method="com"
        )

        assert result.shape == (1, 2)
        np.testing.assert_allclose(result[0], [1.0, 2.0])

    def test_2d_center_of_mass_uniform_distribution(self):
        """Test 2D center of mass with uniform distribution."""
        posterior = np.ones((1, 3, 2))  # Uniform distribution

        result = position_estimator(
            posterior, self.y_centers, self.x_centers, method="com"
        )

        # Should be at the center of the grid
        expected_x = np.mean(self.x_centers)  # 0.5
        expected_y = np.mean(self.y_centers)  # 1.0
        np.testing.assert_allclose(result[0], [expected_x, expected_y])

    def test_2d_maximum_method(self):
        """Test 2D maximum a posteriori method."""
        posterior = np.zeros((1, 3, 2))
        posterior[0, 1, 0] = 0.8  # y=1, x=0
        posterior[0, 0, 1] = 0.2  # y=0, x=1

        result = position_estimator(
            posterior, self.y_centers, self.x_centers, method="max"
        )

        # Should return position of maximum probability
        np.testing.assert_allclose(result[0], [0.0, 1.0])

    def test_2d_multiple_time_bins(self):
        """Test 2D with multiple time bins."""
        n_time = 3
        posterior = np.zeros((n_time, 3, 2))

        # Different peak positions for each time bin
        posterior[0, 0, 0] = 1.0  # (x=0, y=0)
        posterior[1, 1, 1] = 1.0  # (x=1, y=1)
        posterior[2, 2, 0] = 1.0  # (x=0, y=2)

        result = position_estimator(
            posterior, self.y_centers, self.x_centers, method="com"
        )

        expected = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 2.0]])
        np.testing.assert_allclose(result, expected)

    # Error handling and edge cases
    def test_zero_probability_handling_1d(self):
        """Test 1D handling of time bins with zero probability."""
        posterior = np.zeros((2, 5))
        posterior[0, 2] = 1.0  # Valid probability for first time bin
        # Second time bin has all zeros

        result = position_estimator(posterior, self.bin_centers_1d, method="com")

        # First time bin should have valid position
        np.testing.assert_allclose(result[0], 2.0)
        # Second time bin should be NaN
        assert np.isnan(result[1])

    def test_zero_probability_handling_2d(self):
        """Test 2D handling of time bins with zero probability."""
        posterior = np.zeros((2, 3, 2))
        posterior[0, 1, 1] = 1.0  # Valid probability for first time bin
        # Second time bin has all zeros

        result = position_estimator(
            posterior, self.y_centers, self.x_centers, method="com"
        )

        # First time bin should have valid position
        np.testing.assert_allclose(result[0], [1.0, 1.0])
        # Second time bin should be NaN
        assert np.isnan(result[1]).all()

    def test_invalid_method(self):
        """Test error handling for invalid method."""
        posterior = np.ones((1, 5))

        with pytest.raises(ValueError, match="Method 'invalid' not recognized"):
            position_estimator(posterior, self.bin_centers_1d, method="invalid")

    def test_dimension_mismatch_1d(self):
        """Test error handling for 1D dimension mismatch."""
        posterior = np.ones((1, 5))
        wrong_bin_centers = np.array([0, 1, 2])  # Wrong length

        with pytest.raises(
            ValueError, match="Posterior shape 5 doesn't match bin_centers length 3"
        ):
            position_estimator(posterior, wrong_bin_centers)

    def test_dimension_mismatch_2d(self):
        """Test error handling for 2D dimension mismatch."""
        posterior = np.ones((1, 3, 2))
        wrong_y_centers = np.array([0, 1])  # Wrong length

        with pytest.raises(
            ValueError, match="Posterior shape \\(3, 2\\) doesn't match"
        ):
            position_estimator(posterior, wrong_y_centers, self.x_centers)

    def test_wrong_number_of_bin_centers_2d(self):
        """Test error handling when wrong number of bin_centers provided for 2D."""
        posterior = np.ones((1, 3, 2))

        with pytest.raises(
            ValueError, match="For 2D decoding, provide exactly 2 bin_centers"
        ):
            position_estimator(posterior, self.y_centers)  # Only one bin_centers array

    def test_unsupported_dimensions(self):
        """Test error handling for unsupported dimensions."""
        posterior = np.ones((1, 3, 2, 4))  # 3D spatial + 1D time = 4D total

        with pytest.raises(
            ValueError, match="Only 1D and 2D decoding supported, got 3D"
        ):
            position_estimator(
                posterior, self.y_centers, self.x_centers, np.array([0, 1, 2, 3])
            )
