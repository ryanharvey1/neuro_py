import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import tempfile
import os
from scipy.io import savemat
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI warnings

# Patch matplotlib to prevent GUI creation
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

from neuro_py.behavior.circle_maze import (
    CircularTrackLinearizer,
    load_epoch,
    load_animal_behavior,
    run_circular_linearization,
)


class TestCircularTrackLinearizer:
    """Test the CircularTrackLinearizer class."""

    @classmethod
    def setup_class(cls):
        """Set up class-level patches to prevent matplotlib GUI creation."""
        # Patch matplotlib to prevent GUI creation
        cls.patcher1 = patch('matplotlib.pyplot.show')
        cls.patcher2 = patch('matplotlib.pyplot.subplots')
        cls.patcher3 = patch('matplotlib.patches.Circle')
        cls.patcher4 = patch('matplotlib.pyplot.scatter')
        
        cls.mock_show = cls.patcher1.start()
        cls.mock_subplots = cls.patcher2.start()
        cls.mock_circle = cls.patcher3.start()
        cls.mock_scatter = cls.patcher4.start()
        
        # Set up mock returns
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        cls.mock_subplots.return_value = (mock_fig, mock_ax)
        cls.mock_circle.return_value = MagicMock()
        cls.mock_scatter.return_value = MagicMock()

    @classmethod
    def teardown_class(cls):
        """Clean up patches."""
        cls.patcher1.stop()
        cls.patcher2.stop()
        cls.patcher3.stop()
        cls.patcher4.stop()

    def setup_method(self):
        """Set up test data for each test method."""
        # Generate synthetic circular track data
        n_points = 100
        theta_true = np.linspace(0, 2 * np.pi, n_points)
        radius = 50
        center_x, center_y = 100, 150

        # Add some noise
        noise_level = 2
        self.x_data = (
            center_x
            + radius * np.cos(theta_true)
            + np.random.normal(0, noise_level, n_points)
        )
        self.y_data = (
            center_y
            + radius * np.sin(theta_true)
            + np.random.normal(0, noise_level, n_points)
        )

    def test_initialization(self):
        """Test CircularTrackLinearizer initialization."""
        linearizer = CircularTrackLinearizer(self.x_data, self.y_data)
        
        assert linearizer.x_data.shape == self.x_data.shape
        assert linearizer.y_data.shape == self.y_data.shape
        assert linearizer.basepath is None
        assert linearizer.epoch is None
        assert linearizer.interval is None
        assert hasattr(linearizer, 'center_x')
        assert hasattr(linearizer, 'center_y')
        assert hasattr(linearizer, 'radius')

    def test_initialization_with_nan_values(self):
        """Test initialization with NaN values."""
        x_with_nan = self.x_data.copy()
        y_with_nan = self.y_data.copy()
        x_with_nan[10] = np.nan
        y_with_nan[20] = np.nan

        linearizer = CircularTrackLinearizer(x_with_nan, y_with_nan)
        
        # Check that NaN values are filtered out
        assert len(linearizer.x_valid) == len(self.x_data) - 2
        assert len(linearizer.y_valid) == len(self.y_data) - 2
        assert not np.any(np.isnan(linearizer.x_valid))
        assert not np.any(np.isnan(linearizer.y_valid))

    def test_reset_circle(self):
        """Test circle reset functionality."""
        linearizer = CircularTrackLinearizer(self.x_data, self.y_data)
        
        # Store initial values
        initial_center_x = linearizer.center_x
        initial_center_y = linearizer.center_y
        initial_radius = linearizer.radius
        
        # Modify circle parameters
        linearizer.center_x = 0
        linearizer.center_y = 0
        linearizer.radius = 100
        
        # Reset circle
        linearizer.reset_circle()
        
        # Check that values are different from modified ones
        assert linearizer.center_x != 0
        assert linearizer.center_y != 0
        assert linearizer.radius != 100
        
        # Check that values are reasonable
        assert np.isfinite(linearizer.center_x)
        assert np.isfinite(linearizer.center_y)
        assert np.isfinite(linearizer.radius)
        assert linearizer.radius > 0

    def test_linearize_positions(self):
        """Test position linearization."""
        linearizer = CircularTrackLinearizer(self.x_data, self.y_data)
        
        linear_pos, theta, x_centered, y_centered = linearizer.linearize_positions()
        
        # Check output shapes
        assert linear_pos.shape == self.x_data.shape
        assert theta.shape == self.x_data.shape
        assert x_centered.shape == self.x_data.shape
        assert y_centered.shape == self.y_data.shape
        
        # Check that theta is in [0, 2π]
        assert np.all(theta >= 0)
        assert np.all(theta <= 2 * np.pi)
        
        # Check that linear position is positive
        assert np.all(linear_pos >= 0)
        
        # Check that centered coordinates are reasonable
        assert np.all(np.isfinite(x_centered))
        assert np.all(np.isfinite(y_centered))

    def test_linearize_positions_with_nan_input(self):
        """Test linearization with NaN input values."""
        x_with_nan = self.x_data.copy()
        y_with_nan = self.y_data.copy()
        x_with_nan[10] = np.nan
        y_with_nan[20] = np.nan

        linearizer = CircularTrackLinearizer(x_with_nan, y_with_nan)
        linear_pos, theta, x_centered, y_centered = linearizer.linearize_positions()
        
        # Check that NaN inputs produce NaN outputs for x coordinate
        assert np.isnan(linear_pos[10])
        assert np.isnan(theta[10])
        assert np.isnan(x_centered[10])
        # Note: y_centered[10] might not be NaN if only x[10] is NaN
        
        # Check that NaN inputs produce NaN outputs for y coordinate
        assert np.isnan(linear_pos[20])
        assert np.isnan(theta[20])
        assert np.isnan(y_centered[20])
        # Note: x_centered[20] might not be NaN if only y[20] is NaN

    def test_linearize_and_save_no_basepath(self):
        """Test linearize_and_save without basepath."""
        linearizer = CircularTrackLinearizer(self.x_data, self.y_data)
        
        results_df = linearizer.linearize_and_save()
        
        # Check that results DataFrame has expected columns
        expected_columns = [
            'x_original', 'y_original', 'x_centered', 'y_centered',
            'theta', 'linear_position', 'circle_center_x', 'circle_center_y', 'circle_radius'
        ]
        assert all(col in results_df.columns for col in expected_columns)
        
        # Check that results match input data
        np.testing.assert_array_equal(results_df['x_original'], self.x_data)
        np.testing.assert_array_equal(results_df['y_original'], self.y_data)

    def test_plot_results(self):
        """Test plotting results."""
        linearizer = CircularTrackLinearizer(self.x_data, self.y_data)
        linear_pos, theta, x_centered, y_centered = linearizer.linearize_positions()
        
        results_df = pd.DataFrame({
            'x_original': self.x_data,
            'y_original': self.y_data,
            'x_centered': x_centered,
            'y_centered': y_centered,
            'theta': theta,
            'linear_position': linear_pos,
        })
        
        # This should not raise an error
        linearizer.plot_results(results_df)

    def test_mouse_event_handlers(self):
        """Test mouse event handling methods."""
        linearizer = CircularTrackLinearizer(self.x_data, self.y_data)
        
        # Test mouse press events
        mock_event_center = MagicMock()
        mock_event_center.inaxes = linearizer.ax
        mock_event_center.button = 1  # Left click
        mock_event_center.xdata = 100
        mock_event_center.ydata = 150
        
        mock_event_radius = MagicMock()
        mock_event_radius.inaxes = linearizer.ax
        mock_event_radius.button = 3  # Right click
        mock_event_radius.xdata = 150
        mock_event_radius.ydata = 200
        
        # Test center drag
        linearizer.on_mouse_press(mock_event_center)
        assert linearizer.dragging
        assert linearizer.drag_mode == 'center'
        assert linearizer.last_mouse_pos == (100, 150)
        
        # Test radius drag
        linearizer.on_mouse_press(mock_event_radius)
        assert linearizer.dragging
        assert linearizer.drag_mode == 'radius'
        assert linearizer.last_mouse_pos == (150, 200)
        
        # Test mouse release
        linearizer.on_mouse_release(mock_event_center)
        assert not linearizer.dragging
        assert linearizer.drag_mode is None
        assert linearizer.last_mouse_pos is None

    def test_key_press_handlers(self):
        """Test key press event handling."""
        linearizer = CircularTrackLinearizer(self.x_data, self.y_data)
        
        # Store initial values
        initial_center_x = linearizer.center_x
        initial_center_y = linearizer.center_y
        initial_radius = linearizer.radius
        
        # Modify circle parameters
        linearizer.center_x = 0
        linearizer.center_y = 0
        linearizer.radius = 100
        
        # Test 'r' key reset
        mock_event_r = MagicMock()
        mock_event_r.key = 'r'
        linearizer.on_key_press(mock_event_r)
        
        # Check that circle was reset
        assert linearizer.center_x != 0
        assert linearizer.center_y != 0
        assert linearizer.radius != 100

    def test_update_circle(self):
        """Test circle update functionality."""
        linearizer = CircularTrackLinearizer(self.x_data, self.y_data)
        
        # Modify circle parameters
        new_center_x = 200
        new_center_y = 250
        new_radius = 75
        
        linearizer.center_x = new_center_x
        linearizer.center_y = new_center_y
        linearizer.radius = new_radius
        
        # Update circle
        linearizer.update_circle()
        
        # Check that circle parameters were updated
        assert linearizer.circle.center == (new_center_x, new_center_y)
        assert linearizer.circle.radius == new_radius


class TestUtilityFunctions:
    """Test utility functions."""

    def test_load_epoch(self):
        """Test load_epoch function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock session file with the correct naming pattern
            session_data = {
                'session': {
                    'epochs': [
                        {'startTime': 0.0, 'stopTime': 100.0, 'name': 'epoch1'},
                        {'startTime': 100.0, 'stopTime': 200.0, 'name': 'epoch2'}
                    ]
                }
            }
            
            # Use the directory name as the base name for the file
            base_name = os.path.basename(temp_dir)
            session_file = os.path.join(temp_dir, f'{base_name}.session.mat')
            savemat(session_file, session_data)
            
            # Test loading epochs
            epochs_df = load_epoch(temp_dir)
            
            assert isinstance(epochs_df, pd.DataFrame)
            assert len(epochs_df) == 2
            assert 'startTime' in epochs_df.columns
            assert 'stopTime' in epochs_df.columns
            assert 'name' in epochs_df.columns

    def test_load_animal_behavior(self):
        """Test load_animal_behavior function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a mock behavior file with the correct naming pattern
            behavior_data = {
                'behavior': {
                    'timestamps': np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                    'states': np.array([1, 1, 2, 2, 1]),
                    'position': {
                        'x': np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                        'y': np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                        'z': np.array([0.0, 0.0, 0.0, 0.0, 0.0])
                    }
                }
            }
            
            # Use the directory name as the base name for the file
            base_name = os.path.basename(temp_dir)
            behavior_file = os.path.join(temp_dir, f'{base_name}.animal.behavior.mat')
            savemat(behavior_file, behavior_data)
            
            # Test loading behavior data
            behave_df = load_animal_behavior(temp_dir)
            
            assert isinstance(behave_df, pd.DataFrame)
            assert len(behave_df) == 5
            assert 'time' in behave_df.columns
            assert 'states' in behave_df.columns
            assert 'x' in behave_df.columns
            assert 'y' in behave_df.columns
            assert 'z' in behave_df.columns

    @patch('neuro_py.behavior.circle_maze.CircularTrackLinearizer')
    def test_run_circular_linearization(self, mock_linearizer_class):
        """Test run_circular_linearization function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock behavior data
            behavior_data = {
                'behavior': {
                    'timestamps': np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                    'position': {
                        'x': np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
                        'y': np.array([0.0, 1.0, 2.0, 3.0, 4.0])
                    }
                }
            }
            
            # Use the directory name as the base name for the file
            base_name = os.path.basename(temp_dir)
            behavior_file = os.path.join(temp_dir, f'{base_name}.animal.behavior.mat')
            savemat(behavior_file, behavior_data)
            
            # Create mock session data
            session_data = {
                'session': {
                    'epochs': [
                        {'startTime': 0.0, 'stopTime': 5.0, 'name': 'epoch1'}
                    ]
                }
            }
            
            session_file = os.path.join(temp_dir, f'{base_name}.session.mat')
            savemat(session_file, session_data)
            
            # Mock the linearizer instance
            mock_linearizer = MagicMock()
            mock_linearizer_class.return_value = mock_linearizer
            
            # Test without epoch or interval
            result = run_circular_linearization(temp_dir)
            assert result == mock_linearizer
            
            # Test with epoch
            result = run_circular_linearization(temp_dir, epoch=0)
            assert result == mock_linearizer
            
            # Test with interval
            result = run_circular_linearization(temp_dir, interval=(1.0, 3.0))
            assert result == mock_linearizer


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @classmethod
    def setup_class(cls):
        """Set up class-level patches to prevent matplotlib GUI creation."""
        # Patch matplotlib to prevent GUI creation
        cls.patcher1 = patch('matplotlib.pyplot.show')
        cls.patcher2 = patch('matplotlib.pyplot.subplots')
        cls.patcher3 = patch('matplotlib.patches.Circle')
        cls.patcher4 = patch('matplotlib.pyplot.scatter')
        
        cls.mock_show = cls.patcher1.start()
        cls.mock_subplots = cls.patcher2.start()
        cls.mock_circle = cls.patcher3.start()
        cls.mock_scatter = cls.patcher4.start()
        
        # Set up mock returns
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        cls.mock_subplots.return_value = (mock_fig, mock_ax)
        cls.mock_circle.return_value = MagicMock()
        cls.mock_scatter.return_value = MagicMock()

    @classmethod
    def teardown_class(cls):
        """Clean up patches."""
        cls.patcher1.stop()
        cls.patcher2.stop()
        cls.patcher3.stop()
        cls.patcher4.stop()

    def setup_method(self):
        """Set up test data for each test method."""
        # Generate synthetic circular track data for tests that need it
        n_points = 100
        theta_true = np.linspace(0, 2 * np.pi, n_points)
        radius = 50
        center_x, center_y = 100, 150

        # Add some noise
        noise_level = 2
        self.x_data = (
            center_x
            + radius * np.cos(theta_true)
            + np.random.normal(0, noise_level, n_points)
        )
        self.y_data = (
            center_y
            + radius * np.sin(theta_true)
            + np.random.normal(0, noise_level, n_points)
        )

    def test_empty_data(self):
        """Test with empty data arrays."""
        x_empty = np.array([])
        y_empty = np.array([])
        
        # The actual implementation fails with empty arrays due to numpy operations
        # This test documents the current behavior
        with pytest.raises(ValueError):
            CircularTrackLinearizer(x_empty, y_empty)

    def test_single_point(self):
        """Test with single data point."""
        x_single = np.array([100.0])
        y_single = np.array([150.0])
        
        linearizer = CircularTrackLinearizer(x_single, y_single)
        
        # Should handle single point gracefully
        assert linearizer.center_x == 100.0
        assert linearizer.center_y == 150.0
        assert linearizer.radius == 0.0

    def test_all_nan_data(self):
        """Test with all NaN data."""
        x_all_nan = np.full(10, np.nan)
        y_all_nan = np.full(10, np.nan)
        
        # The actual implementation fails with all-NaN arrays due to numpy operations
        # This test documents the current behavior
        with pytest.raises(ValueError):
            CircularTrackLinearizer(x_all_nan, y_all_nan)

    def test_mismatched_array_lengths(self):
        """Test with mismatched array lengths."""
        x_data = np.array([1, 2, 3])
        y_data = np.array([1, 2, 3, 4])
        
        # The actual implementation fails with mismatched array lengths
        # This test documents the current behavior
        with pytest.raises(ValueError):
            CircularTrackLinearizer(x_data, y_data)

    def test_invalid_circle_parameters(self):
        """Test with invalid circle parameters."""
        linearizer = CircularTrackLinearizer(self.x_data, self.y_data)
        
        # Test with negative radius
        linearizer.radius = -10
        # The actual implementation doesn't validate radius
        linear_pos, theta, x_centered, y_centered = linearizer.linearize_positions()
        assert np.all(np.isfinite(linear_pos))
        
        # Test with infinite values
        linearizer.center_x = np.inf
        linearizer.center_y = np.inf
        linearizer.radius = 50
        
        # The actual implementation doesn't validate center coordinates
        linear_pos, theta, x_centered, y_centered = linearizer.linearize_positions()
        assert np.all(np.isfinite(linear_pos))


class TestIntegration:
    """Integration tests for the complete workflow."""

    @classmethod
    def setup_class(cls):
        """Set up class-level patches to prevent matplotlib GUI creation."""
        # Patch matplotlib to prevent GUI creation
        cls.patcher1 = patch('matplotlib.pyplot.show')
        cls.patcher2 = patch('matplotlib.pyplot.subplots')
        cls.patcher3 = patch('matplotlib.patches.Circle')
        cls.patcher4 = patch('matplotlib.pyplot.scatter')
        
        cls.mock_show = cls.patcher1.start()
        cls.mock_subplots = cls.patcher2.start()
        cls.mock_circle = cls.patcher3.start()
        cls.mock_scatter = cls.patcher4.start()
        
        # Set up mock returns
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        cls.mock_subplots.return_value = (mock_fig, mock_ax)
        cls.mock_circle.return_value = MagicMock()
        cls.mock_scatter.return_value = MagicMock()

    @classmethod
    def teardown_class(cls):
        """Clean up patches."""
        cls.patcher1.stop()
        cls.patcher2.stop()
        cls.patcher3.stop()
        cls.patcher4.stop()

    def test_complete_linearization_workflow(self):
        """Test the complete linearization workflow."""
        # Generate perfect circular data
        n_points = 100
        theta_true = np.linspace(0, 2 * np.pi, n_points)
        radius_true = 50
        center_x_true = 100
        center_y_true = 150
        
        x_perfect = center_x_true + radius_true * np.cos(theta_true)
        y_perfect = center_y_true + radius_true * np.sin(theta_true)
        
        linearizer = CircularTrackLinearizer(x_perfect, y_perfect)
        
        # Reset to auto-fit
        linearizer.reset_circle()
        
        # Linearize positions
        linear_pos, theta, x_centered, y_centered = linearizer.linearize_positions()
        
        # Check that linearization is reasonable
        assert np.all(np.isfinite(linear_pos))
        assert np.all(np.isfinite(theta))
        assert np.all(np.isfinite(x_centered))
        assert np.all(np.isfinite(y_centered))
        
        # Check that linear position covers the full circle
        assert np.max(linear_pos) - np.min(linear_pos) > 0
        
        # Check that theta covers the full range
        assert np.max(theta) - np.min(theta) > np.pi

    def test_noisy_circular_data(self):
        """Test with noisy circular data."""
        # Generate noisy circular data
        n_points = 200
        theta_true = np.linspace(0, 4 * np.pi, n_points)  # Two laps
        radius_true = 75
        center_x_true = 200
        center_y_true = 300
        
        # Add significant noise
        noise_level = 10
        x_noisy = (
            center_x_true
            + radius_true * np.cos(theta_true)
            + np.random.normal(0, noise_level, n_points)
        )
        y_noisy = (
            center_y_true
            + radius_true * np.sin(theta_true)
            + np.random.normal(0, noise_level, n_points)
        )
        
        linearizer = CircularTrackLinearizer(x_noisy, y_noisy)
        
        # Linearize positions
        linear_pos, theta, x_centered, y_centered = linearizer.linearize_positions()
        
        # Check that linearization still works with noise
        assert np.all(np.isfinite(linear_pos))
        assert np.all(np.isfinite(theta))
        
        # Check that the linear position shows some structure
        # (should increase over time for a circular track)
        valid_indices = ~np.isnan(linear_pos)
        if np.sum(valid_indices) > 10:
            # Check that there's some variation in linear position
            assert np.std(linear_pos[valid_indices]) > 0 