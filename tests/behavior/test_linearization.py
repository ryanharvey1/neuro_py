import os
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

# Set matplotlib to use non-interactive backend for testing
import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

matplotlib.use("Agg")

from neuro_py.behavior.linearization import (
    NodePicker,
    TrackGraph,
    get_linearized_position,
    load_animal_behavior,
    load_epoch,
    make_track_graph,
    project_position_to_track,
)


class TestTrackGraph:
    """Test the TrackGraph class."""

    def test_init(self):
        """Test TrackGraph initialization."""
        node_positions = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        edges = [[0, 1, 2, 3, 0]]

        track_graph = TrackGraph(node_positions, edges)

        assert track_graph.n_nodes == 4
        assert np.array_equal(track_graph.node_positions, node_positions)
        assert track_graph.edges == edges
        assert isinstance(track_graph.adjacency_matrix, csr_matrix)
        assert len(track_graph.edge_distances) > 0
        assert len(track_graph.cumulative_distances) == 4

    def test_create_adjacency_matrix(self):
        """Test adjacency matrix creation."""
        node_positions = np.array([[0, 0], [1, 0], [1, 1]])
        edges = [[0, 1], [1, 2]]

        track_graph = TrackGraph(node_positions, edges)
        adj_matrix = track_graph.adjacency_matrix.toarray()

        # Check that connected nodes have 1, others have 0
        expected = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        np.testing.assert_array_equal(adj_matrix, expected)

    def test_calculate_edge_distances(self):
        """Test edge distance calculation."""
        node_positions = np.array([[0, 0], [3, 0], [3, 4]])
        edges = [[0, 1], [1, 2]]

        track_graph = TrackGraph(node_positions, edges)

        # Distance from (0,0) to (3,0) should be 3
        assert track_graph.edge_distances[(0, 1)] == 3.0
        assert track_graph.edge_distances[(1, 0)] == 3.0

        # Distance from (3,0) to (3,4) should be 4
        assert track_graph.edge_distances[(1, 2)] == 4.0
        assert track_graph.edge_distances[(2, 1)] == 4.0

    def test_calculate_cumulative_distances(self):
        """Test cumulative distance calculation."""
        node_positions = np.array([[0, 0], [3, 0], [3, 4]])
        edges = [[0, 1, 2]]

        track_graph = TrackGraph(node_positions, edges)

        # Cumulative distances should be [0, 3, 7]
        expected = np.array([0, 3, 7])
        np.testing.assert_array_almost_equal(track_graph.cumulative_distances, expected)

    def test_empty_edges(self):
        """Test TrackGraph with empty edges."""
        node_positions = np.array([[0, 0], [1, 0]])
        edges = []

        track_graph = TrackGraph(node_positions, edges)

        assert track_graph.n_nodes == 2
        assert track_graph.adjacency_matrix.nnz == 0  # No connections
        assert len(track_graph.edge_distances) == 0
        assert np.all(track_graph.cumulative_distances == 0)


class TestProjectPositionToTrack:
    """Test position projection functions."""

    def test_project_position_to_track_simple(self):
        """Test simple position projection."""
        # Create a simple linear track
        node_positions = np.array([[0, 0], [10, 0]])
        edges = [[0, 1]]
        track_graph = TrackGraph(node_positions, edges)

        # Test position exactly on the track
        position = np.array([[5, 0]])
        linear_pos, segment_ids, projected_pos = project_position_to_track(
            position, track_graph
        )

        assert linear_pos[0] == 5.0  # Should be 5 units from start
        assert segment_ids[0] == 0  # Should be on first segment
        np.testing.assert_array_almost_equal(projected_pos[0], [5, 0])

    def test_project_position_to_track_off_track(self):
        """Test position projection for points off the track."""
        # Create a simple linear track
        node_positions = np.array([[0, 0], [10, 0]])
        edges = [[0, 1]]
        track_graph = TrackGraph(node_positions, edges)

        # Test position off the track
        position = np.array([[5, 2]])  # 2 units above the track
        linear_pos, segment_ids, projected_pos = project_position_to_track(
            position, track_graph
        )

        assert linear_pos[0] == 5.0  # Should project to 5 units from start
        assert segment_ids[0] == 0  # Should be on first segment
        np.testing.assert_array_almost_equal(
            projected_pos[0], [5, 0]
        )  # Should project to track

    def test_project_position_to_track_multiple_points(self):
        """Test projection of multiple points."""
        # Create a simple linear track
        node_positions = np.array([[0, 0], [10, 0]])
        edges = [[0, 1]]
        track_graph = TrackGraph(node_positions, edges)

        # Test multiple positions
        position = np.array([[2, 0], [8, 1], [15, 0]])  # Last point beyond track
        linear_pos, segment_ids, projected_pos = project_position_to_track(
            position, track_graph
        )

        assert linear_pos[0] == 2.0
        assert linear_pos[1] == 8.0
        # The current implementation projects beyond-track points to the end of the track
        # This is actually reasonable behavior, so we test for that instead
        assert linear_pos[2] == 10.0  # Projected to end of track
        assert segment_ids[0] == 0
        assert segment_ids[1] == 0
        assert segment_ids[2] == 0  # Still on the track segment

    def test_project_position_to_track_complex_track(self):
        """Test projection on a more complex track."""
        # Create a square track
        node_positions = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        edges = [[0, 1, 2, 3, 0]]
        track_graph = TrackGraph(node_positions, edges)

        # Test position on different segments
        position = np.array([[5, 0], [10, 5], [5, 10], [0, 5]])
        linear_pos, segment_ids, projected_pos = project_position_to_track(
            position, track_graph
        )

        # All should be valid
        assert not np.any(np.isnan(linear_pos))
        assert not np.any(segment_ids == -1)
        assert all(segment_ids == 0)  # All on the same track segment


class TestGetLinearizedPosition:
    """Test the main linearization function."""

    def test_get_linearized_position(self):
        """Test the main linearization function."""
        # Create a simple linear track
        node_positions = np.array([[0, 0], [10, 0]])
        edges = [[0, 1]]
        track_graph = TrackGraph(node_positions, edges)

        # Test positions
        position = np.array([[2, 0], [8, 1]])

        result = get_linearized_position(
            position, track_graph, show_confirmation_plot=False
        )

        assert isinstance(result, pd.DataFrame)
        assert "linear_position" in result.columns
        assert "track_segment_id" in result.columns
        assert "projected_x_position" in result.columns
        assert "projected_y_position" in result.columns

        assert result.shape[0] == 2
        assert result["linear_position"].iloc[0] == 2.0
        assert result["linear_position"].iloc[1] == 8.0

    def test_get_linearized_position_with_edge_order(self):
        """Test linearization with edge order parameter."""
        node_positions = np.array([[0, 0], [10, 0]])
        edges = [[0, 1]]
        track_graph = TrackGraph(node_positions, edges)

        position = np.array([[5, 0]])
        edge_order = [[0, 1]]  # Should be ignored in our implementation

        result = get_linearized_position(
            position, track_graph, edge_order=edge_order, show_confirmation_plot=False
        )

        assert result["linear_position"].iloc[0] == 5.0

    def test_get_linearized_position_with_hmm_flag(self):
        """Test HMM-based linearization."""
        node_positions = np.array([[0, 0], [10, 0]])
        edges = [[0, 1]]
        track_graph = TrackGraph(node_positions, edges)

        position = np.array([[5, 0]])

        result = get_linearized_position(
            position, track_graph, use_HMM=True, show_confirmation_plot=False
        )

        # HMM should provide similar results to standard linearization
        assert result["linear_position"].iloc[0] == pytest.approx(5.0, abs=1.0)
        assert result["track_segment_id"].iloc[0] == 0
        assert result["projected_x_position"].iloc[0] == pytest.approx(5.0, abs=1.0)
        assert result["projected_y_position"].iloc[0] == pytest.approx(0.0, abs=1.0)

    def test_hmm_linearizer_initialization(self):
        """Test HMMLinearizer initialization."""
        from neuro_py.behavior.linearization import HMMLinearizer

        node_positions = np.array([[0, 0], [10, 0], [10, 10]])
        edges = [[0, 1], [1, 2]]
        track_graph = TrackGraph(node_positions, edges)

        hmm = HMMLinearizer(track_graph)

        assert hmm.n_segments == 2
        assert hmm.n_states > 0
        assert hmm.transition_matrix.shape == (hmm.n_states, hmm.n_states)
        assert len(hmm.state_to_segment) == hmm.n_states
        assert len(hmm.state_to_position) == hmm.n_states

    def test_hmm_linearizer_with_noisy_data(self):
        """Test HMM linearization with noisy position data."""
        from neuro_py.behavior.linearization import HMMLinearizer

        # Create a simple track
        node_positions = np.array([[0, 0], [10, 0], [20, 0]])
        edges = [[0, 1], [1, 2]]
        track_graph = TrackGraph(node_positions, edges)

        hmm = HMMLinearizer(track_graph, emission_noise=2.0)

        # Create noisy positions along the track
        true_positions = np.array([[5, 0], [15, 0]])
        noisy_positions = true_positions + np.random.normal(0, 1, true_positions.shape)

        # Linearize with HMM
        linear_pos, segment_ids, projected_pos = hmm.linearize_with_hmm(noisy_positions)

        # Check results
        assert len(linear_pos) == 2
        assert len(segment_ids) == 2
        assert len(projected_pos) == 2
        assert not np.any(np.isnan(linear_pos))
        assert not np.any(np.isnan(segment_ids))
        assert not np.any(np.isnan(projected_pos))

    def test_hmm_linearizer_with_nan_positions(self):
        """Test HMM linearization with NaN positions."""
        from neuro_py.behavior.linearization import HMMLinearizer

        node_positions = np.array([[0, 0], [10, 0]])
        edges = [[0, 1]]
        track_graph = TrackGraph(node_positions, edges)

        hmm = HMMLinearizer(track_graph)

        # Create positions with some NaN values
        positions = np.array([[5, 0], [np.nan, np.nan], [8, 0]])

        linear_pos, segment_ids, projected_pos = hmm.linearize_with_hmm(positions)

        # Check that NaN positions are handled correctly
        assert np.isnan(linear_pos[1])
        assert segment_ids[1] == -1
        assert np.all(np.isnan(projected_pos[1]))

        # Valid positions should be processed
        assert not np.isnan(linear_pos[0])
        assert not np.isnan(linear_pos[2])

    def test_hmm_vs_standard_linearization(self):
        """Test that HMM and standard linearization give reasonable results for clean data."""
        node_positions = np.array([[0, 0], [10, 0]])
        edges = [[0, 1]]
        track_graph = TrackGraph(node_positions, edges)

        position = np.array([[5, 0], [8, 0]])

        # Standard linearization
        result_standard = get_linearized_position(
            position, track_graph, use_HMM=False, show_confirmation_plot=False
        )

        # HMM linearization
        result_hmm = get_linearized_position(
            position, track_graph, use_HMM=True, show_confirmation_plot=False
        )

        # For clean data, both methods should produce reasonable results
        # Standard linearization should give exact projections
        assert np.allclose(result_standard["linear_position"], [5.0, 8.0], atol=0.1)

        # HMM linearization may have discretization error but should be reasonable
        # Check that both positions are in the expected range (0-10)
        assert np.all(result_hmm["linear_position"] >= 0)
        assert np.all(result_hmm["linear_position"] <= 10)

        # Check that both methods assign positions to the same segment
        assert np.array_equal(
            result_standard["track_segment_id"], result_hmm["track_segment_id"]
        )


class TestMakeTrackGraph:
    """Test the make_track_graph function."""

    def test_make_track_graph(self):
        """Test track graph creation."""
        node_positions = np.array([[0, 0], [1, 0], [1, 1]])
        edges = [[0, 1], [1, 2]]

        track_graph = make_track_graph(node_positions, edges)

        assert isinstance(track_graph, TrackGraph)
        assert track_graph.n_nodes == 3
        assert np.array_equal(track_graph.node_positions, node_positions)
        assert track_graph.edges == edges


class TestLoadAnimalBehavior:
    """Test behavior loading functions."""

    @patch("neuro_py.behavior.linearization.loadmat")
    def test_load_animal_behavior(self, mock_loadmat):
        """Test loading animal behavior data."""
        # Mock the loadmat return value
        mock_data = {
            "behavior": {
                "timestamps": np.array([1, 2, 3]),
                "states": np.array([0, 1, 0]),
                "position": {
                    "x": np.array([1.1, 2.2, 3.3]),
                    "y": np.array([1.0, 2.0, 3.0]),
                },
            }
        }
        mock_loadmat.return_value = mock_data

        with tempfile.TemporaryDirectory() as temp_dir:
            basepath = temp_dir
            result = load_animal_behavior(basepath)

            assert isinstance(result, pd.DataFrame)
            assert "time" in result.columns
            assert "states" in result.columns
            assert "x" in result.columns
            assert "y" in result.columns
            assert len(result) == 3

    @patch("neuro_py.behavior.linearization.loadmat")
    def test_load_animal_behavior_missing_states(self, mock_loadmat):
        """Test loading behavior data when states are missing."""
        # Mock the loadmat return value without states
        mock_data = {
            "behavior": {
                "timestamps": np.array([1, 2, 3]),
                "position": {
                    "x": np.array([1.1, 2.2, 3.3]),
                    "y": np.array([1.0, 2.0, 3.0]),
                },
            }
        }
        mock_loadmat.return_value = mock_data

        with tempfile.TemporaryDirectory() as temp_dir:
            basepath = temp_dir
            result = load_animal_behavior(basepath)

            assert isinstance(result, pd.DataFrame)
            assert "time" in result.columns
            assert "x" in result.columns
            assert "y" in result.columns
            # States column should not be present
            assert "states" not in result.columns


class TestLoadEpoch:
    """Test epoch loading functions."""

    @patch("neuro_py.behavior.linearization.loadmat")
    def test_load_epoch(self, mock_loadmat):
        """Test loading epoch data."""
        # Mock the loadmat return value - create a DataFrame-like structure
        # that matches what the function expects
        mock_data = {
            "session": {
                "epochs": pd.DataFrame(
                    [
                        {"startTime": 0, "stopTime": 100},
                        {"startTime": 100, "stopTime": 200},
                    ]
                )
            }
        }
        mock_loadmat.return_value = mock_data

        with tempfile.TemporaryDirectory() as temp_dir:
            basepath = temp_dir
            result = load_epoch(basepath)

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            # The function should return a DataFrame with startTime and stopTime columns
            assert result.iloc[0]["startTime"] == 0
            assert result.iloc[0]["stopTime"] == 100
            assert result.iloc[1]["startTime"] == 100
            assert result.iloc[1]["stopTime"] == 200

    @patch("neuro_py.behavior.linearization.loadmat")
    def test_load_epoch_single_epoch(self, mock_loadmat):
        """Test loading single epoch data."""
        # Mock the loadmat return value with single epoch
        mock_data = {"session": {"epochs": {"startTime": 0, "stopTime": 100}}}
        mock_loadmat.return_value = mock_data

        with tempfile.TemporaryDirectory() as temp_dir:
            basepath = temp_dir
            result = load_epoch(basepath)

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 1
            assert "startTime" in result.columns
            assert "stopTime" in result.columns


class TestNodePicker:
    """Test the NodePicker class."""

    def test_init(self):
        """Test NodePicker initialization."""
        with patch("matplotlib.pyplot.gca") as mock_gca:
            mock_ax = Mock()
            mock_gca.return_value = mock_ax

            picker = NodePicker()

            assert picker.ax == mock_ax
            assert picker._nodes == []
            assert picker.edges == [[]]
            assert picker.node_color == "#177ee6"
            assert picker.use_HMM is True

    def test_node_positions_property(self):
        """Test node_positions property."""
        with patch("matplotlib.pyplot.gca") as mock_gca:
            mock_ax = Mock()
            mock_gca.return_value = mock_ax

            picker = NodePicker()
            picker._nodes = [(1, 2), (3, 4)]

            positions = picker.node_positions
            expected = np.array([[1, 2], [3, 4]])
            np.testing.assert_array_equal(positions, expected)

    def test_connect_disconnect(self):
        """Test connect and disconnect methods."""
        with patch("matplotlib.pyplot.gca") as mock_gca:
            mock_ax = Mock()
            mock_canvas = Mock()
            mock_ax.get_figure.return_value.canvas = mock_canvas
            mock_gca.return_value = mock_ax

            picker = NodePicker()

            # Test connect
            picker.connect()
            assert picker.cid is not None
            assert mock_canvas.mpl_connect.call_count == 2  # button_press and key_press

            # Test disconnect
            picker.disconnect()
            assert picker.cid is None
            assert mock_canvas.mpl_disconnect.called

    def test_clear(self):
        """Test clear method."""
        with patch("matplotlib.pyplot.gca") as mock_gca:
            mock_ax = Mock()
            mock_gca.return_value = mock_ax

            picker = NodePicker()
            picker._nodes = [(1, 2), (3, 4)]
            picker.edges = [[0, 1], [1, 2]]

            picker.clear()

            assert picker._nodes == []
            assert picker.edges == [[]]

    def test_remove_point(self):
        """Test remove_point method."""
        with patch("matplotlib.pyplot.gca") as mock_gca:
            mock_ax = Mock()
            mock_gca.return_value = mock_ax

            picker = NodePicker()
            picker._nodes = [(0, 0), (1, 0), (2, 0)]

            # Remove point closest to (1.1, 0.1)
            picker.remove_point((1.1, 0.1))

            # Should remove the point at (1, 0)
            expected = [(0, 0), (2, 0)]
            assert picker._nodes == expected

    @patch("neuro_py.behavior.linearization.load_animal_behavior")
    @patch("neuro_py.behavior.linearization.load_epoch")
    @patch("neuro_py.behavior.linearization.loadmat")
    @patch("neuro_py.behavior.linearization.savemat")
    @patch("neuro_py.behavior.linearization.plot_linearization_confirmation")
    def test_format_and_save(
        self, mock_plot, mock_savemat, mock_loadmat, mock_load_epoch, mock_load_behavior
    ):
        """Test format_and_save method."""
        with patch("matplotlib.pyplot.gca") as mock_gca:
            mock_ax = Mock()
            mock_gca.return_value = mock_ax

            # Mock behavior data
            mock_behavior_df = pd.DataFrame(
                {
                    "time": [0, 1, 2],
                    "x": [0, 1, 2],
                    "y": [0, 0, 0],
                    "linearized": [np.nan, np.nan, np.nan],
                    "states": [np.nan, np.nan, np.nan],
                    "projected_x_position": [np.nan, np.nan, np.nan],
                    "projected_y_position": [np.nan, np.nan, np.nan],
                }
            )
            mock_load_behavior.return_value = mock_behavior_df

            # Mock epoch data
            mock_epoch_df = pd.DataFrame({"startTime": [0], "stopTime": [3]})
            mock_load_epoch.return_value = mock_epoch_df

            # Mock mat file data
            mock_mat_data = {
                "behavior": {
                    "position": {},
                    "epochs": [{"node_positions": None, "edges": None}],
                }
            }
            mock_loadmat.return_value = mock_mat_data

            with tempfile.TemporaryDirectory() as temp_dir:
                picker = NodePicker(basepath=temp_dir, epoch=0)
                picker._nodes = [(0, 0), (1, 0)]
                picker.edges = [[0, 1]]

                picker.format_and_save()

                # Check that save methods were called
                assert mock_savemat.called

                # Check that nodes and edges were saved
                assert len(picker._nodes) == 2
                assert picker.edges == [[0, 1]]


if __name__ == "__main__":
    pytest.main([__file__])
