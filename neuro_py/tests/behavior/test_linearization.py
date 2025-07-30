import numpy as np
import pytest
from neuro_py.behavior.linearization import get_linearized_position, make_track_graph


def test_get_linearized_position_with_hmm_flag():
    """Test HMM-based linearization."""
    # Create a simple track graph
    node_positions = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
    track_graph = make_track_graph(node_positions, edges)
    
    # Create some test positions
    positions = np.array([[5, 5], [8, 2], [12, 8], [2, 8]])
    
    # Test HMM linearization
    result_df = get_linearized_position(positions, track_graph, use_HMM=True)
    
    # Verify the result
    assert len(result_df) == 4
    # HMM may return different values than standard linearization, so just check it's reasonable
    assert not np.isnan(result_df["linear_position"].iloc[0])
    assert result_df["linear_position"].iloc[0] >= 0
    assert "track_segment_id" in result_df.columns
    assert "projected_x_position" in result_df.columns
    assert "projected_y_position" in result_df.columns


def test_hmm_linearizer_initialization():
    """Test HMMLinearizer class initialization."""
    from neuro_py.behavior.linearization import HMMLinearizer
    
    # Create a simple track graph
    node_positions = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
    track_graph = make_track_graph(node_positions, edges)
    
    # Initialize HMM linearizer
    hmm_linearizer = HMMLinearizer(track_graph)
    
    # Check basic properties
    assert hmm_linearizer.n_segments == 4
    assert hmm_linearizer.n_states > 0
    assert hmm_linearizer.transition_matrix.shape == (hmm_linearizer.n_states, hmm_linearizer.n_states)
    assert len(hmm_linearizer.state_to_segment) == hmm_linearizer.n_states
    assert len(hmm_linearizer.state_to_position) == hmm_linearizer.n_states


def test_hmm_linearizer_with_noisy_data():
    """Test HMM linearizer with noisy input data."""
    from neuro_py.behavior.linearization import HMMLinearizer
    
    # Create a simple track graph
    node_positions = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
    track_graph = make_track_graph(node_positions, edges)
    
    # Create noisy test positions
    positions = np.array([[5, 5], [8, 2], [12, 8], [2, 8]])
    
    # Initialize HMM linearizer
    hmm_linearizer = HMMLinearizer(track_graph)
    
    # Test linearization
    linear_position, track_segment_id, projected_position = hmm_linearizer.linearize_with_hmm(positions)
    
    # Verify results
    assert len(linear_position) == 4
    assert len(track_segment_id) == 4
    assert projected_position.shape == (4, 2)
    assert not np.any(np.isnan(linear_position))
    assert not np.any(np.isnan(track_segment_id))
    assert not np.any(np.isnan(projected_position))


def test_hmm_linearizer_with_nan_positions():
    """Test HMM linearizer with NaN input positions."""
    from neuro_py.behavior.linearization import HMMLinearizer
    
    # Create a simple track graph
    node_positions = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
    track_graph = make_track_graph(node_positions, edges)
    
    # Create test positions with NaN
    positions = np.array([[5, 5], [np.nan, np.nan], [12, 8], [2, 8]])
    
    # Initialize HMM linearizer
    hmm_linearizer = HMMLinearizer(track_graph)
    
    # Test linearization
    linear_position, track_segment_id, projected_position = hmm_linearizer.linearize_with_hmm(positions)
    
    # Verify results
    assert len(linear_position) == 4
    assert len(track_segment_id) == 4
    assert projected_position.shape == (4, 2)
    assert np.isnan(linear_position[1])  # NaN position should result in NaN output
    # HMM returns -1 for invalid segments, not NaN
    assert track_segment_id[1] == -1
    assert np.all(np.isnan(projected_position[1]))


def test_hmm_vs_standard_linearization():
    """Test comparison between HMM and standard linearization."""
    # Create a simple track graph
    node_positions = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
    track_graph = make_track_graph(node_positions, edges)
    
    # Create clean test positions
    positions = np.array([[5, 5], [8, 2], [12, 8], [2, 8]])
    
    # Test both methods
    result_hmm = get_linearized_position(positions, track_graph, use_HMM=True)
    result_standard = get_linearized_position(positions, track_graph, use_HMM=False)
    
    # Verify both methods produce results (they may differ due to HMM optimization)
    assert len(result_hmm) == len(result_standard)
    # HMM and standard may produce different results, so just check they're both valid
    assert not np.any(np.isnan(result_hmm["linear_position"]))
    assert not np.any(np.isnan(result_standard["linear_position"]))
    # Both should use valid segment IDs
    assert np.all(result_hmm["track_segment_id"] >= 0)
    assert np.all(result_standard["track_segment_id"] >= 0)


def test_plot_linearization_confirmation():
    """Test the linearization confirmation plot function."""
    # Create a simple track graph
    node_positions = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
    track_graph = make_track_graph(node_positions, edges)
    
    # Create some test positions
    positions = np.array([[5, 5], [8, 2], [12, 8], [2, 8]])
    
    # Get linearized positions
    result_df = get_linearized_position(positions, track_graph, use_HMM=False)
    
    # Test the confirmation plot function (with show_plot=False to avoid blocking)
    from neuro_py.behavior.linearization import plot_linearization_confirmation
    plot_linearization_confirmation(positions, result_df, track_graph, show_plot=False)
    
    # Verify the function doesn't raise any errors
    assert True  # If we get here, the function worked without errors


def test_get_linearized_position_with_confirmation_plot():
    """Test linearization with confirmation plot enabled."""
    # Create a simple track graph
    node_positions = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
    track_graph = make_track_graph(node_positions, edges)
    
    # Create some test positions
    positions = np.array([[5, 5], [8, 2], [12, 8], [2, 8]])
    
    # Test with confirmation plot (show_plot=False to avoid blocking)
    # We need to temporarily patch plt.show to avoid blocking
    import matplotlib.pyplot as plt
    original_show = plt.show
    
    def mock_show(*args, **kwargs):
        pass
    
    plt.show = mock_show
    
    try:
        result_df = get_linearized_position(
            positions, track_graph, use_HMM=False, show_confirmation_plot=True
        )
        
        # Verify the result is correct
        assert len(result_df) == 4
        assert "linear_position" in result_df.columns
        assert "track_segment_id" in result_df.columns
        assert "projected_x_position" in result_df.columns
        assert "projected_y_position" in result_df.columns
    finally:
        plt.show = original_show 