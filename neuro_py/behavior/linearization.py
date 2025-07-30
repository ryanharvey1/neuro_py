import os
import pickle
import sys
from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal

# Add Numba for JIT compilation
try:
    from numba import njit, prange, jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorators if Numba is not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def prange(*args):
        return range(*args)

# Numba-optimized emission probability calculation
@njit(parallel=True, fastmath=True)
def _emission_probabilities_numba(observations, emission_centers, emission_covariance):
    """
    Numba-optimized emission probability calculation.
    """
    n_observations = len(observations)
    n_states = len(emission_centers)
    emission_probs = np.zeros((n_observations, n_states))
    
    # Pre-compute covariance matrix components for 2D case
    # For 2x2 covariance matrix: [[a, b], [b, c]]
    a = emission_covariance[0, 0]
    b = emission_covariance[0, 1]
    c = emission_covariance[1, 1]
    
    # Determinant: det = a*c - b*b
    det = a * c - b * b
    
    # Inverse matrix: [[c, -b], [-b, a]] / det
    inv_a = c / det
    inv_b = -b / det
    inv_c = a / det
    
    # Normalization constant
    norm_const = 1.0 / (2 * np.pi * np.sqrt(det))
    
    for t in prange(n_observations):
        obs = observations[t]
        for i in range(n_states):
            center = emission_centers[i]
            dx = obs[0] - center[0]
            dy = obs[1] - center[1]
            
            # Full Gaussian calculation: diff.T @ inv_cov @ diff
            # For 2D: [dx, dy] @ [[inv_a, inv_b], [inv_b, inv_c]] @ [dx, dy]
            # = dx*(inv_a*dx + inv_b*dy) + dy*(inv_b*dx + inv_c*dy)
            # = dx*inv_a*dx + dx*inv_b*dy + dy*inv_b*dx + dy*inv_c*dy
            # = inv_a*dx*dx + 2*inv_b*dx*dy + inv_c*dy*dy
            exponent = -0.5 * (inv_a * dx * dx + 2 * inv_b * dx * dy + inv_c * dy * dy)
            emission_probs[t, i] = norm_const * np.exp(exponent)
    
    return emission_probs

# Numba-optimized Viterbi algorithm
@njit(fastmath=True)
def _viterbi_numba(emission_probs, transition_matrix, n_observations, n_states):
    """
    Numba-optimized Viterbi algorithm.
    """
    # Initialize
    delta = np.zeros((n_observations, n_states))
    psi = np.zeros((n_observations, n_states), dtype=np.int32)
    
    # Initial probabilities (uniform)
    initial_probs = np.ones(n_states) / n_states
    
    # Forward pass
    for t in range(n_observations):
        if t == 0:
            delta[t] = initial_probs * emission_probs[t]
        else:
            # Manual max and argmax operations for Numba compatibility
            max_probs = np.zeros(n_states)
            max_indices = np.zeros(n_states, dtype=np.int32)
            
            for j in range(n_states):
                max_val = -np.inf
                max_idx = 0
                
                for i in range(n_states):
                    val = delta[t - 1, i] * transition_matrix[i, j]
                    if val > max_val:
                        max_val = val
                        max_idx = i
                
                max_probs[j] = max_val
                max_indices[j] = max_idx
            
            delta[t] = max_probs * emission_probs[t]
            psi[t] = max_indices
    
    # Backward pass
    state_sequence = np.zeros(n_observations, dtype=np.int32)
    
    # Find the state with maximum probability at the last time step
    max_val = -np.inf
    max_idx = 0
    for i in range(n_states):
        if delta[-1, i] > max_val:
            max_val = delta[-1, i]
            max_idx = i
    state_sequence[-1] = max_idx
    
    for t in range(n_observations - 2, -1, -1):
        state_sequence[t] = psi[t + 1, state_sequence[t + 1]]
    
    return state_sequence

# Numba-optimized transition matrix building for sparse transitions
@njit(parallel=True)
def _build_sparse_transition_matrix_numba(state_to_segment, state_to_position, n_states, 
                                        transition_smoothness, segments_connected):
    """
    Numba-optimized sparse transition matrix building.
    """
    transition_matrix = np.zeros((n_states, n_states))
    
    for i in prange(n_states):
        seg1 = state_to_segment[i]
        pos1 = state_to_position[i]
        
        # Compute transitions for all states (Numba doesn't handle dynamic lists well)
        for j in range(n_states):
            seg2 = state_to_segment[j]
            pos2 = state_to_position[j]
            
            # Check if this is a valid transition for sparse mode
            valid_transition = False
            
            # Same segment transitions (nearby positions only)
            if seg1 == seg2:
                pos_diff = abs(i - j)
                if pos_diff <= 2:  # Only nearby positions
                    valid_transition = True
            
            # Adjacent segment transitions (boundary positions only)
            elif segments_connected[seg1, seg2]:
                if (seg1 < seg2 and j < 50) or (seg1 > seg2 and j >= n_states - 50):
                    valid_transition = True
            
            if valid_transition:
                # Calculate transition probability
                if seg1 == seg2:
                    # Within same segment - manual norm calculation for Numba
                    dx = pos1[0] - pos2[0]
                    dy = pos1[1] - pos2[1]
                    pos_diff = (dx * dx + dy * dy) ** 0.5
                    prob = np.exp(-pos_diff / transition_smoothness)
                else:
                    # Between segments
                    prob = 0.1  # Lower probability for segment transitions
                
                transition_matrix[i, j] = prob
    
    # Normalize rows
    for i in range(n_states):
        row_sum = 0.0
        for j in range(n_states):
            row_sum += transition_matrix[i, j]
        
        if row_sum > 0:
            for j in range(n_states):
                transition_matrix[i, j] /= row_sum
    
    return transition_matrix

# Numba-optimized position projection
@njit(parallel=True, fastmath=True)
def _project_positions_parallel(positions, node_positions, edges, n_positions):
    """
    Numba-optimized parallel position projection onto track.
    """
    linear_positions = np.full(n_positions, np.nan)
    track_segment_ids = np.full(n_positions, -1, dtype=np.int32)
    projected_positions = np.full((n_positions, 2), np.nan)
    
    for i in prange(n_positions):
        pos = positions[i]
        
        # Find closest edge
        min_distance = np.inf
        best_segment = -1
        best_projection = np.array([np.nan, np.nan])
        
        for edge_idx, edge in enumerate(edges):
            if len(edge) >= 2:
                for j in range(len(edge) - 1):
                    node1, node2 = edge[j], edge[j + 1]
                    p1 = node_positions[node1]
                    p2 = node_positions[node2]
                    
                    # Project position onto this line segment
                    v = p2 - p1
                    u = pos - p1
                    t = np.dot(u, v) / np.dot(v, v)
                    t = np.clip(t, 0, 1)
                    
                    projected = p1 + t * v
                    distance = np.linalg.norm(pos - projected)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_segment = edge_idx
                        best_projection = projected
        
        if best_segment >= 0:
            track_segment_ids[i] = best_segment
            projected_positions[i] = best_projection
            # Calculate linear position (simplified)
            linear_positions[i] = best_segment + best_projection[0] / 100.0  # Rough approximation
    
    return linear_positions, track_segment_ids, projected_positions

# Memory-optimized batch processing
@njit(fastmath=True)
def _process_batch_emissions(observations_batch, emission_centers, emission_covariance, batch_size):
    """
    Process emission probabilities in batches to reduce memory usage.
    """
    n_observations = len(observations_batch)
    n_states = len(emission_centers)
    emission_probs = np.zeros((n_observations, n_states))

    # Pre-compute covariance matrix components for 2D case
    # For 2x2 covariance matrix: [[a, b], [b, c]]
    a = emission_covariance[0, 0]
    b = emission_covariance[0, 1]
    c = emission_covariance[1, 1]
    
    # Determinant: det = a*c - b*b
    det = a * c - b * b
    
    # Inverse matrix: [[c, -b], [-b, a]] / det
    inv_a = c / det
    inv_b = -b / det
    inv_c = a / det
    
    # Normalization constant
    norm_const = 1.0 / (2 * np.pi * np.sqrt(det))

    # Process in batches
    for batch_start in range(0, n_observations, batch_size):
        batch_end = min(batch_start + batch_size, n_observations)

        for t in range(batch_start, batch_end):
            obs = observations_batch[t]
            for i in range(n_states):
                center = emission_centers[i]
                dx = obs[0] - center[0]
                dy = obs[1] - center[1]

                # Full Gaussian calculation: diff.T @ inv_cov @ diff
                # For 2D: [dx, dy] @ [[inv_a, inv_b], [inv_b, inv_c]] @ [dx, dy]
                # = inv_a*dx*dx + 2*inv_b*dx*dy + inv_c*dy*dy
                exponent = -0.5 * (inv_a * dx * dx + 2 * inv_b * dx * dy + inv_c * dy * dy)
                emission_probs[t, i] = norm_const * np.exp(exponent)

    return emission_probs


class TrackGraph:
    """
    A simple track graph implementation for linearization.

    Parameters
    ----------
    node_positions : np.ndarray
        Array of node positions (n_nodes, 2)
    edges : list
        List of edge connections between nodes
    """

    def __init__(self, node_positions: np.ndarray, edges: List[List[int]]):
        self.node_positions = np.asarray(node_positions)
        self.edges = edges
        self.n_nodes = len(node_positions)

        # Create adjacency matrix
        self.adjacency_matrix = self._create_adjacency_matrix()

        # Calculate distances between connected nodes
        self.edge_distances = self._calculate_edge_distances()

        # Calculate cumulative distances for linearization
        self.cumulative_distances = self._calculate_cumulative_distances()

    def _create_adjacency_matrix(self) -> csr_matrix:
        """Create sparse adjacency matrix from edges."""
        row_indices = []
        col_indices = []

        for edge in self.edges:
            if len(edge) >= 2:
                for i in range(len(edge) - 1):
                    row_indices.extend([edge[i], edge[i + 1]])
                    col_indices.extend([edge[i + 1], edge[i]])

        data = np.ones(len(row_indices))
        return csr_matrix(
            (data, (row_indices, col_indices)), shape=(self.n_nodes, self.n_nodes)
        )

    def _calculate_edge_distances(self) -> dict:
        """Calculate distances between connected nodes."""
        distances = {}
        for edge in self.edges:
            if len(edge) >= 2:
                for i in range(len(edge) - 1):
                    node1, node2 = edge[i], edge[i + 1]
                    dist = np.linalg.norm(
                        self.node_positions[node1] - self.node_positions[node2]
                    )
                    distances[(node1, node2)] = dist
                    distances[(node2, node1)] = dist
        return distances

    def _calculate_cumulative_distances(self) -> np.ndarray:
        """Calculate cumulative distances along the track."""
        # Find the main path through the track
        # For simplicity, we'll use the first edge as the starting point
        if not self.edges or len(self.edges[0]) < 2:
            return np.zeros(self.n_nodes)

        cumulative = np.zeros(self.n_nodes)
        visited = set()

        # Start from the first edge
        current_edge = self.edges[0]
        if len(current_edge) >= 2:
            for i in range(len(current_edge) - 1):
                node1, node2 = current_edge[i], current_edge[i + 1]
                if node1 not in visited:
                    visited.add(node1)
                if node2 not in visited:
                    visited.add(node2)
                    cumulative[node2] = cumulative[node1] + self.edge_distances.get(
                        (node1, node2), 0
                    )

        return cumulative


class HMMLinearizer:
    """
    Hidden Markov Model for track linearization.

    This class implements an HMM that infers the most likely position on a track
    given noisy 2D position measurements. The hidden states represent positions
    along track segments, and observations are 2D coordinates.

    Parameters
    ----------
    track_graph : TrackGraph
        The track graph defining the track structure
    n_bins_per_segment : int, optional
        Number of position bins per track segment, by default 50
    transition_smoothness : float, optional
        Smoothness parameter for state transitions, by default 0.1
    emission_noise : float, optional
        Standard deviation of emission noise, by default 5.0
    max_iterations : int, optional
        Maximum iterations for Viterbi algorithm, by default 1000

    Attributes
    ----------
    track_graph : TrackGraph
        The track graph
    n_states : int
        Total number of hidden states
    n_segments : int
        Number of track segments
    n_bins_per_segment : int
        Number of position bins per segment
    transition_matrix : np.ndarray
        State transition probability matrix
    emission_centers : np.ndarray
        Center positions for each hidden state
    emission_covariance : np.ndarray
        Covariance matrix for emission model
    """

    def __init__(
        self,
        track_graph: "TrackGraph",
        n_bins_per_segment: int = 50,
        transition_smoothness: float = 0.1,
        emission_noise: float = 5.0,
        max_iterations: int = 1000,
        # Optimization parameters
        use_sparse_transitions: bool = True,
        subsample_positions: bool = False,
        subsample_factor: int = 5,
        # Additional optimization parameters
        use_adaptive_subsampling: bool = True,
        # Advanced optimization parameters
        use_batch_processing: bool = False,
        batch_size: int = 1000,
    ):
        self.track_graph = track_graph
        self.n_bins_per_segment = n_bins_per_segment
        self.transition_smoothness = transition_smoothness
        self.emission_noise = emission_noise
        self.max_iterations = max_iterations

        # Optimization parameters
        self.use_sparse_transitions = use_sparse_transitions
        self.subsample_positions = subsample_positions
        self.subsample_factor = subsample_factor
        self.use_adaptive_subsampling = use_adaptive_subsampling
        self.use_batch_processing = use_batch_processing
        self.batch_size = batch_size

        # Calculate track segment properties
        self.n_segments = len(track_graph.edges)
        self.segment_lengths = []
        self.segment_positions = []

        for edge in track_graph.edges:
            if len(edge) >= 2:
                # Calculate segment length
                segment_length = 0
                for i in range(len(edge) - 1):
                    node1, node2 = edge[i], edge[i + 1]
                    p1 = track_graph.node_positions[node1]
                    p2 = track_graph.node_positions[node2]
                    segment_length += np.linalg.norm(p2 - p1)

                self.segment_lengths.append(segment_length)

                # Generate position bins along this segment
                positions = []
                for i in range(self.n_bins_per_segment):
                    t = i / (self.n_bins_per_segment - 1)  # Parameter along segment
                    pos = self._interpolate_along_segment(edge, t)
                    positions.append(pos)

                self.segment_positions.append(np.array(positions))
            else:
                # Empty segment
                self.segment_lengths.append(0)
                self.segment_positions.append(np.array([]))

        # Total number of states
        self.n_states = sum(len(positions) for positions in self.segment_positions)

        # Build state mapping
        self.state_to_segment = []
        self.state_to_position = []
        state_idx = 0

        for seg_idx, positions in enumerate(self.segment_positions):
            for pos_idx, pos in enumerate(positions):
                self.state_to_segment.append(seg_idx)
                self.state_to_position.append(pos)
            state_idx += len(positions)

        self.state_to_segment = np.array(self.state_to_segment)
        self.state_to_position = np.array(self.state_to_position)

        # Build transition matrix
        self._build_transition_matrix()

        # Build emission model
        self._build_emission_model()

    def _interpolate_along_segment(self, edge: List[int], t: float) -> np.ndarray:
        """Interpolate position along a track segment."""
        if len(edge) < 2:
            return np.array([0, 0])

        # Find which sub-segment we're in
        total_length = 0
        segment_lengths = []

        for i in range(len(edge) - 1):
            node1, node2 = edge[i], edge[i + 1]
            p1 = self.track_graph.node_positions[node1]
            p2 = self.track_graph.node_positions[node2]
            length = np.linalg.norm(p2 - p1)
            segment_lengths.append(length)
            total_length += length

        if total_length == 0:
            return self.track_graph.node_positions[edge[0]]

        # Find target position along the segment
        target_distance = t * total_length
        current_distance = 0

        for i in range(len(edge) - 1):
            node1, node2 = edge[i], edge[i + 1]
            p1 = self.track_graph.node_positions[node1]
            p2 = self.track_graph.node_positions[node2]
            length = segment_lengths[i]

            if current_distance + length >= target_distance:
                # Interpolate within this sub-segment
                local_t = (target_distance - current_distance) / length
                return p1 + local_t * (p2 - p1)

            current_distance += length

        # If we get here, return the last point
        return self.track_graph.node_positions[edge[-1]]

    def _build_transition_matrix(self):
        """Build transition matrix between states."""
        if self.use_sparse_transitions and NUMBA_AVAILABLE:
            # Use Numba-optimized sparse transition matrix building
            # Create segments_connected matrix for Numba
            segments_connected = np.zeros((self.n_segments, self.n_segments), dtype=bool)
            for i in range(self.n_segments):
                for j in range(self.n_segments):
                    segments_connected[i, j] = self._segments_connected(i, j)
            
            self.transition_matrix = _build_sparse_transition_matrix_numba(
                self.state_to_segment, 
                self.state_to_position, 
                self.n_states,
                self.transition_smoothness,
                segments_connected
            )
        else:
            # Fallback to original implementation
            self.transition_matrix = np.zeros((self.n_states, self.n_states))

            for i in range(self.n_states):
                seg1 = self.state_to_segment[i]
                pos1 = self.state_to_position[i]

                # Early termination for sparse transitions
                if self.use_sparse_transitions:
                    # Only compute transitions to nearby states
                    nearby_states = []
                    
                    # Same segment transitions
                    for j in range(self.n_states):
                        if self.state_to_segment[j] == seg1:
                            pos_diff = abs(i - j)
                            if pos_diff <= 2:  # Only nearby positions
                                nearby_states.append(j)
                    
                    # Adjacent segment transitions
                    for j in range(self.n_states):
                        seg2 = self.state_to_segment[j]
                        if seg1 != seg2 and self._segments_connected(seg1, seg2):
                            # Only boundary positions
                            if (seg1 < seg2 and j < 50) or (seg1 > seg2 and j >= self.n_states - 50):
                                nearby_states.append(j)
                    
                    # Compute transitions only for nearby states
                    for j in nearby_states:
                        seg2 = self.state_to_segment[j]
                        pos2 = self.state_to_position[j]
                        self.transition_matrix[i, j] = self._calculate_transition_probability(
                            seg1, pos1, seg2, pos2
                        )
                else:
                    # Full transition matrix (slower but more accurate)
                    for j in range(self.n_states):
                        seg2 = self.state_to_segment[j]
                        pos2 = self.state_to_position[j]
                        self.transition_matrix[i, j] = self._calculate_transition_probability(
                            seg1, pos1, seg2, pos2
                        )

            # Normalize rows
            row_sums = self.transition_matrix.sum(axis=1)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            self.transition_matrix /= row_sums[:, np.newaxis]

    def _calculate_transition_probability(
        self, seg1: int, pos1: np.ndarray, seg2: int, pos2: np.ndarray
    ) -> float:
        """Calculate transition probability between two states."""
        # Distance-based probability
        distance = np.linalg.norm(pos2 - pos1)

        # Same segment: high probability for nearby positions
        if seg1 == seg2:
            # Gaussian kernel for smooth transitions
            prob = np.exp(-(distance**2) / (2 * self.transition_smoothness**2))
        else:
            # Different segments: lower probability, but allow transitions
            # Check if segments are connected
            if self._segments_connected(seg1, seg2):
                prob = 0.1 * np.exp(
                    -(distance**2) / (2 * self.transition_smoothness**2)
                )
            else:
                prob = 0.01 * np.exp(
                    -(distance**2) / (2 * self.transition_smoothness**2)
                )

        return prob

    def _segments_connected(self, seg1: int, seg2: int) -> bool:
        """Check if two segments are connected in the track graph."""
        if seg1 >= len(self.track_graph.edges) or seg2 >= len(self.track_graph.edges):
            return False

        edge1 = self.track_graph.edges[seg1]
        edge2 = self.track_graph.edges[seg2]

        if len(edge1) == 0 or len(edge2) == 0:
            return False

        # Check if segments share any nodes
        nodes1 = set(edge1)
        nodes2 = set(edge2)
        return len(nodes1.intersection(nodes2)) > 0

    def _build_emission_model(self):
        """Build the emission probability model."""
        # Use the state positions as emission centers
        self.emission_centers = self.state_to_position

        # Create covariance matrix for emission noise
        self.emission_covariance = np.eye(2) * self.emission_noise**2

    def _emission_probability(self, observation: np.ndarray, state: int) -> float:
        """Calculate emission probability P(observation | state)."""
        center = self.emission_centers[state]

        # Use multivariate normal distribution
        rv = multivariate_normal(center, self.emission_covariance)
        return rv.pdf(observation)

    def _emission_probabilities_vectorized(self, observations: np.ndarray) -> np.ndarray:
        """
        Calculate emission probabilities for all observations and states at once.
        Much faster than calling _emission_probability repeatedly.
        Uses Numba JIT compilation if available.
        """
        if NUMBA_AVAILABLE:
            if self.use_batch_processing and len(observations) > 100:  # Use batch processing for any significant number of observations
                return _process_batch_emissions(observations, self.emission_centers, self.emission_covariance, self.batch_size)
            else:
                return _emission_probabilities_numba(observations, self.emission_centers, self.emission_covariance)
        else:
            # Fallback to original implementation
            n_observations = len(observations)
            emission_probs = np.zeros((n_observations, self.n_states))
            
            # Pre-compute constants for Gaussian calculation
            det_cov = np.linalg.det(self.emission_covariance)
            inv_cov = np.linalg.inv(self.emission_covariance)
            norm_const = 1.0 / (2 * np.pi * np.sqrt(det_cov))
            
            for t in range(n_observations):
                obs = observations[t]
                for i in range(self.n_states):
                    center = self.emission_centers[i]
                    diff = obs - center
                    # Manual Gaussian calculation (faster than multivariate_normal)
                    exponent = -0.5 * diff.T @ inv_cov @ diff
                    emission_probs[t, i] = norm_const * np.exp(exponent)
            
            return emission_probs

    def linearize_with_hmm(
        self, positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Linearize positions using HMM inference.

        Parameters
        ----------
        positions : np.ndarray
            Array of 2D positions (n_positions, 2)

        Returns
        -------
        linear_positions : np.ndarray
            Linearized positions along the track
        track_segment_ids : np.ndarray
            Track segment IDs for each position
        projected_positions : np.ndarray
            Projected 2D positions on the track
        """
        n_positions = len(positions)

        # Initialize arrays
        linear_positions = np.full(n_positions, np.nan)
        track_segment_ids = np.full(n_positions, -1, dtype=int)
        projected_positions = np.full((n_positions, 2), np.nan)

        # Handle NaN positions
        valid_mask = ~np.isnan(positions).any(axis=1)
        if not np.any(valid_mask):
            return linear_positions, track_segment_ids, projected_positions

        valid_positions = positions[valid_mask]

        # Subsample positions if requested for speed
        if self.subsample_positions and len(valid_positions) > 1000:
            # Adaptive subsampling: use larger factor for bigger datasets
            if self.use_adaptive_subsampling:
                adaptive_factor = max(self.subsample_factor, len(valid_positions) // 2000)
            else:
                adaptive_factor = self.subsample_factor
            subsample_indices = np.arange(0, len(valid_positions), adaptive_factor)
            subsampled_positions = valid_positions[subsample_indices]
            original_indices = np.where(valid_mask)[0][subsample_indices]
        else:
            subsampled_positions = valid_positions
            original_indices = np.where(valid_mask)[0]

        # Run Viterbi algorithm to find most likely state sequence
        state_sequence = self._viterbi(subsampled_positions)

        # Convert states back to linear positions and segment IDs
        for i, state in enumerate(state_sequence):
            global_idx = original_indices[i]

            seg_id = self.state_to_segment[state]
            projected_pos = self.state_to_position[state]

            # Calculate linear position along the track
            linear_pos = self._calculate_linear_position(seg_id, projected_pos)

            linear_positions[global_idx] = linear_pos
            track_segment_ids[global_idx] = seg_id
            projected_positions[global_idx] = projected_pos

        # Interpolate missing values if subsampling was used
        if self.subsample_positions and len(original_indices) < n_positions:
            linear_positions, track_segment_ids, projected_positions = self._interpolate_missing_values(
                linear_positions, track_segment_ids, projected_positions, original_indices, valid_mask
            )

        return linear_positions, track_segment_ids, projected_positions

    def _viterbi(self, observations: np.ndarray) -> np.ndarray:
        """
        Run Viterbi algorithm to find most likely state sequence.
        Vectorized implementation for better performance.
        Uses Numba JIT compilation if available.

        Parameters
        ----------
        observations : np.ndarray
            Array of observations (n_observations, 2)

        Returns
        -------
        np.ndarray
            Most likely state sequence
        """
        n_observations = len(observations)

        # Pre-compute emission probabilities for all observations and states
        emission_probs = self._emission_probabilities_vectorized(observations)

        if NUMBA_AVAILABLE:
            return _viterbi_numba(emission_probs, self.transition_matrix, n_observations, self.n_states)
        else:
            # Fallback to original implementation
            # Initialize
            delta = np.zeros((n_observations, self.n_states))
            psi = np.zeros((n_observations, self.n_states), dtype=int)

            # Initial probabilities (uniform)
            initial_probs = np.ones(self.n_states) / self.n_states

            # Forward pass (vectorized)
            for t in range(n_observations):
                if t == 0:
                    # Initial probabilities
                    delta[t] = initial_probs * emission_probs[t]
                else:
                    # Vectorized recursion: compute all state transitions at once
                    # delta[t-1] * transition_matrix gives us all possible transitions
                    transition_probs = delta[t - 1].reshape(-1, 1) * self.transition_matrix
                    
                    # Find maximum probability and corresponding previous state for each current state
                    max_indices = np.argmax(transition_probs, axis=0)
                    max_probs = np.max(transition_probs, axis=0)
                    
                    # Update delta and psi
                    delta[t] = max_probs * emission_probs[t]
                    psi[t] = max_indices

            # Backward pass
            state_sequence = np.zeros(n_observations, dtype=int)
            state_sequence[-1] = np.argmax(delta[-1])

            for t in range(n_observations - 2, -1, -1):
                state_sequence[t] = psi[t + 1, state_sequence[t + 1]]

            return state_sequence

    def _calculate_linear_position(
        self, segment_id: int, position: np.ndarray
    ) -> float:
        """Calculate linear position along the track for a given segment and position."""
        if segment_id >= len(self.track_graph.edges):
            return 0.0

        # Calculate cumulative distance to this segment
        cumulative_distance = 0.0

        for i in range(segment_id):
            if i < len(self.track_graph.cumulative_distances):
                # Use the track graph's cumulative distances
                edge = self.track_graph.edges[i]
                if len(edge) >= 2:
                    node1, node2 = edge[0], edge[-1]
                    cumulative_distance += self.track_graph.edge_distances.get(
                        (node1, node2), 0
                    )

        # Add distance within current segment
        if segment_id < len(self.track_graph.edges):
            edge = self.track_graph.edges[segment_id]
            if len(edge) >= 2:
                # Find closest point on this segment
                min_distance = np.inf
                segment_position = 0.0

                for i in range(len(edge) - 1):
                    node1, node2 = edge[i], edge[i + 1]
                    p1 = self.track_graph.node_positions[node1]
                    p2 = self.track_graph.node_positions[node2]

                    # Project position onto this line segment
                    v = p2 - p1
                    u = position - p1
                    t = np.dot(u, v) / np.dot(v, v)
                    t = np.clip(t, 0, 1)

                    projected = p1 + t * v
                    distance = np.linalg.norm(position - projected)

                    if distance < min_distance:
                        min_distance = distance
                        segment_position = t * np.linalg.norm(v)

                cumulative_distance += segment_position

        return cumulative_distance

    def _interpolate_missing_values(
        self,
        linear_positions: np.ndarray,
        track_segment_ids: np.ndarray,
        projected_positions: np.ndarray,
        original_indices: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate missing values in subsampled linearization results.
        
        This method fills in the NaN values between linearized points using
        linear interpolation. For rat tracking data, this provides smooth
        trajectories while maintaining the speed benefits of subsampling.
        
        Parameters
        ----------
        linear_positions : np.ndarray
            Array of linear positions with NaN values for missing points
        track_segment_ids : np.ndarray
            Array of track segment IDs with -1 for missing points
        projected_positions : np.ndarray
            Array of projected 2D positions with NaN values for missing points
        original_indices : np.ndarray
            Indices of the original positions that were linearized
        valid_mask : np.ndarray
            Boolean mask indicating which original positions are valid (not NaN)
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Interpolated linear_positions, track_segment_ids, and projected_positions
        """
        from scipy.interpolate import interp1d
        
        # Get the valid linearized indices (where we have actual values)
        valid_linearized = original_indices[~np.isnan(linear_positions[original_indices])]
        
        if len(valid_linearized) < 2:
            # Not enough points to interpolate
            return linear_positions, track_segment_ids, projected_positions
        
        # Get the corresponding values
        valid_linear = linear_positions[valid_linearized]
        valid_segments = track_segment_ids[valid_linearized]
        valid_projected = projected_positions[valid_linearized]
        
        # Find all missing indices that need interpolation
        all_valid_indices = np.where(valid_mask)[0]
        missing_indices = all_valid_indices[~np.isin(all_valid_indices, valid_linearized)]
        
        if len(missing_indices) == 0:
            return linear_positions, track_segment_ids, projected_positions
        
        # Sort by index for proper interpolation
        sort_idx = np.argsort(valid_linearized)
        sorted_indices = valid_linearized[sort_idx]
        sorted_linear = valid_linear[sort_idx]
        sorted_segments = valid_segments[sort_idx]
        sorted_projected = valid_projected[sort_idx]
        
        # Interpolate linear positions
        try:
            linear_interp = interp1d(sorted_indices, sorted_linear, 
                                   kind='linear', bounds_error=False, fill_value='extrapolate')
            linear_positions[missing_indices] = linear_interp(missing_indices)
        except Exception:
            # Fallback to nearest neighbor if interpolation fails
            for idx in missing_indices:
                # Find closest valid index
                closest_idx = sorted_indices[np.argmin(np.abs(sorted_indices - idx))]
                linear_positions[idx] = linear_positions[closest_idx]
        
        # Interpolate segment IDs (use mode/nearest neighbor for categorical data)
        for idx in missing_indices:
            # Find the two closest valid indices
            distances = np.abs(sorted_indices - idx)
            closest_indices = sorted_indices[np.argsort(distances)[:2]]
            
            if len(closest_indices) == 1:
                track_segment_ids[idx] = track_segment_ids[closest_indices[0]]
            else:
                # Use the closer one, or the one with higher linear position if tied
                if distances[np.argsort(distances)[0]] < distances[np.argsort(distances)[1]]:
                    track_segment_ids[idx] = track_segment_ids[closest_indices[0]]
                else:
                    # If equidistant, use the one with higher linear position
                    if linear_positions[closest_indices[0]] > linear_positions[closest_indices[1]]:
                        track_segment_ids[idx] = track_segment_ids[closest_indices[0]]
                    else:
                        track_segment_ids[idx] = track_segment_ids[closest_indices[1]]
        
        # Interpolate projected positions
        try:
            # Interpolate x and y coordinates separately
            x_interp = interp1d(sorted_indices, sorted_projected[:, 0], 
                              kind='linear', bounds_error=False, fill_value='extrapolate')
            y_interp = interp1d(sorted_indices, sorted_projected[:, 1], 
                              kind='linear', bounds_error=False, fill_value='extrapolate')
            
            projected_positions[missing_indices, 0] = x_interp(missing_indices)
            projected_positions[missing_indices, 1] = y_interp(missing_indices)
        except Exception:
            # Fallback to nearest neighbor if interpolation fails
            for idx in missing_indices:
                closest_idx = sorted_indices[np.argmin(np.abs(sorted_indices - idx))]
                projected_positions[idx] = projected_positions[closest_idx]
        
        return linear_positions, track_segment_ids, projected_positions


def project_position_to_track(
    position: np.ndarray, track_graph: TrackGraph
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 2D positions onto the track graph.

    Parameters
    ----------
    position : np.ndarray
        Array of 2D positions (n_positions, 2)
    track_graph : TrackGraph
        Track graph object

    Returns
    -------
    linear_position : np.ndarray
        Linearized positions along the track
    track_segment_id : np.ndarray
        Track segment IDs for each position
    projected_position : np.ndarray
        Projected 2D positions on the track
    """
    n_positions = position.shape[0]
    linear_position = np.full(n_positions, np.nan)
    track_segment_id = np.full(n_positions, -1, dtype=int)
    projected_position = np.full((n_positions, 2), np.nan)

    for i, pos in enumerate(position):
        # Find closest node
        distances_to_nodes = np.linalg.norm(track_graph.node_positions - pos, axis=1)
        closest_node = np.argmin(distances_to_nodes)

        # Find closest edge
        min_distance = np.inf
        best_segment = -1
        best_projection = None

        for edge_idx, edge in enumerate(track_graph.edges):
            if len(edge) >= 2:
                for j in range(len(edge) - 1):
                    node1, node2 = edge[j], edge[j + 1]
                    p1 = track_graph.node_positions[node1]
                    p2 = track_graph.node_positions[node2]

                    # Project point onto line segment
                    v = p2 - p1
                    u = pos - p1
                    t = np.dot(u, v) / np.dot(v, v)
                    t = np.clip(t, 0, 1)

                    projection = p1 + t * v
                    distance = np.linalg.norm(pos - projection)

                    if distance < min_distance:
                        min_distance = distance
                        best_segment = edge_idx
                        best_projection = projection

        if best_segment >= 0 and best_projection is not None:
            # Calculate linear position
            edge = track_graph.edges[best_segment]
            for j in range(len(edge) - 1):
                node1, node2 = edge[j], edge[j + 1]
                p1 = track_graph.node_positions[node1]
                p2 = track_graph.node_positions[node2]

                # Check if projection is on this segment
                v = p2 - p1
                u = best_projection - p1
                t = np.dot(u, v) / np.dot(v, v)

                if 0 <= t <= 1:
                    # Linear position is cumulative distance to node1 + distance along segment
                    linear_position[i] = track_graph.cumulative_distances[
                        node1
                    ] + t * np.linalg.norm(v)
                    track_segment_id[i] = best_segment
                    projected_position[i] = best_projection
                    break

    return linear_position, track_segment_id, projected_position


def plot_linearization_confirmation(
    original_positions: np.ndarray,
    linearized_df: pd.DataFrame,
    track_graph: TrackGraph,
    title: str = "Linearization Confirmation",
    show_plot: bool = True,
) -> None:
    """
    Create a confirmation plot showing the linearization results.
    
    Parameters
    ----------
    original_positions : np.ndarray
        Original 2D positions (n_positions, 2)
    linearized_df : pd.DataFrame
        DataFrame with linearization results from get_linearized_position
    track_graph : TrackGraph
        Track graph object used for linearization
    title : str, optional
        Title for the plot, by default "Linearization Confirmation"
    show_plot : bool, optional
        Whether to display the plot, by default True
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)
    
    # Create color map for segments
    unique_segments = sorted(linearized_df['track_segment_id'].unique())
    if len(unique_segments) > 0:
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_segments)))
        segment_colors = dict(zip(unique_segments, colors))
    else:
        segment_colors = {}
    
    # Plot 1: Original 2D positions with track graph (color coded by segment)
    ax1 = axes[0, 0]
    
    # Color code original positions by their corresponding segment
    if len(segment_colors) > 0:
        # Get segment IDs for original positions (assuming they correspond to linearized_df order)
        segment_ids = linearized_df['track_segment_id'].values
        valid_mask = segment_ids >= 0  # Only plot valid segments
        
        if np.any(valid_mask):
            valid_positions = original_positions[valid_mask]
            valid_segments = segment_ids[valid_mask]
            colors_for_positions = [segment_colors.get(seg, 'gray') for seg in valid_segments]
            
            ax1.scatter(valid_positions[:, 0], valid_positions[:, 1], 
                        c=colors_for_positions, s=1, alpha=0.6)
        else:
            ax1.scatter(original_positions[:, 0], original_positions[:, 1], 
                        c='lightblue', s=1, alpha=0.6)
    else:
        ax1.scatter(original_positions[:, 0], original_positions[:, 1], 
                    c='lightblue', s=1, alpha=0.6)
    
    # Plot track graph nodes and edges with color coding
    node_positions = track_graph.node_positions
    ax1.scatter(node_positions[:, 0], node_positions[:, 1], 
                c='red', s=50, zorder=5)
    
    # Draw edges with color coding
    for i, edge in enumerate(track_graph.edges):
        start_pos = node_positions[edge[0]]
        end_pos = node_positions[edge[1]]
        edge_color = segment_colors.get(i, 'black')
        ax1.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                 color=edge_color, linewidth=3, alpha=0.8)
    
    ax1.set_xlabel('X Position (cm)')
    ax1.set_ylabel('Y Position (cm)')
    ax1.set_title('Original 2D Positions with Track Graph (Color Coded by Segment)')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Plot 2: Projected positions on track (color coded by segment)
    ax2 = axes[0, 1]
    
    # Color code projected positions by segment
    if len(segment_colors) > 0:
        valid_mask = linearized_df['track_segment_id'] >= 0
        if np.any(valid_mask):
            valid_proj_x = linearized_df.loc[valid_mask, 'projected_x_position']
            valid_proj_y = linearized_df.loc[valid_mask, 'projected_y_position']
            valid_segments = linearized_df.loc[valid_mask, 'track_segment_id']
            colors_for_proj = [segment_colors.get(seg, 'gray') for seg in valid_segments]
            
            ax2.scatter(valid_proj_x, valid_proj_y, 
                        c=colors_for_proj, s=1, alpha=0.6)
        else:
            ax2.scatter(linearized_df['projected_x_position'], linearized_df['projected_y_position'], 
                        c='green', s=1, alpha=0.6)
    else:
        ax2.scatter(linearized_df['projected_x_position'], linearized_df['projected_y_position'], 
                    c='green', s=1, alpha=0.6)
    
    # Plot track graph nodes and edges with color coding
    ax2.scatter(node_positions[:, 0], node_positions[:, 1], 
                c='red', s=50, zorder=5)
    
    # Draw edges with color coding
    for i, edge in enumerate(track_graph.edges):
        start_pos = node_positions[edge[0]]
        end_pos = node_positions[edge[1]]
        edge_color = segment_colors.get(i, 'black')
        ax2.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], 
                 color=edge_color, linewidth=3, alpha=0.8)
    
    ax2.set_xlabel('X Position (cm)')
    ax2.set_ylabel('Y Position (cm)')
    ax2.set_title('Projected Positions on Track (Color Coded by Segment)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Plot 3: Linear position over time (color coded by segment)
    ax3 = axes[1, 0]
    time_points = np.arange(len(linearized_df))
    
    if len(segment_colors) > 0:
        # Plot each segment separately with different colors using scatter
        for segment_id in unique_segments:
            if segment_id >= 0:  # Only plot valid segments
                segment_mask = linearized_df['track_segment_id'] == segment_id
                if np.any(segment_mask):
                    segment_times = time_points[segment_mask]
                    segment_positions = linearized_df.loc[segment_mask, 'linear_position']
                    ax3.scatter(segment_times, segment_positions, 
                               c=segment_colors[segment_id], s=1, alpha=0.7)
    else:
        ax3.scatter(time_points, linearized_df['linear_position'], c='blue', s=1, alpha=0.7)
    
    ax3.set_xlabel('Time Point')
    ax3.set_ylabel('Linear Position (cm)')
    ax3.set_title('Linear Position Over Time (Color Coded by Segment)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Track segment distribution (color coded)
    ax4 = axes[1, 1]
    segment_counts = linearized_df['track_segment_id'].value_counts().sort_index()
    
    if len(segment_colors) > 0:
        # Color code the bars by segment
        bar_colors = [segment_colors.get(seg, 'gray') for seg in segment_counts.index]
        ax4.bar(segment_counts.index, segment_counts.values, alpha=0.7, color=bar_colors)
    else:
        ax4.bar(segment_counts.index, segment_counts.values, alpha=0.7, color='orange')
    
    ax4.set_xlabel('Track Segment ID')
    ax4.set_ylabel('Number of Positions')
    ax4.set_title('Distribution Across Track Segments (Color Coded)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if show_plot:
        plt.show(block=True)


def get_linearized_position(
    position: np.ndarray,
    track_graph: TrackGraph,
    edge_order: Optional[List[List[int]]] = None,
    use_HMM: bool = False,
    show_confirmation_plot: bool = True,
    # HMM optimization parameters
    n_bins_per_segment: int = 30,  # Reduced from 50 for better speed
    use_sparse_transitions: bool = True,
    subsample_positions: bool = False,  # Keep False for accuracy
    subsample_factor: int = 5,
    use_adaptive_subsampling: bool = True,
    # Advanced optimization parameters
    use_batch_processing: bool = False,
    batch_size: int = 1000,
) -> pd.DataFrame:
    """
    Get linearized position from 2D positions using track graph.

    Parameters
    ----------
    position : np.ndarray
        Array of 2D positions (n_positions, 2)
    track_graph : TrackGraph
        Track graph object
    edge_order : list, optional
        Order of edges to use (for compatibility)
    use_HMM : bool, optional
        Whether to use HMM-based linearization for smoother, more robust results
    show_confirmation_plot : bool, optional
        Whether to show a confirmation plot of the linearization results, by default True

    Returns
    -------
    pd.DataFrame
        DataFrame with linearized position, track segment ID, and projected positions
        
    Notes
    -----
    When using HMM with subsampling (subsample_positions=True), interpolation is automatically
    applied to fill missing values, providing 100% coverage while maintaining speed benefits.
    This is particularly useful for rat tracking data from DeepLabCut where smooth trajectories
    are desired.
    """
    if use_HMM:
        hmm_linearizer = HMMLinearizer(
            track_graph,
            n_bins_per_segment=n_bins_per_segment,
            use_sparse_transitions=use_sparse_transitions,
            subsample_positions=subsample_positions,
            subsample_factor=subsample_factor,
            use_adaptive_subsampling=use_adaptive_subsampling,
            use_batch_processing=use_batch_processing,
            batch_size=batch_size,
        )
        linear_position, track_segment_id, projected_position = (
            hmm_linearizer.linearize_with_hmm(position)
        )
    else:
        linear_position, track_segment_id, projected_position = (
            project_position_to_track(position, track_graph)
        )

    result_df = pd.DataFrame(
        {
            "linear_position": linear_position,
            "track_segment_id": track_segment_id,
            "projected_x_position": projected_position[:, 0],
            "projected_y_position": projected_position[:, 1],
        }
    )
    
    if show_confirmation_plot:
        method_name = "HMM-based" if use_HMM else "Standard"
        plot_linearization_confirmation(
            position, 
            result_df, 
            track_graph, 
            title=f"Linearization Confirmation ({method_name})"
        )
    
    return result_df


def make_track_graph(node_positions: np.ndarray, edges: List[List[int]]) -> TrackGraph:
    """
    Create a track graph from node positions and edges.

    Parameters
    ----------
    node_positions : np.ndarray
        Array of node positions (n_nodes, 2)
    edges : list
        List of edge connections between nodes

    Returns
    -------
    TrackGraph
        Track graph object
    """
    return TrackGraph(node_positions, edges)


class NodePicker:
    """
    Interactive creation of track graph by looking at video frames.

    Parameters
    ----------
    ax : plt.Axes, optional
        The matplotlib axes to draw on, by default None.
    basepath : str, optional
        The base path for saving data, by default None.
    node_color : str, optional
        The color of the nodes, by default "#177ee6".
    node_size : int, optional
        The size of the nodes, by default 100.
    epoch : int, optional
        The epoch number, by default None.
    interval : Tuple[float, float], optional
        Time interval to process, by default None.

    Attributes
    ----------
    ax : plt.Axes
        The matplotlib axes to draw on.
    canvas : plt.FigureCanvas
        The matplotlib figure canvas.
    cid : int
        The connection id for the event handler.
    _nodes : list
        The list of node positions.
    node_color : str
        The color of the nodes.
    _nodes_plot : plt.scatter
        The scatter plot of the nodes.
    edges : list
        The list of edges.
    basepath : str
        The base path for saving data.
    epoch : int
        The epoch number.
    use_HMM : bool
        Whether to use the hidden markov model.

    Methods
    -------
    node_positions
        Get the positions of the nodes.
    connect
        Connect the event handlers.
    disconnect
        Disconnect the event handlers.
    process_key
        Process key press events.
    click_event
        Process mouse click events.
    redraw
        Redraw the nodes and edges.
    remove_point
        Remove a point from the nodes.
    clear
        Clear all nodes and edges.
    format_and_save
        Format the data and save it to disk.
    save_nodes_edges
        Save the nodes and edges to a pickle file.
    save_nodes_edges_to_behavior
        Store nodes and edges into behavior file.

    Examples
    --------
    # in command line
    >>> python linearization.py path/to/session

    # for a specific epoch
    >>> python linearization.py path/to/session 1

    # for a specific interval
    >>> python linearization.py path/to/session 0 100
    """

    def __init__(
        self,
        ax: Optional[plt.Axes] = None,
        basepath: Optional[str] = None,
        node_color: str = "#177ee6",
        node_size: int = 100,
        epoch: Optional[int] = None,
        interval: Optional[Tuple[float, float]] = None,
    ):
        """
        Initialize the NodePicker.

        Parameters
        ----------
        ax : plt.Axes, optional
            The matplotlib axes to draw on, by default None.
        basepath : str, optional
            The base path for saving data, by default None.
        node_color : str, optional
            The color of the nodes, by default "#177ee6".
        node_size : int, optional
            The size of the nodes, by default 100.
        epoch : int, optional
            The epoch number, by default None.
        interval : Tuple[float, float], optional
            Time interval to process, by default None.
        """
        if ax is None:
            ax = plt.gca()
        self.ax = ax
        self.canvas = ax.get_figure().canvas
        self.cid = None
        self._nodes = []
        self.node_color = node_color
        self._nodes_plot = ax.scatter([], [], zorder=5, s=node_size, color=node_color)
        self.edges = [[]]
        self.basepath = basepath
        self.epoch = epoch
        self.interval = interval
        self.use_HMM = True

        if self.epoch is not None:
            self.epoch = int(self.epoch)

        ax.set_title(
            "Left click to place node.\nRight click to remove node."
            "\nShift+Left click to clear nodes.\nCntrl+Left click two nodes to place an edge"
            "\nEnter to save and exit.",
            fontsize=8,
        )

        self.canvas.draw()
        self.connect()

    @property
    def node_positions(self) -> np.ndarray:
        """
        Get the positions of the nodes.

        Returns
        -------
        np.ndarray
            An array of node positions.
        """
        return np.asarray(self._nodes)

    def connect(self) -> None:
        """Connect the event handlers."""
        print("Connecting to events")
        if self.cid is None:
            self.cid = self.canvas.mpl_connect("button_press_event", self.click_event)
            self.canvas.mpl_connect("key_press_event", self.process_key)
            print("Mouse click event connected!")

    def disconnect(self) -> None:
        """Disconnect the event handlers."""
        if self.cid is not None:
            self.canvas.mpl_disconnect(self.cid)
            self.cid = None

    def process_key(self, event: Any) -> None:
        """
        Process key press events.

        Parameters
        ----------
        event : Any
            The key press event.
        """
        if event.key == "enter":
            self.format_and_save()

    def click_event(self, event: Any) -> None:
        """
        Process mouse click events.

        Parameters
        ----------
        event : Any
            The mouse click event.
        """
        print(
            f"Mouse clicked at: {event.xdata}, {event.ydata}, button: {event.button}, key: {event.key}"
        )
        if not event.inaxes:
            return

        if event.key is None:  # Regular mouse clicks
            if event.button == 1:  # Left click
                self._nodes.append((event.xdata, event.ydata))
            elif event.button == 3:  # Right click
                self.remove_point((event.xdata, event.ydata))

        elif event.key == "shift" and event.button == 1:  # Shift + Left click
            self.clear()

        elif (
            event.key == "control" and event.button == 1
        ):  # Ctrl + Left click (Edge creation)
            if len(self._nodes) == 0:
                return
            point = (event.xdata, event.ydata)
            distance_to_nodes = np.linalg.norm(self.node_positions - point, axis=1)
            closest_node_ind = np.argmin(distance_to_nodes)
            if len(self.edges[-1]) < 2:
                self.edges[-1].append(closest_node_ind)
            else:
                self.edges.append([closest_node_ind])

        elif event.key == "enter":  # Pressing Enter
            self.format_and_save()

        self.redraw()

    def redraw(self) -> None:
        """Redraw the nodes and edges."""
        # Draw Node Circles
        if len(self.node_positions) > 0:
            self._nodes_plot.set_offsets(self.node_positions)
        else:
            self._nodes_plot.set_offsets([])

        # Draw Node Numbers
        for ind, (x, y) in enumerate(self.node_positions):
            self.ax.text(
                x,
                y,
                ind,
                zorder=6,
                fontsize=10,
                horizontalalignment="center",
                verticalalignment="center",
                clip_on=True,
                bbox=None,
                transform=self.ax.transData,
            )
        # Draw Edges
        for edge in self.edges:
            if len(edge) > 1:
                x1, y1 = self.node_positions[edge[0]]
                x2, y2 = self.node_positions[edge[1]]
                self.ax.plot(
                    [x1, x2], [y1, y2], color="#1f8e4f", linewidth=3, zorder=1000
                )
        self.canvas.draw()

    def remove_point(self, point: Tuple[float, float]) -> None:
        """
        Remove a point from the nodes.

        Parameters
        ----------
        point : Tuple[float, float]
            The point to remove.
        """
        if len(self._nodes) > 0:
            distance_to_nodes = np.linalg.norm(self.node_positions - point, axis=1)
            closest_node_ind = np.argmin(distance_to_nodes)
            self._nodes.pop(closest_node_ind)

    def clear(self) -> None:
        """Clear all nodes and edges."""
        self._nodes = []
        self.edges = [[]]
        self.redraw()

    def format_and_save(self) -> None:
        """Format the data and save it to disk."""
        behave_df = load_animal_behavior(self.basepath)

        if self.epoch is not None:
            epochs = load_epoch(self.basepath)

            cur_epoch = (
                ~np.isnan(behave_df.x)
                & (behave_df.time >= epochs.iloc[self.epoch].startTime)
                & (behave_df.time <= epochs.iloc[self.epoch].stopTime)
            )
        elif self.interval is not None:
            cur_epoch = (
                ~np.isnan(behave_df.x)
                & (behave_df.time >= self.interval[0])
                & (behave_df.time <= self.interval[1])
            )
        else:
            cur_epoch = ~np.isnan(behave_df.x)

        print("running linearization...")
        track_graph = make_track_graph(self.node_positions, self.edges)

        position = np.vstack(
            [behave_df[cur_epoch].x.values, behave_df[cur_epoch].y.values]
        ).T

        position_df = get_linearized_position(
            position=position,
            track_graph=track_graph,
            edge_order=self.edges,
            use_HMM=self.use_HMM,
            show_confirmation_plot=True,
            # HMM optimization parameters
            n_bins_per_segment=30,  # Reduced from 50 for better speed
            use_sparse_transitions=True,
            subsample_positions=False,  # Keep False for accuracy
            subsample_factor=5,
            use_adaptive_subsampling=True,
            # Advanced optimization parameters
            use_batch_processing=False,
            batch_size=1000,
        )

        print("saving to disk...")
        behave_df.loc[cur_epoch, "linearized"] = position_df.linear_position.values
        behave_df.loc[cur_epoch, "states"] = position_df.track_segment_id.values
        behave_df.loc[cur_epoch, "projected_x_position"] = (
            position_df.projected_x_position.values
        )
        behave_df.loc[cur_epoch, "projected_y_position"] = (
            position_df.projected_y_position.values
        )

        filename = os.path.join(
            self.basepath, os.path.basename(self.basepath) + ".animal.behavior.mat"
        )

        data = loadmat(filename, simplify_cells=True)

        data["behavior"]["position"]["linearized"] = behave_df.linearized.values
        data["behavior"]["states"] = behave_df.states.values
        data["behavior"]["position"]["projected_x"] = (
            behave_df.projected_x_position.values
        )
        data["behavior"]["position"]["projected_y"] = (
            behave_df.projected_y_position.values
        )

        # store nodes and edges within behavior file
        data = self.save_nodes_edges_to_behavior(data, behave_df)

        savemat(filename, data, long_field_names=True)

        self.save_nodes_edges()
        self.disconnect()

    def save_nodes_edges(self) -> None:
        """Save the nodes and edges to a pickle file."""
        results = {"node_positions": self.node_positions, "edges": self.edges}
        save_file = os.path.join(self.basepath, "linearization_nodes_edges.pkl")
        with open(save_file, "wb") as f:
            pickle.dump(results, f)

    def save_nodes_edges_to_behavior(self, data: dict, behave_df: pd.DataFrame) -> dict:
        """
        Store nodes and edges into behavior file.
        Searches to find epochs with valid linearized coords.
        Nodes and edges are stored within behavior.epochs{n}.{node_positions and edges}

        Parameters
        ----------
        data : dict
            The behavior data dictionary.
        behave_df : pd.DataFrame
            The DataFrame containing behavior data.

        Returns
        -------
        dict
            The updated behavior data dictionary.
        """
        if self.epoch is None and self.interval is None:
            # load epochs
            epochs = load_epoch(self.basepath)
            # iter over each epoch
            for epoch_i, ep in enumerate(epochs.itertuples()):
                # locate index for given epoch
                idx = behave_df.time.between(ep.startTime, ep.stopTime)
                # if linearized is not all nan, add nodes and edges
                if not all(np.isnan(behave_df[idx].linearized)) & (
                    behave_df[idx].shape[0] != 0
                ):
                    # adding nodes and edges
                    data["behavior"]["epochs"][epoch_i]["node_positions"] = (
                        self.node_positions
                    )
                    data["behavior"]["epochs"][epoch_i]["edges"] = self.edges
        elif self.interval is not None:
            # if interval was used, add nodes and edges just the epochs within that interval
            epochs = load_epoch(self.basepath)
            for epoch_i, ep in enumerate(epochs.itertuples()):
                # amount of overlap between interval and epoch
                start_overlap = max(self.interval[0], ep.startTime)
                end_overlap = min(self.interval[1], ep.stopTime)
                overlap = max(0, end_overlap - start_overlap)

                # if overlap is greater than 1 second, add nodes and edges
                if overlap > 1:
                    data["behavior"]["epochs"][epoch_i]["node_positions"] = (
                        self.node_positions
                    )
                    data["behavior"]["epochs"][epoch_i]["edges"] = self.edges
        else:
            # if epoch was used, add nodes and edges just that that epoch
            data["behavior"]["epochs"][self.epoch]["node_positions"] = (
                self.node_positions
            )
            data["behavior"]["epochs"][self.epoch]["edges"] = self.edges

        return data


def load_animal_behavior(basepath: str) -> pd.DataFrame:
    """
    Load animal behavior data from a .mat file.

    Parameters
    ----------
    basepath : str
        The base path where the .mat file is located.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the animal behavior data.
    """
    filename = os.path.join(
        basepath, os.path.basename(basepath) + ".animal.behavior.mat"
    )
    data = loadmat(filename, simplify_cells=True)
    df = pd.DataFrame()
    df["time"] = data["behavior"]["timestamps"]
    try:
        df["states"] = data["behavior"]["states"]
    except Exception:
        pass
    for key in data["behavior"]["position"].keys():
        try:
            df[key] = data["behavior"]["position"][key]
        except Exception:
            pass
    return df


def load_epoch(basepath: str) -> pd.DataFrame:
    """
    Load epoch info from cell explorer basename.session and store in a DataFrame.

    Parameters
    ----------
    basepath : str
        The base path where the .session.mat file is located.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the epoch information.
    """
    filename = os.path.join(basepath, os.path.basename(basepath) + ".session.mat")
    data = loadmat(filename, simplify_cells=True)
    try:
        return pd.DataFrame(data["session"]["epochs"])
    except Exception:
        return pd.DataFrame([data["session"]["epochs"]])


def run(
    basepath: str,
    epoch: Optional[int] = None,
    interval: Optional[Tuple[float, float]] = None,
) -> None:
    """
    Run the linearization pipeline.

    Parameters
    ----------
    basepath : str
        The base path where the data files are located.
    epoch : int, optional
        The epoch number to process, by default None.
    interval : Tuple[float, float], optional
        Time interval to process, by default None.

    Returns
    -------
    None
    """
    plt.close("all")
    print("here is the file,", basepath)

    with plt.style.context("dark_background"):
        plt.ioff()

        _, ax = plt.subplots(figsize=(5, 5))

        behave_df = load_animal_behavior(basepath)

        if epoch is not None:
            epochs = load_epoch(basepath)

            behave_df = behave_df[
                behave_df["time"].between(
                    epochs.iloc[epoch].startTime, epochs.iloc[epoch].stopTime
                )
            ]
        elif interval is not None:
            behave_df = behave_df[behave_df["time"].between(interval[0], interval[1])]

        ax.scatter(behave_df.x, behave_df.y, color="white", s=0.5, alpha=0.5)
        ax.axis("equal")
        ax.set_axisbelow(True)
        ax.yaxis.grid(color="gray", linestyle="dashed")
        ax.xaxis.grid(color="gray", linestyle="dashed")
        ax.set_ylabel("y (cm)")
        ax.set_xlabel("x (cm)")

        picker = NodePicker(ax=ax, basepath=basepath, epoch=epoch, interval=interval)
        picker.connect()  # Ensure connection

        plt.show(block=True)


if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) == 2:
        run(sys.argv[1])
    elif len(sys.argv) == 3:
        run(sys.argv[1], epoch=int(sys.argv[2]))
    elif len(sys.argv) == 4:
        run(sys.argv[1], interval=(float(sys.argv[2]), float(sys.argv[3])))
