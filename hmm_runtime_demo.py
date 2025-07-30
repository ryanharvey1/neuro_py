#!/usr/bin/env python3
"""
HMM Linearization Runtime Comparison Demo

This script demonstrates the runtime differences between various HMM optimization configurations.
"""

import time
import numpy as np
from neuro_py.behavior.linearization import HMMLinearizer, TrackGraph, make_track_graph


def create_simple_track():
    """Create a simple track with multiple segments."""
    # Define a more complex track with curves
    node_positions = np.array([
        [0, 0],      # Start
        [10, 0],     # Straight segment
        [20, 0],     # 
        [30, 0],     # End of first straight
        [40, 10],    # Curve up
        [50, 20],    # 
        [60, 20],    # Straight segment
        [70, 20],    # 
        [80, 10],    # Curve down
        [90, 0],     # 
        [100, 0],    # End
    ])
    
    # Define edges connecting the nodes
    edges = [
        [0, 1, 2, 3],      # First straight segment
        [3, 4, 5],         # Curve up
        [5, 6, 7],         # Second straight segment
        [7, 8, 9, 10],     # Curve down
    ]
    
    return make_track_graph(node_positions, edges)


def generate_test_positions(track_graph, n_positions=5000, noise_level=2.0):
    """Generate test positions along the track with noise."""
    positions = []
    
    # Generate positions along each segment
    for edge_idx, edge in enumerate(track_graph.edges):
        if len(edge) >= 2:
            # Generate positions along this edge
            n_segment_positions = n_positions // len(track_graph.edges)
            
            for i in range(n_segment_positions):
                # Parameter along the edge (0 to 1)
                t = i / (n_segment_positions - 1)
                
                # Interpolate position along the edge
                if len(edge) == 2:
                    # Simple linear interpolation
                    p1 = track_graph.node_positions[edge[0]]
                    p2 = track_graph.node_positions[edge[1]]
                    pos = p1 + t * (p2 - p1)
                else:
                    # Multi-node edge - use piecewise linear
                    total_length = 0
                    for j in range(len(edge) - 1):
                        p1 = track_graph.node_positions[edge[j]]
                        p2 = track_graph.node_positions[edge[j + 1]]
                        segment_length = np.linalg.norm(p2 - p1)
                        total_length += segment_length
                    
                    # Find which segment we're in
                    current_length = 0
                    target_length = t * total_length
                    
                    for j in range(len(edge) - 1):
                        p1 = track_graph.node_positions[edge[j]]
                        p2 = track_graph.node_positions[edge[j + 1]]
                        segment_length = np.linalg.norm(p2 - p1)
                        
                        if current_length + segment_length >= target_length:
                            # We're in this segment
                            local_t = (target_length - current_length) / segment_length
                            pos = p1 + local_t * (p2 - p1)
                            break
                        current_length += segment_length
                    else:
                        # Fallback to last segment
                        p1 = track_graph.node_positions[edge[-2]]
                        p2 = track_graph.node_positions[edge[-1]]
                        pos = p2
                
                # Add noise
                noise = np.random.normal(0, noise_level, 2)
                pos += noise
                
                positions.append(pos)
    
    return np.array(positions)


def test_hmm_configurations():
    """Test different HMM configurations and measure runtime."""
    print("HMM Linearization Runtime Comparison")
    print("=" * 50)
    
    # Create test data
    track_graph = create_simple_track()
    positions = generate_test_positions(track_graph, n_positions=5000)
    
    print(f"Test data: {len(positions)} positions, {len(track_graph.edges)} segments\n")
    
    # Define configurations to test
    configurations = [
        {
            "name": "Ultra Fast (Numba + Sparse + Adaptive + Batch + Interpolation)",
            "params": {
                "n_bins_per_segment": 25,  # Increased from 20 for better accuracy
                "use_sparse_transitions": True,
                "subsample_positions": True,
                "subsample_factor": 5,  # Reduced from 10 for better accuracy (process 20% instead of 10%)
                "use_adaptive_subsampling": True,
                "use_batch_processing": True,
                "batch_size": 500,  # Smaller batch size for subsampled data
            }
        },
        {
            "name": "Ultra Fast (Vectorized + Sparse + Adaptive)",
            "params": {
                "n_bins_per_segment": 30,
                "use_sparse_transitions": True,
                "subsample_positions": False,
                "subsample_factor": 5,
                "use_adaptive_subsampling": True,
                "use_batch_processing": False,
                "batch_size": 1000,
            }
        },
        {
            "name": "Fast (Sparse + Subsample + Interpolation)",
            "params": {
                "n_bins_per_segment": 30,
                "use_sparse_transitions": True,
                "subsample_positions": True,
                "subsample_factor": 5,
                "use_adaptive_subsampling": False,
                "use_batch_processing": False,
                "batch_size": 1000,
            }
        },
        {
            "name": "Medium (Sparse only)",
            "params": {
                "n_bins_per_segment": 30,
                "use_sparse_transitions": True,
                "subsample_positions": False,
                "subsample_factor": 5,
                "use_adaptive_subsampling": False,
                "use_batch_processing": False,
                "batch_size": 1000,
            }
        },
        {
            "name": "Default",
            "params": {
                "n_bins_per_segment": 50,
                "use_sparse_transitions": False,
                "subsample_positions": False,
                "subsample_factor": 5,
                "use_adaptive_subsampling": False,
                "use_batch_processing": False,
                "batch_size": 1000,
            }
        },
        {
            "name": "High Quality (Full matrix)",
            "params": {
                "n_bins_per_segment": 50,
                "use_sparse_transitions": False,
                "subsample_positions": False,
                "subsample_factor": 1,
                "use_adaptive_subsampling": False,
                "use_batch_processing": False,
                "batch_size": 1000,
            }
        },
    ]
    
    results = []
    
    for config in configurations:
        print(f"Testing: {config['name']}")
        
        # Create HMM linearizer with specific parameters
        hmm_linearizer = HMMLinearizer(
            track_graph=track_graph,
            **config["params"]
        )
        
        # Measure runtime
        start_time = time.time()
        try:
            linear_positions, track_segment_ids, projected_positions = hmm_linearizer.linearize_with_hmm(positions)
            runtime = time.time() - start_time
            
            # Calculate success rate (non-NaN positions)
            success_rate = np.sum(~np.isnan(linear_positions)) / len(linear_positions) * 100
            
            print(f"  Runtime: {runtime:.2f}s")
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  States: {hmm_linearizer.n_states}")
            print(f"  Observations processed: {len(linear_positions)}")
            print(f"  Valid positions: {np.sum(~np.isnan(linear_positions))}")
            print()
            
            results.append({
                "name": config["name"],
                "runtime": runtime,
                "success_rate": success_rate,
                "states": hmm_linearizer.n_states,
            })
            
        except Exception as e:
            print(f"  Error: {e}")
            print()
            results.append({
                "name": config["name"],
                "runtime": float('inf'),
                "success_rate": 0.0,
                "states": 0,
            })
    
    # Print summary table
    print("Summary:")
    print("-" * 80)
    print(f"{'Configuration':<50} {'Runtime (s)':<12} {'Success Rate':<12} {'States':<8}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['name']:<50} {result['runtime']:<12.2f} {result['success_rate']:<12.1f} {result['states']:<8}")
    
    print()
    print("Recommendations:")
    print("- For quick testing: Use 'Ultra Fast (Numba + Sparse + Adaptive + Batch + Interpolation)' configuration")
    print("- For production: Use 'Medium' configuration (good speed/accuracy balance)")
    print("- For highest quality: Use 'High Quality' configuration (but expect longer runtime)")
    print()
    print("Note: The 'Ultra Fast' configurations use Numba JIT compilation, vectorized operations,")
    print("sparse transitions, adaptive subsampling, and batch processing for maximum speed.")
    print("Batch processing is particularly effective for very large datasets (>10k positions).")
    print()
    print("NEW: Subsampled configurations now include interpolation to fill missing values,")
    print("providing 100% coverage while maintaining speed benefits.")


if __name__ == "__main__":
    test_hmm_configurations() 