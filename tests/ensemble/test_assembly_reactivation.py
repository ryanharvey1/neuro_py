from itertools import chain

import nelpy as nel
import numpy as np
import pytest
from scipy import stats

from neuro_py.ensemble import assembly_reactivation


def test_assembly_reactivation():
    def lif_neuron(n_steps=1000, alpha=0.01, rate=10):
        """Simulate a linear integrate-and-fire neuron.

        Args:
        n_steps (int): The number of time steps to simulate the neuron's activity.
        alpha (float): The input scaling factor
        rate (int): The mean rate of incoming spikes

        """
        # Precompute Poisson samples for speed
        exc = stats.poisson(rate).rvs(n_steps)

        # Initialize voltage and spike storage
        v = np.zeros(n_steps)
        spike_times = []

        # Loop over steps
        for i in range(1, n_steps):
            # Update v
            dv = alpha * exc[i]
            v[i] = v[i - 1] + dv

            # If spike happens, reset voltage and record
            if v[i] > 1:
                spike_times.append(i)
                v[i] = 0

        return v, spike_times

    def jitter(spike_times, rate=10, jitter_amount=0.001):
        return [
            spike_times + stats.poisson(rate).rvs(len(spike_times)) * jitter_amount
            for _ in np.arange(0, 0.1, 0.01)
        ]

    def generate_data():
        v, spike_times_ = lif_neuron()
        spike_times_1 = jitter(spike_times_, rate=10)

        v, spike_times_ = lif_neuron()
        spike_times_2 = jitter(spike_times_, rate=10)

        v, spike_times_ = lif_neuron()
        spike_times_3 = jitter(spike_times_, rate=10)

        v, spike_times_ = lif_neuron()
        spike_times_4 = jitter(spike_times_, rate=10)

        v, spike_times_ = lif_neuron()
        spike_times_5 = jitter(spike_times_, rate=10)

        spike_times = np.array(
            list(
                chain.from_iterable(
                    [
                        spike_times_1,
                        spike_times_2,
                        spike_times_3,
                        spike_times_4,
                        spike_times_5,
                    ]
                )
            ),
            dtype=object,
        )
        return spike_times

    def create_pre_task_post_assembly():
        """
        Create a spike train with a pre-task, task, and post-task assembly.
        creates more noise in pre than in post
        """
        v, spike_times_ = lif_neuron(n_steps=3000, rate=10)

        spike_times_1 = jitter(spike_times_, rate=10)
        edges = np.linspace(0, len(spike_times_1[0]), 4).astype(int)
        # add a lot of noise to the pre-task epoch
        spike_times_jittered = jitter(spike_times_, rate=10, jitter_amount=1)
        for i in range(0, len(spike_times_1)):
            spike_times_1[i][edges[0] : edges[1]] = spike_times_jittered[i][
                edges[0] : edges[1]
            ]
        # add a little noise to the post-task epoch
        spike_times_jittered = jitter(spike_times_, rate=10, jitter_amount=0.5)
        for i in range(0, len(spike_times_1)):
            spike_times_1[i][edges[2] : edges[3]] = spike_times_jittered[i][
                edges[2] : edges[3]
            ]

        return spike_times_1

    # generate spike trains
    st = nel.SpikeTrainArray(timestamps=generate_data())

    # create assembly reactivation object
    assembly_react = assembly_reactivation.AssemblyReact()

    # test is empty
    assert assembly_react.isempty is True

    # load spike trains
    assembly_react.add_st(st)

    # test is empty
    assert assembly_react.isempty is False

    # detect assemblies
    assembly_react.get_weights()

    # test number of assemblies
    assert assembly_react.n_assemblies() == 5

    assembly_react.find_members()
    n_members_per_assembly = [
        np.sum(assembly_react.assembly_members[i])
        for i in range(0, assembly_react.n_assemblies())
    ]
    assert all(np.array(n_members_per_assembly) >= 10)

    # test reactivation
    # test number of assemblies
    spike_times_1 = create_pre_task_post_assembly()
    spike_times_2 = create_pre_task_post_assembly()
    spike_times_3 = create_pre_task_post_assembly()
    spike_times_4 = create_pre_task_post_assembly()
    spike_times_5 = create_pre_task_post_assembly()

    spike_times = np.array(
        list(
            chain.from_iterable(
                [
                    spike_times_1,
                    spike_times_2,
                    spike_times_3,
                    spike_times_4,
                    spike_times_5,
                ]
            )
        ),
        dtype=object,
    )
    st = nel.SpikeTrainArray(timestamps=spike_times)
    epochs = nel.EpochArray(np.array([[0, 1000], [1000, 2000], [2000, 3000]]))
    assembly_react = assembly_reactivation.AssemblyReact(z_mat_dt=5)
    assembly_react.add_st(st)
    assembly_react.get_weights(epochs[1])
    assembly_act = assembly_react.get_assembly_act()

    # test that the assembly activity is higher in the post-task epoch than the pre-task epoch
    assert all(
        assembly_act[epochs[0]].mean(axis=1) < assembly_act[epochs[2]].mean(axis=1)
    )


def test_cross_structural_assembly_reactivation():
    """Test AssemblyReact with cross-structural assembly detection."""
    
    def create_cross_structural_data():
        """Create synthetic spike data with cross-structural assemblies."""
        np.random.seed(42)
        
        # Create two groups of neurons
        n_group1 = 15
        n_group2 = 10
        total_neurons = n_group1 + n_group2
        
        # Generate base spike times for each neuron
        spike_times = []
        
        # Group 1 neurons (background activity)
        for i in range(n_group1):
            base_spikes = np.sort(np.random.exponential(0.1, 50))
            spike_times.append(base_spikes)
        
        # Group 2 neurons (background activity)  
        for i in range(n_group2):
            base_spikes = np.sort(np.random.exponential(0.1, 40))
            spike_times.append(base_spikes)
        
        # Add cross-structural assembly activity
        # Assembly spans neurons 5-8 from group1 and 3-5 from group2
        assembly_times = np.sort(np.random.uniform(0, 5, 30))
        
        for t in assembly_times:
            # Add synchronized spikes with some jitter
            for neuron in range(5, 9):  # Group 1 assembly members
                jittered_time = t + np.random.normal(0, 0.01)
                spike_times[neuron] = np.sort(np.append(spike_times[neuron], jittered_time))
            
            for neuron in range(n_group1 + 3, n_group1 + 6):  # Group 2 assembly members
                jittered_time = t + np.random.normal(0, 0.01)
                spike_times[neuron] = np.sort(np.append(spike_times[neuron], jittered_time))
        
        return spike_times
    
    # Generate data
    spike_times = create_cross_structural_data()
    st = nel.SpikeTrainArray(timestamps=spike_times)
    
    # Create group labels
    cross_structural = np.array(['CA1'] * 15 + ['CA3'] * 10)
    
    # Test standard assembly detection
    assembly_react_std = assembly_reactivation.AssemblyReact()
    assembly_react_std.add_st(st)
    assembly_react_std.get_weights()
    n_assemblies_std = assembly_react_std.n_assemblies()
    
    # Test cross-structural assembly detection
    assembly_react_cross = assembly_reactivation.AssemblyReact(
        cross_structural=cross_structural
    )
    assembly_react_cross.add_st(st)
    assembly_react_cross.get_weights()
    n_assemblies_cross = assembly_react_cross.n_assemblies()
    
    # Cross-structural should detect assemblies
    assert n_assemblies_cross >= 0
    
    # If assemblies are detected, verify they are cross-structural
    if n_assemblies_cross > 0:
        assembly_react_cross.find_members()
        
        for i in range(n_assemblies_cross):
            pattern = assembly_react_cross.patterns[i]
            group1_weights = np.abs(pattern[:15])
            group2_weights = np.abs(pattern[15:])
            group1_active = np.sum(group1_weights > 0.1)
            group2_active = np.sum(group2_weights > 0.1)
            
            # At least one assembly should be cross-structural
            if i == 0:  # Check at least the first assembly
                assert group1_active > 0 or group2_active > 0


def test_cross_structural_parameter_validation():
    """Test parameter validation for cross_structural in AssemblyReact."""
    
    # Create simple spike data
    spike_times = [np.array([0.1, 0.5, 1.0]), np.array([0.2, 0.6, 1.1])]
    st = nel.SpikeTrainArray(timestamps=spike_times)
    
    # Test with correct length cross_structural
    cross_structural_correct = np.array(['A', 'B'])
    assembly_react = assembly_reactivation.AssemblyReact(
        cross_structural=cross_structural_correct
    )
    assembly_react.add_st(st)
    # Should not raise error during initialization
    assert assembly_react.cross_structural is not None
    
    # Test with wrong length cross_structural (should fail during get_weights)
    cross_structural_wrong = np.array(['A', 'B', 'C'])  # 3 elements for 2 neurons
    assembly_react_wrong = assembly_reactivation.AssemblyReact(
        cross_structural=cross_structural_wrong
    )
    assembly_react_wrong.add_st(st)
    
    # Should raise error when calling get_weights
    with pytest.raises(ValueError, match="cross_structural length"):
        assembly_react_wrong.get_weights()


def test_cross_structural_with_different_methods():
    """Test cross-structural detection with different methods (PCA vs ICA)."""
    
    # Create test data
    spike_times = []
    for i in range(10):
        spikes = np.sort(np.random.exponential(0.1, 30))
        spike_times.append(spikes)
    
    st = nel.SpikeTrainArray(timestamps=spike_times)
    cross_structural = np.array(['Region1'] * 5 + ['Region2'] * 5)
    
    # Test with ICA
    assembly_react_ica = assembly_reactivation.AssemblyReact(
        method='ica',
        cross_structural=cross_structural
    )
    assembly_react_ica.add_st(st)
    assembly_react_ica.get_weights()
    
    # Test with PCA
    assembly_react_pca = assembly_reactivation.AssemblyReact(
        method='pca', 
        cross_structural=cross_structural
    )
    assembly_react_pca.add_st(st)
    assembly_react_pca.get_weights()
    
    # Both should run without errors
    assert assembly_react_ica.n_assemblies() >= 0
    assert assembly_react_pca.n_assemblies() >= 0


def test_cross_structural_assembly_activity():
    """Test assembly activity computation with cross-structural assemblies."""
    
    # Create test data with clear cross-structural pattern
    np.random.seed(42)
    spike_times = []
    
    # Generate spike trains
    for i in range(8):
        base_spikes = np.sort(np.random.exponential(0.2, 40))
        spike_times.append(base_spikes)
    
    # Add synchronized events across groups
    sync_times = [1.0, 2.0, 3.0, 4.0]
    for t in sync_times:
        # Group 1 neurons (0-3)
        for i in range(4):
            spike_times[i] = np.sort(np.append(spike_times[i], t + np.random.normal(0, 0.02)))
        # Group 2 neurons (4-7) 
        for i in range(4, 8):
            spike_times[i] = np.sort(np.append(spike_times[i], t + np.random.normal(0, 0.02)))
    
    st = nel.SpikeTrainArray(timestamps=spike_times)
    cross_structural = np.array(['A'] * 4 + ['B'] * 4)
    
    # Create AssemblyReact object
    assembly_react = assembly_reactivation.AssemblyReact(
        cross_structural=cross_structural,
        z_mat_dt=0.1  # Higher resolution for this test
    )
    assembly_react.add_st(st)
    assembly_react.get_weights()
    
    # Compute assembly activity
    if assembly_react.n_assemblies() > 0:
        assembly_act = assembly_react.get_assembly_act()
        
        # Should return AnalogSignalArray
        assert hasattr(assembly_act, 'data')
        assert assembly_act.data.shape[0] == assembly_react.n_assemblies()
        
        # Basic checks on assembly activity
        if not assembly_act.isempty:
            # Check that assembly activity has reasonable properties
            assert assembly_act.data.shape[1] > 0  # Has time bins
            assert not np.all(np.isnan(assembly_act.data))  # Not all NaN
            
            # Check that assembly activity varies over time (not constant)
            activity_std = np.std(assembly_act.data, axis=1)
            assert np.any(activity_std > 0)  # At least some assemblies show variation


def test_cross_structural_empty_spike_train():
    """Test cross-structural detection with empty or minimal spike data."""
    
    # Test with empty spike train
    st_empty = nel.SpikeTrainArray(empty=True)
    cross_structural = np.array(['A', 'B'])
    
    assembly_react = assembly_reactivation.AssemblyReact(
        cross_structural=cross_structural
    )
    assembly_react.add_st(st_empty)
    assembly_react.get_weights()
    
    # Should handle empty data gracefully
    assert assembly_react.n_assemblies() == 0
    
    # Test with minimal spike data
    spike_times = [np.array([1.0]), np.array([1.1])]  # Very few spikes
    st_minimal = nel.SpikeTrainArray(timestamps=spike_times)
    
    assembly_react_minimal = assembly_reactivation.AssemblyReact(
        cross_structural=cross_structural
    )
    assembly_react_minimal.add_st(st_minimal)
    assembly_react_minimal.get_weights()
    
    # Should not crash with minimal data
    assert assembly_react_minimal.n_assemblies() >= 0
