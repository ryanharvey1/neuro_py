from itertools import chain

import nelpy as nel
import numpy as np
from neuro_py.ensemble import assembly_reactivation
from scipy import stats


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
