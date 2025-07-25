from itertools import chain

import nelpy as nel
import numpy as np
import pytest
from scipy import stats

from neuro_py.ensemble.explained_variance import ExplainedVariance


def test_explained_variance():
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

    def test_init():
        # Test initialization of ExplainedVariance object
        # generate spike trains
        st = nel.SpikeTrainArray(timestamps=generate_data())
        epochs = nel.EpochArray(np.array([[0, 1000], [1000, 2000], [2000, 3000]]))

        ev = ExplainedVariance(
            st=st,
            template=epochs[1],
            matching=epochs[2],
            control=epochs[0],
            bin_size=0.1,
            window=200,
            slideby=100,
        )

        assert isinstance(ev, ExplainedVariance)
        assert isinstance(ev.st, nel.core._eventarray.SpikeTrainArray)
        assert isinstance(ev.template, nel.core._intervalarray.EpochArray)
        assert isinstance(ev.matching, nel.core._intervalarray.EpochArray)
        assert isinstance(ev.control, nel.core._intervalarray.EpochArray)
        assert isinstance(ev.bin_size, (float, int))
        assert isinstance(ev.window, (int, type(None)))
        assert isinstance(ev.slideby, (int, type(None)))

    def test_calculate_statistics():
        # Test calculation of explained variance statistics
        st = nel.SpikeTrainArray(timestamps=generate_data())
        epochs = nel.EpochArray(np.array([[0, 1000], [1000, 2000], [2000, 3000]]))

        ev = ExplainedVariance(
            st=st,
            template=epochs[1],
            matching=epochs[2],
            control=epochs[0],
            bin_size=0.1,
            window=200,
            slideby=100,
        )
        assert (ev.ev) is not None
        assert (ev.rev) is not None
        assert (ev.ev_std) is not None
        assert (ev.rev_std) is not None
        assert (ev.partial_corr) is not None
        assert (ev.rev_partial_corr) is not None
        assert isinstance(ev.n_pairs, int)
        assert (ev.matching_time) is not None
        assert (ev.control_time) is not None

    def test_validate_input():
        # Test validation of input parameters
        with pytest.raises(AssertionError):
            ExplainedVariance(None, None, None, None)

    def test_examples():
        # Load data
        st = nel.SpikeTrainArray(timestamps=generate_data())
        epochs = nel.EpochArray(np.array([[0, 1000], [1000, 2000], [2000, 3000]]))

        # Example 1
        # test single variable outcoume
        ev = ExplainedVariance(
            st=st,
            template=epochs[1],
            matching=epochs[2],
            control=epochs[0],
            window=None,
            slideby=None,
        )

        assert ev.ev > ev.rev
        assert ev.pvalue() < 0.05

        # Example 2
        # test time resolved
        ev = ExplainedVariance(
            st=st,
            template=epochs[1],
            matching=epochs,
            control=epochs[0],
            window=200,
        )
        assert isinstance(ev, ExplainedVariance)
        assert (np.array(epochs.duration) / 200) - 1 == len(ev.ev)
        idx = (ev.matching_time >= epochs[2].start) & (
            ev.matching_time <= epochs[2].stop
        )
        assert ev.ev[idx].mean() > ev.rev[idx].mean()

        # Example 3
        # test time resolved with sliding window
        ev = ExplainedVariance(
            st=st,
            template=epochs[1],
            matching=epochs,
            control=epochs[0],
            window=200,
            slideby=100,
        )
        assert isinstance(ev, ExplainedVariance)
        assert (np.array(epochs.duration) / 200) * 2 - 1 == len(ev.ev)
        idx = (ev.matching_time >= epochs[2].start) & (
            ev.matching_time <= epochs[2].stop
        )
        assert ev.ev[idx].mean() > ev.rev[idx].mean()

    test_init()
    test_calculate_statistics()
    test_validate_input()
    test_examples()
