import unittest

import numpy as np

from neuro_py.ensemble.replay import PairwiseBias


class TestPairwiseBiasAnalysis(unittest.TestCase):
    def setUp(self):
        """Set up parameters for tests."""
        self.nneurons = 15
        self.minseqduration = 0.05
        self.maxseqduration = 0.15
        self.duration = 0.5
        np.random.seed(0)  # Set seed for reproducibility

    def simulate_sequential_spikes(
        self,
        nneurons=30,
        minseqduration=0.05,
        maxseqduration=0.15,
        duration=1.0,
        jitter=0.01,
        random=False,
    ):
        spikes = []
        neuron_ids = []
        max_nsequences = np.ceil(duration / minseqduration)
        sequence_durations = np.random.uniform(
            minseqduration, maxseqduration, int(max_nsequences)
        )

        last_sequence = np.where(np.cumsum(sequence_durations) <= duration)[0][-1]
        sequence_durations = sequence_durations[: last_sequence + 1]
        sequence_epochs = np.cumsum(sequence_durations)
        sequence_epochs = np.asarray(
            (np.r_[0, sequence_epochs][:-1], sequence_epochs)
        ).T  # shape (nsequences, 2)

        for seq_start, seq_end in sequence_epochs:
            spike_ts = np.linspace(seq_start, seq_end, nneurons)
            spike_ts += np.random.uniform(-jitter, jitter, nneurons)
            spike_ts = np.sort(spike_ts)
            spike_ts = np.clip(spike_ts, seq_start, seq_end)
            spikes.append(spike_ts)
            neuron_ids.append(np.arange(nneurons))

        spikes = np.concatenate(spikes)
        neuron_ids = np.concatenate(neuron_ids)

        if random:
            neuron_ids = np.random.permutation(neuron_ids)

        return spikes, neuron_ids, sequence_epochs

    def test_simulate_sequential_spikes(self):
        """Test the spike simulation function."""
        task_spikes, task_neurons, task_seq_epochs = self.simulate_sequential_spikes(
            nneurons=self.nneurons,
            minseqduration=self.minseqduration,
            maxseqduration=self.maxseqduration,
            duration=self.duration,
            random=False,
        )

        # Check if the number of spikes is correct
        self.assertEqual(len(task_spikes), self.nneurons * len(task_seq_epochs))

        # Check if neuron IDs are within expected range
        self.assertTrue(np.all(np.isin(task_neurons, np.arange(self.nneurons))))

    def test_pairwise_bias_analysis(self):
        """Test the PairwiseBias analysis."""
        task_spikes, task_neurons, task_seq_epochs = self.simulate_sequential_spikes(
            nneurons=self.nneurons,
            minseqduration=self.minseqduration,
            maxseqduration=self.maxseqduration,
            duration=self.duration,
            random=False,
        )

        post_spikes_sig, post_neurons_sig, post_sig_seq_epochs = (
            self.simulate_sequential_spikes(
                nneurons=self.nneurons,
                minseqduration=self.minseqduration,
                maxseqduration=self.maxseqduration,
                duration=self.duration,
                random=False,
            )
        )

        post_spikes_nonsig, post_neurons_nonsig, post_nonsig_seq_epochs = (
            self.simulate_sequential_spikes(
                nneurons=self.nneurons,
                minseqduration=self.minseqduration,
                maxseqduration=self.maxseqduration,
                duration=self.duration,
                random=True,
            )
        )

        transformer = PairwiseBias()

        # Analyze significant replay
        z_score_sig, p_value_sig, cosine_val_sig = transformer.fit_transform(
            task_spikes,
            task_neurons,
            task_seq_epochs,
            post_spikes_sig,
            post_neurons_sig,
            post_sig_seq_epochs,
        )

        # Analyze non-significant replay
        z_score_nonsig, p_value_nonsig, cosine_val_nonsig = transformer.transform(
            post_spikes_nonsig, post_neurons_nonsig, post_nonsig_seq_epochs
        )

        # Check results are of expected shape and values are valid
        self.assertEqual(len(z_score_sig), len(post_sig_seq_epochs))
        self.assertEqual(len(z_score_nonsig), len(post_nonsig_seq_epochs))

        # Check that significant replay has lower p-values than non-significant replay
        self.assertTrue(np.all(p_value_sig < 0.05))
        self.assertTrue(np.mean(p_value_sig) < np.mean(p_value_nonsig))


if __name__ == "__main__":
    unittest.main()
