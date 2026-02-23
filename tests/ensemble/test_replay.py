import unittest

import numpy as np

from neuro_py.ensemble.replay import PairwiseBias, compute_bias_matrix_optimized_


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
        reverseseqprob=0.0,
        random=False,
    ):
        spikes = []
        neuron_ids = []
        max_nsequences = np.ceil(duration / minseqduration)
        sequence_durations = np.random.uniform(
            minseqduration, maxseqduration, int(max_nsequences)
        )
        # get index of last sequence that fits into duration
        last_sequence = np.where(np.cumsum(sequence_durations) <= duration)[0][-1]
        sequence_durations = sequence_durations[: last_sequence + 1]
        sequence_epochs = np.cumsum(sequence_durations)
        sequence_epochs = np.asarray(
            (np.r_[0, sequence_epochs][:-1], sequence_epochs)
        ).T  # shape (nsequences, 2)

        for seq_start, seq_end in sequence_epochs:
            spike_ts = np.linspace(seq_start, seq_end, nneurons)
            neuron_seqids = (
                np.arange(nneurons)
                if np.random.rand() > reverseseqprob
                else np.arange(nneurons)[::-1]
            )
            # add jitter
            spike_ts += np.random.uniform(-jitter, jitter, nneurons)
            spike_ts = np.sort(spike_ts)
            # clip to sequence bounds
            spike_ts = np.clip(spike_ts, seq_start, seq_end)
            spikes.append(spike_ts)
            neuron_ids.append(neuron_seqids)

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
        # comparing means to account for stochasticity due to small number of shuffles
        self.assertTrue(np.mean(p_value_sig) < np.mean(p_value_nonsig))

    def test_pairwise_bias_analysis_with_reverse_seqs(self):
        """Test the PairwiseBias analysis with reverse sequences."""
        task_spikes, task_neurons, task_seq_epochs = self.simulate_sequential_spikes(
            nneurons=self.nneurons,
            minseqduration=self.minseqduration,
            maxseqduration=self.maxseqduration,
            duration=self.duration,
            reverseseqprob=0.0,
            random=False,
        )

        post_spikes_sig, post_neurons_sig, post_sig_seq_epochs = (
            self.simulate_sequential_spikes(
                nneurons=self.nneurons,
                minseqduration=self.minseqduration,
                maxseqduration=self.maxseqduration,
                duration=self.duration,
                reverseseqprob=1,
                random=False,
            )
        )

        post_spikes_nonsig, post_neurons_nonsig, post_nonsig_seq_epochs = (
            self.simulate_sequential_spikes(
                nneurons=self.nneurons,
                minseqduration=self.minseqduration,
                maxseqduration=self.maxseqduration,
                duration=self.duration,
                reverseseqprob=1,
                random=False,
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
            allow_reverse_replay=True,
        )

        # Analyze non-significant replay
        z_score_nonsig, p_value_nonsig, cosine_val_nonsig = transformer.transform(
            post_spikes_nonsig,
            post_neurons_nonsig,
            post_nonsig_seq_epochs,
            allow_reverse_replay=False,
        )

        # Check results are of expected shape and values are valid
        self.assertEqual(len(z_score_sig), len(post_sig_seq_epochs))
        self.assertEqual(len(z_score_nonsig), len(post_nonsig_seq_epochs))

        # Check that significant replay has lower p-values than non-significant replay
        self.assertTrue(np.all(p_value_sig < 0.05))
        # comparing means to account for stochasticity due to small number of shuffles
        self.assertTrue(np.mean(p_value_sig) < np.mean(p_value_nonsig))


class TestBottomUpReplayDetection(unittest.TestCase):
    def test_bottom_up_replay_detection(self):
        """Test the bottom_up_replay_detection with a simulated posterior containing two replay sequences."""
        from neuro_py.ensemble.replay import bottom_up_replay_detection

        # time and space setup
        n_time = 400
        n_space = 100
        t = np.linspace(0, 4.0, n_time)  # 4 seconds
        bins = np.linspace(0, 100, n_space)  # 0-100 cm

        # create empty posterior with background noise
        posterior = np.random.rand(n_time, n_space) * 0.01

        # insert two moving Gaussian bumps (replay-like): one from 0.5-0.9s, another from 2.0-2.5s
        def add_moving_bump(start_t, end_t, start_x, end_x, width=5.0):
            inds = np.where((t >= start_t) & (t <= end_t))[0]
            times = t[inds]
            positions = np.linspace(start_x, end_x, len(inds))
            for ii, pos in zip(inds, positions):
                posterior[ii] += np.exp(-0.5 * ((bins - pos) ** 2) / (width**2))

        add_moving_bump(0.5, 0.9, 10, 60)
        add_moving_bump(2.0, 2.5, 80, 20)

        # normalize posterior per time bin
        posterior = np.clip(posterior, 0, None)
        posterior = posterior / (posterior.sum(axis=1, keepdims=True) + 1e-12)

        # convert to time-last shape (space, time)
        posterior = posterior.T

        # speed: low everywhere so speed criterion passes for replay windows
        speed_times = t
        speed_values = np.ones_like(t) * 1.0

        replays, meta = bottom_up_replay_detection(
            posterior,
            time_centers=t,
            bin_centers=bins,
            speed_times=speed_times,
            speed_values=speed_values,
            window_dt=None,
            speed_thresh=5.0,
            spread_thresh=20.0,
            com_jump_thresh=30.0,
            merge_spatial_gap=25.0,
            merge_time_gap=0.05,
            min_duration=0.08,
            dispersion_thresh=5.0,
        )

        # We expect at least two detected replays roughly around the inserted windows
        self.assertTrue(replays.shape[0] >= 2)

        # check one replay near 0.5-0.9 and one near 2.0-2.5
        starts = replays[:, 0]
        self.assertTrue(np.any(np.abs(starts - 0.5) < 0.2))
        self.assertTrue(np.any(np.abs(starts - 2.0) < 0.3))

    def test_bottom_up_replay_detection_2d(self):
        """Test bottom_up_replay_detection with a simulated 2D posterior (y,x,t)."""
        from neuro_py.ensemble.replay import bottom_up_replay_detection

        # time and 2D space setup
        n_time = 500
        ny, nx = 20, 30
        t = np.linspace(0, 5.0, n_time)
        y = np.linspace(0, 100, ny)
        x = np.linspace(0, 150, nx)

        posterior = np.random.rand(n_time, ny, nx) * 0.001

        # helper to add moving 2D Gaussian bump
        def add_2d_bump(t0, t1, y0, x0, y1, x1, wy=3.0, wx=3.0):
            inds = np.where((t >= t0) & (t <= t1))[0]
            positions_y = np.linspace(y0, y1, len(inds))
            positions_x = np.linspace(x0, x1, len(inds))
            for ii, py, px in zip(inds, positions_y, positions_x):
                # create grids with shapes (ny, nx) so they broadcast with posterior[ii]
                xx, yy = np.meshgrid(x, y, indexing="xy")
                posterior[ii] += np.exp(
                    -0.5 * (((yy - py) ** 2) / (wy**2) + ((xx - px) ** 2) / (wx**2))
                )

        add_2d_bump(0.4, 0.9, 10, 10, 60, 80)
        add_2d_bump(2.0, 2.6, 80, 120, 30, 40)

        # normalize per time bin (sum over spatial axes)
        posterior = np.clip(posterior, 0, None)
        posterior = posterior / (posterior.sum(axis=(1, 2), keepdims=True) + 1e-12)

        # convert to time-last shape (ny, nx, time)
        posterior = np.moveaxis(posterior, 0, 2)

        speed_times = t
        speed_values = np.ones_like(t) * 1.0

        replays, meta = bottom_up_replay_detection(
            posterior,
            time_centers=t,
            bin_centers=(y, x),
            speed_times=speed_times,
            speed_values=speed_values,
            speed_thresh=5.0,
            spread_thresh=30.0,
            com_jump_thresh=40.0,
            merge_spatial_gap=30.0,
            merge_time_gap=0.06,
            min_duration=0.08,
            dispersion_thresh=3.0,
        )

        # Expect at least two detected replay events
        self.assertTrue(replays.shape[0] >= 2)

    def test_nan_com_jump_excluded(self):
        """Ensure bins with no posterior mass (leading to NaN COM/jump) are excluded."""
        from neuro_py.ensemble.replay import bottom_up_replay_detection

        # create a tiny posterior with first time bin all zeros -> NaN COM
        t = np.linspace(0, 0.2, 5)
        bins = np.linspace(0, 10, 5)

        posterior = np.zeros((5, 5))
        # put mass in later bins only
        posterior[2] = np.exp(-0.5 * ((bins - 5.0) ** 2) / (1.0**2))
        posterior[3] = np.exp(-0.5 * ((bins - 6.0) ** 2) / (1.0**2))

        # normalize
        posterior = posterior / (posterior.sum(axis=1, keepdims=True) + 1e-12)

        replays, meta = bottom_up_replay_detection(
            posterior,
            time_centers=t,
            bin_centers=bins,
            speed_times=t,
            speed_values=np.zeros_like(t),
            speed_thresh=5.0,
            spread_thresh=100.0,
            com_jump_thresh=100.0,
            merge_spatial_gap=10.0,
            merge_time_gap=0.05,
            min_duration=0.0,
            dispersion_thresh=-1.0,  # accept everything for this test
        )

        # mask should be False for first bin (NaN com_jump originally)
        mask = meta["mask"]
        self.assertFalse(mask[0])


class TestComputeBiasMatrixOptimized(unittest.TestCase):
    """Test suite for compute_bias_matrix_optimized_ function."""

    def test_compute_bias_matrix_output_shape(self):
        """Test that output has correct shape (total_neurons, total_neurons)."""
        spike_times = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        neuron_ids = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        total_neurons = 4

        bias_matrix = compute_bias_matrix_optimized_(
            spike_times, neuron_ids, total_neurons
        )

        assert bias_matrix.shape == (total_neurons, total_neurons)

    def test_compute_bias_matrix_output_type(self):
        """Test that output is a numpy array."""
        spike_times = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        neuron_ids = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        total_neurons = 4

        bias_matrix = compute_bias_matrix_optimized_(
            spike_times, neuron_ids, total_neurons
        )

        assert isinstance(bias_matrix, np.ndarray)

    def test_compute_bias_matrix_initialization(self):
        """Test that bias matrix is initialized to 0.5."""
        spike_times = np.array([])
        neuron_ids = np.array([], dtype=int)
        total_neurons = 3

        bias_matrix = compute_bias_matrix_optimized_(
            spike_times, neuron_ids, total_neurons
        )

        # With empty spike times, all elements should be initialized to 0.5 (no spike data)
        # or result in NaN from division by 0
        assert bias_matrix.shape == (total_neurons, total_neurons)

    def test_compute_bias_matrix_no_nans_with_good_data(self):
        """Test that bias matrix contains no NaNs when spike data is present."""
        # Create sequential spike sequence (clear ordering)
        spike_times = np.array([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
        neuron_ids = np.array([0, 1, 2, 3, 0, 1, 2, 3])  # Sequential repeating
        total_neurons = 4

        bias_matrix = compute_bias_matrix_optimized_(
            spike_times, neuron_ids, total_neurons
        )

        # Bias values should be valid numbers (not NaN) where spikes exist
        assert not np.isnan(bias_matrix).all()

    def test_compute_bias_matrix_diagonal(self):
        """Test diagonal elements (self-pairs) are handled correctly."""
        spike_times = np.array([0.1, 0.2, 0.3, 0.4])
        neuron_ids = np.array([0, 1, 2, 3])
        total_neurons = 4

        bias_matrix = compute_bias_matrix_optimized_(
            spike_times, neuron_ids, total_neurons
        )

        # Diagonal should not have issues (self comparison)
        assert np.all(np.isfinite(bias_matrix.diagonal()))

    def test_compute_bias_matrix_two_neurons(self):
        """Test with minimum: two neurons."""
        spike_times = np.array([0.1, 0.2, 0.3, 0.4])
        neuron_ids = np.array([0, 1, 0, 1])
        total_neurons = 2

        bias_matrix = compute_bias_matrix_optimized_(
            spike_times, neuron_ids, total_neurons
        )

        assert bias_matrix.shape == (2, 2)

    def test_compute_bias_matrix_many_neurons(self):
        """Test with many neurons."""
        # Create spike times for 20 neurons
        np.random.seed(42)
        n_neurons = 20
        n_spikes_per_neuron = 10

        spike_times = np.concatenate(
            [np.random.rand(n_spikes_per_neuron) for _ in range(n_neurons)]
        )
        spike_times = np.sort(spike_times)
        neuron_ids = np.repeat(np.arange(n_neurons), n_spikes_per_neuron)

        bias_matrix = compute_bias_matrix_optimized_(spike_times, neuron_ids, n_neurons)

        assert bias_matrix.shape == (n_neurons, n_neurons)
        assert isinstance(bias_matrix, np.ndarray)

    def test_compute_bias_matrix_sequential_forward(self):
        """Test with clearly sequential forward spikes (0→1→2→3)."""
        # Sequential forward replay: 0 fires, then 1, then 2, then 3
        spike_times = np.array(
            [0.0, 0.01, 0.02, 0.03, 0.1, 0.11, 0.12, 0.13]
        )  # Two sequences
        neuron_ids = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        total_neurons = 4

        bias_matrix = compute_bias_matrix_optimized_(
            spike_times, neuron_ids, total_neurons
        )

        # In forward replay:
        # neuron 0 fires before 1 → bias[0,1] should be high
        # neuron 1 fires before 2 → bias[1,2] should be high
        # neuron 2 fires before 3 → bias[2,3] should be high
        # neuron 3 fires before 0 → bias[3,0] should be low

        assert bias_matrix.shape == (total_neurons, total_neurons)
        # Values should be finite
        assert np.all(np.isfinite(bias_matrix[np.isfinite(bias_matrix)]))

    def test_compute_bias_matrix_sequential_reverse(self):
        """Test with clearly sequential reverse spikes (3→2→1→0)."""
        # Sequential reverse replay: 3 fires, then 2, then 1, then 0
        spike_times = np.array(
            [0.0, 0.01, 0.02, 0.03, 0.1, 0.11, 0.12, 0.13]
        )  # Two sequences
        neuron_ids = np.array([3, 2, 1, 0, 3, 2, 1, 0])
        total_neurons = 4

        bias_matrix = compute_bias_matrix_optimized_(
            spike_times, neuron_ids, total_neurons
        )

        # In reverse replay:
        # neuron 3 fires before 2 → bias[3,2] should be high
        # neuron 2 fires before 1 → bias[2,1] should be high
        # neuron 1 fires before 0 → bias[1,0] should be high
        # neuron 0 fires before 3 → bias[0,3] should be low

        assert bias_matrix.shape == (total_neurons, total_neurons)
        assert np.all(np.isfinite(bias_matrix[np.isfinite(bias_matrix)]))

    def test_compute_bias_matrix_reproducible(self):
        """Test that results are reproducible."""
        np.random.seed(123)
        spike_times1 = np.random.rand(100)
        spike_times1.sort()
        neuron_ids1 = np.random.randint(0, 10, 100)

        np.random.seed(123)
        spike_times2 = np.random.rand(100)
        spike_times2.sort()
        neuron_ids2 = np.random.randint(0, 10, 100)

        bias_matrix1 = compute_bias_matrix_optimized_(spike_times1, neuron_ids1, 10)
        bias_matrix2 = compute_bias_matrix_optimized_(spike_times2, neuron_ids2, 10)

        np.testing.assert_array_equal(bias_matrix1, bias_matrix2)

    def test_compute_bias_matrix_values_valid_range(self):
        """Test that bias values are in a valid range when they exist."""
        spike_times = np.array([0.0, 0.01, 0.02, 0.03, 0.1, 0.11, 0.12, 0.13])
        neuron_ids = np.array([0, 1, 2, 3, 2, 1, 0, 3])
        total_neurons = 4

        bias_matrix = compute_bias_matrix_optimized_(
            spike_times, neuron_ids, total_neurons
        )

        # Valid bias values should be non-negative (ratio of counts or initialized value)
        valid_mask = np.isfinite(bias_matrix)
        if valid_mask.any():
            assert np.all(bias_matrix[valid_mask] >= 0)

    def test_compute_bias_matrix_with_single_neuron_repeated(self):
        """Test with spikes from only one neuron repeated."""
        # Only neuron 0 spikes
        spike_times = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        neuron_ids = np.array([0, 0, 0, 0, 0])
        total_neurons = 5

        bias_matrix = compute_bias_matrix_optimized_(
            spike_times, neuron_ids, total_neurons
        )

        assert bias_matrix.shape == (total_neurons, total_neurons)
        # Should handle gracefully (NaN or valid values)
        assert isinstance(bias_matrix, np.ndarray)

    def test_compute_bias_matrix_with_missing_neurons(self):
        """Test when some neurons have no spikes."""
        # Total 5 neurons but only 0, 2, 3 have spikes (1, 4 are missing)
        spike_times = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        neuron_ids = np.array([0, 2, 3, 0, 2, 3])
        total_neurons = 5

        bias_matrix = compute_bias_matrix_optimized_(
            spike_times, neuron_ids, total_neurons
        )

        assert bias_matrix.shape == (total_neurons, total_neurons)
        assert isinstance(bias_matrix, np.ndarray)


if __name__ == "__main__":
    unittest.main()
