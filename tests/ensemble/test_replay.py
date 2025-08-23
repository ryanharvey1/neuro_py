import unittest

import numpy as np

from neuro_py.ensemble.replay import PairwiseBias, find_replay_score


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


class TestFindReplayScore(unittest.TestCase):
    def test_find_replay_score_1d_simple(self):
        # simple 1D case: nSpace=5, nTime=4; create a clear diagonal trajectory
        mat = np.zeros((5, 4), dtype=float)
        # set a line from space 0 -> 4 across time
        mat[0, 0] = 1.0
        mat[1, 1] = 1.0
        mat[2, 2] = 1.0
        mat[3, 3] = 1.0

        r, st, sp = find_replay_score(mat, threshold=0, circular=False)
        # best score should be 1.0 (all mass on the trajectory)
        self.assertAlmostEqual(r, 1.0)
        # start should be index 0 and end index 3 or 4 depending on nTime; expect 0->3 here
        self.assertEqual(st, 0)
        # end should be within spatial range
        self.assertIn(sp, range(5))

    def test_find_replay_score_2d_simple(self):
        # simple 2D case: nX=3, nY=3, nTime=3; create a straight line across grid
        mat = np.zeros((3, 3, 3), dtype=float)
        # positions: (0,0)->(1,1)->(2,2)
        mat[0, 0, 0] = 1.0
        mat[1, 1, 1] = 1.0
        mat[2, 2, 2] = 1.0

        r, st, sp = find_replay_score(mat, threshold=0, circular=False)
        self.assertAlmostEqual(r, 1.0)
        # start should be (0,0) and end should be (2,2)
        self.assertEqual(st, (0, 0))
        self.assertEqual(sp, (2, 2))

    def test_find_replay_score_2d_normal_size(self):
        # normal 2D case: nX=5, nY=5, nTime=5; create a diagonal-like trajectory
        # construct a trajectory that spans the full spatial grid from
        # (0,0) to (max_x, max_y) across time so the end point should be
        # (max_x, max_y) == (nX-1, nY-1)
        nX, nY, nTime = 80, 90, 15
        mat = np.zeros((nX, nY, nTime), dtype=float)
        for t in range(nTime):
            x = int(round(t * (nX - 1) / (nTime - 1)))
            y = int(round(t * (nY - 1) / (nTime - 1)))
            mat[x, y, t] = 1.0

        r, st, sp = find_replay_score(mat, threshold=0, circular=False)
        self.assertAlmostEqual(r, 1.0)
        self.assertEqual(st, (0, 0))
        # function returns (x, y) ordering; final bin is (nX-1, nY-1)
        self.assertEqual(sp, (nX - 1, nY - 1))

    def test_find_replay_score_2d_with_noise(self):
        # 2D case with noise: nX=50, nY=95, nTime=20; add noise to the trajectory
        nX, nY, nTime = 50, 95, 20
        mat = np.zeros((nX, nY, nTime), dtype=float)
        for t in range(nTime):
            x = int(round(t * (nX - 1) / (nTime - 1)))
            y = int(round(t * (nY - 1) / (nTime - 1)))
            mat[x, y, t] = 1.0
        # add noise
        mat += np.random.normal(0, 0.1, mat.shape)

        r, st, sp = find_replay_score(mat, threshold=0, circular=False)
        self.assertAlmostEqual(r, 1.0, delta=0.1)
        self.assertEqual(st, (0, 0))
        self.assertEqual(sp, (nX - 1, nY - 1))

    def test_find_replay_score_pruned_vs_full_small(self):
        # small 2D test where we can brute-force and compare to the function
        nX, nY, nTime = 4, 4, 5
        mat = np.zeros((nX, nY, nTime), dtype=float)
        # place a clear diagonal-like trajectory across time
        for t in range(nTime):
            x = t % nX
            y = t % nY
            mat[x, y, t] = 1.0

        M = nX * nY

        # full search via function (force full by setting candidate_k=M)
        r_full, st_full, sp_full = find_replay_score(mat, threshold=0, circular=False, candidate_k=M)

        # brute-force reference (pure Python)
        def brute_best(mat):
            nX, nY, nTime = mat.shape
            best = -np.inf
            best_trip = (None, None, None)
            for sx in range(nX):
                for sy in range(nY):
                    for ex in range(nX):
                        for ey in range(nY):
                            ssum = 0.0
                            for ti in range(nTime):
                                if nTime == 1:
                                    tfrac = 0.0
                                else:
                                    tfrac = ti / (nTime - 1)
                                xt = sx + (ex - sx) * tfrac
                                yt = sy + (ey - sy) * tfrac
                                xi = int(round(xt))
                                yi = int(round(yt))
                                xi = max(0, min(nX - 1, xi))
                                yi = max(0, min(nY - 1, yi))
                                ssum += mat[xi, yi, ti]
                            score = ssum / nTime
                            if score > best:
                                best = score
                                best_trip = (score, (sx, sy), (ex, ey))
            return best_trip

        brute_score, brute_st, brute_sp = brute_best(mat)

        # compare full function result with brute reference
        self.assertAlmostEqual(r_full, brute_score)
        self.assertEqual(st_full, brute_st)
        self.assertEqual(sp_full, brute_sp)

        # pruned search with small K should not exceed full-search score
        r_pruned, st_p, sp_p = find_replay_score(mat, threshold=0, circular=False, candidate_k=5)
        self.assertLessEqual(r_pruned, r_full + 1e-12)


if __name__ == "__main__":
    unittest.main()
