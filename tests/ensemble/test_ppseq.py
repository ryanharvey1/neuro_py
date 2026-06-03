import numpy as np
import pytest
from scipy.stats import spearmanr

from neuro_py.ensemble.ppseq import (
    PPSeq,
    PPSeqResult,
    bin_spikes,
    ppseq_seed_stability,
    select_ppseq_template_count,
    template_set_similarity,
)


def make_sequence_counts(
    n_neurons: int = 8,
    n_time: int = 180,
    template_duration: int = 16,
    starts: list[int] | None = None,
    neuron_order: np.ndarray | None = None,
    background: float = 0.01,
) -> np.ndarray:
    """Create compact synthetic sequence count data."""
    rng = np.random.default_rng(0)
    if starts is None:
        starts = [10, 45, 80, 115, 145]
    if neuron_order is None:
        neuron_order = np.arange(n_neurons)

    data = rng.poisson(background, size=(n_neurons, n_time)).astype(float)
    offsets = np.linspace(1, template_duration - 2, n_neurons).round().astype(int)
    for start in starts:
        for neuron_rank, neuron in enumerate(neuron_order):
            t = start + offsets[neuron_rank]
            if t < n_time:
                data[neuron, t] += 2.0
    return data


def test_ppseq_recovers_synthetic_sequence_order():
    n_neurons = 8
    model = PPSeq(
        n_templates=1,
        template_duration=16,
        n_neurons=n_neurons,
        random_state=1,
        sequence_frac=0.8,
    )
    data = make_sequence_counts(n_neurons=n_neurons)

    result = model.fit(data, num_iter=35)

    corr, _ = spearmanr(np.arange(n_neurons), model.template_offsets_[0])
    assert corr > 0.7
    assert result.amplitudes.shape == (1, data.shape[1])
    assert result.rates.shape == data.shape
    assert result.log_likelihood.shape == (35,)
    assert model.templates.shape == (1, n_neurons, 16)
    assert np.all(np.isfinite(result.log_likelihood))


def test_ppseq_scores_matching_ripple_events_above_shuffled_controls():
    n_neurons = 8
    task = make_sequence_counts(n_neurons=n_neurons)
    post = make_sequence_counts(
        n_neurons=n_neurons,
        n_time=90,
        starts=[10, 50],
        background=0.005,
    )
    shuffled = make_sequence_counts(
        n_neurons=n_neurons,
        n_time=90,
        starts=[10, 50],
        neuron_order=np.array([0, 2, 4, 6, 1, 3, 5, 7]),
        background=0.005,
    )
    time_centers = np.arange(post.shape[1]) * 0.02 + 0.01
    intervals = np.array([[0.16, 0.56], [0.96, 1.36]])

    model = PPSeq(
        n_templates=1,
        template_duration=16,
        n_neurons=n_neurons,
        bin_size=0.02,
        random_state=1,
        sequence_frac=0.8,
    )
    model.fit(task, num_iter=35)

    matching_scores = model.score_events(post, intervals, time_centers, num_iter=20)
    shuffled_scores = model.score_events(shuffled, intervals, time_centers, num_iter=20)

    expected_columns = {
        "start",
        "stop",
        "duration",
        "best_template",
        "likelihood_gain",
        "template_0_sum",
        "template_0_mean",
        "template_0_max",
    }
    assert expected_columns.issubset(matching_scores.columns)
    assert len(matching_scores) == 2
    assert np.all(matching_scores["best_template"] == 0)
    assert matching_scores["template_0_max"].mean() > shuffled_scores[
        "template_0_max"
    ].mean()
    assert matching_scores["likelihood_gain"].mean() > shuffled_scores[
        "likelihood_gain"
    ].mean()


def test_ppseq_score_events_supports_per_event_mode():
    n_neurons = 8
    task = make_sequence_counts(n_neurons=n_neurons)
    post = make_sequence_counts(
        n_neurons=n_neurons,
        n_time=50,
        starts=[10],
        background=0.005,
    )
    time_centers = np.arange(post.shape[1]) * 0.02 + 0.01
    intervals = np.array([[0.16, 0.56]])

    model = PPSeq(
        n_templates=1,
        template_duration=16,
        n_neurons=n_neurons,
        bin_size=0.02,
        random_state=1,
        sequence_frac=0.8,
    )
    model.fit(task, num_iter=15)

    scores = model.score_events(
        post, intervals, time_centers, num_iter=5, method="per_event"
    )

    assert len(scores) == 1
    assert np.isfinite(scores["likelihood_gain"].iloc[0])

    with pytest.raises(ValueError, match="method"):
        model.score_events(post, intervals, time_centers, method="unknown")


def test_ppseq_score_events_reuses_existing_transform_result(monkeypatch):
    n_neurons = 8
    task = make_sequence_counts(n_neurons=n_neurons)
    post = make_sequence_counts(
        n_neurons=n_neurons,
        n_time=50,
        starts=[10],
        background=0.005,
    )
    time_centers = np.arange(post.shape[1]) * 0.02 + 0.01
    intervals = np.array([[0.16, 0.56]])

    model = PPSeq(
        n_templates=1,
        template_duration=16,
        n_neurons=n_neurons,
        bin_size=0.02,
        random_state=1,
        sequence_frac=0.8,
    )
    model.fit(task, num_iter=15)
    result = model.transform(post, num_iter=5)

    def fail_transform(*args, **kwargs):
        raise AssertionError("score_events should reuse the supplied result")

    monkeypatch.setattr(model, "transform", fail_transform)
    scores = model.score_events(
        post, intervals, time_centers, num_iter=5, result=result
    )

    assert len(scores) == 1
    assert np.isfinite(scores["likelihood_gain"].iloc[0])

    with pytest.raises(ValueError, match="result cannot be used"):
        model.score_events(
            post,
            intervals,
            time_centers,
            method="per_event",
            result=result,
        )

    bad_result = PPSeqResult(
        log_likelihood=result.log_likelihood,
        amplitudes=result.amplitudes[:, :-1],
        rates=result.rates,
    )
    with pytest.raises(ValueError, match="result.amplitudes"):
        model.score_events(post, intervals, time_centers, result=bad_result)


def test_ppseq_score_replay_events_detects_forward_and_reverse_replay():
    n_neurons = 8
    task = make_sequence_counts(n_neurons=n_neurons)
    forward = make_sequence_counts(
        n_neurons=n_neurons,
        n_time=60,
        starts=[10],
        background=0.005,
    )
    reverse = make_sequence_counts(
        n_neurons=n_neurons,
        n_time=60,
        starts=[10],
        neuron_order=np.arange(n_neurons)[::-1],
        background=0.005,
    )
    shuffled = make_sequence_counts(
        n_neurons=n_neurons,
        n_time=60,
        starts=[10],
        neuron_order=np.array([0, 2, 4, 6, 1, 3, 5, 7]),
        background=0.005,
    )
    time_centers = np.arange(forward.shape[1]) * 0.02 + 0.01
    intervals = np.array([[0.16, 0.56]])

    model = PPSeq(
        n_templates=1,
        template_duration=16,
        n_neurons=n_neurons,
        bin_size=0.02,
        random_state=1,
        sequence_frac=0.8,
    )
    model.fit(task, num_iter=35)

    forward_scores, _ = model.score_replay_events(
        forward, intervals, time_centers, num_iter=20
    )
    reverse_scores, _ = model.score_replay_events(
        reverse, intervals, time_centers, num_iter=20
    )
    shuffled_scores, _ = model.score_replay_events(
        shuffled, intervals, time_centers, num_iter=20
    )

    assert forward_scores["template_0_forward_sum"].iloc[0] > forward_scores[
        "template_0_reverse_sum"
    ].iloc[0]
    assert reverse_scores["template_0_reverse_sum"].iloc[0] > reverse_scores[
        "template_0_forward_sum"
    ].iloc[0]
    assert forward_scores["best_direction"].iloc[0] == "forward"
    assert reverse_scores["best_direction"].iloc[0] == "reverse"
    assert reverse_scores["likelihood_gain"].iloc[0] > shuffled_scores[
        "likelihood_gain"
    ].iloc[0]


def test_ppseq_score_replay_events_reuses_result_and_preserves_templates(monkeypatch):
    n_neurons = 8
    task = make_sequence_counts(n_neurons=n_neurons)
    post = make_sequence_counts(
        n_neurons=n_neurons,
        n_time=50,
        starts=[10],
        background=0.005,
    )
    time_centers = np.arange(post.shape[1]) * 0.02 + 0.01
    intervals = np.array([[0.16, 0.56]])

    model = PPSeq(
        n_templates=1,
        template_duration=16,
        n_neurons=n_neurons,
        bin_size=0.02,
        random_state=1,
        sequence_frac=0.8,
    )
    model.fit(task, num_iter=15)
    scales = model.template_scales_.copy()
    offsets = model.template_offsets_.copy()
    widths = model.template_widths_.copy()
    scores, result = model.score_replay_events(post, intervals, time_centers, num_iter=5)

    def fail_transform(*args, **kwargs):
        raise AssertionError("score_replay_events should reuse the supplied result")

    monkeypatch.setattr(model, "_run_fixed_template_transform", fail_transform)
    reused_scores, reused_result = model.score_replay_events(
        post, intervals, time_centers, result=result
    )

    assert reused_result is result
    np.testing.assert_allclose(
        reused_scores["likelihood_gain"], scores["likelihood_gain"]
    )
    np.testing.assert_allclose(model.template_scales_, scales)
    np.testing.assert_allclose(model.template_offsets_, offsets)
    np.testing.assert_allclose(model.template_widths_, widths)

    bad_result = PPSeqResult(
        log_likelihood=result.log_likelihood,
        amplitudes=result.amplitudes[:-1],
        rates=result.rates,
    )
    with pytest.raises(ValueError, match="result.amplitudes"):
        model.score_replay_events(post, intervals, time_centers, result=bad_result)
    with pytest.raises(ValueError, match="include_forward"):
        model.score_replay_events(
            post,
            intervals,
            time_centers,
            include_forward=False,
            include_reverse=False,
        )


def test_ppseq_vectorized_core_matches_slow_reference():
    data = make_sequence_counts(n_neurons=5, n_time=24, template_duration=6)
    model = PPSeq(
        n_templates=2,
        template_duration=6,
        n_neurons=5,
        random_state=3,
        sequence_frac=0.7,
    )
    model.fit(data, num_iter=0)

    rng = np.random.default_rng(4)
    amplitudes = rng.gamma(shape=1.2, scale=0.5, size=(2, data.shape[1]))
    ratio = rng.uniform(0.1, 2.0, size=data.shape)
    templates = model.templates

    expected_rates = _slow_reconstruct(model, amplitudes, templates)
    expected_amplitudes = _slow_update_amplitudes(
        model, amplitudes, ratio, templates
    )
    expected_scales, expected_offsets, expected_widths = _slow_update_templates(
        model, amplitudes, ratio
    )

    np.testing.assert_allclose(model._reconstruct(amplitudes, templates), expected_rates)
    np.testing.assert_allclose(
        model._update_amplitudes(amplitudes, ratio, templates),
        expected_amplitudes,
    )

    model._update_templates(amplitudes, ratio)
    np.testing.assert_allclose(model.template_scales_, expected_scales)
    np.testing.assert_allclose(model.template_offsets_, expected_offsets)
    np.testing.assert_allclose(model.template_widths_, expected_widths)


def test_ppseq_random_state_is_deterministic():
    data = make_sequence_counts()

    model_a = PPSeq(n_templates=1, template_duration=16, random_state=4)
    model_b = PPSeq(n_templates=1, template_duration=16, random_state=4)

    result_a = model_a.fit(data, num_iter=10)
    result_b = model_b.fit(data, num_iter=10)

    np.testing.assert_allclose(model_a.template_offsets_, model_b.template_offsets_)
    np.testing.assert_allclose(result_a.amplitudes, result_b.amplitudes)


def test_ppseq_empty_fit_raises_clear_error():
    model = PPSeq(n_templates=1, template_duration=4, n_neurons=3)

    with pytest.raises(ValueError, match="at least one spike"):
        model.fit(np.zeros((3, 10)), num_iter=1)


def test_bin_spikes_helper_uses_seconds_and_neuron_ids():
    spike_times = np.array([0.01, 0.03, 0.041, 0.07])
    neuron_ids = np.array([0, 1, 1, 0])

    counts, centers = bin_spikes(
        spike_times,
        neuron_ids,
        bin_size=0.02,
        start=0.0,
        stop=0.08,
        n_neurons=2,
    )

    assert counts.shape == (2, 4)
    np.testing.assert_allclose(centers, [0.01, 0.03, 0.05, 0.07])
    np.testing.assert_array_equal(counts[0], [1, 0, 0, 1])
    np.testing.assert_array_equal(counts[1], [0, 1, 1, 0])


def test_select_ppseq_template_count_returns_heldout_metrics():
    data = make_sequence_counts(n_time=120)
    train = data[:, :80]
    test = data[:, 80:]

    scores = select_ppseq_template_count(
        train,
        test,
        [1, 2],
        template_duration=16,
        fit_num_iter=5,
        transform_num_iter=3,
        random_state=10,
        sequence_frac=0.8,
    )

    assert list(scores["n_templates"]) == [1, 2]
    assert {
        "train_log_likelihood",
        "heldout_log_likelihood",
        "heldout_log_likelihood_per_bin",
        "heldout_log_likelihood_per_spike",
    }.issubset(scores.columns)
    assert np.all(np.isfinite(scores["heldout_log_likelihood"]))


def test_ppseq_seed_stability_returns_pairwise_template_similarity():
    data = make_sequence_counts(n_time=120)

    summary, pairwise = ppseq_seed_stability(
        data,
        n_templates=1,
        seeds=[1, 2, 3],
        template_duration=16,
        fit_num_iter=5,
        sequence_frac=0.8,
    )

    assert len(summary) == 1
    assert len(pairwise) == 3
    assert np.all(pairwise["mean_matched_template_similarity"].between(-1, 1))
    assert np.isfinite(summary["mean_stability"].iloc[0])


def test_template_set_similarity_is_permutation_invariant():
    templates = np.zeros((2, 3, 4))
    templates[0, 0, 1] = 1
    templates[1, 2, 3] = 1

    similarity = template_set_similarity(templates, templates[::-1])

    assert similarity == pytest.approx(1.0)


def _slow_reconstruct(
    model: PPSeq, amplitudes: np.ndarray, templates: np.ndarray
) -> np.ndarray:
    n_time = amplitudes.shape[1]
    rates = np.repeat(model.base_rates_[:, np.newaxis], n_time, axis=1)
    for k in range(model.n_templates):
        for n in range(model.n_neurons):
            rates[n] += np.convolve(amplitudes[k], templates[k, n], mode="full")[
                :n_time
            ]
    return np.clip(rates, model.min_rate, None)


def _slow_update_amplitudes(
    model: PPSeq,
    amplitudes: np.ndarray,
    ratio: np.ndarray,
    templates: np.ndarray,
) -> np.ndarray:
    n_time = amplitudes.shape[1]
    updated = np.zeros_like(amplitudes)
    template_mass = np.clip(templates.sum(axis=(1, 2)), model.min_rate, None)
    for k in range(model.n_templates):
        drive = np.zeros(n_time, dtype=float)
        for n in range(model.n_neurons):
            for delay in range(model.template_duration):
                if delay < n_time:
                    drive[: n_time - delay] += (
                        templates[k, n, delay] * ratio[n, delay:]
                    )
        updated[k] = (
            amplitudes[k] * drive + model.alpha_amplitude
        ) / (template_mass[k] + model.beta_amplitude)
    return np.clip(updated, model.min_amplitude, None)


def _slow_update_templates(
    model: PPSeq, amplitudes: np.ndarray, ratio: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_time = amplitudes.shape[1]
    templates = model.templates
    targets = np.full_like(templates, model.alpha_template)
    for k in range(model.n_templates):
        amp_sum = max(float(np.sum(amplitudes[k])), model.min_rate)
        for n in range(model.n_neurons):
            for delay in range(model.template_duration):
                if delay < n_time:
                    expected = float(
                        np.dot(amplitudes[k, : n_time - delay], ratio[n, delay:])
                    )
                else:
                    expected = 0.0
                targets[k, n, delay] += templates[k, n, delay] * expected / amp_sum

    bins = np.arange(model.template_duration, dtype=float)
    scales = targets.sum(axis=2)
    scale_sums = np.clip(scales.sum(axis=1, keepdims=True), model.min_rate, None)
    template_scales = np.clip(scales / scale_sums, model.min_rate, None)
    template_scales /= np.clip(
        template_scales.sum(axis=1, keepdims=True), model.min_rate, None
    )
    target_sums = np.clip(targets.sum(axis=2), model.min_rate, None)
    offsets = np.sum(targets * bins[np.newaxis, np.newaxis, :], axis=2) / target_sums
    variances = (
        np.sum(
            targets * (bins[np.newaxis, np.newaxis, :] - offsets[:, :, np.newaxis]) ** 2,
            axis=2,
        )
        / target_sums
    )
    template_offsets = np.clip(offsets, 0, model.template_duration - 1)
    template_widths = np.clip(np.sqrt(variances), model.min_width, None)
    return template_scales, template_offsets, template_widths
