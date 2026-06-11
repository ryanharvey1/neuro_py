import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from neuro_py.ensemble.ppseq import (
    EventSummaryInfo,
    NormalInvChisq,
    PPSeq,
    PPSeqResult,
    RateGamma,
    ScaledInvChisq,
    SeqEvent,
    SeqEventList,
    SeqGlobals,
    SeqModel,
    SeqPriors,
    Spike,
    SymmetricDirichlet,
    bin_spikes,
    fit_ppseq_candidates,
    gauss_info_logZ,
    log_joint,
    log_like,
    log_p_latents,
    log_posterior_predictive,
    log_prior,
    posterior_normal_inv_chisq,
    posterior_rate_gamma,
    posterior_symmetric_dirichlet,
    load_ppseq_songbird_spikes,
    plot_ppseq_diagnostics,
    plot_ppseq_raster,
    ppseq_assignment_raster,
    ppseq_behavior_diagnostics,
    ppseq_seed_stability,
    ppseq_spike_responsibilities,
    score_ppseq_replay_nulls,
    summarize_ppseq_replay_scores,
    select_ppseq_template_count,
    specify_gamma,
    template_set_similarity,
)

matplotlib.use("Agg")


def make_sequence_counts(
    n_neurons: int = 10,
    n_time: int = 220,
    template_duration: int = 20,
    starts: list[int] | None = None,
    neuron_order: np.ndarray | None = None,
    background: float = 0.0,
) -> np.ndarray:
    """Create repeated continuous-time-like sequential count data."""

    rng = np.random.default_rng(0)
    if starts is None:
        starts = [12, 52, 92, 132, 172]
    if neuron_order is None:
        neuron_order = np.arange(n_neurons)
    data = rng.poisson(background, size=(n_neurons, n_time)).astype(float)
    offsets = np.linspace(1, template_duration - 2, n_neurons).round().astype(int)
    for start in starts:
        for rank, neuron in enumerate(neuron_order):
            t = start + offsets[rank]
            if 0 <= t < n_time:
                data[neuron, t] += 2.0
    return data


def make_bidirectional_counts(
    n_neurons: int = 12,
    n_laps: int = 8,
    lap_bins: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create outbound/inbound linear-track sequence counts."""

    n_time = n_laps * lap_bins
    data = np.zeros((n_neurons, n_time), dtype=float)
    position = np.zeros(n_time, dtype=float)
    offsets = np.linspace(3, lap_bins - 4, n_neurons).round().astype(int)
    for lap in range(n_laps):
        start = lap * lap_bins
        stop = start + lap_bins
        if lap % 2 == 0:
            position[start:stop] = np.linspace(0, 1, lap_bins)
            order = np.arange(n_neurons)
        else:
            position[start:stop] = np.linspace(1, 0, lap_bins)
            order = np.arange(n_neurons)[::-1]
        for rank, neuron in enumerate(order):
            data[neuron, start + offsets[rank]] += 2.0
    time_centers = np.arange(n_time) * 0.02 + 0.01
    return data, time_centers, position


def test_julia_style_distribution_helpers_and_log_probabilities():
    gamma = specify_gamma(4.0, 2.0)
    assert gamma.alpha == pytest.approx(8.0)
    assert gamma.beta == pytest.approx(2.0)
    assert posterior_rate_gamma(3, gamma, exposure=2).alpha == pytest.approx(11.0)
    assert posterior_rate_gamma(3, gamma, exposure=2).beta == pytest.approx(4.0)

    dirichlet = SymmetricDirichlet(0.5, 3)
    posterior = posterior_symmetric_dirichlet(np.array([1, 2, 3]), dirichlet)
    np.testing.assert_allclose(posterior, [1.5, 2.5, 3.5])
    assert np.isfinite(dirichlet.logpdf(np.array([0.2, 0.3, 0.5])))

    prior = NormalInvChisq(1.0, 0.0, 2.0, 1.0)
    nic = posterior_normal_inv_chisq(2, 1.0, 1.0, prior)
    assert nic.k == pytest.approx(3.0)
    assert nic.nu == pytest.approx(4.0)
    assert np.isfinite(ScaledInvChisq(2.0, 1.0).logpdf(1.5))
    assert gauss_info_logZ(0.5, 2.0) == pytest.approx(
        0.5 * (np.log(2 * np.pi) - np.log(2.0) + 0.25 / 2.0)
    )

    priors = SeqPriors(
        seq_event_rate=1.0,
        seq_type_proportions=SymmetricDirichlet(1.0, 1),
        seq_event_amplitude=RateGamma(2.0, 1.0),
        neuron_response_proportions=SymmetricDirichlet(1.0, 1),
        neuron_response_profile=NormalInvChisq(1.0, 0.0, 2.0, 1.0),
        bkgd_amplitude=RateGamma(2.0, 1.0),
        bkgd_proportions=SymmetricDirichlet(1.0, 1),
        warp_values=np.ones(1),
        warp_log_proportions=np.zeros(1),
    )
    globals_ = SeqGlobals(
        seq_type_log_proportions=np.zeros(1),
        neuron_response_log_proportions=np.zeros((1, 1)),
        neuron_response_offsets=np.zeros((1, 1)),
        neuron_response_widths=np.ones((1, 1)) * 0.01,
        bkgd_amplitude=0.5,
        bkgd_log_proportions=np.zeros(1),
    )
    event = SeqEvent(
        spike_count=1,
        summed_potentials=np.zeros((1, 1)),
        summed_precisions=np.ones((1, 1)),
        summed_logZ=np.zeros((1, 1)),
        seq_type_posterior=np.zeros((1, 1)),
        sampled_type=0,
        sampled_warp=0,
        sampled_timestamp=0.05,
        sampled_amplitude=1.0,
    )
    model = SeqModel(
        max_time=1.0,
        max_sequence_length=0.2,
        priors=priors,
        globals=globals_,
        sequence_events=SeqEventList([event], [0]),
        new_cluster_log_prob=0.0,
        bkgd_log_prob=0.0,
    )
    spike = Spike(0, 0.05)
    assert isinstance(EventSummaryInfo(0, 0.05, 0, 1.0, 1.0), EventSummaryInfo)
    assert np.isfinite(log_posterior_predictive(event, spike, model))
    assert np.isfinite(log_posterior_predictive(spike, None, model))
    assert np.isfinite(log_prior(model))
    assert np.isfinite(log_p_latents(model))
    assert np.isfinite(log_like(model, [spike]))
    assert np.isfinite(log_joint(model, [spike]))


def test_ppseq_recovers_sequence_events_offsets_and_spike_assignments():
    data = make_sequence_counts()
    model = PPSeq(
        n_templates=1,
        template_duration=20,
        n_neurons=data.shape[0],
        bin_size=0.02,
        random_state=1,
        sequence_frac=0.8,
    )

    result = model.fit(data, num_iter=10, init="peak_time")

    corr = spearmanr(np.arange(data.shape[0]), model.template_offsets_[0]).statistic
    assert corr > 0.8
    assert result.events is not None
    assert result.spike_assignments is not None
    assert len(result.events) >= 4
    assert result.spike_assignments["prob_sequence"].mean() > 0.5
    assert result.amplitudes.shape == (1, data.shape[1])
    assert result.rates.shape == data.shape
    assert model.templates.shape == (1, data.shape[0], 20)
    assert result.initial_assignments is not None
    assert result.final_assignments is not None
    assert result.assignment_hist.shape[0] == len(result.final_assignments)
    assert len(result.latent_event_hist) == len(result.log_p_hist)
    assert len(result.globals_hist) == len(result.log_p_hist)
    assert result.anneal_assignment_hist.shape[1] == 0
    assert hasattr(model, "seq_model_")


def test_ppseq_transform_freezes_templates_and_assigns_new_spikes():
    task = make_sequence_counts()
    post = make_sequence_counts(n_time=80, starts=[12], background=0.0)
    model = PPSeq(
        n_templates=1,
        template_duration=20,
        n_neurons=task.shape[0],
        bin_size=0.02,
        random_state=2,
        sequence_frac=0.8,
    )
    model.fit(task, num_iter=8)
    offsets = model.template_offsets_.copy()

    result = model.transform(post, num_iter=5)
    responsibilities = ppseq_spike_responsibilities(model, post, result)

    np.testing.assert_allclose(model.template_offsets_, offsets)
    assert not responsibilities.empty
    assert responsibilities["assigned_template_id"].max() == 0


def test_ppseq_scores_forward_and_reverse_replay_events():
    task = make_sequence_counts()
    forward = make_sequence_counts(n_time=70, starts=[12])
    reverse = make_sequence_counts(
        n_time=70,
        starts=[12],
        neuron_order=np.arange(task.shape[0])[::-1],
    )
    time_centers = np.arange(forward.shape[1]) * 0.02 + 0.01
    intervals = np.array([[0.20, 0.70]])
    model = PPSeq(
        n_templates=1,
        template_duration=20,
        n_neurons=task.shape[0],
        bin_size=0.02,
        random_state=3,
        sequence_frac=0.8,
    )
    model.fit(task, num_iter=10)

    forward_scores, forward_result = model.score_replay_events(
        forward, intervals, time_centers, num_iter=10, method="event", base_rates="data"
    )
    reverse_scores, _ = model.score_replay_events(
        reverse, intervals, time_centers, num_iter=10, method="event", base_rates="data"
    )

    assert forward_result.amplitudes.shape[0] == 2
    assert forward_scores["template_0_forward_sum"].iloc[0] > forward_scores[
        "template_0_reverse_sum"
    ].iloc[0]
    assert reverse_scores["template_0_reverse_sum"].iloc[0] > reverse_scores[
        "template_0_forward_sum"
    ].iloc[0]
    assert np.isfinite(forward_scores["likelihood_gain_per_spike"].iloc[0])


def test_score_events_compatibility_returns_template_columns():
    task = make_sequence_counts()
    post = make_sequence_counts(n_time=70, starts=[12])
    time_centers = np.arange(post.shape[1]) * 0.02 + 0.01
    intervals = np.array([[0.20, 0.70]])
    model = PPSeq(
        n_templates=1,
        template_duration=20,
        n_neurons=task.shape[0],
        bin_size=0.02,
        random_state=4,
    )
    model.fit(task, num_iter=8)

    scores = model.score_events(post, intervals, time_centers, method="per_event")

    assert {"template_0_sum", "template_0_mean", "template_0_max"}.issubset(
        scores.columns
    )
    assert len(scores) == 1


def test_behavior_diagnostics_pass_for_bidirectional_task_and_fail_for_noise():
    data, time_centers, position = make_bidirectional_counts()
    model = PPSeq(
        n_templates=2,
        template_duration=32,
        n_neurons=data.shape[0],
        bin_size=0.02,
        random_state=5,
        sequence_frac=0.85,
        shared_width=True,
        max_width_bins=4,
    )
    result = model.fit(
        data,
        num_iter=12,
        init="lap_direction",
        init_position=position,
        tol=1e-2,
        patience=3,
    )
    diagnostics = ppseq_behavior_diagnostics(
        model,
        data,
        time_centers,
        position=position,
        result=result,
        min_events=5,
        min_spike_fraction=0.2,
        min_direction_purity=0.6,
    )

    assert diagnostics.passed
    assert diagnostics.per_template["direction_purity"].min() >= 0.6
    assert "offset_position_corr" in diagnostics.per_template.columns

    rng = np.random.default_rng(10)
    noise = rng.poisson(0.02, size=data.shape).astype(float)
    noise_model = PPSeq(2, 20, n_neurons=data.shape[0], bin_size=0.02, random_state=6)
    noise_result = noise_model.fit(noise, num_iter=3)
    noise_diag = ppseq_behavior_diagnostics(
        noise_model,
        noise,
        time_centers,
        position=position,
        result=noise_result,
        min_events=20,
        min_spike_fraction=0.4,
        min_direction_purity=0.9,
    )
    assert not noise_diag.passed
    assert noise_diag.summary["failure_reasons"].iloc[0]


def test_dense_task_background_does_not_collapse_sequence_assignment():
    data = make_sequence_counts(
        n_neurons=18,
        n_time=260,
        template_duration=24,
        starts=[20, 70, 120, 170, 220],
        background=0.08,
    )
    model = PPSeq(
        n_templates=1,
        template_duration=24,
        n_neurons=data.shape[0],
        bin_size=0.02,
        random_state=11,
        sequence_frac=0.8,
    )

    result = model.fit(data, num_iter=8, init="peak_time")
    diagnostics = ppseq_behavior_diagnostics(
        model,
        data,
        np.arange(data.shape[1]) * 0.02 + 0.01,
        result=result,
        min_events=3,
        min_spike_fraction=0.05,
        min_participating_neurons=0.25,
        min_direction_purity=0.0,
    )

    assert diagnostics.summary["assigned_spike_fraction"].iloc[0] > 0.05
    assert diagnostics.summary["min_participating_neurons"].iloc[0] >= 5


def test_fit_candidates_and_model_selection_outputs_behavior_columns():
    data, time_centers, position = make_bidirectional_counts(n_laps=6)

    model, result, candidates = fit_ppseq_candidates(
        data,
        time_centers,
        n_templates_grid=[2],
        template_duration_s_grid=[0.4, 0.64],
        bin_size=0.02,
        seeds=[0, 1],
        init_grid=["lap_direction"],
        position=position,
        fit_num_iter=6,
        transform_num_iter=3,
        sequence_frac=0.8,
    )

    assert isinstance(model, PPSeq)
    assert isinstance(result, PPSeqResult)
    assert {
        "passed_behavior_gate",
        "blocking_failure",
        "event_count_passed",
        "assignment_passed",
        "participation_passed",
        "direction_passed",
        "direction_purity_score",
        "template_direction_balance",
        "selection_reason",
    }.issubset(candidates.columns)
    assert candidates["selection_score"].is_monotonic_decreasing


def test_fit_candidates_returns_best_diagnostic_model_when_no_candidate_passes():
    data = make_sequence_counts(
        n_neurons=12,
        n_time=180,
        template_duration=24,
        starts=[20, 80, 140],
        background=0.02,
    )
    time_centers = np.arange(data.shape[1]) * 0.02 + 0.01

    model, result, candidates = fit_ppseq_candidates(
        data,
        time_centers,
        n_templates_grid=[1],
        template_duration_s_grid=[0.48],
        bin_size=0.02,
        seeds=[0],
        init_grid=["peak_time"],
        fit_num_iter=5,
        transform_num_iter=2,
        min_events=50,
        min_spike_fraction=0.95,
        min_direction_purity=0.0,
        sequence_frac=0.7,
    )
    diagnostics = ppseq_behavior_diagnostics(
        model,
        data,
        time_centers,
        result=result,
        min_events=50,
        min_spike_fraction=0.95,
        min_participating_neurons=0.20,
        min_direction_purity=0.0,
    )

    assert isinstance(model, PPSeq)
    assert isinstance(result, PPSeqResult)
    assert not bool(candidates["passed_behavior_gate"].iloc[0])
    assert candidates["blocking_failure"].iloc[0]
    assert candidates["full_assigned_spike_fraction"].iloc[0] == pytest.approx(
        diagnostics.summary["assigned_spike_fraction"].iloc[0]
    )


def test_fit_candidates_returns_full_fit_result_for_behavior_gate(monkeypatch):
    data = make_sequence_counts(
        n_neurons=14,
        n_time=260,
        template_duration=24,
        starts=[20, 70, 120, 170, 220],
        background=0.06,
    )
    time_centers = np.arange(data.shape[1]) * 0.02 + 0.01
    calls = {"transform": 0}
    original_transform = PPSeq.transform

    def counted_transform(self, data, **kwargs):
        calls["transform"] += 1
        return original_transform(self, data, **kwargs)

    monkeypatch.setattr(PPSeq, "transform", counted_transform)

    model, result, candidates = fit_ppseq_candidates(
        data,
        time_centers,
        n_templates_grid=[1],
        template_duration_s_grid=[0.48],
        bin_size=0.02,
        seeds=[0],
        init_grid=["peak_time"],
        fit_num_iter=6,
        transform_num_iter=3,
        min_events=20,
        min_spike_fraction=0.05,
        min_direction_purity=0.0,
        sequence_frac=0.8,
    )
    diagnostics = ppseq_behavior_diagnostics(
        model,
        data,
        time_centers,
        result=result,
        min_events=int(candidates["scaled_min_events"].iloc[0]),
        min_spike_fraction=0.05,
        min_participating_neurons=0.25,
        min_direction_purity=0.0,
    )

    assert candidates["full_assigned_spike_fraction"].iloc[0] == pytest.approx(
        diagnostics.summary["assigned_spike_fraction"].iloc[0]
    )
    assert candidates["full_n_events"].iloc[0] == diagnostics.summary["n_events"].iloc[0]
    assert calls["transform"] == 1  # heldout screening only, not task-fit validation
    assert result is not None


def test_ppseq_tutorial_is_slim_and_songbird_first():
    notebook = json.loads(
        Path("tutorials/ppseq_sequence_detection.ipynb").read_text(encoding="utf-8-sig")
    )
    all_source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    songbird_idx = all_source.index("Songbird PPSeq parity demo")
    simulated_idx = all_source.index("Simulated task and replay data")

    assert songbird_idx < simulated_idx
    assert "load_ppseq_songbird_spikes" in all_source
    assert "plot_ppseq_behavior_with_position" in all_source
    assert "score_ppseq_pre_post_ripples" in all_source
    assert "def get_position_time_and_track_position" not in all_source
    assert "def replay_spike_responsibilities" not in all_source
    banned = [
        "raise ValueError(\"Behavioral PPSeq templates failed validation",
        "raise ValueError(\"Pre-task sleep contains no ripples",
        "raise ValueError(\"Post-task sleep contains no ripples",
        "raise ValueError(\"Replay null scoring did not produce replay_z",
        "raise ValueError(\"No post-task ripples are available",
        "raise ValueError(\"No post-task ripples passed",
        "raise ValueError(\n        \"No high-confidence PPSeq events were detected",
    ]
    for text in banned:
        assert text not in all_source
    in_real_section = False
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "Real CellExplorer / nelpy session" in source:
            in_real_section = True
        if in_real_section and cell.get("cell_type") == "code":
            assert len(source.splitlines()) <= 60


def test_replay_nulls_are_deterministic_and_add_z_and_p():
    task = make_sequence_counts()
    post = make_sequence_counts(n_time=80, starts=[12, 42])
    time_centers = np.arange(post.shape[1]) * 0.02 + 0.01
    intervals = np.array([[0.20, 0.70], [0.80, 1.30]])
    model = PPSeq(
        n_templates=1,
        template_duration=20,
        n_neurons=task.shape[0],
        bin_size=0.02,
        random_state=7,
        sequence_frac=0.8,
    )
    model.fit(task, num_iter=8)

    scores, nulls = score_ppseq_replay_nulls(
        model,
        post,
        intervals,
        time_centers,
        n_null=5,
        random_state=0,
        method="event",
        base_rates="data",
        null_method="fast",
        num_iter=6,
    )
    scores_again, nulls_again = score_ppseq_replay_nulls(
        model,
        post,
        intervals,
        time_centers,
        n_null=5,
        random_state=0,
        method="event",
        base_rates="data",
        null_method="fast",
        num_iter=6,
    )

    assert {"replay_z", "replay_p"}.issubset(scores.columns)
    assert len(nulls) == 10
    np.testing.assert_allclose(scores["replay_p"], scores_again["replay_p"])
    np.testing.assert_allclose(
        nulls["likelihood_gain_per_spike"],
        nulls_again["likelihood_gain_per_spike"],
    )


def test_api_helpers_shapes_determinism_and_validation():
    data = make_sequence_counts(n_time=120)
    train = data[:, :80]
    test = data[:, 80:]

    model_a = PPSeq(1, 20, n_neurons=data.shape[0], bin_size=0.02, random_state=8)
    model_b = PPSeq(1, 20, n_neurons=data.shape[0], bin_size=0.02, random_state=8)
    result_a = model_a.fit(data, num_iter=5)
    result_b = model_b.fit(data, num_iter=5)
    np.testing.assert_allclose(model_a.template_offsets_, model_b.template_offsets_)
    np.testing.assert_allclose(result_a.amplitudes, result_b.amplitudes)

    with pytest.raises(ValueError, match="at least one spike"):
        PPSeq(1, 4, n_neurons=3).fit(np.zeros((3, 10)), num_iter=1)

    counts, centers = bin_spikes(
        np.array([0.01, 0.03, 0.041, 0.07]),
        np.array([0, 1, 1, 0]),
        bin_size=0.02,
        start=0.0,
        stop=0.08,
        n_neurons=2,
    )
    assert counts.shape == (2, 4)
    np.testing.assert_allclose(centers, [0.01, 0.03, 0.05, 0.07])

    scores = select_ppseq_template_count(
        train,
        test,
        [1, 2],
        template_duration=20,
        fit_num_iter=3,
        transform_num_iter=2,
        random_state=9,
    )
    assert {"heldout_log_likelihood", "heldout_log_likelihood_per_bin"}.issubset(
        scores.columns
    )

    summary, pairwise = ppseq_seed_stability(
        data,
        n_templates=1,
        seeds=[1, 2, 3],
        template_duration=20,
        fit_num_iter=3,
    )
    assert len(summary) == 1
    assert len(pairwise) == 3

    assert template_set_similarity(model_a.templates, model_b.templates) == pytest.approx(1.0)


def test_fit_and_transform_accept_continuous_spike_inputs():
    data = make_sequence_counts(n_time=90)
    spike_times = []
    neuron_ids = []
    for neuron in range(data.shape[0]):
        for bin_id in np.flatnonzero(data[neuron] > 0):
            count = int(data[neuron, bin_id])
            spike_times.extend([bin_id * 0.02 + 0.01] * count)
            neuron_ids.extend([neuron] * count)
    spike_times = np.asarray(spike_times)
    neuron_ids = np.asarray(neuron_ids)

    model = PPSeq(1, 20, n_neurons=data.shape[0], bin_size=0.02, random_state=12)
    result = model.fit(spike_times, neuron_ids=neuron_ids, start=0.0, stop=1.8, num_iter=5)
    transformed = model.transform(
        spike_times,
        neuron_ids=neuron_ids,
        start=0.0,
        stop=1.8,
        num_iter=3,
    )

    assert result.assignment_hist.shape[0] == spike_times.size
    assert transformed.rates.shape == data.shape


def test_songbird_loader_assignment_raster_and_plot_helpers():
    spike_times, neuron_ids = load_ppseq_songbird_spikes(
        Path("tutorials/data/songbird_spikes.txt")
    )

    assert spike_times.ndim == 1
    assert neuron_ids.ndim == 1
    assert spike_times.shape == neuron_ids.shape
    assert np.all(np.isfinite(spike_times))
    assert neuron_ids.min() == 0

    data = make_sequence_counts(n_time=100)
    model = PPSeq(1, 20, n_neurons=data.shape[0], bin_size=0.02, random_state=13)
    result = model.fit(data, num_iter=4)
    raster = ppseq_assignment_raster(result, threshold=0.3)
    empty = ppseq_assignment_raster(
        PPSeqResult(np.asarray([]), np.zeros((1, 1)), np.zeros((1, 1)))
    )

    assert {"time", "neuron_id", "assigned_template_id"}.issubset(raster.columns)
    assert empty.empty
    ax = plot_ppseq_raster(raster["time"], raster["neuron_id"], raster)
    axes = plot_ppseq_diagnostics(result)
    assert ax is not None
    assert len(axes) == 2


def test_continuous_spike_fit_preserves_assignment_times():
    spike_times = np.array([0.011, 0.037, 0.109, 0.223])
    neuron_ids = np.array([0, 1, 0, 1])
    model = PPSeq(1, 8, n_neurons=2, bin_size=0.02, random_state=14)
    result = model.fit(
        spike_times,
        neuron_ids=neuron_ids,
        start=0.0,
        stop=0.3,
        num_iter=2,
    )

    assert np.allclose(result.spike_assignments["time"].to_numpy(), spike_times)


def test_raster_colors_assignments_by_spike_id_not_row_order():
    spike_times = np.array([0.0, 0.1, 0.2])
    neuron_ids = np.array([0, 1, 2])
    assignments = pd.DataFrame(
        {
            "spike_id": [2, 0, 1],
            "time": spike_times[[2, 0, 1]],
            "neuron_id": neuron_ids[[2, 0, 1]],
            "assigned_template_id": [0, -1, -1],
        }
    )

    ax = plot_ppseq_raster(spike_times, neuron_ids, assignments)
    assigned_offsets = ax.collections[1].get_offsets()

    assert assigned_offsets.shape[0] == 1
    assert float(assigned_offsets[0, 0]) == pytest.approx(0.2)


def test_summarize_replay_scores_handles_empty_and_prepost_scores():
    empty = summarize_ppseq_replay_scores(pd.DataFrame())
    scores = pd.DataFrame(
        {
            "start": [0.0, 1.0, 2.0, 3.0],
            "sleep_epoch": ["pre", "pre", "post", "post"],
            "replay_z": [0.0, 1.0, 2.0, 3.0],
            "replay_p": [0.5, 0.5, 0.01, 0.5],
            "likelihood_gain_per_spike": [0.1, 0.2, 0.3, 0.4],
        }
    )
    summary = summarize_ppseq_replay_scores(scores)

    assert list(empty.index) == ["pre", "post"]
    assert summary.loc["post", "n_ripples"] == 2
    assert summary.loc["post", "fraction_replay_like"] > 0
