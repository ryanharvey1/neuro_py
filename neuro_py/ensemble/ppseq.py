from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from warnings import warn

import nelpy as nel
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import gammaln, logsumexp
from scipy.stats import spearmanr

from neuro_py.process.peri_event import count_in_interval


@dataclass(frozen=True)
class Spike:
    """
    Continuous-time spike used by the Julia PPSeq model.

    Parameters
    ----------
    neuron : int
        Zero-based neuron index.
    timestamp : float
        Spike time in seconds.
    """

    neuron: int
    timestamp: float


@dataclass(frozen=True)
class EventSummaryInfo:
    """
    Summary of a latent sequence event.

    This mirrors the Julia ``EventSummaryInfo`` type, using zero-based
    ``seq_type`` values for Python indexing.
    """

    assignment_id: int
    timestamp: float
    seq_type: int
    seq_warp: float
    amplitude: float


@dataclass(frozen=True)
class RateGamma:
    """Gamma distribution parameterized by shape and rate."""

    alpha: float
    beta: float

    def logpdf(self, x: float) -> float:
        x = float(x)
        if x <= 0:
            return -np.inf
        return (
            self.alpha * np.log(self.beta)
            - gammaln(self.alpha)
            + (self.alpha - 1.0) * np.log(x)
            - self.beta * x
        )

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.gamma(self.alpha, 1.0 / self.beta))


@dataclass(frozen=True)
class SymmetricDirichlet:
    """Symmetric Dirichlet distribution."""

    conc: float
    dim: int

    def logpdf(self, p: np.ndarray) -> float:
        p = np.asarray(p, dtype=float)
        if p.shape != (self.dim,) or np.any(p <= 0):
            return -np.inf
        return float(
            gammaln(self.dim * self.conc)
            - self.dim * gammaln(self.conc)
            + (self.conc - 1.0) * np.sum(np.log(p))
        )

    def sample(self, rng: np.random.Generator, size: int | None = None) -> np.ndarray:
        shape = self.dim if size is None else (size, self.dim)
        draws = rng.gamma(self.conc, 1.0, size=shape)
        return draws / np.clip(draws.sum(axis=-1, keepdims=True), np.finfo(float).tiny, None)


@dataclass(frozen=True)
class ScaledInvChisq:
    """Scaled inverse chi-squared distribution."""

    nu: float
    s2: float

    def logpdf(self, x: float) -> float:
        x = float(x)
        if x <= 0:
            return -np.inf
        half_nu = 0.5 * self.nu
        return float(
            half_nu * np.log(half_nu)
            - gammaln(half_nu)
            + half_nu * np.log(self.s2)
            - (half_nu + 1.0) * np.log(x)
            - half_nu * self.s2 / x
        )

    def sample(self, rng: np.random.Generator) -> float:
        return float(self.nu * self.s2 / rng.chisquare(self.nu))


@dataclass(frozen=True)
class NormalInvChisq:
    """Normal-inverse-chi-squared prior over Gaussian mean and variance."""

    k: float
    m: float
    nu: float
    s2: float

    def logpdf(self, mu: float, sigma2: float) -> float:
        if sigma2 <= 0:
            return -np.inf
        var_mu = sigma2 / self.k
        return float(
            ScaledInvChisq(self.nu, self.s2).logpdf(sigma2)
            - 0.5 * (np.log(2.0 * np.pi * var_mu) + ((mu - self.m) ** 2) / var_mu)
        )

    def sample(self, rng: np.random.Generator) -> tuple[float, float]:
        sigma2 = ScaledInvChisq(self.nu, self.s2).sample(rng)
        mu = float(self.m + rng.normal() * np.sqrt(sigma2 / self.k))
        return mu, sigma2


@dataclass
class SeqPriors:
    """Prior distributions for a Julia-style PPSeq model."""

    seq_event_rate: float
    seq_type_proportions: SymmetricDirichlet
    seq_event_amplitude: RateGamma
    neuron_response_proportions: SymmetricDirichlet
    neuron_response_profile: NormalInvChisq
    bkgd_amplitude: RateGamma
    bkgd_proportions: SymmetricDirichlet
    warp_values: np.ndarray
    warp_log_proportions: np.ndarray


@dataclass
class SeqGlobals:
    """Global variables for a Julia-style PPSeq model."""

    seq_type_log_proportions: np.ndarray
    neuron_response_log_proportions: np.ndarray
    neuron_response_offsets: np.ndarray
    neuron_response_widths: np.ndarray
    bkgd_amplitude: float
    bkgd_log_proportions: np.ndarray


@dataclass
class SeqEvent:
    """Latent event sufficient statistics and sampled latent state."""

    spike_count: int
    summed_potentials: np.ndarray
    summed_precisions: np.ndarray
    summed_logZ: np.ndarray
    seq_type_posterior: np.ndarray
    sampled_type: int
    sampled_warp: int
    sampled_timestamp: float
    sampled_amplitude: float


@dataclass
class SeqEventList:
    """Dynamic event storage mirroring Julia ``SeqEventList``."""

    events: list[SeqEvent]
    indices: list[int]


@dataclass
class SeqModel:
    """Julia-style PPSeq model container."""

    max_time: float
    max_sequence_length: float
    priors: SeqPriors
    globals: SeqGlobals
    sequence_events: SeqEventList
    new_cluster_log_prob: float
    bkgd_log_prob: float


@dataclass
class PPSeqResult:
    """
    Results from PPSeq fitting or fixed-template inference.

    Parameters
    ----------
    log_likelihood : np.ndarray
        Approximate model score or log-posterior trace.
    amplitudes : np.ndarray
        Event/template activity over bins with shape
        ``(n_templates, n_time_bins)`` for compatibility.
    rates : np.ndarray
        Reconstructed rates/counts with shape ``(n_neurons, n_time_bins)``.
    base_rates : np.ndarray, optional
        Background rates used for reconstruction and likelihood scoring.
    events : pandas.DataFrame, optional
        Sequence event table.
    spike_assignments : pandas.DataFrame, optional
        Spike-level assignment probabilities and labels.
    templates : np.ndarray, optional
        Template tensor with shape ``(n_templates, n_neurons, duration)``.
    initial_assignments, final_assignments, assignment_hist, latent_event_hist,
    globals_hist, log_p_hist, anneal_assignment_hist, anneal_latent_event_hist,
    anneal_globals_hist, anneal_log_p_hist : optional
        Julia-style sampler outputs.
    """

    log_likelihood: np.ndarray
    amplitudes: np.ndarray
    rates: np.ndarray
    base_rates: np.ndarray | None = None
    events: pd.DataFrame | None = None
    spike_assignments: pd.DataFrame | None = None
    templates: np.ndarray | None = None
    initial_assignments: np.ndarray | None = None
    final_assignments: np.ndarray | None = None
    assignment_hist: np.ndarray | None = None
    latent_event_hist: list[list[EventSummaryInfo]] | None = None
    globals_hist: list[SeqGlobals] | None = None
    log_p_hist: np.ndarray | None = None
    anneal_assignment_hist: np.ndarray | None = None
    anneal_latent_event_hist: list[list[EventSummaryInfo]] | None = None
    anneal_globals_hist: list[SeqGlobals] | None = None
    anneal_log_p_hist: np.ndarray | None = None


@dataclass
class PPSeqBehaviorDiagnostics:
    """
    Behavioral quality diagnostics for fitted PPSeq templates.

    Parameters
    ----------
    summary : pandas.DataFrame
        One-row summary with pass/fail fields.
    per_template : pandas.DataFrame
        Template-level event, direction, and participation metrics.
    event_table : pandas.DataFrame
        Detected behavioral sequence events.
    passed : bool
        Whether supplied validation thresholds were passed.
    """

    summary: pd.DataFrame
    per_template: pd.DataFrame
    event_table: pd.DataFrame
    passed: bool


def specify_gamma(mean: float, var: float) -> RateGamma:
    """
    Specify a rate-parameterized Gamma distribution by mean and variance.

    Parameters
    ----------
    mean : float
        Distribution mean.
    var : float
        Distribution variance.

    Returns
    -------
    RateGamma
        Shape/rate Gamma distribution.
    """

    if mean <= 0 or var <= 0:
        raise ValueError("mean and var must be positive")
    beta = mean / var
    alpha = mean * beta
    return RateGamma(float(alpha), float(beta))


def posterior_rate_gamma(count: float, prior: RateGamma, exposure: float = 1.0) -> RateGamma:
    """Poisson-Gamma posterior in rate parameterization."""

    return RateGamma(float(prior.alpha + count), float(prior.beta + exposure))


def posterior_symmetric_dirichlet(counts: np.ndarray, prior: SymmetricDirichlet) -> np.ndarray:
    """Dirichlet posterior concentration vector."""

    counts = np.asarray(counts, dtype=float)
    if counts.shape != (prior.dim,):
        raise ValueError("counts must match prior dimension")
    return counts + prior.conc


def posterior_normal_inv_chisq(
    n: int,
    sum_x: float,
    sumsq_x: float,
    prior: NormalInvChisq,
) -> NormalInvChisq:
    """Normal-inverse-chi-squared posterior."""

    if n <= 0:
        return prior
    k = prior.k + n
    nu = prior.nu + n
    mean_x = sum_x / n
    m = (sum_x + prior.k * prior.m) / k
    s2 = (
        prior.nu * prior.s2
        + (sumsq_x - n * mean_x * mean_x)
        + (prior.k * n * (mean_x - prior.m) ** 2) / k
    ) / nu
    return NormalInvChisq(float(k), float(m), float(nu), float(max(s2, np.finfo(float).tiny)))


def gauss_info_logZ(potential: float, precision: float) -> float:
    """Log normalizer of a univariate Gaussian in information form."""

    if precision <= 0:
        return np.inf
    return float(0.5 * (np.log(2.0 * np.pi) - np.log(precision) + potential * potential / precision))


def log_posterior_predictive(
    event: SeqEvent | Spike,
    spike: Spike | None,
    model: SeqModel,
) -> float:
    """
    Julia-style posterior predictive probability for assigning a spike.

    If ``event`` is a ``Spike`` and ``spike`` is None, compute the singleton
    predictive used for creating a new event.
    """

    if isinstance(event, Spike) and spike is None:
        spike = event
        log_prob = (
            model.globals.seq_type_log_proportions
            + model.globals.neuron_response_log_proportions[spike.neuron]
        )
        return float(logsumexp(log_prob) - np.log(model.max_time))
    if not isinstance(event, SeqEvent) or spike is None:
        raise TypeError("expected (SeqEvent, Spike, SeqModel) or (Spike, None, SeqModel)")
    if event.spike_count <= 0:
        raise ValueError("event must contain at least one spike")
    n = spike.neuron
    log_prob = np.empty_like(event.seq_type_posterior)
    for r in range(log_prob.shape[0]):
        for w in range(log_prob.shape[1]):
            m = event.summed_potentials[r, w]
            v = max(event.summed_precisions[r, w], np.finfo(float).tiny)
            mu = m / v
            sigma2 = 1.0 / v
            warp = model.priors.warp_values[w]
            offset = model.globals.neuron_response_offsets[n, r] * warp
            width = max(model.globals.neuron_response_widths[n, r] * (warp**2), np.finfo(float).tiny)
            var = sigma2 + width
            log_prob[r, w] = (
                event.seq_type_posterior[r, w]
                + model.globals.neuron_response_log_proportions[n, r]
                - 0.5 * (np.log(2.0 * np.pi * var) + ((spike.timestamp - (mu + offset)) ** 2) / var)
            )
    return float(logsumexp(log_prob))


def log_prior(model: SeqModel) -> float:
    """Log prior probability of Julia-style global parameters."""

    priors = model.priors
    globals_ = model.globals
    lp = priors.bkgd_amplitude.logpdf(globals_.bkgd_amplitude)
    lp += priors.bkgd_proportions.logpdf(np.exp(globals_.bkgd_log_proportions))
    for r in range(priors.seq_type_proportions.dim):
        lp += priors.neuron_response_proportions.logpdf(
            np.exp(globals_.neuron_response_log_proportions[:, r])
        )
        for n in range(priors.bkgd_proportions.dim):
            lp += priors.neuron_response_profile.logpdf(
                globals_.neuron_response_offsets[n, r],
                globals_.neuron_response_widths[n, r],
            )
    return float(lp)


def log_p_latents(model: SeqModel) -> float:
    """Log probability of current latent events given globals and priors."""

    lp = 0.0
    for event in model.sequence_events.events:
        lp += model.priors.seq_event_amplitude.logpdf(event.sampled_amplitude)
        lp += model.globals.seq_type_log_proportions[event.sampled_type]
        lp += model.priors.warp_log_proportions[event.sampled_warp]
    return float(lp)


def log_like(model: SeqModel, spikes: list[Spike] | np.ndarray) -> float:
    """Point-process log likelihood for spikes under a Julia-style model."""

    if isinstance(spikes, np.ndarray):
        spikes = [Spike(int(n), float(t)) for n, t in spikes]
    globals_ = model.globals
    ll = 0.0
    for spike in spikes:
        intensity = globals_.bkgd_amplitude * np.exp(globals_.bkgd_log_proportions[spike.neuron])
        for event in model.sequence_events.events:
            warp = model.priors.warp_values[event.sampled_warp]
            mu = event.sampled_timestamp + globals_.neuron_response_offsets[spike.neuron, event.sampled_type] * warp
            var = max(globals_.neuron_response_widths[spike.neuron, event.sampled_type] * warp * warp, np.finfo(float).tiny)
            scale = np.exp(globals_.neuron_response_log_proportions[spike.neuron, event.sampled_type])
            intensity += event.sampled_amplitude * scale * np.exp(
                -0.5 * ((spike.timestamp - mu) ** 2) / var
            ) / np.sqrt(2.0 * np.pi * var)
        ll += np.log(max(float(intensity), np.finfo(float).tiny))
    ll -= globals_.bkgd_amplitude * model.max_time
    for event in model.sequence_events.events:
        ll -= event.sampled_amplitude
    return float(ll)


def log_joint(model: SeqModel, spikes: list[Spike] | np.ndarray) -> float:
    """Julia-style joint log probability up to sampler constants."""

    return float(log_prior(model) + log_p_latents(model) + log_like(model, spikes))


class PPSeq:
    """
    Continuous-time PPSeq-style sequence detector.

    This implementation follows the replay-facing pieces of the PP-Seq paper
    and Julia package: sequence events are represented in continuous time,
    spikes receive probabilistic background/sequence assignments, and learned
    task templates can be frozen for forward/reverse ripple scoring. Version 1
    implements a compact Gibbs-like coordinate update over event assignments and
    template offsets; split-merge proposals, distributed chains, and learned
    warps are intentionally left for a later full-parity port.

    Parameters
    ----------
    n_templates : int
        Number of sequence types.
    template_duration : int
        Template duration in bins. The duration in seconds is
        ``template_duration * bin_size``.
    n_neurons : int, optional
        Number of neurons. If None, inferred from data.
    bin_size : float, optional
        Bin size in seconds used for compatibility outputs and event scoring.
    sequence_frac : float, optional
        Initial fraction of spikes assigned to sequence events.
    min_rate : float, optional
        Numerical floor for background and template probabilities.
    min_width : float, optional
        Minimum template width in bins.
    random_state : int, optional
        Seed for deterministic initialization.
    """

    def __init__(
        self,
        n_templates: int,
        template_duration: int,
        n_neurons: int | None = None,
        bin_size: float = 0.02,
        sequence_frac: float = 0.5,
        concentration: float = 10.0,
        min_rate: float = 1e-7,
        min_amplitude: float = 1e-9,
        min_width: float = 1e-3,
        alpha_amplitude: float = 1e-3,
        beta_amplitude: float = 1e-3,
        alpha_base: float = 1e-3,
        beta_base: float = 1e-3,
        alpha_template: float = 1e-4,
        shared_width: bool = False,
        min_width_bins: float | None = None,
        max_width_bins: float | None = None,
        min_template_participation: int | float | None = None,
        random_state: int | None = None,
    ) -> None:
        if n_templates <= 0:
            raise ValueError("n_templates must be positive")
        if template_duration <= 0:
            raise ValueError("template_duration must be positive")
        if bin_size <= 0:
            raise ValueError("bin_size must be positive")
        if not 0 <= sequence_frac <= 1:
            raise ValueError("sequence_frac must be between 0 and 1")

        self.n_templates = int(n_templates)
        self.template_duration = int(template_duration)
        self.n_neurons = None if n_neurons is None else int(n_neurons)
        self.bin_size = float(bin_size)
        self.sequence_frac = float(sequence_frac)
        self.concentration = float(concentration)
        self.min_rate = float(min_rate)
        self.min_amplitude = float(min_amplitude)
        self.min_width = float(min_width)
        self.alpha_amplitude = float(alpha_amplitude)
        self.beta_amplitude = float(beta_amplitude)
        self.alpha_base = float(alpha_base)
        self.beta_base = float(beta_base)
        self.alpha_template = float(alpha_template)
        self.shared_width = bool(shared_width)
        self.min_width_bins = min_width_bins
        self.max_width_bins = max_width_bins
        self.min_template_participation = min_template_participation
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    @property
    def template_duration_s(self) -> float:
        """Template duration in seconds."""

        return self.template_duration * self.bin_size

    @property
    def templates(self) -> np.ndarray:
        """
        Discretized template tensor.

        Returns
        -------
        np.ndarray
            Template tensor with shape
            ``(n_templates, n_neurons, template_duration)``.
        """

        self._check_initialized()
        return self._templates_from_parameters(
            self.template_offsets_,
            self.template_widths_,
            self.template_scales_,
            self.template_duration,
        )

    def fit(
        self,
        data: np.ndarray,
        *,
        neuron_ids: np.ndarray | None = None,
        start: float | None = None,
        stop: float | None = None,
        num_iter: int = 50,
        fit_templates: bool = True,
        fit_base_rates: bool = True,
        tol: float | None = None,
        patience: int = 5,
        init: str = "random",
        init_position: np.ndarray | None = None,
    ) -> PPSeqResult:
        """
        Fit task sequence templates.

        Parameters
        ----------
        data : np.ndarray
            Spike count matrix with shape ``(n_neurons, n_time_bins)`` or
            spike timestamps in seconds when ``neuron_ids`` is provided.
        neuron_ids : np.ndarray, optional
            Neuron id per spike for continuous-time spike inputs.
        start, stop : float, optional
            Time bounds for continuous-time spike inputs.
        num_iter : int, optional
            Number of coordinate/Gibbs-like update sweeps.
        fit_templates : bool, optional
            Whether to update template offsets, widths, and weights.
        fit_base_rates : bool, optional
            Whether to update background rates.
        tol : float, optional
            Early stopping tolerance on score improvement.
        patience : int, optional
            Consecutive low-improvement iterations before stopping.
        init : {"random", "peak_time", "lap_direction"}, optional
            Initialization strategy.
        init_position : np.ndarray, optional
            Position per time bin for direction-aware initialization.

        Returns
        -------
        PPSeqResult
            Fitted templates, events, spike assignments, and compatibility
            arrays.
        """

        data, spike_times, neuron_ids = self._coerce_input_data_and_spikes(
            data, neuron_ids=neuron_ids, start=start, stop=stop
        )
        data = self._validate_data(data)
        if np.sum(data) <= 0:
            raise ValueError("data must contain at least one spike")
        self._validate_early_stopping(tol, patience)
        time_bins = data.shape[1]
        self._initialize_from_spikes(
            spike_times,
            neuron_ids,
            time_bins,
            init=init,
            position=init_position,
        )
        result = self._run_sequence_updates(
            data,
            spike_times,
            neuron_ids,
            num_iter=num_iter,
            fit_templates=fit_templates,
            fit_base_rates=fit_base_rates,
            tol=tol,
            patience=patience,
            fixed_templates=False,
        )
        self.amplitudes_ = result.amplitudes
        self.log_likelihood_ = result.log_likelihood
        self.events_ = result.events
        self.spike_assignments_ = result.spike_assignments
        return result

    def transform(
        self,
        data: np.ndarray,
        *,
        neuron_ids: np.ndarray | None = None,
        start: float | None = None,
        stop: float | None = None,
        num_iter: int = 25,
        fit_templates: bool = False,
        fit_base_rates: bool = False,
        tol: float | None = None,
        patience: int = 5,
    ) -> PPSeqResult:
        """
        Infer sequence events in new data with learned templates frozen.

        Parameters
        ----------
        data : np.ndarray
            Spike count matrix with shape ``(n_neurons, n_time_bins)`` or
            spike timestamps in seconds when ``neuron_ids`` is provided.
        neuron_ids : np.ndarray, optional
            Neuron id per spike for continuous-time spike inputs.
        start, stop : float, optional
            Time bounds for continuous-time spike inputs.
        num_iter : int, optional
            Number of fixed-template update sweeps.
        fit_templates : bool, optional
            Ignored unless explicitly True; defaults to frozen templates.
        fit_base_rates : bool, optional
            Whether to estimate background from the new data.
        tol : float, optional
            Early stopping tolerance.
        patience : int, optional
            Early stopping patience.

        Returns
        -------
        PPSeqResult
            Fixed-template inference result.
        """

        self._check_initialized()
        data, spike_times, neuron_ids = self._coerce_input_data_and_spikes(
            data, neuron_ids=neuron_ids, start=start, stop=stop
        )
        data = self._validate_data(data)
        self._validate_early_stopping(tol, patience)
        result = self._run_sequence_updates(
            data,
            spike_times,
            neuron_ids,
            num_iter=num_iter,
            fit_templates=fit_templates,
            fit_base_rates=fit_base_rates,
            tol=tol,
            patience=patience,
            fixed_templates=not fit_templates,
        )
        return result

    def fit_transform(self, data: np.ndarray, **kwargs) -> PPSeqResult:
        """
        Fit templates and return the fitted result.

        Parameters
        ----------
        data : np.ndarray
            Spike count matrix with shape ``(n_neurons, n_time_bins)``.
        **kwargs
            Passed to :meth:`fit`.

        Returns
        -------
        PPSeqResult
            Fitted PPSeq result.
        """

        return self.fit(data, **kwargs)

    def reconstruct(self, result: PPSeqResult | None = None) -> np.ndarray:
        """
        Return the reconstructed count/rate matrix.

        Parameters
        ----------
        result : PPSeqResult, optional
            Result to reconstruct from. If omitted, uses the last fit.

        Returns
        -------
        np.ndarray
            Reconstructed rates/counts.
        """

        if result is None:
            if not hasattr(self, "amplitudes_"):
                raise ValueError("PPSeq must be fit before reconstruct is used")
            return self._reconstruct_from_amplitudes(self.amplitudes_)
        return result.rates

    def score(self, data: np.ndarray, result: PPSeqResult | None = None) -> float:
        """
        Approximate Poisson log likelihood for a result.

        Parameters
        ----------
        data : np.ndarray
            Spike count matrix.
        result : PPSeqResult, optional
            Result to score. If omitted, transforms the data.

        Returns
        -------
        float
            Poisson log likelihood.
        """

        data = self._validate_data(data)
        if result is None:
            result = self.transform(data, num_iter=10)
        rates = np.clip(result.rates, self.min_rate, None)
        return float(np.sum(data * np.log(rates) - rates - gammaln(data + 1.0)))

    def score_events(
        self,
        data: np.ndarray,
        event_intervals: np.ndarray,
        time_centers: np.ndarray,
        *,
        num_iter: int = 25,
        method: str = "full",
        result: PPSeqResult | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Score fixed task templates in event intervals.

        This compatibility method scores forward templates only. For replay,
        prefer :meth:`score_replay_events`.
        """

        scores, _ = self.score_replay_events(
            data,
            event_intervals,
            time_centers,
            num_iter=num_iter,
            method="event" if method == "per_event" else method,
            result=result,
            include_reverse=False,
            **kwargs,
        )
        forward_cols = [c for c in scores.columns if c.endswith("_forward_sum")]
        for col in forward_cols:
            k = col.split("_")[1]
            scores[f"template_{k}_sum"] = scores[col]
            scores[f"template_{k}_mean"] = scores[col.replace("_sum", "_mean")]
            scores[f"template_{k}_max"] = scores[col.replace("_sum", "_max")]
        return scores

    def score_replay_events(
        self,
        data: np.ndarray,
        event_intervals: np.ndarray,
        time_centers: np.ndarray,
        *,
        num_iter: int = 25,
        method: str = "full",
        result: PPSeqResult | None = None,
        include_forward: bool = True,
        include_reverse: bool = True,
        template_names: list[str] | None = None,
        replay_duration_s: float | None = None,
        event_pad_s: float | None = None,
        base_rates: str | np.ndarray = "model",
        tol: float | None = None,
        patience: int = 5,
    ) -> tuple[pd.DataFrame, PPSeqResult]:
        """
        Score forward/reverse replay during event intervals.

        Parameters
        ----------
        data : np.ndarray
            Spike count matrix.
        event_intervals : np.ndarray
            Event intervals with shape ``(n_events, 2)`` in seconds.
        time_centers : np.ndarray
            Time center per data bin in seconds.
        num_iter : int, optional
            Fixed-template inference sweeps.
        method : {"full", "event"}, optional
            Whether to infer on full data or concatenated event windows.
        result : PPSeqResult, optional
            Precomputed replay-bank inference result.
        include_forward, include_reverse : bool, optional
            Include forward/reverse task templates.
        template_names : list of str, optional
            Template labels.
        replay_duration_s : float, optional
            Compress task templates to this duration for replay scoring.
        event_pad_s : float, optional
            Padding around events for event-local inference.
        base_rates : {"model", "data"} or np.ndarray, optional
            Background rates used for likelihood-gain scoring.
        tol : float, optional
            Early stopping tolerance.
        patience : int, optional
            Early stopping patience.

        Returns
        -------
        tuple of pandas.DataFrame and PPSeqResult
            Event-level replay scores and replay-bank inference result.
        """

        self._check_initialized()
        data = self._validate_data(data)
        event_intervals = self._validate_intervals(event_intervals)
        time_centers = np.asarray(time_centers, dtype=float)
        if time_centers.ndim != 1 or time_centers.shape[0] != data.shape[1]:
            raise ValueError("time_centers must be 1D and match data time bins")
        if method not in {"full", "event"}:
            raise ValueError("method must be 'full' or 'event'")
        replay_templates, replay_labels = self._make_replay_template_bank(
            include_forward=include_forward,
            include_reverse=include_reverse,
            template_names=template_names,
            replay_duration_s=replay_duration_s,
        )
        if method == "event":
            if result is not None:
                raise ValueError("result cannot be used with method='event'")
            if event_pad_s is None:
                event_pad_s = (
                    replay_duration_s
                    if replay_duration_s is not None
                    else self.template_duration_s
                )
            event_data, event_time, event_slices = self._event_local_data(
                data, event_intervals, time_centers, float(event_pad_s)
            )
            inference_base_rates = self._resolve_base_rates(event_data, base_rates)
            result = self._infer_with_template_bank(
                event_data,
                replay_templates,
                num_iter=num_iter,
                base_rates=inference_base_rates,
                tol=tol,
                patience=patience,
            )
            scores = self._summarize_replay_slices(
                event_data,
                event_intervals,
                result,
                replay_labels,
                event_slices=event_slices,
            )
            return scores, result
        if result is None:
            inference_base_rates = self._resolve_base_rates(data, base_rates)
            result = self._infer_with_template_bank(
                data,
                replay_templates,
                num_iter=num_iter,
                base_rates=inference_base_rates,
                tol=tol,
                patience=patience,
            )
        else:
            if result.amplitudes.shape != (len(replay_labels), data.shape[1]):
                raise ValueError("result.amplitudes shape does not match replay bank")
            if result.rates.shape != data.shape:
                raise ValueError("result.rates must have the same shape as data")
        event_slices = [
            slice(
                int(np.searchsorted(time_centers, start, side="left")),
                int(np.searchsorted(time_centers, stop, side="right")),
            )
            for start, stop in event_intervals
        ]
        scores = self._summarize_replay_slices(
            data, event_intervals, result, replay_labels, event_slices=event_slices
        )
        return scores, result

    def bin_spikes(
        self,
        spike_times: np.ndarray | nel.SpikeTrainArray,
        neuron_ids: np.ndarray | None = None,
        *,
        start: float | None = None,
        stop: float | None = None,
        bin_size: float | None = None,
        n_neurons: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Bin spikes into a count matrix.

        Parameters
        ----------
        spike_times : np.ndarray or nel.SpikeTrainArray
            Spike timestamps in seconds, or a nelpy spike train array.
        neuron_ids : np.ndarray, optional
            Neuron ID per spike for array inputs.
        start, stop : float, optional
            Time range in seconds.
        bin_size : float, optional
            Bin size in seconds. Defaults to ``self.bin_size``.
        n_neurons : int, optional
            Number of neurons for array inputs.

        Returns
        -------
        counts : np.ndarray
            Count matrix with shape ``(n_neurons, n_time_bins)``.
        time_centers : np.ndarray
            Bin centers in seconds.
        """

        return bin_spikes(
            spike_times,
            neuron_ids,
            start=start,
            stop=stop,
            bin_size=self.bin_size if bin_size is None else bin_size,
            n_neurons=self.n_neurons if n_neurons is None else n_neurons,
        )

    def _initialize_from_spikes(
        self,
        spike_times: np.ndarray,
        neuron_ids: np.ndarray,
        time_bins: int,
        *,
        init: str,
        position: np.ndarray | None,
    ) -> None:
        if self.n_neurons is None:
            self.n_neurons = int(neuron_ids.max()) + 1 if neuron_ids.size else 0
        duration = self.template_duration_s
        event_times = self._candidate_event_times_from_counts(
            self._spikes_to_counts(spike_times, neuron_ids, time_bins)
        )
        if event_times.size == 0:
            event_times = np.array([spike_times.min()], dtype=float)
        if init == "peak_time":
            assignments = self._cluster_event_windows(
                spike_times, neuron_ids, event_times, duration
            )
        else:
            assignments = self._initial_event_template_assignments(
                event_times, init=init, position=position, time_bins=time_bins
            )
        offsets = np.zeros((self.n_templates, self.n_neurons), dtype=float)
        widths = np.full((self.n_templates, self.n_neurons), duration / 12.0)
        scales = np.full((self.n_templates, self.n_neurons), 1.0 / self.n_neurons)
        for k in range(self.n_templates):
            event_subset = event_times[assignments == k]
            if event_subset.size == 0:
                offsets[k] = np.linspace(0.05 * duration, 0.95 * duration, self.n_neurons)
                if init == "random":
                    self._rng.shuffle(offsets[k])
                continue
            for n in range(self.n_neurons):
                lags = []
                neuron_spikes = spike_times[neuron_ids == n]
                for event_time in event_subset:
                    local = neuron_spikes[
                        (neuron_spikes >= event_time)
                        & (neuron_spikes <= event_time + duration)
                    ]
                    if local.size:
                        lags.extend((local - event_time).tolist())
                if lags:
                    offsets[k, n] = float(np.median(lags))
                    widths[k, n] = max(float(np.std(lags)), self.bin_size)
                    scales[k, n] = len(lags)
                else:
                    offsets[k, n] = (n + 0.5) / self.n_neurons * duration
                    scales[k, n] = self.min_rate
            if np.sum(scales[k]) > 0:
                scales[k] = scales[k] / np.sum(scales[k])
        self.template_offsets_ = np.clip(offsets, 0.0, duration)
        self.template_widths_ = self._clip_widths(widths)
        self.template_scales_ = self._normalize_scales(scales)
        total_duration = max(time_bins * self.bin_size, self.bin_size)
        counts = np.bincount(neuron_ids, minlength=self.n_neurons).astype(float)
        self.base_rates_ = np.clip(counts / total_duration, self.min_rate, None)
        if init == "peak_time" and event_times.size:
            self._event_times_ = np.repeat(event_times, self.n_templates)
            self._event_templates_ = np.tile(np.arange(self.n_templates), event_times.size)
        else:
            self._event_times_ = event_times
            self._event_templates_ = assignments

    def _cluster_event_windows(
        self,
        spike_times: np.ndarray,
        neuron_ids: np.ndarray,
        event_times: np.ndarray,
        duration: float,
    ) -> np.ndarray:
        if event_times.size == 0:
            return np.asarray([], dtype=int)
        features = np.zeros((event_times.size, self.n_neurons), dtype=float)
        for i, event_time in enumerate(event_times):
            in_window = (spike_times >= event_time) & (spike_times <= event_time + duration)
            if np.any(in_window):
                features[i] = np.bincount(
                    neuron_ids[in_window], minlength=self.n_neurons
                ).astype(float)
        row_sums = features.sum(axis=1, keepdims=True)
        features = features / np.clip(row_sums, self.min_rate, None)
        if event_times.size < self.n_templates or np.all(features <= 0):
            return np.arange(event_times.size) % self.n_templates
        centered = features - features.mean(axis=0, keepdims=True)
        try:
            scores = np.linalg.svd(centered, full_matrices=False)[0][:, 0]
        except np.linalg.LinAlgError:
            scores = event_times.copy()
        if np.allclose(scores, scores[0]):
            scores = event_times.copy()
        edges = np.quantile(scores, np.linspace(0, 1, self.n_templates + 1))
        labels = np.searchsorted(edges[1:-1], scores, side="right")
        return np.clip(labels, 0, self.n_templates - 1).astype(int)

    def _run_sequence_updates(
        self,
        data: np.ndarray,
        spike_times: np.ndarray,
        neuron_ids: np.ndarray,
        *,
        num_iter: int,
        fit_templates: bool,
        fit_base_rates: bool,
        tol: float | None,
        patience: int,
        fixed_templates: bool,
    ) -> PPSeqResult:
        if num_iter < 0:
            raise ValueError("num_iter must be nonnegative")
        if fixed_templates:
            events = self._detect_events_for_templates(data)
            if events.empty:
                events = self._events_from_population(data)
        else:
            rows = []
            for i, (time, template) in enumerate(
                zip(self._event_times_, self._event_templates_)
            ):
                start = int(np.floor(float(time) / self.bin_size))
                stop = min(data.shape[1], start + self.template_duration)
                local_mass = float(np.sum(data[:, max(0, start) : stop]))
                rows.append(
                    {
                        "event_id": i,
                        "time": float(time),
                        "template_id": int(template),
                        "amplitude": max(local_mass, 1.0),
                        "duration": self.template_duration_s,
                        "log_probability": 0.0,
                    }
                )
            events = pd.DataFrame(rows)
        log_scores = []
        best_assignments = pd.DataFrame()
        initial_assignments: np.ndarray | None = None
        assignment_hist: list[np.ndarray] = []
        latent_event_hist: list[list[EventSummaryInfo]] = []
        globals_hist: list[SeqGlobals] = []
        stale = 0
        previous = -np.inf
        for _ in range(max(num_iter, 1)):
            assignments = self._assign_spikes(spike_times, neuron_ids, events)
            if initial_assignments is None:
                initial_assignments = self._assignment_array(assignments)
            if fit_templates and not events.empty:
                self._update_templates_from_assignments(assignments, events)
                assignments = self._assign_spikes(spike_times, neuron_ids, events)
            if fit_base_rates:
                self._update_base_rates_from_assignments(assignments, data.shape[1])
            score = self._assignment_score(assignments)
            log_scores.append(score)
            best_assignments = assignments
            assignment_hist.append(self._assignment_array(assignments))
            latent_event_hist.append(self._event_summary(events))
            globals_hist.append(self._seq_globals_snapshot())
            if tol is not None:
                if score - previous < tol:
                    stale += 1
                else:
                    stale = 0
                if stale >= patience:
                    break
                previous = score
        amplitudes = self._amplitudes_from_events(events, data.shape[1])
        rates = self._reconstruct_from_events(events, data.shape[1])
        final_assignments = self._assignment_array(best_assignments)
        if initial_assignments is None:
            initial_assignments = final_assignments.copy()
        assignment_hist_array = (
            np.column_stack(assignment_hist)
            if assignment_hist
            else np.zeros((spike_times.size, 0), dtype=int)
        )
        log_p_hist = np.asarray(log_scores, dtype=float)
        self.seq_model_ = self._seq_model_snapshot(data.shape[1], events)
        return PPSeqResult(
            log_likelihood=np.asarray(log_scores, dtype=float),
            amplitudes=amplitudes,
            rates=rates,
            base_rates=self.base_rates_.copy(),
            events=events.copy(),
            spike_assignments=best_assignments.copy(),
            templates=self.templates.copy(),
            initial_assignments=initial_assignments.copy(),
            final_assignments=final_assignments.copy(),
            assignment_hist=assignment_hist_array,
            latent_event_hist=latent_event_hist,
            globals_hist=globals_hist,
            log_p_hist=log_p_hist.copy(),
            anneal_assignment_hist=np.zeros((spike_times.size, 0), dtype=int),
            anneal_latent_event_hist=[],
            anneal_globals_hist=[],
            anneal_log_p_hist=np.asarray([], dtype=float),
        )

    def _detect_events_for_templates(self, data: np.ndarray) -> pd.DataFrame:
        templates = self.templates
        rows = []
        n_time = data.shape[1]
        for k in range(self.n_templates):
            scores = np.zeros(max(n_time - self.template_duration + 1, 1), dtype=float)
            valid_len = scores.size
            for d in range(self.template_duration):
                if d >= n_time:
                    break
                scores += templates[k, :, d] @ data[:, d : d + valid_len]
            if scores.size == 0:
                continue
            threshold = np.percentile(scores, 90.0)
            min_sep = max(1, self.template_duration // 2)
            order = np.argsort(scores)[::-1]
            selected: list[int] = []
            for idx in order:
                if scores[idx] <= max(threshold, self.min_rate):
                    break
                if all(abs(int(idx) - old) >= min_sep for old in selected):
                    selected.append(int(idx))
                if len(selected) >= max(1, n_time // max(self.template_duration, 1)):
                    break
            for idx in selected:
                rows.append(
                    {
                        "event_id": len(rows),
                        "time": idx * self.bin_size,
                        "template_id": k,
                        "amplitude": float(scores[idx]),
                        "duration": self.template_duration_s,
                        "log_probability": float(np.log(scores[idx] + self.min_rate)),
                    }
                )
        return pd.DataFrame(rows)

    def _events_from_population(self, data: np.ndarray) -> pd.DataFrame:
        event_times = self._candidate_event_times_from_counts(data)
        rows = []
        for i, time in enumerate(event_times):
            rows.append(
                {
                    "event_id": i,
                    "time": float(time),
                    "template_id": int(i % self.n_templates),
                    "amplitude": 1.0,
                    "duration": self.template_duration_s,
                    "log_probability": 0.0,
                }
            )
        return pd.DataFrame(rows)

    def _candidate_event_times_from_counts(self, data: np.ndarray) -> np.ndarray:
        pop = np.asarray(data, dtype=float).sum(axis=0)
        if pop.size == 0 or np.max(pop) <= 0:
            return np.asarray([], dtype=float)
        active_bins = np.flatnonzero(pop > 0)
        gap = max(2, self.template_duration // 5)
        if active_bins.size < 0.5 * pop.size:
            runs = (
                np.split(active_bins, np.flatnonzero(np.diff(active_bins) > gap) + 1)
                if active_bins.size
                else []
            )
            starts = [run[0] for run in runs if run.size >= 2]
            if starts:
                return np.asarray(starts, dtype=float) * self.bin_size
        smoothed = np.convolve(pop, np.ones(3) / 3.0, mode="same")
        threshold = max(np.percentile(smoothed, 75.0), np.mean(smoothed))
        min_sep = max(1, self.template_duration // 2)
        order = np.argsort(smoothed)[::-1]
        selected: list[int] = []
        for idx in order:
            if smoothed[idx] < threshold:
                break
            if all(abs(int(idx) - old) >= min_sep for old in selected):
                selected.append(int(idx))
        selected = sorted(selected)
        return np.asarray(selected, dtype=float) * self.bin_size

    def _initial_event_template_assignments(
        self,
        event_times: np.ndarray,
        *,
        init: str,
        position: np.ndarray | None,
        time_bins: int,
    ) -> np.ndarray:
        if init not in {"random", "peak_time", "lap_direction"}:
            raise ValueError("init must be 'random', 'peak_time', or 'lap_direction'")
        if event_times.size == 0:
            return np.asarray([], dtype=int)
        if init == "random":
            return self._rng.integers(0, self.n_templates, size=event_times.size)
        if init == "lap_direction":
            if position is None:
                raise ValueError("init_position is required for init='lap_direction'")
            position = np.asarray(position, dtype=float)
            if position.shape[0] != time_bins:
                raise ValueError("init_position must have one value per time bin")
            bins = np.clip((event_times / self.bin_size).astype(int), 0, time_bins - 1)
            window = max(1, self.template_duration // 2)
            directions = []
            for b in bins:
                lo = max(0, b - window // 2)
                hi = min(time_bins - 1, b + window // 2)
                directions.append(position[hi] - position[lo])
            directions = np.asarray(directions)
            labels = np.zeros(event_times.size, dtype=int)
            if self.n_templates == 1:
                return labels
            labels[directions < 0] = 1
            if self.n_templates > 2:
                quant = np.quantile(event_times, np.linspace(0, 1, self.n_templates // 2 + 1))
                for i in range(event_times.size):
                    half = 0 if directions[i] >= 0 else self.n_templates // 2
                    sub = np.searchsorted(quant, event_times[i], side="right") - 1
                    labels[i] = min(half + sub, self.n_templates - 1)
            return labels % self.n_templates
        return np.arange(event_times.size) % self.n_templates

    def _assign_spikes(
        self, spike_times: np.ndarray, neuron_ids: np.ndarray, events: pd.DataFrame
    ) -> pd.DataFrame:
        if spike_times.size == 0:
            return pd.DataFrame(
                columns=[
                    "spike_id",
                    "time",
                    "neuron_id",
                    "event_id",
                    "template_id",
                    "prob_sequence",
                    "prob_background",
                    "sequence_rate",
                    "background_rate",
                    "lag",
                    "template_offset",
                ]
            )
        rows = []
        event_records = events.to_dict("records") if not events.empty else []
        for spike_id, (time, neuron) in enumerate(zip(spike_times, neuron_ids)):
            bg = float(self.base_rates_[neuron])
            best = {
                "event_id": -1,
                "template_id": -1,
                "density": 0.0,
                "lag": np.nan,
                "offset": np.nan,
            }
            for event in event_records:
                k = int(event["template_id"])
                lag = time - float(event["time"])
                if lag < -3 * self.bin_size or lag > self.template_duration_s + 3 * self.bin_size:
                    continue
                offset = float(self.template_offsets_[k, neuron])
                width = max(float(self.template_widths_[k, neuron]), self.bin_size)
                scale = float(self.template_scales_[k, neuron])
                density = float(event["amplitude"]) * scale * np.exp(
                    -0.5 * ((lag - offset) / width) ** 2
                ) / (width * np.sqrt(2 * np.pi))
                if density > best["density"]:
                    best = {
                        "event_id": int(event["event_id"]),
                        "template_id": k,
                        "density": density,
                        "lag": lag,
                        "offset": offset,
                    }
            denom = bg + best["density"] + self.min_rate
            prob_sequence = best["density"] / denom
            rows.append(
                {
                    "spike_id": spike_id,
                    "time": float(time),
                    "neuron_id": int(neuron),
                    "event_id": int(best["event_id"]),
                    "template_id": int(best["template_id"]),
                    "prob_sequence": float(prob_sequence),
                    "prob_background": float(1.0 - prob_sequence),
                    "sequence_rate": float(best["density"]),
                    "background_rate": float(bg),
                    "lag": float(best["lag"]) if np.isfinite(best["lag"]) else np.nan,
                    "template_offset": float(best["offset"])
                    if np.isfinite(best["offset"])
                    else np.nan,
                }
            )
        return pd.DataFrame(rows)

    def _update_templates_from_assignments(
        self, assignments: pd.DataFrame, events: pd.DataFrame
    ) -> None:
        if assignments.empty:
            return
        offsets = self.template_offsets_.copy()
        widths = self.template_widths_.copy()
        scales = np.full_like(self.template_scales_, self.min_rate)
        assigned = assignments[assignments["template_id"] >= 0]
        for k in range(self.n_templates):
            subset = assigned[assigned["template_id"] == k]
            for n in range(self.n_neurons):
                local = subset[subset["neuron_id"] == n]
                if len(local):
                    weights = local["prob_sequence"].to_numpy(dtype=float)
                    lags = local["lag"].to_numpy(dtype=float)
                    offsets[k, n] = np.average(lags, weights=weights)
                    var = np.average((lags - offsets[k, n]) ** 2, weights=weights)
                    widths[k, n] = max(np.sqrt(var), self.bin_size)
                    scales[k, n] = float(np.sum(weights))
        self.template_offsets_ = np.clip(offsets, 0.0, self.template_duration_s)
        self.template_widths_ = self._clip_widths(widths)
        self.template_scales_ = self._normalize_scales(scales)
        if not events.empty:
            event_counts = assigned.groupby("event_id")["prob_sequence"].sum()
            for event_id, value in event_counts.items():
                current = float(
                    events.loc[events["event_id"] == event_id, "amplitude"].iloc[0]
                )
                events.loc[events["event_id"] == event_id, "amplitude"] = max(
                    float(value), 0.5 * current, 1.0
                )

    def _update_base_rates_from_assignments(
        self, assignments: pd.DataFrame, n_time_bins: int
    ) -> None:
        duration = max(n_time_bins * self.bin_size, self.bin_size)
        bg_weights = assignments["prob_background"].to_numpy(dtype=float)
        neurons = assignments["neuron_id"].to_numpy(dtype=int)
        counts = np.bincount(neurons, weights=bg_weights, minlength=self.n_neurons)
        self.base_rates_ = np.clip(counts / duration, self.min_rate, None)

    def _assignment_score(self, assignments: pd.DataFrame) -> float:
        if assignments.empty:
            return 0.0
        if {"sequence_rate", "background_rate"}.issubset(assignments.columns):
            rate = (
                assignments["sequence_rate"].to_numpy(dtype=float)
                + assignments["background_rate"].to_numpy(dtype=float)
            )
            return float(np.sum(np.log(np.clip(rate, self.min_rate, None))))
        p = np.clip(assignments["prob_sequence"].to_numpy(dtype=float), self.min_rate, 1)
        q = np.clip(assignments["prob_background"].to_numpy(dtype=float), self.min_rate, 1)
        return float(np.sum(np.log(p + q)))

    def _assignment_array(self, assignments: pd.DataFrame) -> np.ndarray:
        if assignments.empty:
            return np.asarray([], dtype=int)
        return assignments["event_id"].to_numpy(dtype=int).copy()

    def _event_summary(self, events: pd.DataFrame) -> list[EventSummaryInfo]:
        if events.empty:
            return []
        return [
            EventSummaryInfo(
                assignment_id=int(row.event_id),
                timestamp=float(row.time),
                seq_type=int(row.template_id),
                seq_warp=1.0,
                amplitude=float(row.amplitude),
            )
            for row in events.itertuples()
        ]

    def _seq_priors_snapshot(self) -> SeqPriors:
        n_neurons = int(self.n_neurons)
        return SeqPriors(
            seq_event_rate=max(self.min_rate, self.sequence_frac / max(self.template_duration_s, self.bin_size)),
            seq_type_proportions=SymmetricDirichlet(max(self.concentration, self.min_rate), self.n_templates),
            seq_event_amplitude=RateGamma(max(self.alpha_amplitude, self.min_rate), max(self.beta_amplitude, self.min_rate)),
            neuron_response_proportions=SymmetricDirichlet(max(self.concentration, self.min_rate), n_neurons),
            neuron_response_profile=NormalInvChisq(
                k=max(self.alpha_template, self.min_rate),
                m=0.0,
                nu=max(self.alpha_template, self.min_rate),
                s2=max((self.template_duration_s / 12.0) ** 2, self.min_rate),
            ),
            bkgd_amplitude=RateGamma(max(self.alpha_base, self.min_rate), max(self.beta_base, self.min_rate)),
            bkgd_proportions=SymmetricDirichlet(max(self.concentration, self.min_rate), n_neurons),
            warp_values=np.ones(1, dtype=float),
            warp_log_proportions=np.zeros(1, dtype=float),
        )

    def _seq_globals_snapshot(self) -> SeqGlobals:
        self._check_initialized()
        seq_props = np.full(self.n_templates, 1.0 / self.n_templates, dtype=float)
        bkgd_amp = float(np.sum(np.clip(self.base_rates_, self.min_rate, None)))
        bkgd_props = np.clip(self.base_rates_, self.min_rate, None) / max(bkgd_amp, self.min_rate)
        return SeqGlobals(
            seq_type_log_proportions=np.log(np.clip(seq_props, self.min_rate, None)),
            neuron_response_log_proportions=np.log(np.clip(self.template_scales_.T, self.min_rate, None)),
            neuron_response_offsets=self.template_offsets_.T.copy(),
            neuron_response_widths=np.square(self.template_widths_.T.copy()),
            bkgd_amplitude=bkgd_amp,
            bkgd_log_proportions=np.log(np.clip(bkgd_props, self.min_rate, None)),
        )

    def _seq_events_from_summary(self, events: pd.DataFrame) -> SeqEventList:
        event_objs: list[SeqEvent] = []
        indices: list[int] = []
        for summary in self._event_summary(events):
            posterior = np.full((self.n_templates, 1), -np.log(self.n_templates), dtype=float)
            posterior[summary.seq_type, 0] = 0.0
            event_objs.append(
                SeqEvent(
                    spike_count=max(1, int(round(summary.amplitude))),
                    summed_potentials=np.zeros((self.n_templates, 1), dtype=float),
                    summed_precisions=np.ones((self.n_templates, 1), dtype=float),
                    summed_logZ=np.zeros((self.n_templates, 1), dtype=float),
                    seq_type_posterior=posterior,
                    sampled_type=summary.seq_type,
                    sampled_warp=0,
                    sampled_timestamp=summary.timestamp,
                    sampled_amplitude=summary.amplitude,
                )
            )
            indices.append(summary.assignment_id)
        return SeqEventList(event_objs, indices)

    def _seq_model_snapshot(self, n_time_bins: int, events: pd.DataFrame) -> SeqModel:
        priors = self._seq_priors_snapshot()
        globals_ = self._seq_globals_snapshot()
        max_time = max(n_time_bins * self.bin_size, self.bin_size)
        return SeqModel(
            max_time=max_time,
            max_sequence_length=self.template_duration_s,
            priors=priors,
            globals=globals_,
            sequence_events=self._seq_events_from_summary(events),
            new_cluster_log_prob=float(np.log(max(priors.seq_event_rate * max_time, self.min_rate))),
            bkgd_log_prob=float(np.log(max(globals_.bkgd_amplitude * max_time, self.min_rate))),
        )

    def _amplitudes_from_events(self, events: pd.DataFrame, n_time: int) -> np.ndarray:
        amps = np.zeros((self.n_templates, n_time), dtype=float)
        if events.empty:
            return amps
        for event in events.itertuples():
            b = int(round(float(event.time) / self.bin_size))
            if 0 <= b < n_time:
                amps[int(event.template_id), b] += float(event.amplitude)
        return amps

    def _reconstruct_from_events(self, events: pd.DataFrame, n_time: int) -> np.ndarray:
        rates = np.repeat(
            np.clip(self.base_rates_, self.min_rate, None)[:, np.newaxis] * self.bin_size,
            n_time,
            axis=1,
        )
        if events.empty:
            return rates
        templates = self.templates
        for event in events.itertuples():
            k = int(event.template_id)
            start = int(round(float(event.time) / self.bin_size))
            stop = min(n_time, start + templates.shape[2])
            if start < 0 or start >= n_time or stop <= start:
                continue
            rates[:, start:stop] += float(event.amplitude) * templates[k, :, : stop - start]
        return np.clip(rates, self.min_rate, None)

    def _reconstruct_from_amplitudes(self, amplitudes: np.ndarray) -> np.ndarray:
        return self._reconstruct(amplitudes, self.templates, base_rates=self.base_rates_)

    def _infer_with_template_bank(
        self,
        data: np.ndarray,
        templates: np.ndarray,
        *,
        num_iter: int,
        base_rates: np.ndarray,
        tol: float | None,
        patience: int,
    ) -> PPSeqResult:
        self._validate_early_stopping(tol, patience)
        n_templates, _, duration = templates.shape
        n_time = data.shape[1]
        amplitudes = np.zeros((n_templates, n_time), dtype=float)
        scores = []
        stale = 0
        previous = -np.inf
        for _ in range(max(num_iter, 1)):
            rates = self._reconstruct(amplitudes, templates, base_rates=base_rates)
            ratio = data / np.clip(rates, self.min_rate, None)
            drive = np.zeros_like(amplitudes)
            for d in range(duration):
                if d >= n_time:
                    break
                drive[:, : n_time - d] += templates[:, :, d] @ ratio[:, d:]
            amplitudes = np.clip(amplitudes * drive, self.min_amplitude, None)
            if not np.any(amplitudes > self.min_amplitude * 10):
                amplitudes = self._template_bank_matched_filter(data, templates)
            rates = self._reconstruct(amplitudes, templates, base_rates=base_rates)
            ll = float(np.sum(data * np.log(rates) - rates - gammaln(data + 1.0)))
            scores.append(ll)
            if tol is not None:
                if ll - previous < tol:
                    stale += 1
                else:
                    stale = 0
                if stale >= patience:
                    break
                previous = ll
        rates = self._reconstruct(amplitudes, templates, base_rates=base_rates)
        return PPSeqResult(
            log_likelihood=np.asarray(scores, dtype=float),
            amplitudes=amplitudes,
            rates=rates,
            base_rates=base_rates.copy(),
            templates=templates.copy(),
        )

    def _template_bank_matched_filter(
        self, data: np.ndarray, templates: np.ndarray
    ) -> np.ndarray:
        n_templates, _, duration = templates.shape
        n_time = data.shape[1]
        amplitudes = np.zeros((n_templates, n_time), dtype=float)
        for d in range(duration):
            if d >= n_time:
                break
            amplitudes[:, : n_time - d] += templates[:, :, d] @ data[:, d:]
        return np.clip(amplitudes, self.min_amplitude, None)

    def _summarize_replay_slices(
        self,
        data: np.ndarray,
        intervals: np.ndarray,
        result: PPSeqResult,
        replay_labels: list[tuple[int, str, str]],
        *,
        event_slices: list[slice],
    ) -> pd.DataFrame:
        bg = np.clip(result.base_rates, self.min_rate, None)[:, np.newaxis] * self.bin_size
        bg_rates = np.repeat(bg, data.shape[1], axis=1)
        gain_per_bin = (
            data * np.log(np.clip(result.rates, self.min_rate, None) / np.clip(bg_rates, self.min_rate, None))
            - result.rates
            + bg_rates
        ).sum(axis=0)
        rows = []
        for (start, stop), slc in zip(intervals, event_slices):
            amps = result.amplitudes[:, slc]
            row = {
                "start": float(start),
                "stop": float(stop),
                "duration": float(stop - start),
                "n_spikes": int(np.sum(data[:, slc])),
                "likelihood_gain": float(np.sum(gain_per_bin[slc])),
            }
            width = max(slc.stop - slc.start, 1)
            row["likelihood_gain_per_bin"] = row["likelihood_gain"] / width
            row["likelihood_gain_per_spike"] = row["likelihood_gain"] / max(
                row["n_spikes"], 1
            )
            sums = amps.sum(axis=1) if amps.size else np.zeros(len(replay_labels))
            maxes = amps.max(axis=1) if amps.size else np.zeros(len(replay_labels))
            means = amps.mean(axis=1) if amps.size else np.zeros(len(replay_labels))
            best_idx = int(np.argmax(sums)) if sums.size else 0
            second = np.partition(sums, -2)[-2] if sums.size > 1 else 0.0
            for i, (template_id, direction, name) in enumerate(replay_labels):
                prefix = f"template_{template_id}_{direction}"
                row[f"{prefix}_sum"] = float(sums[i])
                row[f"{prefix}_mean"] = float(means[i])
                row[f"{prefix}_max"] = float(maxes[i])
            best_template, best_direction, best_name = replay_labels[best_idx]
            row["best_template"] = int(best_template)
            row["best_direction"] = best_direction
            row["best_replay"] = best_name
            row["best_replay_sum"] = float(sums[best_idx])
            row["best_replay_max"] = float(maxes[best_idx])
            row["best_second_ratio"] = float(sums[best_idx] / max(second, self.min_rate))
            rows.append(row)
        return pd.DataFrame(rows)

    def _make_replay_template_bank(
        self,
        *,
        include_forward: bool,
        include_reverse: bool,
        template_names: list[str] | None,
        replay_duration_s: float | None,
    ) -> tuple[np.ndarray, list[tuple[int, str, str]]]:
        if not include_forward and not include_reverse:
            raise ValueError("At least one of include_forward or include_reverse must be True")
        templates = self.templates
        if replay_duration_s is not None:
            duration = max(1, int(round(replay_duration_s / self.bin_size)))
            templates = self._resample_templates(templates, duration)
        if template_names is None:
            template_names = [f"template_{k}" for k in range(self.n_templates)]
        if len(template_names) != self.n_templates:
            raise ValueError("template_names must match n_templates")
        bank = []
        labels = []
        for k in range(self.n_templates):
            if include_forward:
                bank.append(templates[k])
                labels.append((k, "forward", f"{template_names[k]}_forward"))
            if include_reverse:
                bank.append(templates[k, :, ::-1])
                labels.append((k, "reverse", f"{template_names[k]}_reverse"))
        return np.asarray(bank, dtype=float), labels

    def _resample_templates(self, templates: np.ndarray, duration: int) -> np.ndarray:
        if duration <= 0:
            raise ValueError("duration must be positive")
        old_x = np.linspace(0, 1, templates.shape[2])
        new_x = np.linspace(0, 1, duration)
        out = np.zeros((templates.shape[0], templates.shape[1], duration), dtype=float)
        for k in range(templates.shape[0]):
            for n in range(templates.shape[1]):
                out[k, n] = np.interp(new_x, old_x, templates[k, n])
                mass = templates[k, n].sum()
                new_mass = out[k, n].sum()
                if new_mass > 0:
                    out[k, n] *= mass / new_mass
        return np.clip(out, self.min_rate, None)

    def _reconstruct(
        self,
        amplitudes: np.ndarray,
        templates: np.ndarray,
        *,
        base_rates: np.ndarray,
    ) -> np.ndarray:
        n_time = amplitudes.shape[1]
        rates = np.repeat(
            np.clip(base_rates, self.min_rate, None)[:, np.newaxis] * self.bin_size,
            n_time,
            axis=1,
        )
        for d in range(templates.shape[2]):
            if d >= n_time:
                break
            rates[:, d:] += templates[:, :, d].T @ amplitudes[:, : n_time - d]
        return np.clip(rates, self.min_rate, None)

    def _event_local_data(
        self,
        data: np.ndarray,
        intervals: np.ndarray,
        time_centers: np.ndarray,
        event_pad_s: float,
    ) -> tuple[np.ndarray, np.ndarray, list[slice]]:
        if event_pad_s < 0:
            raise ValueError("event_pad_s must be nonnegative")
        pieces = []
        times = []
        slices = []
        cursor = 0
        for start, stop in intervals:
            lo = int(np.searchsorted(time_centers, start - event_pad_s, side="left"))
            hi = int(np.searchsorted(time_centers, stop + event_pad_s, side="right"))
            lo = max(0, lo)
            hi = min(data.shape[1], hi)
            summary_lo = int(np.searchsorted(time_centers[lo:hi], start, side="left"))
            summary_hi = int(np.searchsorted(time_centers[lo:hi], stop, side="right"))
            pieces.append(data[:, lo:hi])
            times.append(time_centers[lo:hi])
            slices.append(slice(cursor + summary_lo, cursor + summary_hi))
            cursor += hi - lo
        if pieces:
            return np.concatenate(pieces, axis=1), np.concatenate(times), slices
        return data[:, :0], time_centers[:0], slices

    def _resolve_base_rates(self, data: np.ndarray, base_rates: str | np.ndarray) -> np.ndarray:
        if isinstance(base_rates, str):
            if base_rates == "model":
                self._check_initialized()
                return np.clip(self.base_rates_, self.min_rate, None)
            if base_rates == "data":
                duration = max(data.shape[1] * self.bin_size, self.bin_size)
                return np.clip(data.sum(axis=1) / duration, self.min_rate, None)
            raise ValueError("base_rates must be 'model', 'data', or an array")
        base_rates = np.asarray(base_rates, dtype=float)
        if base_rates.shape != (self.n_neurons,):
            raise ValueError("base_rates must have shape (n_neurons,)")
        return np.clip(base_rates, self.min_rate, None)

    def _counts_to_spikes(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        times = []
        neurons = []
        for n in range(data.shape[0]):
            bins = np.flatnonzero(data[n] > 0)
            for b in bins:
                count = int(round(data[n, b]))
                if count <= 0:
                    continue
                times.extend([b * self.bin_size + 0.5 * self.bin_size] * count)
                neurons.extend([n] * count)
        if not times:
            return np.asarray([], dtype=float), np.asarray([], dtype=int)
        order = np.argsort(times)
        return np.asarray(times, dtype=float)[order], np.asarray(neurons, dtype=int)[order]

    def _spikes_to_counts(
        self, spike_times: np.ndarray, neuron_ids: np.ndarray, n_time: int
    ) -> np.ndarray:
        counts = np.zeros((self.n_neurons, n_time), dtype=float)
        bins = np.floor(spike_times / self.bin_size).astype(int)
        valid = (bins >= 0) & (bins < n_time)
        np.add.at(counts, (neuron_ids[valid], bins[valid]), 1.0)
        return counts

    def _validate_data(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError("data must have shape (n_neurons, n_time_bins)")
        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError("data must have at least one neuron and one time bin")
        if np.any(data < 0):
            raise ValueError("data must contain nonnegative counts")
        if self.n_neurons is None:
            self.n_neurons = data.shape[0]
        elif data.shape[0] != self.n_neurons:
            raise ValueError("data neuron count does not match n_neurons")
        return data

    def _validate_intervals(self, event_intervals: np.ndarray) -> np.ndarray:
        event_intervals = np.asarray(event_intervals, dtype=float)
        if event_intervals.ndim != 2 or event_intervals.shape[1] != 2:
            raise ValueError("event_intervals must have shape (n_events, 2)")
        return event_intervals

    def _coerce_input_data(
        self,
        data: np.ndarray,
        *,
        neuron_ids: np.ndarray | None,
        start: float | None,
        stop: float | None,
    ) -> np.ndarray:
        data_array = np.asarray(data)
        if neuron_ids is None:
            return np.asarray(data_array, dtype=float)
        counts, _ = bin_spikes(
            np.asarray(data_array, dtype=float),
            np.asarray(neuron_ids, dtype=int),
            start=start,
            stop=stop,
            bin_size=self.bin_size,
            n_neurons=self.n_neurons,
        )
        return counts

    def _coerce_input_data_and_spikes(
        self,
        data: np.ndarray,
        *,
        neuron_ids: np.ndarray | None,
        start: float | None,
        stop: float | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data_array = np.asarray(data)
        if neuron_ids is None:
            counts = np.asarray(data_array, dtype=float)
            spike_times, spike_neurons = self._counts_to_spikes(counts)
            return counts, spike_times, spike_neurons

        spike_times = np.asarray(data_array, dtype=float)
        spike_neurons = np.asarray(neuron_ids, dtype=int)
        if spike_times.shape[0] != spike_neurons.shape[0]:
            raise ValueError("spike_times and neuron_ids must have the same length")
        if start is None:
            start = float(np.min(spike_times)) if spike_times.size else 0.0
        if stop is None:
            stop = (
                float(np.max(spike_times) + self.bin_size)
                if spike_times.size
                else start + self.bin_size
            )
        counts, _ = bin_spikes(
            spike_times,
            spike_neurons,
            start=start,
            stop=stop,
            bin_size=self.bin_size,
            n_neurons=self.n_neurons,
        )
        valid = (spike_times >= start) & (spike_times < stop)
        rel_times = spike_times[valid] - float(start)
        rel_neurons = spike_neurons[valid]
        order = np.lexsort((rel_neurons, rel_times))
        return counts, rel_times[order], rel_neurons[order]

    def _validate_early_stopping(self, tol: float | None, patience: int) -> None:
        if patience < 1:
            raise ValueError("patience must be at least 1")
        if tol is not None and tol < 0:
            raise ValueError("tol must be nonnegative or None")

    def _clip_widths(self, widths: np.ndarray) -> np.ndarray:
        widths = np.clip(widths, max(self.min_width * self.bin_size, self.bin_size / 2), None)
        if self.min_width_bins is not None:
            widths = np.clip(widths, self.min_width_bins * self.bin_size, None)
        if self.max_width_bins is not None:
            widths = np.clip(widths, None, self.max_width_bins * self.bin_size)
        if self.shared_width:
            shared = np.nanmedian(widths, axis=1, keepdims=True)
            widths = np.repeat(shared, widths.shape[1], axis=1)
        return widths

    def _normalize_scales(self, scales: np.ndarray) -> np.ndarray:
        scales = np.clip(scales, self.min_rate, None)
        denom = scales.sum(axis=1, keepdims=True)
        return scales / np.clip(denom, self.min_rate, None)

    def _templates_from_parameters(
        self,
        offsets: np.ndarray,
        widths: np.ndarray,
        scales: np.ndarray,
        duration: int,
    ) -> np.ndarray:
        bins = np.arange(duration, dtype=float) * self.bin_size + 0.5 * self.bin_size
        z = np.exp(
            -0.5
            * ((bins[np.newaxis, np.newaxis, :] - offsets[:, :, np.newaxis])
               / np.clip(widths[:, :, np.newaxis], self.bin_size / 2, None))
            ** 2
        )
        z = z / np.clip(z.sum(axis=2, keepdims=True), self.min_rate, None)
        return np.clip(scales[:, :, np.newaxis] * z, self.min_rate, None)

    def _check_initialized(self) -> None:
        required = ("template_offsets_", "template_widths_", "template_scales_", "base_rates_")
        if not all(hasattr(self, name) for name in required):
            raise ValueError("PPSeq must be fit before this method is used")


def bin_spikes(
    spike_times: np.ndarray | nel.SpikeTrainArray,
    neuron_ids: np.ndarray | None = None,
    *,
    start: float | None = None,
    stop: float | None = None,
    bin_size: float = 0.02,
    n_neurons: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bin spikes into counts using seconds/bin semantics.

    Parameters
    ----------
    spike_times : np.ndarray or nel.SpikeTrainArray
        Spike timestamps in seconds, or a nelpy spike train array.
    neuron_ids : np.ndarray, optional
        Neuron ID per spike for array inputs.
    start, stop : float, optional
        Time bounds in seconds.
    bin_size : float, optional
        Bin size in seconds.
    n_neurons : int, optional
        Number of neurons for array inputs.

    Returns
    -------
    counts : np.ndarray
        Count matrix with shape ``(n_neurons, n_time_bins)``.
    time_centers : np.ndarray
        Bin centers in seconds.
    """

    if isinstance(spike_times, nel.SpikeTrainArray):
        if start is None or stop is None:
            raise ValueError("start and stop must both be provided for nelpy slicing")
        bins = np.arange(start, stop + bin_size, bin_size)
        counts = count_in_interval(spike_times.data, bins[:-1], bins[1:])
        return np.asarray(counts, dtype=float), bins[:-1] + 0.5 * bin_size
    spike_times = np.asarray(spike_times, dtype=float)
    if neuron_ids is None:
        raise ValueError("neuron_ids is required when spike_times is an array")
    neuron_ids = np.asarray(neuron_ids, dtype=int)
    if spike_times.shape[0] != neuron_ids.shape[0]:
        raise ValueError("spike_times and neuron_ids must have the same length")
    if start is None:
        start = float(np.min(spike_times)) if spike_times.size else 0.0
    if stop is None:
        stop = float(np.max(spike_times) + bin_size) if spike_times.size else start + bin_size
    if stop <= start:
        raise ValueError("stop must be greater than start")
    if n_neurons is None:
        if neuron_ids.size == 0:
            raise ValueError("n_neurons is required for empty spike arrays")
        n_neurons = int(neuron_ids.max()) + 1
    edges = np.arange(start, stop + bin_size, bin_size)
    counts = np.zeros((n_neurons, len(edges) - 1), dtype=float)
    valid = (spike_times >= start) & (spike_times < stop)
    bins = np.searchsorted(edges, spike_times[valid], side="right") - 1
    np.add.at(counts, (neuron_ids[valid], bins), 1.0)
    return counts, edges[:-1] + 0.5 * bin_size


def detect_ppseq_events(
    amplitudes: np.ndarray,
    time_centers: np.ndarray,
    *,
    percentile: float = 90.0,
    min_separation_s: float | None = None,
) -> pd.DataFrame:
    """
    Detect sequence events from template activity.

    Parameters
    ----------
    amplitudes : np.ndarray
        Template activity with shape ``(n_templates, n_time_bins)``.
    time_centers : np.ndarray
        Bin centers in seconds.
    percentile : float, optional
        Per-template threshold percentile.
    min_separation_s : float, optional
        Minimum event separation in seconds.

    Returns
    -------
    pandas.DataFrame
        Event table with ``time``, ``template_id``, and ``amplitude`` columns.
    """

    amplitudes = np.asarray(amplitudes, dtype=float)
    time_centers = np.asarray(time_centers, dtype=float)
    if amplitudes.ndim != 2:
        raise ValueError("amplitudes must have shape (n_templates, n_time_bins)")
    if time_centers.shape[0] != amplitudes.shape[1]:
        raise ValueError("time_centers must match amplitude time bins")
    if min_separation_s is None:
        min_separation_s = float(np.median(np.diff(time_centers))) if len(time_centers) > 1 else 0.0
    min_sep_bins = max(1, int(round(min_separation_s / max(np.median(np.diff(time_centers)), 1e-12)))) if len(time_centers) > 1 else 1
    rows = []
    for k in range(amplitudes.shape[0]):
        threshold = np.percentile(amplitudes[k], percentile)
        selected: list[int] = []
        for idx in np.argsort(amplitudes[k])[::-1]:
            if amplitudes[k, idx] <= threshold:
                break
            if all(abs(int(idx) - old) >= min_sep_bins for old in selected):
                selected.append(int(idx))
                rows.append(
                    {
                        "time": float(time_centers[idx]),
                        "template_id": k,
                        "amplitude": float(amplitudes[k, idx]),
                    }
                )
    return pd.DataFrame(rows).sort_values("time", ignore_index=True) if rows else pd.DataFrame(columns=["time", "template_id", "amplitude"])


def ppseq_spike_responsibilities(
    model: PPSeq,
    data: np.ndarray,
    result: PPSeqResult | None = None,
    *,
    responsibility_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Return spike-level PPSeq sequence/background responsibilities.

    Parameters
    ----------
    model : PPSeq
        Fitted model.
    data : np.ndarray
        Spike count matrix.
    result : PPSeqResult, optional
        Existing result. If omitted, data is transformed.
    responsibility_threshold : float, optional
        Threshold used to create ``assigned_template_id``.

    Returns
    -------
    pandas.DataFrame
        Spike assignment table.
    """

    if result is None:
        result = model.transform(data, num_iter=10)
    assignments = result.spike_assignments.copy()
    if assignments.empty:
        return assignments
    assignments["assigned_template_id"] = np.where(
        assignments["prob_sequence"] >= responsibility_threshold,
        assignments["template_id"],
        -1,
    )
    return assignments


def load_ppseq_songbird_spikes(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the two-column songbird spike file used by the PPSeq.jl demo.

    Parameters
    ----------
    path : str or pathlib.Path
        Text file with neuron ids in the first column and spike times in the
        second column.

    Returns
    -------
    spike_times : np.ndarray
        Spike times in seconds.
    neuron_ids : np.ndarray
        Zero-based neuron ids.
    """

    arr = np.loadtxt(Path(path), dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("songbird spike file must have at least two columns")
    raw_neurons = arr[:, 0].astype(int)
    neuron_ids = raw_neurons - raw_neurons.min()
    spike_times = arr[:, 1].astype(float)
    order = np.argsort(spike_times)
    return spike_times[order], neuron_ids[order]


def ppseq_assignment_raster(
    result: PPSeqResult,
    *,
    threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Return spike assignments formatted for raster plotting.

    Parameters
    ----------
    result : PPSeqResult
        Fitted or transformed PPSeq result with spike assignments.
    threshold : float, optional
        Minimum sequence probability for assigning a spike to a template.

    Returns
    -------
    pandas.DataFrame
        Columns include ``time``, ``neuron_id``, ``template_id``,
        ``prob_sequence``, and ``assigned_template_id``.
    """

    if result.spike_assignments is None or result.spike_assignments.empty:
        return pd.DataFrame(
            columns=[
                "time",
                "neuron_id",
                "template_id",
                "prob_sequence",
                "assigned_template_id",
            ]
        )
    raster = result.spike_assignments.copy()
    raster["assigned_template_id"] = np.where(
        raster["prob_sequence"].to_numpy(dtype=float) >= threshold,
        raster["template_id"].to_numpy(dtype=int),
        -1,
    )
    return raster


def plot_ppseq_raster(
    spike_times: np.ndarray,
    neuron_ids: np.ndarray,
    assignments: pd.DataFrame | None = None,
    *,
    sort_order: np.ndarray | None = None,
    ax=None,
    colors: dict[int, str] | list[str] | None = None,
):
    """
    Plot a PPSeq-style spike raster.

    Parameters
    ----------
    spike_times, neuron_ids : np.ndarray
        Spike times and zero-based neuron ids.
    assignments : pandas.DataFrame, optional
        Assignment table from :func:`ppseq_assignment_raster`.
    sort_order : np.ndarray, optional
        Neuron id ordering for the y-axis.
    ax : matplotlib.axes.Axes, optional
        Existing axes.
    colors : dict or list, optional
        Template colors.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the raster.
    """

    import matplotlib.pyplot as plt

    spike_times = np.asarray(spike_times, dtype=float)
    neuron_ids = np.asarray(neuron_ids, dtype=int)
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3))
    if sort_order is None:
        sort_order = np.arange(int(neuron_ids.max()) + 1 if neuron_ids.size else 0)
    sort_order = np.asarray(sort_order, dtype=int)
    max_id = int(max(neuron_ids.max(initial=0), sort_order.max(initial=0)))
    sort_rank = np.full(max_id + 1, -1, dtype=int)
    sort_rank[sort_order] = np.arange(len(sort_order))
    valid = (neuron_ids >= 0) & (neuron_ids < len(sort_rank)) & (sort_rank[neuron_ids] >= 0)
    x = spike_times[valid]
    y = sort_rank[neuron_ids[valid]]
    assigned_all = np.full(spike_times.shape, -1, dtype=int)
    if assignments is not None and not assignments.empty:
        assigned_values = assignments["assigned_template_id"].to_numpy(dtype=int)
        if "spike_id" in assignments.columns:
            spike_ids = assignments["spike_id"].to_numpy(dtype=int)
            ok = (spike_ids >= 0) & (spike_ids < assigned_all.size)
            assigned_all[spike_ids[ok]] = assigned_values[ok]
        elif len(assignments) == spike_times.size:
            assigned_all = assigned_values.copy()
        else:
            key = pd.DataFrame(
                {
                    "plot_index": np.arange(spike_times.size),
                    "time_key": np.round(spike_times, 9),
                    "neuron_id": neuron_ids,
                }
            )
            assign_key = assignments.copy()
            assign_key["time_key"] = np.round(assign_key["time"].to_numpy(dtype=float), 9)
            merged = key.merge(
                assign_key[["time_key", "neuron_id", "assigned_template_id"]],
                on=["time_key", "neuron_id"],
                how="left",
            )
            hit = merged["assigned_template_id"].notna().to_numpy()
            assigned_all[merged.loc[hit, "plot_index"].to_numpy(dtype=int)] = merged.loc[
                hit, "assigned_template_id"
            ].to_numpy(dtype=int)
    assigned = assigned_all[valid]
    ax.scatter(x[assigned < 0], y[assigned < 0], s=5, c="k", alpha=0.35, linewidths=0)
    template_ids = sorted(int(k) for k in np.unique(assigned) if k >= 0)
    cmap = plt.get_cmap("tab10")
    for template_id in template_ids:
        idx = assigned == template_id
        if isinstance(colors, dict):
            color = colors.get(template_id, cmap(template_id % 10))
        elif isinstance(colors, list):
            color = colors[template_id % len(colors)]
        else:
            color = cmap(template_id % 10)
        ax.scatter(x[idx], y[idx], s=8, c=[color], alpha=0.95, linewidths=0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("neurons")
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_ppseq_diagnostics(result: PPSeqResult, *, ax=None):
    """
    Plot PPSeq log-probability and latent event count traces.

    Parameters
    ----------
    result : PPSeqResult
        Fitted PPSeq result.
    ax : array-like of matplotlib.axes.Axes, optional
        Two axes. If omitted, creates a new figure.

    Returns
    -------
    np.ndarray
        Axes containing the diagnostics.
    """

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 2, figsize=(8, 2.7))
    ax = np.asarray(ax, dtype=object).ravel()
    if ax.size < 2:
        raise ValueError("ax must contain at least two matplotlib axes")
    log_p = result.log_p_hist if result.log_p_hist is not None else result.log_likelihood
    ax[0].plot(np.arange(len(log_p)), log_p, color="0.2", lw=1.5)
    ax[0].set_title("log probability")
    ax[0].set_xlabel("saved sample")
    counts = [
        len(events)
        for events in (result.latent_event_hist if result.latent_event_hist is not None else [])
    ]
    ax[1].plot(np.arange(len(counts)), counts, color="0.2", lw=1.5)
    ax[1].set_title("latent events")
    ax[1].set_xlabel("saved sample")
    for axis in ax:
        axis.spines[["top", "right"]].set_visible(False)
    return ax


def summarize_ppseq_replay_scores(
    ripple_scores: pd.DataFrame,
    *,
    pre_label: str = "pre",
    post_label: str = "post",
    metric: str = "replay_z",
) -> pd.DataFrame:
    """
    Summarize pre/post PPSeq replay scores.

    Parameters
    ----------
    ripple_scores : pandas.DataFrame
        Replay score table with a ``sleep_epoch`` column.
    pre_label, post_label : str, optional
        Labels for pre- and post-task sleep.
    metric : str, optional
        Primary metric for replay-like thresholding.

    Returns
    -------
    pandas.DataFrame
        Per-epoch summary table.
    """

    columns = [
        "n_ripples",
        f"median_{metric}",
        "median_likelihood_gain_per_spike",
        "fraction_replay_like",
        "n_replay_like",
    ]
    if ripple_scores is None or ripple_scores.empty or "sleep_epoch" not in ripple_scores:
        return pd.DataFrame(index=[pre_label, post_label], columns=columns, dtype=float)
    scores = ripple_scores.copy()
    if metric not in scores:
        metric = "likelihood_gain_per_spike"
    pre_values = scores.loc[scores["sleep_epoch"] == pre_label, metric].dropna()
    threshold = pre_values.quantile(0.95) if not pre_values.empty else np.inf
    scores["is_replay_like"] = scores[metric] > threshold
    if "replay_p" in scores:
        scores["is_replay_like"] |= scores["replay_p"] < 0.05
    summary = (
        scores.groupby("sleep_epoch")
        .agg(
            n_ripples=("start", "size"),
            median_metric=(metric, "median"),
            median_likelihood_gain_per_spike=("likelihood_gain_per_spike", "median"),
            fraction_replay_like=("is_replay_like", "mean"),
            n_replay_like=("is_replay_like", "sum"),
        )
        .reindex([pre_label, post_label])
    )
    summary = summary.rename(columns={"median_metric": f"median_{metric}"})
    return summary


def select_ppseq_running_subset(
    counts: np.ndarray,
    time_centers: np.ndarray,
    position: np.ndarray | pd.Series | pd.DataFrame | None,
    *,
    max_bins: int = 3000,
    template_duration_bins: int | None = None,
) -> np.ndarray:
    """
    Select a compact, position-balanced running subset for PPSeq fitting.
    """

    counts = np.asarray(counts, dtype=float)
    if counts.shape[1] <= max_bins:
        return np.arange(counts.shape[1])
    if position is None:
        pop = counts.sum(axis=0)
        order = np.argsort(pop)[::-1]
        return np.sort(order[:max_bins])
    if isinstance(position, pd.DataFrame):
        col = "x" if "x" in position.columns else "linearized"
        pos = position[col].to_numpy(dtype=float)
    else:
        pos = np.asarray(position, dtype=float)
    if pos.shape[0] != counts.shape[1] or np.count_nonzero(np.isfinite(pos)) < 10:
        return np.linspace(0, counts.shape[1] - 1, max_bins, dtype=int)
    smooth = pd.Series(pos).interpolate(limit_direction="both").rolling(
        9, center=True, min_periods=1
    ).median().to_numpy()
    dx = np.gradient(smooth)
    moving = np.abs(dx) >= np.nanpercentile(np.abs(dx[np.isfinite(dx)]), 35)
    selected = []
    per_direction = max_bins // 2
    for sign in (1, -1):
        idx = np.flatnonzero(moving & (np.sign(dx) == sign))
        if idx.size:
            take = idx[np.linspace(0, idx.size - 1, min(per_direction, idx.size), dtype=int)]
            selected.append(take)
    if not selected:
        return np.linspace(0, counts.shape[1] - 1, max_bins, dtype=int)
    out = np.unique(np.concatenate(selected))
    if template_duration_bins is not None and out.size < template_duration_bins:
        return np.arange(min(counts.shape[1], max(max_bins, template_duration_bins)))
    return np.sort(out[:max_bins])


def fit_ppseq_behavior_model(
    counts: np.ndarray,
    time_centers: np.ndarray,
    position: np.ndarray | pd.Series | pd.DataFrame | None = None,
    *,
    candidate_grid: dict | None = None,
    random_state: int = 0,
) -> tuple[PPSeq, PPSeqResult, pd.DataFrame, PPSeqBehaviorDiagnostics]:
    """
    Fit a compact behavior PPSeq model and return diagnostics.
    """

    counts = np.asarray(counts, dtype=float)
    time_centers = np.asarray(time_centers, dtype=float)
    grid = {
        "n_templates_grid": [2],
        "template_duration_s_grid": [1.0, 1.5, 2.0],
        "seeds": [random_state],
        "fit_num_iter": 20,
        "transform_num_iter": 5,
        "sequence_frac": 0.8,
    }
    if candidate_grid:
        grid.update(candidate_grid)
    bin_size = float(np.median(np.diff(time_centers))) if time_centers.size > 1 else 0.02
    pos = None
    if isinstance(position, pd.DataFrame):
        col = "x" if "x" in position.columns else "linearized"
        pos = position[col].to_numpy(dtype=float)
    elif position is not None:
        pos = np.asarray(position, dtype=float)
    model, result, candidates = fit_ppseq_candidates(
        counts,
        time_centers,
        bin_size=bin_size,
        position=pos if pos is not None and pos.shape[0] == counts.shape[1] else None,
        init_grid=["lap_direction", "peak_time"] if pos is not None else ["peak_time"],
        **grid,
    )
    diagnostics = ppseq_behavior_diagnostics(
        model,
        counts,
        time_centers,
        position=pos if pos is not None and pos.shape[0] == counts.shape[1] else None,
        result=result,
        min_events=2 * model.n_templates,
        min_spike_fraction=0.20,
        min_participating_neurons=0.20,
        min_direction_purity=0.75 if pos is not None else 0.0,
    )
    return model, result, candidates, diagnostics


def plot_ppseq_behavior_with_position(
    model: PPSeq,
    result: PPSeqResult,
    st,
    epoch,
    position: pd.DataFrame,
    *,
    window_s: float = 180,
    threshold: float = 0.3,
):
    """
    Plot a task raster colored by PPSeq assignment with position below.
    """

    import matplotlib.pyplot as plt
    from neuro_py.spikes import spike_tools

    spike_df = spike_tools.get_spindices(st[epoch].data)
    spike_times = spike_df.spike_times.to_numpy()
    neuron_ids = spike_df.spike_id.to_numpy(dtype=int)
    if spike_times.size:
        start = float(spike_times.min())
        stop = min(float(spike_times.max()), start + window_s)
        keep = (spike_times >= start) & (spike_times <= stop)
        spike_times = spike_times[keep]
        neuron_ids = neuron_ids[keep]
    assignments = ppseq_assignment_raster(result, threshold=threshold)
    sort_order = np.argsort(model.template_offsets_[0]) if hasattr(model, "template_offsets_") else None
    fig, axes = plt.subplots(
        2, 1, figsize=(11, 4.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True
    )
    plot_ppseq_raster(spike_times, neuron_ids, assignments, sort_order=sort_order, ax=axes[0])
    pos_col = "x" if "x" in position.columns else "linearized"
    time_col = next((c for c in ("time", "timestamps", "timestamp", "t") if c in position.columns), None)
    pos_time = position[time_col].to_numpy(dtype=float) if time_col else position.index.to_numpy(dtype=float)
    mask = (pos_time >= axes[0].get_xlim()[0]) & (pos_time <= axes[0].get_xlim()[1])
    axes[1].plot(pos_time[mask], position[pos_col].to_numpy(dtype=float)[mask], color="0.2", lw=1.2)
    axes[1].set_ylabel(pos_col)
    axes[1].set_xlabel("time (s)")
    axes[1].spines[["top", "right"]].set_visible(False)
    return fig, axes


def score_ppseq_pre_post_ripples(
    model: PPSeq,
    pre_counts: np.ndarray,
    post_counts: np.ndarray,
    pre_intervals: np.ndarray,
    post_intervals: np.ndarray,
    pre_time_centers: np.ndarray,
    post_time_centers: np.ndarray,
    *,
    replay_duration_s: float = 0.30,
    n_null: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Score pre/post ripple replay with fixed task PPSeq templates.
    """

    if len(pre_intervals) == 0 or len(post_intervals) == 0:
        empty = pd.DataFrame()
        return empty, empty, summarize_ppseq_replay_scores(empty)
    pre_scores, pre_nulls = score_ppseq_replay_nulls(
        model,
        pre_counts,
        pre_intervals,
        pre_time_centers,
        n_null=n_null,
        random_state=0,
        null_method="fast",
        method="event",
        base_rates="data",
        replay_duration_s=replay_duration_s,
        event_pad_s=replay_duration_s,
        num_iter=10,
    )
    post_scores, post_nulls = score_ppseq_replay_nulls(
        model,
        post_counts,
        post_intervals,
        post_time_centers,
        n_null=n_null,
        random_state=1,
        null_method="fast",
        method="event",
        base_rates="data",
        replay_duration_s=replay_duration_s,
        event_pad_s=replay_duration_s,
        num_iter=10,
    )
    pre_scores["sleep_epoch"] = "pre"
    post_scores["sleep_epoch"] = "post"
    pre_nulls["sleep_epoch"] = "pre"
    post_nulls["sleep_epoch"] = "post"
    scores = pd.concat([pre_scores, post_scores], ignore_index=True)
    nulls = pd.concat([pre_nulls, post_nulls], ignore_index=True)
    return scores, nulls, summarize_ppseq_replay_scores(scores)


def ppseq_behavior_diagnostics(
    model: PPSeq,
    data: np.ndarray,
    time_centers: np.ndarray,
    *,
    position: np.ndarray | None = None,
    result: PPSeqResult | None = None,
    event_percentile: float = 90.0,
    responsibility_threshold: float = 0.3,
    min_events: int = 20,
    min_spike_fraction: float = 0.3,
    min_participating_neurons: int | float = 0.3,
    min_direction_purity: float = 0.75,
    direction_window_s: float | None = None,
    min_event_displacement: float = 1e-6,
) -> PPSeqBehaviorDiagnostics:
    """
    Compute behavioral quality diagnostics for PPSeq templates.

    Parameters
    ----------
    model : PPSeq
        Fitted PPSeq model.
    data : np.ndarray
        Spike count matrix.
    time_centers : np.ndarray
        Bin centers in seconds.
    position : np.ndarray, optional
        Linear-track coordinate per time bin.
    result : PPSeqResult, optional
        Existing PPSeq result.
    event_percentile : float, optional
        Event threshold percentile for compatibility.
    responsibility_threshold : float, optional
        Spike assignment threshold.
    min_events : int, optional
        Minimum total sequence events.
    min_spike_fraction : float, optional
        Minimum fraction of spikes assigned to sequences.
    min_participating_neurons : int or float, optional
        Minimum participating neurons or fraction of cells.
    min_direction_purity : float, optional
        Minimum direction purity.
    direction_window_s : float, optional
        Window around each event for displacement direction.
    min_event_displacement : float, optional
        Ignore events with smaller absolute displacement.

    Returns
    -------
    PPSeqBehaviorDiagnostics
        Diagnostics tables and pass flag.
    """

    data = np.asarray(data, dtype=float)
    time_centers = np.asarray(time_centers, dtype=float)
    if result is None:
        result = model.transform(data, num_iter=10)
    events = result.events.copy() if result.events is not None else pd.DataFrame()
    assignments = (
        result.spike_assignments.copy()
        if result.spike_assignments is not None
        else pd.DataFrame()
    )
    if assignments.empty:
        assigned = assignments
        spike_fraction = 0.0
    else:
        assigned = assignments[
            (assignments["template_id"] >= 0)
            & (assignments["prob_sequence"] >= responsibility_threshold)
        ]
        spike_fraction = float(len(assigned) / max(len(assignments), 1))
    if isinstance(min_participating_neurons, float) and min_participating_neurons < 1:
        min_cells = int(np.ceil(min_participating_neurons * model.n_neurons))
    else:
        min_cells = int(min_participating_neurons)
    position = None if position is None else np.asarray(position, dtype=float)
    if position is not None and position.shape[0] != time_centers.shape[0]:
        raise ValueError("position must have one value per time bin")
    if direction_window_s is None:
        direction_window_s = model.template_duration_s / 2.0
    per_rows = []
    group_counts = np.zeros(2, dtype=int)
    for k in range(model.n_templates):
        local_assign = assigned[assigned["template_id"] == k] if not assigned.empty else assigned
        local_events = events[events["template_id"] == k] if not events.empty else events
        participating = int(local_assign["neuron_id"].nunique()) if len(local_assign) else 0
        increasing = 0
        decreasing = 0
        if position is not None and not local_events.empty:
            for event_time in local_events["time"].to_numpy(dtype=float):
                lo = np.searchsorted(time_centers, event_time - direction_window_s / 2, side="left")
                hi = np.searchsorted(time_centers, event_time + direction_window_s / 2, side="right") - 1
                lo = max(0, lo)
                hi = min(len(position) - 1, hi)
                disp = position[hi] - position[lo]
                if abs(disp) < min_event_displacement:
                    continue
                if disp >= 0:
                    increasing += 1
                else:
                    decreasing += 1
        total_dir = increasing + decreasing
        purity = max(increasing, decreasing) / total_dir if total_dir else 0.0
        group_counts[0 if increasing >= decreasing else 1] += len(local_events)
        offset_position_corr = np.nan
        if position is not None and len(local_assign):
            peaks = np.full(model.n_neurons, np.nan)
            for n in range(model.n_neurons):
                neuron_spikes = local_assign[local_assign["neuron_id"] == n]
                if len(neuron_spikes):
                    bins = np.clip(
                        np.searchsorted(time_centers, neuron_spikes["time"].to_numpy(dtype=float)),
                        0,
                        len(position) - 1,
                    )
                    peaks[n] = np.nanmedian(position[bins])
            valid = np.isfinite(peaks)
            if np.count_nonzero(valid) >= 3:
                offset_position_corr = float(
                    spearmanr(model.template_offsets_[k, valid], peaks[valid]).statistic
                )
        per_rows.append(
            {
                "template_id": k,
                "n_events": int(len(local_events)),
                "n_assigned_spikes": int(len(local_assign)),
                "n_participating_neurons": participating,
                "direction_purity": float(purity),
                "increasing_events": int(increasing),
                "decreasing_events": int(decreasing),
                "offset_position_corr": offset_position_corr,
            }
        )
    per_template = pd.DataFrame(per_rows)
    n_templates_with_events = int(np.sum(per_template["n_events"].to_numpy() > 0))
    min_purity = float(per_template["direction_purity"].min()) if len(per_template) else 0.0
    grouped_direction_purity = float(group_counts.max() / group_counts.sum()) if group_counts.sum() else 0.0
    balance = float(group_counts.min() / group_counts.max()) if group_counts.max() else 0.0
    failure_reasons = []
    if len(events) < min_events:
        failure_reasons.append("too_few_events")
    if spike_fraction < min_spike_fraction:
        failure_reasons.append("low_assigned_spike_fraction")
    if len(per_template) and per_template["n_participating_neurons"].min() < min_cells:
        failure_reasons.append("low_participation")
    if n_templates_with_events < min(2, model.n_templates):
        failure_reasons.append("too_few_templates_with_events")
    direction_pass = min_purity >= min_direction_purity
    if model.n_templates >= 4:
        direction_pass = direction_pass or grouped_direction_purity >= min_direction_purity
    if position is not None and not direction_pass:
        failure_reasons.append("low_direction_purity")
    passed = len(failure_reasons) == 0
    summary = pd.DataFrame(
        [
            {
                "passed": passed,
                "n_events": int(len(events)),
                "assigned_spike_fraction": spike_fraction,
                "min_participating_neurons": int(per_template["n_participating_neurons"].min())
                if len(per_template)
                else 0,
                "min_direction_purity": min_purity,
                "grouped_direction_purity": grouped_direction_purity,
                "n_templates_with_events": n_templates_with_events,
                "template_direction_balance": balance,
                "failure_reasons": ",".join(failure_reasons),
                "event_threshold_percentile": float(event_percentile),
                "responsibility_threshold": float(responsibility_threshold),
            }
        ]
    )
    return PPSeqBehaviorDiagnostics(summary, per_template, events, passed)


def fit_ppseq_candidates(
    data: np.ndarray,
    time_centers: np.ndarray,
    *,
    n_templates_grid: list[int],
    template_duration_s_grid: list[float],
    bin_size: float,
    seeds: list[int],
    init_grid: list[str] | None = None,
    position: np.ndarray | None = None,
    fit_num_iter: int = 25,
    transform_num_iter: int = 10,
    holdout_fraction: float = 0.25,
    min_events: int = 20,
    min_spike_fraction: float = 0.20,
    min_direction_purity: float = 0.70,
    **ppseq_kwargs,
) -> tuple[PPSeq, PPSeqResult, pd.DataFrame]:
    """
    Fit and rank PPSeq candidate models.

    Returns
    -------
    tuple
        ``(best_model, best_result, candidate_table)``.
    """

    data = np.asarray(data, dtype=float)
    time_centers = np.asarray(time_centers, dtype=float)
    position = None if position is None else np.asarray(position, dtype=float)
    if init_grid is None:
        init_grid = ["peak_time", "lap_direction"] if position is not None else ["peak_time"]
    split = max(1, int(np.floor(data.shape[1] * (1.0 - holdout_fraction))))
    split = min(split, data.shape[1] - 1)
    train = data[:, :split]
    heldout = data[:, split:]
    train_position = None if position is None else position[:split]
    rows = []
    best: tuple[float, PPSeq, PPSeqResult] | None = None
    for n_templates in n_templates_grid:
        for template_duration_s in template_duration_s_grid:
            duration_bins = max(1, int(round(template_duration_s / bin_size)))
            scaled_min_events = max(
                8,
                min(
                    min_events,
                    int(np.floor((data.shape[1] * bin_size) / max(template_duration_s, bin_size))),
                ),
            )
            for init in init_grid:
                if init == "lap_direction" and train_position is None:
                    continue
                for seed in seeds:
                    screen_model = PPSeq(
                        n_templates=n_templates,
                        template_duration=duration_bins,
                        n_neurons=data.shape[0],
                        bin_size=bin_size,
                        random_state=seed,
                        **ppseq_kwargs,
                    )
                    screen_result = screen_model.fit(
                        train,
                        num_iter=fit_num_iter,
                        tol=1e-2,
                        patience=3,
                        init=init,
                        init_position=train_position,
                    )
                    heldout_result = screen_model.transform(
                        heldout,
                        num_iter=transform_num_iter,
                        tol=1e-2,
                        patience=3,
                    )
                    screen_diagnostics = ppseq_behavior_diagnostics(
                        screen_model,
                        train,
                        time_centers[:split],
                        position=train_position,
                        result=screen_result,
                        min_events=scaled_min_events,
                        min_spike_fraction=min_spike_fraction,
                        min_direction_purity=0.0 if train_position is None else min_direction_purity,
                    )
                    full_model = PPSeq(
                        n_templates=n_templates,
                        template_duration=duration_bins,
                        n_neurons=data.shape[0],
                        bin_size=bin_size,
                        random_state=seed,
                        **ppseq_kwargs,
                    )
                    full_result = full_model.fit(
                        data,
                        num_iter=fit_num_iter,
                        tol=1e-2,
                        patience=3,
                        init=init,
                        init_position=position if init == "lap_direction" else None,
                    )
                    full_diagnostics = ppseq_behavior_diagnostics(
                        full_model,
                        data,
                        time_centers,
                        position=position,
                        result=full_result,
                        min_events=scaled_min_events,
                        min_spike_fraction=min_spike_fraction,
                        min_direction_purity=0.0 if position is None else min_direction_purity,
                    )
                    screen_diag = screen_diagnostics.summary.iloc[0]
                    full_diag = full_diagnostics.summary.iloc[0]
                    min_participating_neurons = max(10, int(np.ceil(0.20 * data.shape[0])))
                    event_floor = max(1, 2 * n_templates)
                    grouped_purity = float(full_diag["grouped_direction_purity"])
                    direction_score = float(full_diag["min_direction_purity"])
                    event_count_passed = int(full_diag["n_events"]) >= event_floor
                    assignment_passed = float(full_diag["assigned_spike_fraction"]) >= min_spike_fraction
                    participation_passed = (
                        int(full_diag["min_participating_neurons"]) >= min_participating_neurons
                    )
                    direction_passed = (
                        position is None
                        or direction_score >= min_direction_purity
                        or (n_templates >= 4 and grouped_purity >= min_direction_purity)
                    )
                    template_count_passed = int(full_diag["n_templates_with_events"]) >= min(2, n_templates)
                    blocking_reasons = []
                    if not event_count_passed:
                        blocking_reasons.append("too_few_events")
                    if not assignment_passed:
                        blocking_reasons.append("low_assigned_spike_fraction")
                    if not participation_passed:
                        blocking_reasons.append("low_participation")
                    if not direction_passed:
                        blocking_reasons.append("low_direction_purity")
                    if not template_count_passed:
                        blocking_reasons.append("too_few_templates_with_events")
                    passed_gate = not blocking_reasons
                    heldout_ll = screen_model.score(heldout, heldout_result)
                    balance = float(full_diag["template_direction_balance"])
                    event_score = min(1.0, int(full_diag["n_events"]) / max(event_floor, 1))
                    participation_score = min(
                        1.0,
                        int(full_diag["min_participating_neurons"]) / max(min_participating_neurons, 1),
                    )
                    score = (
                        (1_000_000.0 if passed_gate else 0.0)
                        + 100_000.0 * float(full_diag["assigned_spike_fraction"])
                        + 10_000.0 * participation_score
                        + 1_000.0 * max(direction_score, grouped_purity)
                        + 100.0 * event_score
                        + 10.0 * balance
                        + heldout_ll / max(heldout.shape[1], 1)
                    )
                    reason = "passed_behavior_gate" if passed_gate else ",".join(blocking_reasons)
                    rows.append(
                        {
                            "n_templates": n_templates,
                            "template_duration_s": template_duration_s,
                            "template_duration": duration_bins,
                            "seed": seed,
                            "init": init,
                            "train_log_likelihood": float(screen_result.log_likelihood[-1]),
                            "heldout_log_likelihood": heldout_ll,
                            "heldout_log_likelihood_per_bin": heldout_ll / max(heldout.shape[1], 1),
                            "assigned_spike_fraction": float(full_diag["assigned_spike_fraction"]),
                            "n_events": int(full_diag["n_events"]),
                            "min_direction_purity": direction_score,
                            "grouped_direction_purity": grouped_purity,
                            "passed_behavior_gate": passed_gate,
                            "strict_passed_behavior_gate": bool(full_diag["passed"]),
                            "direction_purity_score": direction_score,
                            "n_templates_with_events": int(full_diag["n_templates_with_events"]),
                            "template_direction_balance": balance,
                            "scaled_min_events": int(scaled_min_events),
                            "event_count_passed": event_count_passed,
                            "assignment_passed": assignment_passed,
                            "participation_passed": participation_passed,
                            "direction_passed": direction_passed,
                            "blocking_failure": reason,
                            "screen_assigned_spike_fraction": float(screen_diag["assigned_spike_fraction"]),
                            "screen_n_events": int(screen_diag["n_events"]),
                            "screen_passed_behavior_gate": bool(screen_diag["passed"]),
                            "full_assigned_spike_fraction": float(full_diag["assigned_spike_fraction"]),
                            "full_n_events": int(full_diag["n_events"]),
                            "full_min_participating_neurons": int(full_diag["min_participating_neurons"]),
                            "full_passed_behavior_gate": passed_gate,
                            "selection_reason": reason,
                            "selection_score": float(score),
                        }
                    )
                    if best is None or score > best[0]:
                        best = (score, full_model, full_result)
    if best is None:
        raise ValueError("No PPSeq candidates were fit")
    return best[1], best[2], pd.DataFrame(rows).sort_values("selection_score", ascending=False, ignore_index=True)


def score_ppseq_replay_nulls(
    model: PPSeq,
    data: np.ndarray,
    event_intervals: np.ndarray,
    time_centers: np.ndarray,
    *,
    n_null: int = 25,
    random_state: int | None = None,
    replay_duration_s: float | None = None,
    null_method: str = "fast",
    max_null_work: float = 5e8,
    **score_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Score replay events against shuffled-template null controls.

    Returns
    -------
    tuple of pandas.DataFrame
        Observed scores with ``replay_z``/``replay_p`` and long-form null
        scores.
    """

    if null_method not in {"fast", "em"}:
        raise ValueError("null_method must be 'fast' or 'em'")
    observed, observed_result = model.score_replay_events(
        data,
        event_intervals,
        time_centers,
        replay_duration_s=replay_duration_s,
        **score_kwargs,
    )
    work = (
        float(n_null)
        * float(observed_result.amplitudes.shape[1])
        * float(model.n_templates)
        * float(model.n_neurons)
    )
    if work > max_null_work:
        raise ValueError("Requested replay null scoring is too large for the runtime guard")
    rng = np.random.default_rng(random_state)
    original_offsets = model.template_offsets_.copy()
    original_widths = model.template_widths_.copy()
    original_scales = model.template_scales_.copy()
    null_rows = []
    try:
        for null_id in range(n_null):
            for k in range(model.n_templates):
                perm = rng.permutation(model.n_neurons)
                model.template_offsets_[k] = original_offsets[k, perm]
                model.template_widths_[k] = original_widths[k, perm]
                model.template_scales_[k] = original_scales[k, perm]
            if null_method == "em":
                null_scores, _ = model.score_replay_events(
                    data,
                    event_intervals,
                    time_centers,
                    replay_duration_s=replay_duration_s,
                    **score_kwargs,
                )
            else:
                replay_templates, replay_labels = model._make_replay_template_bank(
                    include_forward=score_kwargs.get("include_forward", True),
                    include_reverse=score_kwargs.get("include_reverse", True),
                    template_names=score_kwargs.get("template_names", None),
                    replay_duration_s=replay_duration_s,
                )
                null_rates = model._reconstruct(
                    observed_result.amplitudes,
                    replay_templates,
                    base_rates=observed_result.base_rates,
                )
                null_result = PPSeqResult(
                    log_likelihood=np.asarray([], dtype=float),
                    amplitudes=observed_result.amplitudes,
                    rates=null_rates,
                    base_rates=observed_result.base_rates,
                    templates=replay_templates,
                )
                method = score_kwargs.get("method", "full")
                if method == "event":
                    event_pad_s = score_kwargs.get("event_pad_s", replay_duration_s or model.template_duration_s)
                    event_data, _, event_slices = model._event_local_data(
                        np.asarray(data, dtype=float),
                        np.asarray(event_intervals, dtype=float),
                        np.asarray(time_centers, dtype=float),
                        float(event_pad_s),
                    )
                    null_scores = model._summarize_replay_slices(
                        event_data,
                        np.asarray(event_intervals, dtype=float),
                        null_result,
                        replay_labels,
                        event_slices=event_slices,
                    )
                else:
                    null_scores, _ = model.score_replay_events(
                        data,
                        event_intervals,
                        time_centers,
                        replay_duration_s=replay_duration_s,
                        result=null_result,
                        **score_kwargs,
                    )
            null_scores = null_scores[["start", "stop", "likelihood_gain_per_spike"]].copy()
            null_scores["null_id"] = null_id
            null_scores["null_method"] = null_method
            null_rows.append(null_scores)
    finally:
        model.template_offsets_ = original_offsets
        model.template_widths_ = original_widths
        model.template_scales_ = original_scales
    null_scores = pd.concat(null_rows, ignore_index=True) if null_rows else pd.DataFrame()
    if not null_scores.empty:
        grouped = null_scores.groupby(["start", "stop"])["likelihood_gain_per_spike"]
        replay_z = []
        replay_p = []
        for _, row in observed.iterrows():
            key = (row.start, row.stop)
            values = grouped.get_group(key).to_numpy(dtype=float)
            metric = float(row.likelihood_gain_per_spike)
            std = float(np.std(values, ddof=1)) if values.size > 1 else np.nan
            replay_z.append((metric - float(np.mean(values))) / std if std > 0 else np.nan)
            replay_p.append((1.0 + np.sum(values >= metric)) / (values.size + 1.0))
        observed["replay_z"] = np.asarray(replay_z, dtype=float)
        observed["replay_p"] = np.asarray(replay_p, dtype=float)
    else:
        observed["replay_z"] = np.nan
        observed["replay_p"] = np.nan
    return observed, null_scores


def select_ppseq_template_count(
    train_data: np.ndarray,
    test_data: np.ndarray,
    template_counts: list[int] | np.ndarray,
    *,
    template_duration: int,
    bin_size: float = 0.02,
    fit_num_iter: int = 50,
    transform_num_iter: int = 25,
    random_state: int | None = None,
    **ppseq_kwargs,
) -> pd.DataFrame:
    """
    Compare PPSeq template counts by held-out log likelihood.
    """

    rows = []
    for n_templates in template_counts:
        model = PPSeq(
            int(n_templates),
            template_duration,
            n_neurons=np.asarray(train_data).shape[0],
            bin_size=bin_size,
            random_state=random_state,
            **ppseq_kwargs,
        )
        fit_result = model.fit(train_data, num_iter=fit_num_iter)
        test_result = model.transform(test_data, num_iter=transform_num_iter)
        rows.append(
            {
                "n_templates": int(n_templates),
                "train_log_likelihood": float(fit_result.log_likelihood[-1]),
                "heldout_log_likelihood": model.score(test_data, test_result),
                "heldout_log_likelihood_per_bin": model.score(test_data, test_result)
                / max(np.asarray(test_data).shape[1], 1),
                "heldout_log_likelihood_per_spike": model.score(test_data, test_result)
                / max(float(np.sum(test_data)), 1.0),
            }
        )
    return pd.DataFrame(rows)


def ppseq_seed_stability(
    data: np.ndarray,
    *,
    seeds: list[int],
    n_templates: int,
    template_duration: int,
    bin_size: float = 0.02,
    num_iter: int = 50,
    fit_num_iter: int | None = None,
    **ppseq_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit multiple seeds and summarize template stability.
    """

    models = []
    rows = []
    if fit_num_iter is not None:
        num_iter = fit_num_iter
    for seed in seeds:
        model = PPSeq(
            n_templates,
            template_duration,
            n_neurons=np.asarray(data).shape[0],
            bin_size=bin_size,
            random_state=seed,
            **ppseq_kwargs,
        )
        result = model.fit(data, num_iter=num_iter)
        models.append(model)
        rows.append({"seed": seed, "log_likelihood": float(result.log_likelihood[-1])})
    per_seed = pd.DataFrame(rows)
    pair_rows = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            pair_rows.append(
                {
                    "seed_a": seeds[i],
                    "seed_b": seeds[j],
                    "mean_matched_template_similarity": template_set_similarity(
                        models[i].templates, models[j].templates
                    ),
                }
            )
    pairwise = pd.DataFrame(pair_rows)
    summary = pd.DataFrame(
        [
            {
                "n_seeds": len(seeds),
                "mean_log_likelihood": float(per_seed["log_likelihood"].mean()),
                "mean_stability": float(pairwise["mean_matched_template_similarity"].mean())
                if not pairwise.empty
                else np.nan,
            }
        ]
    )
    return summary, pairwise


def template_set_similarity(templates_a: np.ndarray, templates_b: np.ndarray) -> float:
    """
    Match two template sets and return mean cosine similarity.
    """

    a = np.asarray(templates_a, dtype=float).reshape(templates_a.shape[0], -1)
    b = np.asarray(templates_b, dtype=float).reshape(templates_b.shape[0], -1)
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    sim = a @ b.T
    rows, cols = linear_sum_assignment(-sim)
    return float(np.mean(sim[rows, cols]))
