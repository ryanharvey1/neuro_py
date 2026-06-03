from __future__ import annotations

from dataclasses import dataclass

import nelpy as nel
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import gammaln

from neuro_py.process.peri_event import count_in_interval


@dataclass
class PPSeqResult:
    """
    Results from fitting or transforming a PPSeq model.

    Parameters
    ----------
    log_likelihood : np.ndarray
        Poisson log-likelihood after each EM iteration.
    amplitudes : np.ndarray
        Template amplitudes with shape ``(n_templates, n_time_bins)``.
    rates : np.ndarray
        Reconstructed rates with shape ``(n_neurons, n_time_bins)``.
    """

    log_likelihood: np.ndarray
    amplitudes: np.ndarray
    rates: np.ndarray


class PPSeq:
    """
    Compact discrete-time PPSeq-style sequence detector.

    This class fits Poisson sequence templates to binned spike counts. Each
    template is represented by per-neuron response scales, offsets, and widths,
    and each sequence occurrence is represented by a nonnegative amplitude over
    time. The model is intended for fitting templates during behavior and then
    scoring the strength of those fixed templates during post-task events such
    as sharp wave ripples.

    Parameters
    ----------
    n_templates : int
        Number of sequence templates to fit.
    template_duration : int
        Template duration in time bins.
    n_neurons : int, optional
        Number of neurons. If None, inferred from the first data matrix passed
        to :meth:`fit`.
    bin_size : float, optional
        Bin size in seconds, used by spike-time conversion helpers and event
        scoring. By default 0.02.
    sequence_frac : float, optional
        Initial fraction of spikes attributed to sequence templates rather than
        background. By default 0.5.
    concentration : float, optional
        Dirichlet concentration scale for initial neuron template weights. By
        default 10.0.
    min_rate : float, optional
        Numerical floor for rates. By default 1e-7.
    min_amplitude : float, optional
        Numerical floor for amplitudes. By default 1e-9.
    min_width : float, optional
        Numerical floor for template widths in bins. By default 1e-3.
    alpha_amplitude : float, optional
        Gamma prior shape-like pseudocount for amplitude updates. By default
        1e-3.
    beta_amplitude : float, optional
        Gamma prior rate-like pseudocount for amplitude updates. By default
        1e-3.
    alpha_base : float, optional
        Gamma prior shape-like pseudocount for background updates. By default
        1e-3.
    beta_base : float, optional
        Gamma prior rate-like pseudocount for background updates. By default
        1e-3.
    alpha_template : float, optional
        Pseudocount for template updates. By default 1e-4.
    random_state : int, optional
        Seed for deterministic initialization. By default None.
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
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    @property
    def templates(self) -> np.ndarray:
        """
        Template tensor with shape ``(n_templates, n_neurons, duration)``.

        Returns
        -------
        np.ndarray
            Gaussian templates normalized so each template has total mass one.
        """
        self._check_initialized()
        bins = np.arange(self.template_duration, dtype=float)
        widths = np.clip(self.template_widths_, self.min_width, None)
        z = np.exp(
            -0.5
            * (
                (bins[np.newaxis, np.newaxis, :] - self.template_offsets_[:, :, np.newaxis])
                / widths[:, :, np.newaxis]
            )
            ** 2
        )
        z_sum = np.sum(z, axis=2, keepdims=True)
        z = np.divide(
            z,
            z_sum,
            out=np.full_like(z, 1.0 / self.template_duration),
            where=np.isfinite(z_sum) & (z_sum > 0),
        )
        return self.template_scales_[:, :, np.newaxis] * z

    def fit(
        self,
        data: np.ndarray,
        *,
        num_iter: int = 50,
        fit_templates: bool = True,
        fit_base_rates: bool = True,
    ) -> PPSeqResult:
        """
        Fit PPSeq templates to binned spike counts.

        Parameters
        ----------
        data : np.ndarray
            Spike count matrix with shape ``(n_neurons, n_time_bins)``.
        num_iter : int, optional
            Number of EM iterations. By default 50.
        fit_templates : bool, optional
            Whether to update template scales, offsets, and widths. By default
            True.
        fit_base_rates : bool, optional
            Whether to update background rates. By default True.

        Returns
        -------
        PPSeqResult
            Fitted likelihood history, amplitudes, and reconstructed rates.
        """
        data = self._validate_data(data)
        if np.sum(data) <= 0:
            raise ValueError("data must contain at least one spike")

        self._initialize(data)
        result = self._run_em(
            data,
            num_iter=num_iter,
            fit_templates=fit_templates,
            fit_base_rates=fit_base_rates,
            initialization="default",
        )
        self.amplitudes_ = result.amplitudes
        self.log_likelihood_ = result.log_likelihood
        return result

    def transform(
        self,
        data: np.ndarray,
        *,
        num_iter: int = 25,
        fit_templates: bool = False,
        fit_base_rates: bool = False,
    ) -> PPSeqResult:
        """
        Estimate template amplitudes for new data.

        Parameters
        ----------
        data : np.ndarray
            Spike count matrix with shape ``(n_neurons, n_time_bins)``.
        num_iter : int, optional
            Number of EM iterations. By default 25.
        fit_templates : bool, optional
            Whether to update templates while transforming. By default False.
        fit_base_rates : bool, optional
            Whether to update background rates while transforming. By default
            False.

        Returns
        -------
        PPSeqResult
            Likelihood history, inferred amplitudes, and reconstructed rates.
        """
        self._check_initialized()
        data = self._validate_data(data)
        return self._run_em(
            data,
            num_iter=num_iter,
            fit_templates=fit_templates,
            fit_base_rates=fit_base_rates,
            initialization="population",
        )

    def fit_transform(
        self,
        task_data: np.ndarray,
        post_data: np.ndarray,
        *,
        fit_num_iter: int = 50,
        transform_num_iter: int = 25,
    ) -> PPSeqResult:
        """
        Fit templates on task data and transform post-task data.

        Parameters
        ----------
        task_data : np.ndarray
            Task spike count matrix with shape ``(n_neurons, n_time_bins)``.
        post_data : np.ndarray
            Post-task spike count matrix with shape ``(n_neurons, n_time_bins)``.
        fit_num_iter : int, optional
            Number of task fitting iterations. By default 50.
        transform_num_iter : int, optional
            Number of post-task transform iterations. By default 25.

        Returns
        -------
        PPSeqResult
            Transform result for ``post_data``.
        """
        self.fit(task_data, num_iter=fit_num_iter)
        return self.transform(post_data, num_iter=transform_num_iter)

    def reconstruct(self, amplitudes: np.ndarray) -> np.ndarray:
        """
        Reconstruct firing rates from template amplitudes.

        Parameters
        ----------
        amplitudes : np.ndarray
            Template amplitudes with shape ``(n_templates, n_time_bins)``.

        Returns
        -------
        np.ndarray
            Reconstructed rates with shape ``(n_neurons, n_time_bins)``.
        """
        self._check_initialized()
        amplitudes = self._validate_amplitudes(amplitudes)
        return self._reconstruct(amplitudes, self.templates)

    def score(self, data: np.ndarray, amplitudes: np.ndarray | None = None) -> float:
        """
        Compute Poisson log-likelihood for data under the model.

        Parameters
        ----------
        data : np.ndarray
            Spike count matrix with shape ``(n_neurons, n_time_bins)``.
        amplitudes : np.ndarray, optional
            Template amplitudes. If None, amplitudes are inferred with
            :meth:`transform`.

        Returns
        -------
        float
            Poisson log-likelihood.
        """
        data = self._validate_data(data)
        if amplitudes is None:
            amplitudes = self.transform(data).amplitudes
        rates = self.reconstruct(amplitudes)
        return self._poisson_log_likelihood(data, rates)

    def score_events(
        self,
        data: np.ndarray,
        event_intervals: np.ndarray,
        time_centers: np.ndarray,
        *,
        num_iter: int = 25,
        method: str = "full",
        result: PPSeqResult | None = None,
    ) -> pd.DataFrame:
        """
        Score fixed task templates in event intervals.

        Parameters
        ----------
        data : np.ndarray
            Post-task spike count matrix with shape ``(n_neurons, n_time_bins)``.
        event_intervals : np.ndarray
            Event intervals in seconds with shape ``(n_events, 2)``.
        time_centers : np.ndarray
            Time center for each bin in ``data``.
        num_iter : int, optional
            Number of transform iterations. By default 25.
        method : {"full", "per_event"}, optional
            If "full", transform the full ``data`` matrix once and summarize the
            inferred amplitudes inside each event. If "per_event", run a
            separate transform within each event interval. By default "full".
        result : PPSeqResult, optional
            Existing full-data transform result to summarize. When provided,
            ``score_events`` skips transform inference and uses
            ``result.amplitudes`` and ``result.rates`` directly.

        Returns
        -------
        pd.DataFrame
            Event-level template strength table with ``start``, ``stop``,
            ``duration``, ``best_template``, ``likelihood_gain``, and per-template
            ``template_{k}_sum``, ``template_{k}_mean``, and ``template_{k}_max``
            columns.
        """
        self._check_initialized()
        data = self._validate_data(data)
        event_intervals = np.asarray(event_intervals, dtype=float)
        time_centers = np.asarray(time_centers, dtype=float)

        if event_intervals.ndim != 2 or event_intervals.shape[1] != 2:
            raise ValueError("event_intervals must have shape (n_events, 2)")
        if time_centers.ndim != 1 or time_centers.shape[0] != data.shape[1]:
            raise ValueError("time_centers must be 1D and match data time bins")
        if method not in {"full", "per_event"}:
            raise ValueError("method must be 'full' or 'per_event'")
        if result is not None and method == "per_event":
            raise ValueError("result cannot be used with method='per_event'")
        if result is not None:
            if result.amplitudes.shape != (self.n_templates, data.shape[1]):
                raise ValueError(
                    "result.amplitudes must have shape "
                    "(n_templates, n_time_bins)"
                )
            if result.rates.shape != data.shape:
                raise ValueError("result.rates must have the same shape as data")

        rows = []
        full_result = None
        likelihood_gain_cumsum = None
        if len(event_intervals) > 0 and method == "full":
            full_result = result
            if full_result is None:
                full_result = self.transform(data, num_iter=num_iter)
            likelihood_gain_bins = self._poisson_log_likelihood_per_bin(
                data, full_result.rates
            ) - self._poisson_log_likelihood_per_bin(
                data, self.base_rates_[:, np.newaxis]
            )
            likelihood_gain_cumsum = np.concatenate(
                [[0.0], np.cumsum(likelihood_gain_bins)]
            )

        for start, stop in event_intervals:
            start_idx = int(np.searchsorted(time_centers, start, side="left"))
            stop_idx = int(np.searchsorted(time_centers, stop, side="right"))
            event_data = data[:, start_idx:stop_idx]
            row = {
                "start": float(start),
                "stop": float(stop),
                "duration": float(stop - start),
            }

            if event_data.shape[1] == 0:
                strengths = np.zeros((self.n_templates, 0), dtype=float)
                likelihood_gain = np.nan
            elif method == "per_event":
                result = self.transform(event_data, num_iter=num_iter)
                strengths = result.amplitudes
                background_rates = np.clip(
                    np.repeat(self.base_rates_[:, np.newaxis], event_data.shape[1], axis=1),
                    self.min_rate,
                    None,
                )
                likelihood_gain = result.log_likelihood[-1] - self._poisson_log_likelihood(
                    event_data, background_rates
                )
            else:
                strengths = full_result.amplitudes[:, start_idx:stop_idx]
                likelihood_gain = (
                    likelihood_gain_cumsum[stop_idx] - likelihood_gain_cumsum[start_idx]
                )

            sums = strengths.sum(axis=1) if strengths.size else np.zeros(self.n_templates)
            means = (
                strengths.mean(axis=1) if strengths.shape[1] > 0 else np.zeros(self.n_templates)
            )
            maxes = (
                strengths.max(axis=1) if strengths.shape[1] > 0 else np.zeros(self.n_templates)
            )
            row["best_template"] = int(np.argmax(sums)) if sums.size else -1
            row["likelihood_gain"] = float(likelihood_gain)
            for k in range(self.n_templates):
                row[f"template_{k}_sum"] = float(sums[k])
                row[f"template_{k}_mean"] = float(means[k])
                row[f"template_{k}_max"] = float(maxes[k])
            rows.append(row)

        return pd.DataFrame(rows)

    def score_replay_events(
        self,
        data: np.ndarray,
        event_intervals: np.ndarray,
        time_centers: np.ndarray,
        *,
        num_iter: int = 25,
        result: PPSeqResult | None = None,
        include_forward: bool = True,
        include_reverse: bool = True,
        template_names: list[str] | None = None,
    ) -> tuple[pd.DataFrame, PPSeqResult]:
        """
        Score forward and reverse replay hypotheses in event intervals.

        Parameters
        ----------
        data : np.ndarray
            Sleep spike count matrix with shape ``(n_neurons, n_time_bins)``.
        event_intervals : np.ndarray
            Event intervals in seconds with shape ``(n_events, 2)``.
        time_centers : np.ndarray
            Time center for each bin in ``data``.
        num_iter : int, optional
            Number of fixed-template transform iterations when ``result`` is
            None. By default 25.
        result : PPSeqResult, optional
            Existing replay-bank transform result. When provided, inference is
            skipped and event scoring summarizes the supplied amplitudes/rates.
        include_forward : bool, optional
            Whether to include the learned task templates. By default True.
        include_reverse : bool, optional
            Whether to include time-reversed task templates. By default True.
        template_names : list of str, optional
            Names for fitted task templates. If None, templates are named
            ``template_{k}``.

        Returns
        -------
        tuple of pandas.DataFrame and PPSeqResult
            Event-level replay scores and the full replay-bank transform result.
        """
        self._check_initialized()
        data = self._validate_data(data)
        event_intervals = np.asarray(event_intervals, dtype=float)
        time_centers = np.asarray(time_centers, dtype=float)

        if event_intervals.ndim != 2 or event_intervals.shape[1] != 2:
            raise ValueError("event_intervals must have shape (n_events, 2)")
        if time_centers.ndim != 1 or time_centers.shape[0] != data.shape[1]:
            raise ValueError("time_centers must be 1D and match data time bins")

        replay_templates, replay_labels = self._make_replay_template_bank(
            include_forward=include_forward,
            include_reverse=include_reverse,
            template_names=template_names,
        )
        n_replay_templates = replay_templates.shape[0]

        if result is None:
            result = self._run_fixed_template_transform(
                data,
                replay_templates,
                num_iter=num_iter,
            )
        else:
            if result.amplitudes.shape != (n_replay_templates, data.shape[1]):
                raise ValueError(
                    "result.amplitudes must match the replay template bank and "
                    "data time bins"
                )
            if result.rates.shape != data.shape:
                raise ValueError("result.rates must have the same shape as data")

        likelihood_gain_bins = self._poisson_log_likelihood_per_bin(
            data, result.rates
        ) - self._poisson_log_likelihood_per_bin(data, self.base_rates_[:, np.newaxis])
        likelihood_gain_cumsum = np.concatenate([[0.0], np.cumsum(likelihood_gain_bins)])

        rows = []
        for start, stop in event_intervals:
            start_idx = int(np.searchsorted(time_centers, start, side="left"))
            stop_idx = int(np.searchsorted(time_centers, stop, side="right"))
            strengths = result.amplitudes[:, start_idx:stop_idx]

            if strengths.shape[1] == 0:
                sums = np.zeros(n_replay_templates, dtype=float)
                means = np.zeros(n_replay_templates, dtype=float)
                maxes = np.zeros(n_replay_templates, dtype=float)
                best_idx = -1
                likelihood_gain = np.nan
            else:
                sums = strengths.sum(axis=1)
                means = strengths.mean(axis=1)
                maxes = strengths.max(axis=1)
                best_idx = int(np.argmax(sums))
                likelihood_gain = (
                    likelihood_gain_cumsum[stop_idx] - likelihood_gain_cumsum[start_idx]
                )

            if best_idx >= 0:
                best_label = replay_labels[best_idx]
                best_template = int(best_label["template_id"])
                best_direction = str(best_label["direction"])
                best_replay = str(best_label["name"])
                best_replay_sum = float(sums[best_idx])
                best_replay_max = float(maxes[best_idx])
            else:
                best_template = -1
                best_direction = ""
                best_replay = ""
                best_replay_sum = 0.0
                best_replay_max = 0.0

            row = {
                "start": float(start),
                "stop": float(stop),
                "duration": float(stop - start),
                "best_template": best_template,
                "best_direction": best_direction,
                "best_replay": best_replay,
                "best_replay_sum": best_replay_sum,
                "best_replay_max": best_replay_max,
                "likelihood_gain": float(likelihood_gain),
            }
            for idx, label in enumerate(replay_labels):
                prefix = str(label["name"])
                row[f"{prefix}_sum"] = float(sums[idx])
                row[f"{prefix}_mean"] = float(means[idx])
                row[f"{prefix}_max"] = float(maxes[idx])
            rows.append(row)

        return pd.DataFrame(rows), result

    def bin_spikes(
        self,
        spike_times: np.ndarray | nel.SpikeTrainArray,
        neuron_ids: np.ndarray | None = None,
        *,
        start: float | None = None,
        stop: float | None = None,
        n_neurons: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Bin spike times into a count matrix.

        Parameters
        ----------
        spike_times : np.ndarray or nel.SpikeTrainArray
            Spike timestamps in seconds, or a nelpy spike train array.
        neuron_ids : np.ndarray, optional
            Integer neuron identifier for each spike when ``spike_times`` is an
            array.
        start : float, optional
            Start time in seconds. If None, inferred from the data.
        stop : float, optional
            Stop time in seconds. If None, inferred from the data.
        n_neurons : int, optional
            Number of neurons. If None, inferred from data or the model.

        Returns
        -------
        tuple of np.ndarray
            ``(counts, time_centers)`` where counts has shape
            ``(n_neurons, n_time_bins)``.
        """
        if isinstance(spike_times, nel.SpikeTrainArray):
            return self._bin_nelpy_spikes(spike_times, start=start, stop=stop)

        if neuron_ids is None:
            raise ValueError("neuron_ids is required when spike_times is an array")
        spike_times = np.asarray(spike_times, dtype=float)
        neuron_ids = np.asarray(neuron_ids, dtype=int)
        if spike_times.shape[0] != neuron_ids.shape[0]:
            raise ValueError("spike_times and neuron_ids must have the same length")

        if n_neurons is None:
            if self.n_neurons is not None:
                n_neurons = self.n_neurons
            elif neuron_ids.size:
                n_neurons = int(np.max(neuron_ids)) + 1
            else:
                raise ValueError("n_neurons is required for empty spike arrays")

        if start is None:
            start = float(np.min(spike_times)) if spike_times.size else 0.0
        if stop is None:
            stop = float(np.max(spike_times)) + self.bin_size if spike_times.size else start
        if stop <= start:
            raise ValueError("stop must be greater than start")

        edges = np.arange(start, stop + self.bin_size, self.bin_size)
        valid = (
            (spike_times >= start)
            & (spike_times < edges[-1])
            & (neuron_ids >= 0)
            & (neuron_ids < int(n_neurons))
        )
        spike_trains = np.empty(int(n_neurons), dtype=object)
        for neuron in range(int(n_neurons)):
            spike_trains[neuron] = np.sort(spike_times[valid & (neuron_ids == neuron)])

        counts = count_in_interval(
            spike_trains,
            edges[:-1],
            edges[1:],
            par_type="counts",
        )

        return counts, edges[:-1] + self.bin_size / 2

    def _run_em(
        self,
        data: np.ndarray,
        *,
        num_iter: int,
        fit_templates: bool,
        fit_base_rates: bool,
        initialization: str,
    ) -> PPSeqResult:
        if num_iter < 0:
            raise ValueError("num_iter must be nonnegative")

        amplitudes = self._initialize_amplitudes(data, initialization)
        log_likelihood = []

        for _ in range(num_iter):
            templates = self.templates
            rates = self._reconstruct(amplitudes, templates)
            ratio = data / np.clip(rates, self.min_rate, None)
            amplitudes = self._update_amplitudes(amplitudes, ratio, templates)

            if fit_base_rates:
                rates = self._reconstruct(amplitudes, templates)
                ratio = data / np.clip(rates, self.min_rate, None)
                self._update_base_rates(data, ratio)

            if fit_templates:
                rates = self._reconstruct(amplitudes, self.templates)
                ratio = data / np.clip(rates, self.min_rate, None)
                self._update_templates(amplitudes, ratio)

            rates = self._reconstruct(amplitudes, self.templates)
            log_likelihood.append(self._poisson_log_likelihood(data, rates))

        rates = self._reconstruct(amplitudes, self.templates)
        return PPSeqResult(
            log_likelihood=np.asarray(log_likelihood, dtype=float),
            amplitudes=amplitudes,
            rates=rates,
        )

    def _run_fixed_template_transform(
        self,
        data: np.ndarray,
        templates: np.ndarray,
        *,
        num_iter: int,
    ) -> PPSeqResult:
        if num_iter < 0:
            raise ValueError("num_iter must be nonnegative")
        if templates.ndim != 3:
            raise ValueError("templates must have shape (n_templates, n_neurons, duration)")
        if templates.shape[1] != data.shape[0]:
            raise ValueError("templates neuron count must match data")
        if templates.shape[2] != self.template_duration:
            raise ValueError("templates duration must match template_duration")

        amplitudes = self._initialize_fixed_template_amplitudes(data, templates.shape[0])
        log_likelihood = []
        for _ in range(num_iter):
            rates = self._reconstruct(amplitudes, templates)
            ratio = data / np.clip(rates, self.min_rate, None)
            amplitudes = self._update_amplitudes(amplitudes, ratio, templates)
            rates = self._reconstruct(amplitudes, templates)
            log_likelihood.append(self._poisson_log_likelihood(data, rates))

        rates = self._reconstruct(amplitudes, templates)
        return PPSeqResult(
            log_likelihood=np.asarray(log_likelihood, dtype=float),
            amplitudes=amplitudes,
            rates=rates,
        )

    def _initialize(self, data: np.ndarray) -> None:
        n_neurons = data.shape[0]
        if self.n_neurons is None:
            self.n_neurons = n_neurons
        elif self.n_neurons != n_neurons:
            raise ValueError("data neuron count does not match n_neurons")

        avg_rate = np.clip(data.mean(axis=1), self.min_rate, None)
        self.base_rates_ = np.clip(avg_rate * (1.0 - self.sequence_frac), self.min_rate, None)

        concentration = np.clip(self.concentration * avg_rate, self.min_rate, None)
        self.template_scales_ = self._rng.dirichlet(concentration, size=self.n_templates)
        self.template_offsets_ = self._rng.uniform(
            0, self.template_duration - 1, size=(self.n_templates, self.n_neurons)
        )
        self.template_widths_ = np.ones((self.n_templates, self.n_neurons), dtype=float)

    def _initialize_amplitudes(self, data: np.ndarray, initialization: str) -> np.ndarray:
        total_spikes = float(np.sum(data))
        n_time = data.shape[1]

        if initialization == "population":
            population = np.clip(data.sum(axis=0), self.min_amplitude, None)
            amplitudes = np.tile(population, (self.n_templates, 1))
            scale = max(total_spikes * self.sequence_frac, self.min_amplitude)
            amplitudes *= scale / max(float(np.sum(amplitudes)), self.min_amplitude)
            return np.clip(amplitudes, self.min_amplitude, None)

        amplitudes = self._rng.gamma(
            shape=1.0,
            scale=1.0,
            size=(self.n_templates, n_time),
        )
        scale = max(total_spikes * self.sequence_frac, self.min_amplitude)
        amplitudes *= scale / max(float(np.sum(amplitudes)), self.min_amplitude)
        return np.clip(amplitudes, self.min_amplitude, None)

    def _initialize_fixed_template_amplitudes(
        self, data: np.ndarray, n_templates: int
    ) -> np.ndarray:
        total_spikes = float(np.sum(data))
        population = np.clip(data.sum(axis=0), self.min_amplitude, None)
        amplitudes = np.tile(population, (n_templates, 1))
        scale = max(total_spikes * self.sequence_frac, self.min_amplitude)
        amplitudes *= scale / max(float(np.sum(amplitudes)), self.min_amplitude)
        return np.clip(amplitudes, self.min_amplitude, None)

    def _make_replay_template_bank(
        self,
        *,
        include_forward: bool,
        include_reverse: bool,
        template_names: list[str] | None,
    ) -> tuple[np.ndarray, list[dict[str, int | str]]]:
        if not include_forward and not include_reverse:
            raise ValueError("At least one of include_forward or include_reverse must be True")
        if template_names is None:
            template_names = [f"template_{k}" for k in range(self.n_templates)]
        if len(template_names) != self.n_templates:
            raise ValueError("template_names must match n_templates")

        base_templates = self.templates
        template_bank = []
        labels = []
        directions = []
        if include_forward:
            directions.append(("forward", base_templates))
        if include_reverse:
            directions.append(("reverse", base_templates[:, :, ::-1]))

        for direction, templates in directions:
            for template_id, template_name in enumerate(template_names):
                template_bank.append(templates[template_id])
                labels.append(
                    {
                        "template_id": template_id,
                        "template_name": str(template_name),
                        "direction": direction,
                        "name": f"{template_name}_{direction}",
                    }
                )

        return np.asarray(template_bank, dtype=float), labels

    def _reconstruct(self, amplitudes: np.ndarray, templates: np.ndarray) -> np.ndarray:
        n_time = amplitudes.shape[1]
        rates = np.repeat(self.base_rates_[:, np.newaxis], n_time, axis=1)

        template_duration = templates.shape[2]
        for delay in range(min(template_duration, n_time)):
            rates[:, delay:] += templates[:, :, delay].T @ amplitudes[:, : n_time - delay]

        return np.clip(rates, self.min_rate, None)

    def _update_amplitudes(
        self, amplitudes: np.ndarray, ratio: np.ndarray, templates: np.ndarray
    ) -> np.ndarray:
        n_time = amplitudes.shape[1]
        updated = np.zeros_like(amplitudes)
        template_mass = np.clip(templates.sum(axis=(1, 2)), self.min_rate, None)

        drive = np.zeros_like(amplitudes)
        for delay in range(min(self.template_duration, n_time)):
            drive[:, : n_time - delay] += templates[:, :, delay] @ ratio[:, delay:]

        updated = (
            amplitudes * drive + self.alpha_amplitude
        ) / (template_mass[:, np.newaxis] + self.beta_amplitude)

        return np.clip(updated, self.min_amplitude, None)

    def _update_base_rates(self, data: np.ndarray, ratio: np.ndarray) -> None:
        n_time = data.shape[1]
        numerator = self.base_rates_ * ratio.sum(axis=1) + self.alpha_base
        denominator = n_time + self.beta_base
        self.base_rates_ = np.clip(numerator / denominator, self.min_rate, None)

    def _update_templates(self, amplitudes: np.ndarray, ratio: np.ndarray) -> None:
        n_time = amplitudes.shape[1]
        templates = self.templates
        targets = np.full_like(templates, self.alpha_template)

        amp_sums = np.clip(np.sum(amplitudes, axis=1), self.min_rate, None)
        for delay in range(min(self.template_duration, n_time)):
            expected = amplitudes[:, : n_time - delay] @ ratio[:, delay:].T
            targets[:, :, delay] += (
                templates[:, :, delay] * expected / amp_sums[:, np.newaxis]
            )

        bins = np.arange(self.template_duration, dtype=float)
        scales = targets.sum(axis=2)
        scale_sums = np.clip(scales.sum(axis=1, keepdims=True), self.min_rate, None)
        self.template_scales_ = np.clip(scales / scale_sums, self.min_rate, None)
        self.template_scales_ /= np.clip(
            self.template_scales_.sum(axis=1, keepdims=True), self.min_rate, None
        )

        target_sums = np.clip(targets.sum(axis=2), self.min_rate, None)
        offsets = np.sum(targets * bins[np.newaxis, np.newaxis, :], axis=2) / target_sums
        variances = (
            np.sum(
                targets * (bins[np.newaxis, np.newaxis, :] - offsets[:, :, np.newaxis]) ** 2,
                axis=2,
            )
            / target_sums
        )
        self.template_offsets_ = np.clip(offsets, 0, self.template_duration - 1)
        self.template_widths_ = np.clip(np.sqrt(variances), self.min_width, None)

    def _validate_data(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=float)
        if data.ndim != 2:
            raise ValueError("data must have shape (n_neurons, n_time_bins)")
        if data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError("data must have at least one neuron and one time bin")
        if np.any(data < 0):
            raise ValueError("data must contain nonnegative counts")
        if self.n_neurons is not None and data.shape[0] != self.n_neurons:
            raise ValueError("data neuron count does not match n_neurons")
        return data

    def _validate_amplitudes(self, amplitudes: np.ndarray) -> np.ndarray:
        amplitudes = np.asarray(amplitudes, dtype=float)
        if amplitudes.ndim != 2 or amplitudes.shape[0] != self.n_templates:
            raise ValueError("amplitudes must have shape (n_templates, n_time_bins)")
        if np.any(amplitudes < 0):
            raise ValueError("amplitudes must be nonnegative")
        return amplitudes

    def _check_initialized(self) -> None:
        required = ("base_rates_", "template_scales_", "template_offsets_", "template_widths_")
        if not all(hasattr(self, name) for name in required):
            raise ValueError("PPSeq must be fit before this method is used")

    def _poisson_log_likelihood(self, data: np.ndarray, rates: np.ndarray) -> float:
        rates = np.clip(rates, self.min_rate, None)
        return float(np.sum(data * np.log(rates) - rates - gammaln(data + 1.0)))

    def _poisson_log_likelihood_per_bin(
        self, data: np.ndarray, rates: np.ndarray
    ) -> np.ndarray:
        rates = np.clip(rates, self.min_rate, None)
        return np.sum(data * np.log(rates) - rates - gammaln(data + 1.0), axis=0)

    def _bin_nelpy_spikes(
        self,
        st: nel.SpikeTrainArray,
        *,
        start: float | None,
        stop: float | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if start is None and stop is None:
            binned = st.bin(ds=self.bin_size)
        else:
            if start is None or stop is None:
                raise ValueError("start and stop must both be provided for nelpy slicing")
            binned = st[nel.EpochArray([[start, stop]])].bin(ds=self.bin_size)
        return np.asarray(binned.data, dtype=float), np.asarray(binned.bin_centers, dtype=float)


def bin_spikes(
    spike_times: np.ndarray | nel.SpikeTrainArray,
    neuron_ids: np.ndarray | None = None,
    *,
    bin_size: float = 0.02,
    start: float | None = None,
    stop: float | None = None,
    n_neurons: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bin spike times into a count matrix.

    Parameters
    ----------
    spike_times : np.ndarray or nel.SpikeTrainArray
        Spike timestamps in seconds, or a nelpy spike train array.
    neuron_ids : np.ndarray, optional
        Integer neuron identifier for each spike when ``spike_times`` is an
        array.
    bin_size : float, optional
        Bin size in seconds. By default 0.02.
    start : float, optional
        Start time in seconds. If None, inferred from the data.
    stop : float, optional
        Stop time in seconds. If None, inferred from the data.
    n_neurons : int, optional
        Number of neurons. If None, inferred from data.

    Returns
    -------
    tuple of np.ndarray
        ``(counts, time_centers)`` where counts has shape
        ``(n_neurons, n_time_bins)``.
    """
    model = PPSeq(
        n_templates=1,
        template_duration=1,
        n_neurons=n_neurons,
        bin_size=bin_size,
    )
    return model.bin_spikes(
        spike_times,
        neuron_ids,
        start=start,
        stop=stop,
        n_neurons=n_neurons,
    )


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
    Compare PPSeq template counts by held-out log-likelihood.

    Parameters
    ----------
    train_data : np.ndarray
        Training spike count matrix with shape ``(n_neurons, n_time_bins)``.
    test_data : np.ndarray
        Held-out spike count matrix with shape ``(n_neurons, n_time_bins)``.
    template_counts : list of int or np.ndarray
        Candidate numbers of templates to evaluate.
    template_duration : int
        Template duration in bins.
    bin_size : float, optional
        Bin size in seconds. By default 0.02.
    fit_num_iter : int, optional
        Number of fitting EM iterations for each candidate. By default 50.
    transform_num_iter : int, optional
        Number of held-out amplitude inference iterations. By default 25.
    random_state : int, optional
        Base seed. Candidate ``k`` uses ``random_state + i`` where ``i`` is the
        candidate index. If None, stochastic initialization uses OS entropy.
    **ppseq_kwargs
        Additional keyword arguments passed to :class:`PPSeq`.

    Returns
    -------
    pd.DataFrame
        One row per candidate with columns ``n_templates``,
        ``train_log_likelihood``, ``heldout_log_likelihood``,
        ``heldout_log_likelihood_per_bin``, and ``heldout_log_likelihood_per_spike``.
    """
    train_data = np.asarray(train_data, dtype=float)
    test_data = np.asarray(test_data, dtype=float)
    if train_data.ndim != 2 or test_data.ndim != 2:
        raise ValueError("train_data and test_data must be 2D count matrices")
    if train_data.shape[0] != test_data.shape[0]:
        raise ValueError("train_data and test_data must have the same number of neurons")

    rows = []
    for idx, n_templates in enumerate(template_counts):
        seed = None if random_state is None else int(random_state) + idx
        model = PPSeq(
            n_templates=int(n_templates),
            template_duration=template_duration,
            n_neurons=train_data.shape[0],
            bin_size=bin_size,
            random_state=seed,
            **ppseq_kwargs,
        )
        fit_result = model.fit(train_data, num_iter=fit_num_iter)
        heldout_result = model.transform(test_data, num_iter=transform_num_iter)
        heldout_ll = float(heldout_result.log_likelihood[-1])
        heldout_spikes = float(np.sum(test_data))

        rows.append(
            {
                "n_templates": int(n_templates),
                "train_log_likelihood": float(fit_result.log_likelihood[-1]),
                "heldout_log_likelihood": heldout_ll,
                "heldout_log_likelihood_per_bin": heldout_ll / test_data.shape[1],
                "heldout_log_likelihood_per_spike": (
                    heldout_ll / heldout_spikes if heldout_spikes > 0 else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def ppseq_seed_stability(
    data: np.ndarray,
    n_templates: int,
    seeds: list[int] | np.ndarray,
    *,
    template_duration: int,
    bin_size: float = 0.02,
    fit_num_iter: int = 50,
    **ppseq_kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Estimate PPSeq template stability across random seeds.

    Parameters
    ----------
    data : np.ndarray
        Spike count matrix with shape ``(n_neurons, n_time_bins)``.
    n_templates : int
        Number of templates to fit for each seed.
    seeds : list of int or np.ndarray
        Random seeds to evaluate.
    template_duration : int
        Template duration in bins.
    bin_size : float, optional
        Bin size in seconds. By default 0.02.
    fit_num_iter : int, optional
        Number of fitting EM iterations per seed. By default 50.
    **ppseq_kwargs
        Additional keyword arguments passed to :class:`PPSeq`.

    Returns
    -------
    tuple of pd.DataFrame
        ``(summary, pairwise)``. ``summary`` contains aggregate stability
        statistics. ``pairwise`` contains one row per seed pair with the
        permutation-matched mean template cosine similarity.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be a 2D count matrix")

    fitted_templates = []
    for seed in seeds:
        model = PPSeq(
            n_templates=n_templates,
            template_duration=template_duration,
            n_neurons=data.shape[0],
            bin_size=bin_size,
            random_state=int(seed),
            **ppseq_kwargs,
        )
        model.fit(data, num_iter=fit_num_iter)
        fitted_templates.append(model.templates)

    rows = []
    for i in range(len(fitted_templates)):
        for j in range(i + 1, len(fitted_templates)):
            similarity = template_set_similarity(fitted_templates[i], fitted_templates[j])
            rows.append(
                {
                    "seed_a": int(seeds[i]),
                    "seed_b": int(seeds[j]),
                    "mean_matched_template_similarity": similarity,
                }
            )

    pairwise = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "n_templates": int(n_templates),
                "n_seeds": len(seeds),
                "mean_stability": (
                    float(pairwise["mean_matched_template_similarity"].mean())
                    if len(pairwise)
                    else np.nan
                ),
                "median_stability": (
                    float(pairwise["mean_matched_template_similarity"].median())
                    if len(pairwise)
                    else np.nan
                ),
                "min_stability": (
                    float(pairwise["mean_matched_template_similarity"].min())
                    if len(pairwise)
                    else np.nan
                ),
            }
        ]
    )
    return summary, pairwise


def template_set_similarity(templates_a: np.ndarray, templates_b: np.ndarray) -> float:
    """
    Compute permutation-matched cosine similarity between two template sets.

    Parameters
    ----------
    templates_a : np.ndarray
        Template tensor with shape ``(n_templates, n_neurons, n_time_bins)``.
    templates_b : np.ndarray
        Template tensor with shape ``(n_templates, n_neurons, n_time_bins)``.

    Returns
    -------
    float
        Mean cosine similarity after optimal one-to-one template matching.
    """
    templates_a = np.asarray(templates_a, dtype=float)
    templates_b = np.asarray(templates_b, dtype=float)
    if templates_a.ndim != 3 or templates_b.ndim != 3:
        raise ValueError("templates_a and templates_b must be 3D tensors")
    if templates_a.shape[1:] != templates_b.shape[1:]:
        raise ValueError("template sets must have matching neuron/time dimensions")
    if templates_a.shape[0] != templates_b.shape[0]:
        raise ValueError("template sets must have the same number of templates")

    flat_a = templates_a.reshape(templates_a.shape[0], -1)
    flat_b = templates_b.reshape(templates_b.shape[0], -1)
    flat_a /= np.clip(np.linalg.norm(flat_a, axis=1, keepdims=True), 1e-12, None)
    flat_b /= np.clip(np.linalg.norm(flat_b, axis=1, keepdims=True), 1e-12, None)
    similarity = flat_a @ flat_b.T
    row_ind, col_ind = linear_sum_assignment(-similarity)
    return float(np.mean(similarity[row_ind, col_ind]))
