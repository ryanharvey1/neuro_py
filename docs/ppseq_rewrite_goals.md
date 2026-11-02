# PPSeq Rewrite Goals

## Primary Goal

Fit PP-Seq on task-running spikes and use the learned task sequence types as a
drop-in replay measure for pre-task and post-task sharp-wave ripples.

The analysis should:

- Detect behaviorally meaningful task-running sequences.
- Preserve spike-level probabilistic assignments to sequence events or
  background.
- Freeze task templates before sleep scoring.
- Score pre/post ripples for forward and reverse expression of task templates.
- Compare pre/post sleep with null-normalized replay metrics.

## Acceptance Criteria

- Behavioral raster shows lap-locked sequences when neurons are sorted by the
  learned template offsets.
- Behavioral diagnostics report enough sequence events, participating neurons,
  assigned spikes, direction purity, and offset-position structure.
- Replay scoring reports event-level forward/reverse scores, assigned replay
  spike counts, likelihood gain per spike, replay z-scores, and empirical
  p-values.
- The tutorial runs end-to-end for valid sessions. If task-running templates
  fail behavioral validation, replay scoring is skipped or labeled not
  interpretable with visible status tables rather than a notebook exception.

## Model Direction

The production PPSeq implementation should follow the continuous-time Bayesian
point-process model from the PP-Seq paper and `lindermanlab/PPSeq.jl`, not the
earlier compact binned EM approximation.

Version 1 targets the core sampler/replay workflow and exposes Julia-style
result histories:

- Continuous-time spike inputs in seconds.
- Sequence event times and sequence identities.
- Spike-event/background assignments.
- Per-template neuron offsets, widths, and weights.
- Assignment, latent event, global parameter, and log probability histories.
- Task fitting plus fixed-template replay scoring.

Full Julia parity items such as split-merge proposals, distributed chains, and
learned time warps are future extensions.

## Backend Choice

Use NumPy and Numba-compatible kernels where needed. Do not require PyTorch for
the core implementation.

## Scientific Guardrail

Do not interpret replay when task-running templates fail behavioral validation.
In that case, report that PPSeq did not find reliable behavioral sequences for
the session, keep the notebook running, and diagnose the failure rather than
forcing a pre/post replay claim.
