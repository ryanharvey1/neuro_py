# Delta wave detection

`detect_delta_waves` returns a pandas DataFrame and saves a CellExplorer
`basename.deltaWaves.events.mat` file by default. It never replaces an existing
file. When comparing against a previous detection, choose a different event name:

```python
from neuro_py.detectors import detect_delta_waves

events = detect_delta_waves(
    basepath=r"S:\data\HMC\HMC1\day8",
    event_name="deltaWavesPython",
)
```

The default session workflow uses the `Delta` channel tag, or selects the channel
with the greatest relative 1–6 Hz power among the requested cortical regions.
Channels tagged `Bad` are excluded. Public channel arguments use **zero-based**
indexing; MATLAB session tags are converted automatically. For example,
`channel=53` is MATLAB channel 54. A list of channels restricts automatic selection
to those candidates.

Detection is restricted to NREM and EMG below 0.6 for at least one second. The
session must contain `.session.mat`, `.lfp` (or `.eeg`),
`.SleepState.states.mat`, and `.EMGFromLFP.LFP.mat`. Set `nrem_restrict=False`
and/or `emg_threshold=float("inf")` to disable these restrictions explicitly.
Use `detection_epochs` to further restrict the recording and `ignore_intervals`
to exclude stimulation or other unwanted periods; both accept seconds-based
arrays of shape `(n_intervals, 2)` or `nelpy.EpochArray` objects.

To inspect detection without saving:

```python
events = detect_delta_waves(basepath, channel=53, save_mat=False)
events[["start", "peaks", "stop", "duration", "amplitude"]].head()
```

For a supplied single-channel trace, explicitly disable session restrictions:

```python
events = detect_delta_waves(
    lfp=trace,
    fs=1250,
    timestamps=sample_times,  # Optional; defaults to arange(n_samples)/fs.
    nrem_restrict=False,
    emg_threshold=float("inf"),
    save_mat=False,
)
```

## Algorithm and differences from neurocode

The morphology detector follows `FindDeltaWaves.m`: subtract a fifth-order
0.5 Hz Chebyshev-II lowpass from the raw LFP, lowpass at 9 Hz, differentiate,
and lowpass the derivative at 9 Hz using order eight. Both filtering stages
use forward/backward filtering with 20 dB stopband attenuation. Upward/downward/
upward derivative zero crossings identify the initial trough, positive peak,
and final trough. Wave amplitudes use sample standard deviations pooled across
the usable segments. The initial trough must be below the peak, and waves must
last 150–650 ms. Acceptance requires either a peak above 3 SD with a nonpositive
final trough, or a peak at least 1 SD with a final trough below -1.5 SD. The
peak-to-final-trough difference must also exceed `threshold=3`.

`filter_band` controls morphology filtering; `passband` controls channel selection
only. This distinction reflects what the MATLAB code actually does.

The Python implementation uses stable second-order sections at the native
sampling rate. It filters each contiguous usable segment separately, excludes
NaNs and artifacts instead of interpolating over them, and rejects events within
`edge_buffer=1` second of segment boundaries. Artifacts use the MATLAB wrapper's
6 SD amplitude and 10 SD difference criteria, with 0.5 s and 0.1 s padding.
The difference threshold is in SD units of the raw signal, as in `CleanLFP.m`.
These choices prevent events across gaps but can exclude waves retained by MATLAB.
The Python sample clock starts at zero, consistent with other neuro_py loaders;
the MATLAB wrapper starts at `1/fs`.

Firing verification defaults to `False`, following the wrapper's documented
default (its executable default is `True`). Enable `verify_firing=True` to raise
the amplitude cutoff based on cortical population firing suppression, optionally
supplying `spike_times` as one seconds-based array per unit. Otherwise cortical
spikes are loaded through `loading.load_spikes`. This heuristic uses 10 ms bins
over ±0.5 s, Gaussian smoothing with sigma one bin, per-event z scores averaged
over ±50 ms, and a moving mean over up to 100 amplitude-ordered events. It uses
the lowest amplitude cutoff with mean suppression ≤-0.5 and never lowers the
requested amplitude threshold. Missing spikes or absent suppression produce a
warning and retain the LFP detections; metadata records verification as unsuccessful.
No interactive debugger or channel picker runs.

## Output and export

Each row contains `start`, `stop`, `timestamps`, `peaks`, `amplitude`,
`amplitudeUnits`, `values`, `eventID`, `eventIDlabels`, `eventIDbinary`, `center`,
and `duration`. Times are seconds; amplitude is the peak-to-final-trough
difference in z units. The `values` tuple holds standardized initial trough,
peak, and final trough values, also exposed as `start_value`, `peak_value`, and
`stop_value` columns. All events have ID 1 and label `delta`.

Shared metadata lives in `events.attrs["detectorinfo"]`, including the usable
detection intervals, parameters, effective amplitude threshold, UTC detection
date, and detection channel in both indexing conventions. The saved MATLAB
struct follows the [CellExplorer events format](https://cellexplorer.org/datastructure/data-structure-and-format/#events),
including `(n_events, 2)` timestamps, column vectors for event measurements, and
an `(n_events, 3)` values array. Zero-event and one-event exports preserve these shapes.

You can save a reviewed DataFrame later:

```python
from neuro_py.detectors import save_delta_wave_events

filename = save_delta_wave_events(events, basepath, event_name="deltaWavesReviewed")
```

Both saving entry points use exclusive file creation. An existing destination
raises `FileExistsError`; choose another name or use `save_mat=False`.
