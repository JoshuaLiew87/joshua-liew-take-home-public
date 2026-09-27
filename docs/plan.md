# Sub-pixel spot localization take-home plan

## 1. Objective and scope

Build a small, reproducible Python study of sub-pixel laser-spot localization under sensor noise. The submission should demonstrate four things:

1. A configurable image simulator with a physically interpretable noise model.
2. Two estimators with measured bias, precision, failure behavior, and runtime.
3. Recovery of a moving spot and a periodic disturbance from a noisy sequence.
4. Clear reasoning about theoretical limits, real-time tradeoffs, assumptions, and outdoor failure modes.

The work is planned for one week. The baseline is more important than feature count. No optional feature should be started until the simulator, estimator comparison, dynamic experiment, report, and reproducible command are complete.

The primary implementation will use Python, NumPy, SciPy, Matplotlib, and pytest. All experiments will be deterministic when given a seed.

## 2. Definition of done

The final repository must provide:

- `python run_all.py` as the single command that regenerates the final results and figures.
- A simulator whose parameters are configurable without editing experiment code.
- An explicit `target_snr` control and `snr_sweep` that map requested SNR values to the simulator's physical total spot flux.
- At least two estimators:
  - a windowed, background-corrected intensity centroid;
  - a pixel-integrated, noise-weighted 2D Gaussian fit.
- Monte Carlo curves showing signed bias, standard deviation, RMSE, and failure rate versus SNR.
- A theoretical position-precision reference derived from the simulator's forward model.
- Runtime measurements including tail latency, not only average runtime.
- A bounded target-validation runtime record on the NVIDIA Jetson Orin, kept separate from the Mac reference because timing is machine-dependent.
- A dynamic scenario with drift, jitter, and one periodic disturbance: 2,000 frames for development and up to 5,000 frames for the final result if the runtime remains acceptable.
- Frequency and amplitude recovery results for the injected disturbance.
- A written analysis of outdoor failure modes and possible detection/mitigation strategies.
- Tests for the forward model, noise statistics, estimator sanity, and reproducibility.
- A report that states assumptions and limitations explicitly.
- A report-level assumption register and assignment traceability matrix showing where every required item is addressed.

Supplemental explanation checks are part of the final review run: a small
controlled ablation at SNR 1 and 3 to explain any estimator-ranking reversal,
a multi-seed repeat to quantify disturbance-amplitude variation, and a
non-unity electrons/DN sensor-chain test. These checks support the report's
interpretation but do not replace the primary static or five-second dynamic
results.

The submission will deliberately distinguish a clean theoretical reference from a stressed sensor scenario. A CRLB derived for an ideal continuous-valued model will not be presented as an exact bound for clipped, quantized, hot-pixel-contaminated data.

## 3. Design principles

### Modular and configurable

All tunable values should be held in structured configuration objects rather than scattered through experiment scripts. The configuration should be divided into independent sections:

The dataclasses in `config.py` remain the source of truth and validation
schema. Optional YAML files under `configs/` act only as strict overrides for
`run_all.py`; omitted values inherit the named profile defaults, and unknown
sections or fields are rejected.

- `SensorConfig`: frame dimensions, exposure time, bit depth, conversion gain, full-well capacity, ADC range, and read noise.
- `SpotConfig`: total spot flux, nominal sigma, sigma variation, and allowed position margin.
- `BackgroundConfig`: offset, gradient/plane coefficients, and temporal variation.
- `PatternConfig`: PRNU statistics, dark-current map, and hot-pixel settings.
- `NoiseConfig`: independent switches for photon shot noise, read noise, dark current, background, PRNU, and hot pixels.
- `EstimatorConfig`: ROI size, background model, parameter bounds, optimizer tolerances, and iteration limits.
- `ExperimentConfig`: target SNR values/sweep, trial counts, frame counts, confidence interval settings, and output paths.
- `TrajectoryConfig`: selectable drift model, drift RMS/correlation time,
  valid-position bounds, jitter, periodic disturbance, and dynamic-sequence
  seed. The final dynamic experiment will use a discrete
  Ornstein–Uhlenbeck drift, equivalent to a first-order AR(1)/low-pass
  filtered random process;
  the current sinusoidal drift remains available as a deterministic sanity
  mode for the GUI.

Use named scenario profiles such as `ideal`, `sensor`, and `dynamic`. A scenario should override configuration values, not duplicate simulator or estimator logic.

### Reproducibility

- Use one master seed and derive independent random-number-generator streams for spot positions, shot noise, read noise, fixed-pattern maps, and trajectories.
- Save the effective configuration and seed alongside every result file.
- Keep fixed sensor maps fixed for an entire sequence. PRNU, hot-pixel locations, and fixed-pattern background must not be redrawn on every frame.
- Save machine-readable results such as CSV or JSON in addition to figures.
- Use a non-interactive Matplotlib backend in `run_all.py`.
- Pin minimum dependency versions in `requirements.txt` and record the benchmark machine in the report.

### Assumption discipline

Every non-trivial assumption must be recorded when it is introduced. At minimum, record:

- an identifier;
- the assumption or chosen parameter;
- its numerical value and units, where applicable;
- why it is reasonable for this simplified exercise;
- what result it can affect;
- how it is tested, calibrated, or stress-tested;
- whether it is a baseline assumption, measured value, or known limitation.

`decisions.md` is the working decision log. Before submission, all relevant entries must be consolidated into an assumptions section in `report.md`. No value should appear only in code or only in a plot caption.

### Documentation as part of implementation

Documentation must be updated during the phase in which a decision or result is introduced. Do not defer all documentation until the end of the week. Each phase must update the relevant parts of:

- `../report.md`: methodology, assumptions, results, figures, and limitations;
- `decisions.md`: decision and assumption IDs with rationale and consequences;
- the assignment traceability matrix: requirement evidence and status;
- `../README.md`: setup, commands, configuration profiles, and reproduction instructions when they change;
- saved configuration/result metadata: parameter values, seeds, software versions, and scenario names.

At every phase checkpoint, the documentation should be sufficient for another person to understand what was added, why it was added, how it was validated, and what remains incomplete.

### Fair estimator comparison

The estimator input contract must be explicit. The main comparison will use a full 32×32 image for peak detection, followed by a configurable local ROI for estimation. Both estimators will use the same peak-detection and ROI-selection policy. A separate tracking-mode benchmark may use the previous estimate as initialization, but it must not be mixed with the acquisition-mode result.

At low SNR, an estimator is allowed to fail. It must return an invalid result or a quality flag rather than an arbitrary bounded position. Report lock/failure rate and outlier rate in addition to conditional error statistics.

## 4. Repository structure

```text
transcelestial-spot-localization/
├── README.md
├── requirements.txt
├── config.py
├── run_all.py
├── sim/
│   ├── psf.py
│   ├── noise.py
│   ├── simulator.py
│   └── trajectory.py
├── estimators/
│   ├── common.py
│   ├── centroid.py
│   └── gaussian_fit.py
├── theory/
│   └── crlb.py
├── experiments/
│   ├── characterize_snr.py
│   ├── profile_realtime.py
│   ├── dynamic_tracking.py
│   └── live_gui.py
├── tests/
├── results/
├── report.md
└── docs/
    ├── assignment_spec.md
    ├── plan.md
    ├── decisions.md
    └── methodology_flowchart.md
```

The final `run_all.py` workflow stores each execution under
`results/runs/<UTC timestamp>/`, with separate `results/` and `figures/`
subdirectories plus a manifest. It refreshes `results/latest/` with the most
recent run so report links remain stable without discarding run history.

The optional parabolic estimator is intentionally excluded from the initial implementation. It can be added only if the baseline is complete and it contributes meaningfully to the real-time tradeoff.

## 5. Coordinate system and sensor conventions

Use pixel-center coordinates consistently. Pixel centers are integer coordinates, and pixel boundaries are half a pixel from their centers. The simulator, estimator, trajectory generator, and theoretical model must all use the same convention.

The spot model will be parameterized by total expected spot electrons rather than an arbitrary image amplitude. This makes flux, shot noise, SNR, and sigma variation easier to interpret. If sigma changes while total flux is fixed, the peak pixel value changes naturally.

The sensor model must define these units explicitly:

- electrons for photocharge, dark current, and read noise;
- DN for ADC output;
- electrons per DN for conversion gain;
- full-well capacity in electrons;
- ADC saturation level of `2**bit_depth - 1` DN.

Read noise is added in electrons before conversion to DN. The full-well and
ADC limits are applied as an equivalent charge-domain clip before conversion;
round-to-nearest quantization and the final DN-range clamp are applied after
conversion.

## 6. Phase 0: foundations and quantitative conventions

### 6.1 Spot-width conversion

Use the convention

\[
I(r)=I_0\exp\left(-\frac{2r^2}{w^2}\right)
\]

where (w) is the 1/e² radius. Comparing with

\[
I(r)=I_0\exp\left(-\frac{r^2}{2\sigma^2}\right)
\]

gives `sigma = w / 2`. A 7-pixel diameter at 1/e² corresponds to `w = 3.5` pixels and a nominal sigma of `1.75` pixels.

Use a small configurable sigma variation, for example 5% RMS, in the stressed scenario. The clean reference scenario will hold sigma fixed so that the estimator and CRLB comparison are easy to interpret.

### 6.2 SNR definition

Expose the requested SNR as the experiment-level control through `target_snr` and `snr_sweep`. Internally, use the total expected spot flux, `N_total`, in electrons as the simulator control. A calibration step maps each requested SNR to the corresponding `N_total`. This makes the photon budget physically meaningful while still satisfying the requirement for an explicit SNR sweep.

The expected peak-pixel signal, `S_peak`, is derived from `N_total`, sigma, pixel integration, and position.

For plotting and experiment labels, define a reference peak SNR using a spot centered on a pixel:

\[
\mathrm{SNR}_{peak}=
\frac{S_{peak}}
{\sqrt{S_{peak}+B_{peak}+D_{peak}+\sigma_r^2}}.
\]

Here `B_peak` is expected background charge, `D_peak` is expected dark charge during the exposure, and `sigma_r` is read noise in electrons RMS. This is a reference SNR label, not a claim that every sub-pixel position has exactly the same measured peak SNR. The simulator must record the actual expected peak value as a diagnostic.

For the clean SNR experiment, hold sigma and the background fixed while varying `N_total`. Enable sigma variation only in the stressed scenario so that the SNR sweep remains interpretable.

The SNR-to-flux calibration must not silently create impossible sensor settings. Before the final sweep, calculate the corresponding peak charge, conversion gain, full-well, and ADC level. Save the requested SNR, calibrated `N_total`, reference SNR, and saturation status in the result metadata. Use two explicitly named characterization modes if necessary:

- `ideal_reference`: no ADC quantization or clipping, allowing a broad SNR sweep such as 1–1000 for comparison with the theoretical model;
- `sensor_realistic`: fixed 12-bit ADC, conversion gain, and full well, with an SNR range that is physically achievable without saturation.

The final SNR experiment therefore follows this flow:

```text
requested target SNR → calibrate N_total → render frame → estimate position
```

The report will explain this distinction rather than overlaying one CRLB curve on incompatible sensor regimes.

### 6.3 Theoretical reference

Derive a Fisher-information reference for the pixel-integrated forward model. The parameter vector should include position and relevant nuisance parameters, at minimum:

\[
\theta=(x_0,y_0,A,\sigma,b_0,b_x,b_y).
\]

For the clean Poisson-plus-read-noise approximation, compute the expected pixel means and their derivatives. Use the diagonal variance approximation implemented by `experiments/theory.py`, include nuisance parameters in the Fisher matrix, and use the pseudoinverse or Schur complement to obtain the position covariance reference. Label it as a clean-model Fisher approximation rather than an exact bound for the full Poisson-Gaussian convolution.

The report and `experiments/theory.py` should explain:

1. the likelihood/noise approximation;
2. the expected pixel mean;
3. the derivative with respect to position;
4. assembly of the Fisher matrix;
5. treatment of nuisance parameters;
6. why quantization, clipping, and hot-pixel model mismatch are not represented by the clean bound.

The final result will be described as a theoretical precision reference, not a universal accuracy guarantee. If a Gaussian approximation with parameter-dependent variance is used, document which terms are retained and why.

## 7. Phase 1: simulator

### 7.1 Pixel-integrated PSF

Render the Gaussian by integrating it over each pixel area, using the separability of a circular Gaussian and error functions. Do not evaluate the Gaussian only at pixel centers.

The simulator must expose both the noiseless expected image and the sampled sensor image. This allows tests and experiments to distinguish forward-model error from noise-estimation error.

Add a lightweight visual sanity-check tool that reuses the simulator API. It
must display or save the expected spot, background/dark current, sampled ADC
frame, and charge residual, with the true position, peak pixel, SNR metadata,
sigma, total flux, and saturation count shown. This is a development and
debugging tool, not the authoritative Monte Carlo pipeline.

Test and document:

- total integrated flux;
- response to sub-pixel position changes;
- behavior near the edge of the frame;
- sigma variation with fixed total flux;
- consistency of the pixel-coordinate convention.

### 7.2 Noise and sensor chain

Use this documented simplified sensor chain:

1. Generate spot, background, and dark-current expected charge in electrons.
2. Apply a fixed per-pixel gain/PRNU realization to the expected charge.
3. Draw one Poisson realization of the total expected photocharge and dark charge.
4. Represent hot pixels as fixed locations with elevated dark-current means or fixed additive offsets, with the choice stated explicitly.
5. Add independent Gaussian read noise in electrons.
6. Combine full-well and ADC limits into an equivalent charge limit and clip
   the charge in electrons.
7. Apply electrons-to-DN conversion.
8. Quantize to the configured bit depth and clamp to the final DN range.

The exact PRNU convention will be recorded in `decisions.md`; the important requirement is that the same convention is used by the simulator, tests, and report.

The following effects must be independently toggleable:

- photon shot noise;
- dark current;
- read noise;
- background gradient;
- PRNU;
- hot pixels;
- quantization;
- saturation/clipping;
- sigma variation.

These switches are implemented through `NoiseConfig` so ablation tests can
disable one effect without changing the simulator pipeline or experiment code.

### 7.3 Initial configuration

Start with:

- 32×32 pixels;
- 1 ms exposure and 1 kHz nominal frame rate;
- nominal sigma 1.75 pixels;
- spot position at least 8 pixels from every boundary in characterization;
- 12-bit ADC in the realistic sensor profile;
- configurable conversion gain and full-well capacity;
- a modest background gradient and approximately 1% PRNU in the stressed profile;
- a small number of persistent hot pixels.

The exact read-noise and dark-current values will be stated as engineering assumptions, not presented as measurements of a particular camera.

## 8. Phase 2: estimators

### 8.1 Windowed centroid

Implement a common full-frame peak-detection and local ROI-selection helper. The full 32×32 image is used to locate a candidate peak; neither estimator may use the known simulator position. Estimation then operates on a cropped, odd-sized ROI.

Within the ROI:

- estimate the local background from a border or annulus;
- optionally fit a local plane when a gradient is enabled;
- subtract the background and clamp only clearly negative residuals according to a documented policy;
- compute the intensity-weighted centroid;
- return a quality flag containing peak level, saturation status, total positive signal, and background estimate.

The estimator should expose its ROI and background policy through `EstimatorConfig`. Initial defaults are an 11×11 or 13×13 centroid ROI and a 15×15 Gaussian-fit ROI. The centroid ROI will be selected after a small ROI-sensitivity experiment; 11×11 is not assumed to be unbiased.

Expected behavior to explain: the centroid is cheap and often effective at moderate/high SNR, but is sensitive to background gradients, hot pixels, saturation, truncation, and pixel-phase effects.

### 8.2 Pixel-integrated Gaussian fit

Fit a pixel-integrated Gaussian jointly with position, total flux, sigma, and a background model. Use bounded parameters and robust initialization from the centroid/peak detector. Use a 15×15 default fit ROI to retain the spot's tails while reducing optimizer cost relative to fitting the full frame.

Use SciPy's bounded `least_squares` with an analytical Jacobian for the pixel-integrated model. Validate the analytical Jacobian against finite differences at multiple parameter settings before using it for the final experiments.

Use noise-weighted nonlinear least squares with expected variance approximately

\[
V_i \approx \mu_i + \sigma_r^2,
\]

or an explicitly documented Poisson/read-noise likelihood approximation. If weights depend on fitted parameters, use a small fixed number of iteratively reweighted steps rather than an unbounded inner loop, and document whether the weight derivatives are included in the residual Jacobian.

This will be called a noise-weighted fit or quasi-MLE unless the implemented likelihood exactly includes the sensor model. Quantization, clipping, and hot pixels will be treated as model mismatch in the baseline fit, with saturated/hot pixels optionally masked using quality flags.

Expected behavior to explain: the fit uses the spatial shape and can estimate background and sigma jointly, so it should approach the clean-model precision reference when the model is correct. It is more computationally expensive and can fail or become biased under saturation, bad initialization, strong gradients, or an incorrect PSF.

## 9. Phase 3: Monte Carlo characterization

### 9.1 Experiment design

Use a short pilot sweep first to validate behavior and runtime. For final results, use approximately 10–12 log-spaced SNR points and about 1,000 trials per point, reducing the number only if runtime becomes unreasonable and documenting the change.

Use independent random sub-pixel positions within the safe frame margin. Also run a smaller pixel-phase sweep with positions covering a regular grid of sub-pixel phases so systematic phase bias is not hidden by averaging.

Run a small ROI-sensitivity check at representative low, medium, and high SNR values using centroid ROIs of 11×11, 13×13, and 15×15. Select the smallest ROI whose truncation bias and failure rate are acceptably close to the larger reference ROI. The Gaussian-fit ROI will default to 15×15 unless this check demonstrates a safe smaller value.

Run at least these scenarios:

1. **Clean reference:** known background model, no PRNU/hot pixels, no clipping, no quantization or a sufficiently fine ADC.
2. **Stressed sensor:** gradient, read noise, dark current, persistent PRNU, hot pixels, quantization, and sigma variation.

The same generated frames or equivalent random streams must be used for both estimators wherever practical.

### 9.2 Metrics

For each estimator, SNR, and axis, report:

- signed bias;
- standard deviation;
- RMSE;
- median and 95th-percentile absolute error;
- radial error statistics;
- covariance/correlation between x and y errors;
- invalid-result and outlier rate;
- conditional results for successful locks.

Use error bars such as bootstrap confidence intervals or standard errors, and state which one is used. Plot bias and precision separately rather than showing only one error curve.

Overlay the clean theoretical position reference where it is applicable. Explain deviations in terms of nuisance parameters, pixelation, estimator bias, model mismatch, saturation, and finite sample uncertainty.

Phase 3 implementation checkpoint: `experiments/characterize_snr.py` now
provides a reproducible SNR sweep, shared trial positions and frames for both
estimators, unconditional and locked-conditional metrics, covariance and
correlation, CSV/JSON output, and bias/precision/RMSE/lock-loss plots.
`experiments/theory.py` provides the phase-averaged clean Poisson Fisher
reference and `experiments/roi_sensitivity.py` evaluates the candidate ROI
sizes. The final ideal and stressed sweeps use 1,000 trials per SNR point; the
stressed ROI study uses 200 trials at representative SNR values. The artifacts
and their interpretation are recorded in `report.md`; Phase 3 is complete.

## 10. Phase 4: real-time measurement

Benchmark estimators on already-generated frames so simulator cost is not confused with estimator cost. Include warm-up iterations and report:

- median time;
- p95 and p99 time;
- worst observed time in the benchmark;
- failure/convergence rate;
- cold-start and warm-start fit timing if both are considered.

Benchmark the full Python implementation on the Mac reference development
machine and on the NVIDIA Jetson Orin target-validation device. Record Python,
NumPy, SciPy, platform, and JetPack/OS versions for both. Compare each
estimator time to the 1 ms frame interval, while explicitly noting that camera
transfer, preprocessing, synchronization, filtering, and gimbal output also
consume the real system budget.

The engineering recommendation will distinguish:

- the simplest method that is comfortably fast;
- the most accurate method under clean conditions;
- a practical hybrid such as centroid initialization followed by a bounded/warm-started fit when confidence is high.

No claim will be made that a Python timing directly proves embedded real-time suitability.

Phase 4 implementation checkpoint: `experiments/benchmark_runtime.py` now
generates frames before timing and reports median, p95, p99, worst-case,
failure, optimizer-iteration, and 1 ms deadline-miss statistics. It measures
centroid cold-start and Gaussian cold/warm-start paths and records Python,
NumPy, SciPy, and platform versions. The initial 200-frame SNR-10 benchmark
has been run for both ideal and stressed profiles; final interpretation is
documented in `report.md`. The same final stressed benchmark was subsequently
run on the Jetson Orin; its runtime and dynamic-summary artifacts are retained
under `results/performance/jetson_orin/`.

### Phase 4 follow-up: configurable Gaussian tracking speed study

Keep the acquisition-quality Gaussian defaults unchanged, but expose the
performance-sensitive choices in `EstimatorConfig`: Gaussian ROI size, IRLS
pass count, optimizer evaluation limit, previous-position ROI placement, and
whether centroid reinitialization is skipped when a previous fit is valid.
The optional `configs/fast_gaussian.yaml` profile uses a 13 × 13 ROI, one IRLS
pass, a 30-evaluation limit, and tracking-mode reuse of the previous estimate.

Benchmark both configurations on a sequential dynamic trajectory in addition
to the fixed-position acquisition benchmark. Report the two input contracts
separately because a warm start on an acquisition benchmark does not answer
the same question as a warm start on consecutive tracking frames. Validate
the fast profile against accuracy, lock loss, p95/p99 latency, and worst-case
time before considering any default change.

Follow-up checkpoint: the fast profile is reproducible from the benchmark
module, its raw JSON/CSV outputs are saved, and the report states whether the
reduced configuration meets a useful latency margin rather than only its
median time.

## 11. Phase 5: dynamic tracking

### 11.1 Scenario

Generate 2,000 frames at 1 kHz during development. For the final result, use up to 5,000 frames if the runtime remains acceptable and the additional frequency resolution improves the disturbance measurement. Keep the trajectory inside the valid ROI with margin. Use independent position components where appropriate, with the periodic disturbance injected primarily on one axis so it can be identified clearly.

The final trajectory will contain:

- a slow, bounded, correlated stochastic drift generated by a discrete
  Ornstein–Uhlenbeck process, implemented as a first-order AR(1)/low-pass
  filtered random process. This is preferred over unconstrained Brownian
  motion because it does not wander indefinitely toward the ROI edge;
- zero-mean jitter, with a documented RMS and temporal correlation;
- one sinusoidal disturbance below Nyquist. For the primary experiment,
  choose a frequency aligned to an FFT bin so the baseline is easy to
  interpret; optionally add a non-bin-aligned case as a leakage stress test;
- an amplitude chosen after the static pilot so it is challenging but
  detectable, ideally comparable to the localization noise or jitter rather
  than overwhelmingly large.

The discrete OU drift should expose a nominal stationary RMS before any hard
bound, correlation time, and position bounds. Its random seed must be saved with the dynamic
scenario. The current GUI and early sanity checks may retain the bounded
sinusoidal drift because it makes the direction and scale of motion easy to
inspect, but the final report must identify it as a sanity mode rather than
the main dynamic-scene model.

The final report must state the numerical drift, jitter RMS, disturbance frequency, disturbance amplitude, SNR, and relationship between those quantities.

### 11.2 Recovery and analysis

For every frame:

1. render using the same persistent sensor maps as the static experiment;
2. estimate the position and record quality flags;
3. compare the raw estimate with the known trajectory;
4. optionally apply a causal tracking filter as a separate control-oriented result.

Use the raw estimates as the primary disturbance-analysis signal. Detrend the recovered signal, use a Hann-windowed FFT or periodogram for visualization and initial frequency estimation, then refine frequency and amplitude with sinusoidal least-squares fitting. Correct or explicitly account for Hann coherent gain when reading amplitudes from the spectrum. Document windowing, detrending, FFT resolution, amplitude correction, and the distinction between interpolation and true frequency resolution.

Report:

- position error over time;
- RMS and 95th-percentile trajectory error;
- frequency estimate and absolute/relative error;
- amplitude estimate and absolute/relative error;
- sensitivity to estimator failures and filtering;
- whether the disturbance is above the estimated noise floor.

The filter result, if implemented, must report its latency and attenuation of the disturbance. It must not silently replace the raw estimator result.

Phase 5 implementation checkpoint: `experiments/dynamic_tracking.py` now
generates a shared 5,000-frame dynamic sequence, estimates both trajectories,
saves per-frame results and summary metrics, and recovers the periodic
disturbance using a Hann-windowed FFT followed by sinusoidal least-squares
refinement. The `dynamic` profile uses the final OU/AR(1) drift parameters
recorded in the report. The 73 Hz frequency and 0.05-pixel amplitude were
recovered with zero estimator failures and zero lock loss at SNR 10; the
trajectory and spectrum figure is saved under the run's `figures/` directory
and copied to `results/latest/figures/`.

## 12. Phase 6: real-world conditions analysis

The report will use a table with columns for image signature, effect on estimate, detection signal, and mitigation. Cover at least:

- scintillation as intensity fluctuation, which mainly changes effective SNR and can create temporal amplitude modulation;
- atmospheric beam wander/angle-of-arrival variation, which directly moves the spot and should not be confused with scintillation;
- thermal focus or magnification drift, changing sigma and possibly centroid bias;
- platform vibration, exposure-time motion blur, and gimbal/camera mechanical coupling;
- changing sky/background illumination, gradients, glints, and nearby bright sources;
- saturation, blooming, clipping, and transient reflections;
- dust, contamination, defocus, optical misalignment, and partial occlusion;
- PRNU, dark-current drift, hot pixels, lens distortion, and calibration drift;
- rolling-shutter or timestamp effects if applicable to the camera;
- latency and synchronization errors between camera, estimator, and gimbal.

For each condition, classify the dominant effect as bias, increased variance, outliers/dropouts, latency, or some combination. Suggested detection signals include saturation fraction, peak-to-background ratio, fitted sigma, residual structure, background-plane residuals, centroid/fit disagreement, temporal quality metrics, and persistent bad-pixel maps.

No full robustness implementation is required for this section. A simple scintillation extension may be added only after the baseline is complete; it would modulate spot flux over time while preserving the distinction between intensity fluctuation and beam wander.

Phase 6 implementation checkpoint: `experiments/real_world_conditions.py`
quantifies controlled scintillation, bright-background, defocus, and saturation
cases for both estimators using paired Monte Carlo trials. The report contains
the measured results, the required outdoor-condition table, detection signals,
mitigations, assumptions, and limitations. Effects such as beam wander,
motion blur, blooming, lens distortion, rolling shutter, and end-to-end
latency are addressed analytically because a full implementation of each is
outside the one-week scope. Phase 6 is complete, and the final
`run_all.py` reproducibility workflow is now implemented and verified.

## 13. Post-core live GUI phase

This phase is intentionally scheduled after the simulator, estimator,
characterization, theory, runtime, dynamic tracking, report, and
`run_all.py` workflow are complete. The GUI is a demonstration layer, not a
replacement for the reproducible experiments required by the assignment.

The implemented `experiments/live_gui.py` uses the production simulator,
estimators, and shared trajectory generator. The live display should provide:

- SNR control;
- independent toggles for photon shot noise, read noise, dark current,
  background gradient, PRNU, hot pixels, quantization, and saturation;
- centroid/Gaussian estimator selection;
- drift, jitter, periodic-disturbance amplitude, and frequency controls;
- current image, true and estimated positions, trajectory history, and
  position error;
- a day/night scene-brightness sweep with separate beacon and ambient-light
  scaling, plus an optional bounded automatic exposure/gain response;
- pause/reset controls and an explicit lock/failure status.
- keyboard review shortcuts: arrow-key 0.1-pixel manual nudges, low/high-flux
  presets, background-gradient toggle, and a controlled persistent ROI-edge
  hot pixel; Matplotlib's default keyboard shortcuts are disabled;

The GUI display timer may run below 1 kHz so the visualization remains usable.
It must label itself as a demonstration and must not be used as the evidence
for the 1 kHz requirement; the dedicated runtime benchmark owns that claim.
The live trajectory uses the same drift, correlated-jitter, and one-frequency
disturbance model as the later dynamic experiment so the demonstration stays
consistent with the report.

The illumination sweep is explicitly a GUI-only qualitative extension. It
displays effective SNR, exposure, gain, saturation count, and lock status, but
does not alter the authoritative Monte Carlo, runtime, or dynamic results.

Checkpoint: the GUI can run a moving spot, toggle configured effects, switch
estimators, and show true-versus-estimated behavior without introducing a
second simulator implementation.

## 14. Optional extension policy

There is already a baseline CRLB derivation and a baseline per-frame timing experiment, so neither should be treated as a separate stretch feature. The preferred optional extension is a concise end-to-end latency budget covering exposure, readout, transfer, preprocessing, estimation, filtering, and control output.

If the baseline is not complete before an optional extension is attempted, do not implement the extension. Describe it under “what I would do with more time” instead.

## 15. Tests and validation

Tests should be deterministic and test tolerances rather than exact random values. Include:

- pixel-integrated PSF flux conservation;
- correct response to a known sub-pixel shift;
- SNR calibration against the configured definition;
- requested SNR maps monotonically to total spot flux;
- requested/reference/effective SNR values are recorded consistently;
- Poisson mean/variance behavior over repeated samples;
- read-noise mean and variance;
- dark-current and hot-pixel behavior;
- fixed PRNU and hot-pixel maps persisting across frames;
- analytical Gaussian-fit Jacobian agrees with finite-difference derivatives;
- quantization levels and saturation behavior;
- noiseless estimator recovery with a realistic tolerance;
- invalid result on an intentionally unusable/empty frame;
- reproducibility with the same seed;
- nonnegative/positive-semidefinite CRLB covariance where applicable.

The noiseless centroid test must use a tolerance that accounts for finite ROI and pixel-center assignment. A forward-model fit can have a tighter numerical test; the centroid's residual pixelation bias should be measured, not hidden.

## 16. Documentation deliverables

### README

Explain installation, the one-command reproduction path, configuration profiles, output files, expected runtime, and how to run tests.

### `../report.md`

Recommended structure:

1. Executive summary and main conclusions.
2. Problem framing and coordinate conventions.
3. Sensor/simulator model and assumptions.
4. SNR definition and feasibility of the ADC configuration.
5. Estimator descriptions.
6. Fisher-information reference and limitations.
7. Static Monte Carlo results.
8. Runtime and latency discussion.
9. Dynamic trajectory and disturbance results.
10. Real-world conditions and robustness strategies.
11. Failure modes and limitations.
12. What I would do with more time.
13. Reproduction instructions.

Every important plot should state the scenario, seed/trial count, estimator settings, and whether failures were included or excluded. The report should distinguish the Mac reference run from any embedded-target validation and should record the target platform, software versions, and the estimator-only scope of the timing.

The report must also contain these two explicit tables:

#### Assumption register

Include every material assumption made in the simulator, estimators, theory, dynamic scenario, timing experiment, and real-world analysis. Use columns for ID, assumption/value, units, rationale, expected effect, and validation/limitation.

#### Assignment traceability matrix

Map each assignment requirement to the report section, implementation module, result file or figure, and validation status. The matrix must cover:

- simulator and every requested noise source;
- SNR control and sweep;
- both centroiding methods;
- bias/standard-deviation characterization and theoretical reference;
- runtime measurement and 1 kHz tradeoff;
- dynamic trajectory, recovered trajectory, and disturbance analysis;
- failure modes and real-world conditions;
- reproducibility instructions and all final deliverables.

The report should follow the assignment order where practical: baseline simulator, estimators, characterization, real-time analysis, dynamic tracking, real-world conditions, and optional extensions. This makes it easy for a reviewer to verify coverage.

### `decisions.md`

Maintain a concise decision log containing each non-obvious choice, its justification, and its consequence. Examples include coordinate convention, SNR definition, conversion gain, hot-pixel model, CRLB assumptions, ROI policy, optimizer bounds, trajectory parameters, and frequency-analysis method. Assign stable IDs such as `A-001`, `A-002`, and so on, so the final report can reference every assumption consistently.

## 17. Execution and completion status

The implementation is organized by technical phase rather than a fixed
calendar schedule. This reflects the actual workflow: foundational simulator
work was validated first, followed by estimator development, quantitative
characterization, runtime and dynamic analysis, robustness analysis, optional
GUI work, and final integration. Work within a phase may overlap when results
expose a needed correction in an earlier phase.

### Phase status

| Phase | Scope | Status and evidence |
|---|---|---|
| Foundations and simulator | Units, coordinates, pixel-integrated PSF, sensor/noise chain, SNR calibration, and reproducibility | Complete; `sim/`, `config.py`, foundation tests, and saved manifests |
| Estimators | Windowed background-subtracted centroid and pixel-integrated Gaussian fit with analytical Jacobian | Complete; `estimators/`, estimator tests, and static characterization |
| Theory and static characterization | Fisher-information reference, Monte Carlo SNR sweeps, error bars, lock-loss analysis, ROI sensitivity, and ranking ablation | Complete; `experiments/theory.py`, `experiments/characterize_snr.py`, and `experiments/estimator_ablation.py` |
| Runtime analysis | Fixed-position and sequential timing, tail latency, warm starts, optional fast Gaussian configuration, and Jetson target validation | Complete; the Mac reference and Jetson CPU timings agree on the 1 kHz tradeoff, while the Python Gaussian fit remains above the deterministic 1 ms target |
| Dynamic tracking | Bounded OU drift, correlated jitter, periodic disturbance, trajectory recovery, frequency/amplitude analysis, and multi-seed amplitude repeat | Complete; `experiments/dynamic_tracking.py` and `experiments/dynamic_repeats.py` |
| Real-world analysis | Scintillation, background, defocus, saturation, outdoor failure modes, and mitigations | Complete analysis with selected controlled stress experiments |
| Live GUI | Optional review demonstration using the production simulator and estimators | Implemented; display-only and not authoritative for 1 kHz claims |
| Final integration | YAML-configured `run_all.py`, manifest, report, PDF, README, tests, and traceability | Complete; the latest audited run generated the final result set and the full test suite passes |

The reproducible submission commands are:

```bash
python run_all.py --config configs/final.yaml
python -m pytest -q
```

The first command generates the figures, numerical outputs, dynamic data,
runtime measurements, and run manifest. The second verifies the implementation
independently of the generated results.

## 18. Phase-gate rule

After each phase, stop and record:

- what was implemented;
- what was measured;
- the assumptions introduced;
- why those assumptions are reasonable;
- what failed or remains uncertain;
- whether the next phase should proceed unchanged.

Also update the relevant report sections, assumption/decision entries, traceability status, README instructions, and result metadata before declaring the phase complete. A phase is not complete if its implementation exists but its reasoning and reproducibility documentation has not been updated.

This keeps the work defensible in the live review and prevents a late feature from silently invalidating earlier measurements.
