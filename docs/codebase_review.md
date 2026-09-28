# Codebase review and live-review study guide

This document follows data through the implementation rather than listing files in
isolation. Line references point to the current source tree. The main entry
point is [run_all.py](../run_all.py); its job is orchestration and provenance.
The numerical work is deliberately kept in the sim/, estimators/, and
experiments/ packages.

## 1. Mental model

The system has five layers:

1. config.py resolves a ScenarioConfig from defaults, a named profile, and
   optional YAML overrides.
2. sim/simulator.py turns a known (x, y) position and a brightness control into
   a noisy ADC frame plus metadata.
3. estimators/ turns only the image array into an EstimateResult; it does not
   receive the simulator's true position.
4. experiments/ repeats those calls, computes errors, aggregates statistics,
   performs dynamic spectral analysis, and writes CSV/JSON/PNG outputs.
5. run_all.py creates a timestamped run directory, invokes the experiment
   stages, writes a manifest, and copies the run to results/latest/.

The coordinate contract is important: NumPy arrays are indexed as image[y, x],
while positions are stored as (x, y). Pixel centers are integer coordinates and
pixel boundaries are half a pixel away. The simulator and both estimators use
this convention.

## 2. Entry point and configuration resolution

### 2.1 Command-line path

The final command is:

    python run_all.py --config configs/final.yaml

run_all.py:641-675 parses the output directory, optional run ID, YAML path, and
pipeline overrides. main() at run_all.py:678-737 then:

1. calls load_yaml_overrides() from config.py;
2. separates pipeline values from per-profile profiles values;
3. rejects unknown pipeline keys;
4. gives explicit CLI values precedence over YAML values;
5. passes the profile overrides and pipeline arguments to run_pipeline().

The YAML file supplies the final seed and counts (configs/final.yaml:4-16),
plus the dynamic OU and disturbance settings (configs/final.yaml:23-33).
scenario_config() in config.py:361-427 starts from the dataclass defaults,
selects the ideal, sensor, or dynamic profile, applies strict nested overrides
through apply_scenario_overrides() (config.py:273-315), and calls
ScenarioConfig.validate() (config.py:232-257).

run_pipeline() (run_all.py:319-638) resolves three independent objects:

    ideal = scenario_config("ideal", overrides=...)
    sensor = scenario_config("sensor", overrides=...)
    dynamic = scenario_config("dynamic", overrides=...)

If master_seed is supplied, run_pipeline() injects it into all three profiles
(run_all.py:356-367). It then resolves the trial/frame counts from the command
line, YAML, or profile defaults (run_all.py:371-386). The final run uses 1,000
static trials per SNR, 200 ROI trials, 200 runtime frames plus 20 warm-up calls,
5,000 dynamic frames, 300 outdoor trials, 250 ablation trials, and five
2,500-frame repeat records.

### 2.2 Deterministic random streams

make_rng_streams(master_seed) in common/rng.py:22-39 creates one SeedSequence,
spawns six children, and returns named generators for spot width, positions,
shot noise, read noise, fixed pattern, and trajectory. This prevents adding
draws to one stream from silently changing unrelated streams. It makes seeded
simulation products reproducible for the same code, configuration, and
dependency behavior. It does not make wall-clock latency, plot rasterization,
or a different NumPy/SciPy build invariant.

## 3. Data-flow trace A: static SNR characterization

This is the complete path used for the main bias/precision/RMSE/lock-loss
results. The corresponding experiment is
experiments/characterize_snr.py.

### 3.1 Orchestration call

The first scientific stage is run_all.py:418-443:

    ideal_summaries = run_snr_sweep(ideal, trial_count=static_count)
    ideal_dir = run_results_root / "final_ideal"
    save_sweep_results(ideal_dir, ideal, ideal_summaries,
                       profile="ideal", outlier_threshold_px=1.0)
    ideal_theory = compute_position_crlb(ideal)
    save_theory_results(ideal_dir, ideal, ideal_theory)
    plot_sweep(ideal_summaries,
               run_figures_root / "final_ideal_characterization.png",
               theory=ideal_theory)

The stressed path is the same sequence at run_all.py:471-491, but uses the
sensor profile, writes final_sensor, and does not overlay the clean Fisher
reference. Both calls use run_snr_sweep() with the default SNR tuple from
ExperimentConfig.snr_sweep unless overridden.

### 3.2 Position sampling and frame generation

run_snr_sweep() has the signature:

    def run_snr_sweep(
        config: ScenarioConfig,
        *,
        snr_values: Iterable[float] | None = None,
        trial_count: int | None = None,
        outlier_threshold_px: float = 1.0,
    ) -> list[MetricSummary]:

It lives at experiments/characterize_snr.py:215-279. It validates the
configuration and positive SNR values, constructs one SpotSimulator at line
234, and samples trial_count positions once using _sample_positions()
(characterize_snr.py:194-212). The same positions are therefore reused at every
SNR, which reduces phase-distribution differences between sweep points. The
simulator's fixed-pattern maps also persist for its lifetime.

For each requested SNR and each position, lines 242-251 execute:

    frame = simulator.render(
        (float(position[0]), float(position[1])),
        target_snr=requested_snr,
    )
    centroid_result = estimate_centroid(frame.image_dn, config)
    gaussian_result = estimate_gaussian(frame.image_dn, config)

Both estimators see the exact same rendered frame.image_dn. The simulator does
not pass frame.metadata.position_xy into either estimator; the true position is
retained only by the experiment to calculate the error.

### 3.3 Simulator call chain

The concrete entry point is:

    def SpotSimulator.render(
        self,
        position_xy: tuple[float, float],
        *,
        target_snr: float | None = None,
        total_flux_e: float | None = None,
        sigma_px: float | None = None,
    ) -> FrameResult:

It is implemented at sim/simulator.py:214-326.

1. render() validates the position and rejects simultaneous target_snr and
   total_flux_e (simulator.py:230-234).
2. _choose_sigma() (simulator.py:201-212) uses a supplied width or samples the
   configured fractional width variation from the named spot RNG.
3. If a target SNR is supplied, total_flux_for_target_snr() is called at
   simulator.py:239-243. It uses a pixel-centered reference spot and solves
   S^2 = P^2 / (P + B + D + sigma_read^2) for reference peak signal P, then
   divides by the unit-flux center-pixel probability
   (simulator.py:119-145). The actual frame phase and width can therefore have
   a different effective_snr, which is recorded separately.
4. render_pixel_integrated_gaussian() (sim/psf.py:36-83) evaluates the expected
   spot signal. It computes one-dimensional pixel probabilities as normal-CDF
   differences at pixel edges and returns total_flux_e * outer(y_probability,
   x_probability). This is a finite pixel-aperture integral, not point sampling
   at the pixel center.
5. render_background() (simulator.py:54-68) creates the configured planar
   background. Dark-current expectation is selected from the persistent map
   when hot pixels are enabled (simulator.py:253-270).
6. Signal, background, and dark current are summed (simulator.py:271), then
   PRNU multiplies the expected charge (simulator.py:274-275).
7. Photon shot noise samples a Poisson distribution from that expected charge
   (simulator.py:277-280 and sim/noise.py:78-84). Read noise is then added as
   independent Gaussian electrons (simulator.py:281-287 and
   sim/noise.py:87-96).
8. convert_to_adc() (sim/noise.py:99-130) clips to the minimum of full well
   and ADC maximum when saturation is enabled, converts electrons to DN, and
   rounds to the nearest integer if quantization is enabled.
9. render() computes reference/effective SNR and returns a FrameResult.

The exact return type is the frozen dataclass at sim/simulator.py:40-51:

    FrameResult(
        image_dn: np.ndarray,
        expected_signal_e: np.ndarray,
        expected_background_e: np.ndarray,
        expected_dark_e: np.ndarray,
        expected_charge_after_prnu_e: np.ndarray,
        sampled_charge_e: np.ndarray,
        saturated_mask: np.ndarray,
        metadata: FrameMetadata,
    )

FrameMetadata (simulator.py:23-38) contains the true position, requested,
reference, and effective SNR, total flux, sigma, expected peak components, peak
pixel, and saturated-pixel count. Static estimation uses only image_dn.

### 3.4 Centroid result and error insertion

The centroid signature is:

    def estimate_centroid(
        image_dn: np.ndarray,
        config: ScenarioConfig,
    ) -> EstimateResult

at estimators/centroid.py:17-75. The return is always an EstimateResult
unless an unexpected exception escapes. Its fields are defined in
estimators/common.py:37-48:

    EstimateResult(
        method: str,
        position_xy: tuple[float, float] | None,
        success: bool,
        message: str,
        peak_pixel_xy: tuple[int, int] | None,
        roi_bounds: tuple[int, int, int, int] | None,
        diagnostics: dict[str, float | int | bool | str],
    )

If success is true, run_snr_sweep() stores:

    centroid_result.position_xy - position

in errors["centroid"][trial_index] (characterize_snr.py:252-255). If the
result is unsuccessful, that row remains [nan, nan]. The Gaussian path does
the same at characterize_snr.py:256-259.

### 3.5 Aggregation across trials

After all trials for one SNR, summarize_errors() is called once per estimator
(characterize_snr.py:261-278). The function is at characterize_snr.py:64-191
and returns a MetricSummary dataclass.

The aggregation is not one single error number:

- successful means both error components are finite. NaN rows are estimator
  failures and are excluded from numerical error statistics.
- radial_errors = norm(valid_errors, axis=1).
- failure_count is the number of NaN rows.
- outlier_count counts valid radial errors greater than the supplied 1-pixel
  threshold.
- locked contains only valid errors at or below that threshold.
- Unconditional bias, standard deviation, RMSE, median, p95, covariance, and
  correlation use all finite estimates, including valid but catastrophic
  outliers.
- Conditional bias, standard deviation, and radial RMSE use locked estimates
  only.
- Standard errors are computed for the plotted uncertainty bars. For example,
  bias standard error is std / sqrt(success_count) and lock/failure-rate error
  bars use the binomial standard-error expression.

This distinction explains why an estimator can have failure_rate == 0 but
nonzero lock_failure_rate: it returned finite coordinates, but the resulting
coordinates were more than 1 pixel from truth. The experiment's lock label is
an evaluation rule, not an internal estimator-success flag.

### 3.6 Static files and figure

save_sweep_results() (characterize_snr.py:292-321) converts each
MetricSummary.__dict__ to a row and writes:

    results/runs/<run_id>/results/final_ideal/static_characterization.csv
    results/runs/<run_id>/results/final_ideal/static_characterization.json
    results/runs/<run_id>/results/final_sensor/static_characterization.csv
    results/runs/<run_id>/results/final_sensor/static_characterization.json

The JSON wraps profile, master_seed, trial_count, threshold, configuration, and
summaries. The CSV has one row per estimator/SNR with fields for bias, standard
deviation, RMSE, uncertainty, failure and lock-loss rates, conditional locked
metrics, and x/y covariance/correlation. The figure writer is plot_sweep()
(characterize_snr.py:324-453) and produces:

    results/runs/<run_id>/figures/final_ideal_characterization.png
    results/runs/<run_id>/figures/final_sensor_characterization.png

results/latest/ is an exact directory snapshot of that run, so the stable
paths are results/latest/results/... and results/latest/figures/....

## 4. Data-flow trace B: dynamic trajectory and disturbance recovery

The dynamic implementation is in experiments/dynamic_tracking.py. The
corresponding orchestration is run_all.py:553-569:

    dynamic_result = run_dynamic_experiment(
        dynamic, target_snr=run_snr, frame_count=dynamic_count,
        lock_threshold_px=1.0,
    )
    save_dynamic_results(dynamic_dir, dynamic, dynamic_result,
                         lock_threshold_px=1.0)
    plot_dynamic_results(dynamic, dynamic_result,
                         run_figures_root / "dynamic_tracking.png")

### 4.1 Trajectory generation

generate_dynamic_sequence() has this signature:

    def generate_dynamic_sequence(
        config: ScenarioConfig,
        *,
        target_snr: float,
        frame_count: int,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]

It is at dynamic_tracking.py:59-84.

1. It makes named RNG streams from the profile's master seed (line 69), builds
   a SpotSimulator and a TrajectoryGenerator, and calls
   TrajectoryGenerator.generate(frame_count) (sim/trajectory.py:95-103).
2. TrajectoryGenerator.next_position() (trajectory.py:88-93) calls
   _position_at() for the current frame, increments the internal frame index,
   and returns (x, y).
3. With the final YAML profile, _position_at() uses the OU branch
   (trajectory.py:45-60): each step applies drift = rho * drift + Gaussian
   innovation, then clips both components to ±ou_bound_px. Correlated jitter
   is a second AR(1)-style process at trajectory.py:66-73. The periodic term
   is A*sin(2*pi*f*t) at trajectory.py:75-81 and is added to the configured
   axis. Finally both coordinates are clipped to the sensor margin
   (trajectory.py:83-86).
4. For each true position, the same simulator renders one frame at the target
   SNR (dynamic_tracking.py:80-83). The function returns the true position
   array of shape (frame_count, 2), a list of frame_count image arrays, and one
   saturated-pixel count per frame.

The simulator's dynamic fixed-pattern maps persist because one simulator is used
for the whole sequence. The same frames and truth are subsequently sent to
each estimator.

### 4.2 Per-frame estimation and time-series assembly

estimate_dynamic_trajectory() is:

    def estimate_dynamic_trajectory(
        frames: list[np.ndarray],
        config: ScenarioConfig,
        *,
        estimator: str,
        warm_start: bool = False,
    ) -> np.ndarray

at dynamic_tracking.py:87-115. It allocates an (N, 2) array filled with NaNs.
For each frame:

- centroid calls estimate_centroid(frame, config);
- Gaussian calls estimate_gaussian(frame, config, previous_position=...);
- a successful estimate is copied into that frame's row;
- a failed estimate remains NaN;
- if warm start is enabled, only a successful Gaussian result updates
  previous_position.

The primary run_dynamic_experiment() (dynamic_tracking.py:264-304) runs the
shared sequence once and calls this loop once per estimator with
warm_start=False. It returns the frozen DynamicResult at lines 48-56:

    DynamicResult(
        positions_xy: np.ndarray,
        estimates_xy: dict[str, np.ndarray],
        saturation_counts: np.ndarray,
        summaries: list[DynamicSummary],
        spectra: dict[str, tuple[np.ndarray, np.ndarray]],
    )

### 4.3 Dynamic error and lock metrics

_summarize_dynamic() (dynamic_tracking.py:211-261) computes
errors = estimates_xy - positions_xy. Finite rows are valid; NaNs are
estimator failures. Radial RMS and p95 use finite rows. A second boolean mask
marks finite rows whose radial error is at most the 1-pixel lock threshold.
failure_count counts NaNs; lock_failure_count counts both NaNs and finite
outliers. This is the same valid-estimate versus locked-estimate distinction as
the static experiment.

It then calls recover_disturbance() on the configured disturbance axis and
stores frequency, amplitude, frequency error, amplitude error, spectral noise
floor, and peak-to-noise ratio in DynamicSummary.

### 4.4 FFT path and sinusoidal refinement

recover_disturbance() is at dynamic_tracking.py:170-208:

1. _filled_detrended_signal() (lines 118-127) makes a finite mask, requires at
   least four valid samples, linearly interpolates invalid samples with
   np.interp, and removes a linear trend using SciPy detrend.
2. A Hann window is created with np.hanning(signal.size) (line 180). The
   one-sided real FFT is converted to an amplitude-like spectrum by
   2*abs(rfft(signal*window))/sum(window) (line 182). Frequencies are generated
   at the 1 kHz sample rate.
3. The search mask is 20 Hz <= f < Nyquist (line 186). The lower bound is a
   design choice intended to keep the low-frequency OU drift out of the
   disturbance search. With 1 kHz sampling, Nyquist is 500 Hz.
4. The largest masked spectral bin is selected (lines 188-189). For 5,000
   frames, the bin spacing is 1000/5000 = 0.2 Hz.
5. _fit_sinusoid() (dynamic_tracking.py:130-167) refines that coarse bin. For
   each candidate frequency it constructs a four-column design matrix containing
   an intercept, centered time, sine, and cosine. It solves linear least
   squares with np.linalg.lstsq, computes the residual sum of squares, and
   returns that objective. scipy.optimize.minimize_scalar(method="bounded")
   searches from max(1, peak - half_width) to
   min(Nyquist-1, peak + half_width), where the half-width is
   max(1 Hz, 2*FFT_resolution). The final amplitude is
   hypot(sine_coefficient, cosine_coefficient).
6. The noise floor is the median spectrum in the same search band, excluding
   frequencies within 2*resolution of the refined peak (lines 197-200).

The injected true_frequency_hz argument is carried into the function signature
and used by the caller for the summary's error calculation, but it does not
constrain the FFT search or the least-squares refinement. This is important:
recovery is data-driven, not a fit at the known injected frequency.

### 4.5 Dynamic saved artifacts

save_dynamic_results() (dynamic_tracking.py:307-370) writes:

    results/runs/<run_id>/results/dynamic/dynamic_trajectory.csv
    results/runs/<run_id>/results/dynamic/dynamic_summary.json
    results/runs/<run_id>/results/dynamic/dynamic_summary.csv
    results/runs/<run_id>/figures/dynamic_tracking.png

The trajectory CSV has one row per frame with frame, time_s, true x/y, the
saturated-pixel count, and for each estimator its x/y estimate and radial
error. Failed estimates are written as NaN. The summary CSV has one row per
estimator containing valid/failure/lock counts, RMS and p95 error, injected and
recovered frequency/amplitude, absolute errors, spectral noise floor, and
peak-to-noise ratio. The summary JSON adds the full configuration and lock
threshold. plot_dynamic_results() (dynamic_tracking.py:373-479) visualizes the
true/estimated x and y trajectories, radial error, and the 20–150 Hz portion
of the recovered spectrum.

### 4.6 Multi-seed dynamic repeat

This is a supplemental repeatability analysis, not a second primary trajectory.
run_dynamic_repeats() (experiments/dynamic_repeats.py:48-118) creates seeds
master_seed + repeat_index*1009, runs the dynamic experiment for each, and
stores one RepeatMetric per seed/estimator. It computes RepeatAggregate means,
sample standard deviations, mean amplitude bias, and amplitude standard error.
The save function at dynamic_repeats.py:121-147 writes the repeat rows to CSV
and both repeat rows and aggregates to JSON; the plot is written by
plot_repeat_results().

## 5. Estimator deep dives

### 5.1 Shared peak, ROI, and background code

find_peak() (estimators/common.py:50-63) converts the image to float, requires
a 2-D array with at least one finite pixel, replaces non-finite values with
-inf, and takes np.argmax. It returns integer (x, y), peak value in DN, and a
saturation flag. It does not know the true position.

extract_roi() (common.py:81-107) requires a positive odd ROI size, rounds the
integer center, clamps it to the image, shifts the ROI at borders so the
largest centered crop fits, and returns an ROI containing the image crop,
global x/y coordinate vectors, and half-open bounds
(x_start, x_stop, y_start, y_stop). x_grid and y_grid are broadcast views, not
copies.

fit_background_plane() (common.py:110-153) marks the outer border_width_px
rows/columns, excludes non-finite pixels, and either evaluates

    design = [1, x - cx, y - cy]
    coefficients = np.linalg.lstsq(design, border_values, rcond=None)[0]

or, if planar fitting is disabled, uses the median border value as b0 and sets
slopes to zero. It evaluates the fitted background over the whole ROI as
b0 + bx*(x-cx) + by*(y-cy).

### 5.2 Windowed intensity centroid

estimate_centroid() (estimators/centroid.py:17-75) follows this exact path:

1. Determine the ADC saturation limit if saturation is enabled.
2. Find the brightest finite pixel and extract the configured centroid ROI.
3. Fit the ROI-border background.
4. Compute residual = roi.image_dn - background.
5. If clipping is enabled, execute weights = maximum(residual, 0.0) at
   centroid.py:34-35. This is the exact negative-residual clipping point.
   Non-finite weights are replaced by zero at centroid.py:36-37.
6. Sum the weights. If the sum is no greater than
   config.estimator.minimum_signal_dn, return a failure.
7. Return weighted x/y means:

       x = sum(weights * x_grid) / total_signal
       y = sum(weights * y_grid) / total_signal

The output diagnostics include the peak, saturation flag, total positive signal,
fitted background coefficients, and ROI dimensions. A finite result can still
be badly biased by a hot pixel, clipping, a false peak, or a poor background
fit.

### 5.3 Pixel-integrated Gaussian fit, model and Jacobian

The model/Jacobian function is

    def pixel_integrated_gaussian_model_and_jacobian(
        parameters: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        background_center_xy: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]

at estimators/gaussian_fit.py:44-82. The parameter vector is

    [x0, y0, total_flux_dn, sigma_px, b0, bx, by]

For each x/y coordinate, _integrated_probability_and_derivatives() at
gaussian_fit.py:29-41 evaluates the normal-CDF difference between
coordinate-0.5 and coordinate+0.5. It also returns the analytic derivatives
with respect to center and sigma. The 2-D spot is total_flux * outer(py, px).
The Jacobian columns at gaussian_fit.py:69-80 are, in order, derivatives with
respect to x position, y position, flux, sigma, constant background, x-background
slope, and y-background slope.

This matches the simulator's pixel-integrated forward model. It is more
physically consistent than fitting point-sampled Gaussian values, and the
analytic Jacobian avoids finite-difference calls inside SciPy.

### 5.4 Gaussian estimator setup and failure paths

The public function is

    def estimate_gaussian(
        image_dn: np.ndarray,
        config: ScenarioConfig,
        previous_position: tuple[float, float] | None = None,
    ) -> EstimateResult

at estimators/gaussian_fit.py:85-271.

1. If a valid previous position is supplied and
   gaussian_tracking_use_previous_roi is true, round that position, extract
   the Gaussian ROI, and find the peak inside it (gaussian_fit.py:95-109).
   Otherwise find a full-frame peak and center the ROI there.
2. Fit the ROI-border background. Any ValueError or LinAlgError in peak, ROI,
   or background setup returns failure_result (lines 110-116).
3. Build a finite-pixel mask. If saturation is enabled, pixels with
   image_dn >= adc_max_dn are excluded (lines 118-121). Fewer than eight valid
   pixels returns failure (lines 121-128). The value 8 is a hard-coded minimum,
   not a config field.
4. Initialize x/y from a previous position when the configured skip option is
   active. Otherwise call the centroid; use its position if successful, or fall
   back to the integer peak pixel (lines 130-138).
5. Use positive background-subtracted signal to initialize flux (lines
   140-149). Non-positive signal below minimum_signal_dn returns failure.
6. Construct parameters and bounds (lines 151-185). Position is limited to the
   ROI's outer half-pixel bounds, flux is non-negative, sigma is bounded by
   configured minimum/maximum, and background coefficients are unbounded.
7. For each configured IRLS pass (lines 190-224), evaluate the current model
   and use weighted residuals with SciPy least_squares(method="trf") and the
   analytic Jacobian.

The returned successful EstimateResult stores x/y plus fitted flux, sigma,
background terms, optimizer cost, number of function evaluations, optimality,
valid/saturated pixel counts, and tracking/initialization flags
(gaussian_fit.py:238-260). The short estimate() alias at lines 264-271 simply
forwards to estimate_gaussian().

### 5.5 IRLS weighting and the variance floor

The current code is gaussian_fit.py:192-201:

    model, _ = pixel_integrated_gaussian_model_and_jacobian(...)
    variance_dn = np.maximum(
        model.ravel() / electrons_per_dn,
        config.estimator.gaussian_variance_floor_dn2,
    )
    variance_dn += read_noise_dn * read_noise_dn
    weights = 1.0 / np.sqrt(np.maximum(variance_dn, 1e-9))[valid.ravel()]

This is an approximate heteroscedastic weighting scheme. The Poisson term is
model electrons divided by electrons-per-DN; the read-noise variance is added
in DN². The configurable floor is applied after conversion to DN², so its
units agree with variance_dn. It prevents very small or negative fitted
background-tail means from creating arbitrarily large weights. The separate
1e-9 is a numerical guard after the floor, not the configured physical floor.

If the configured floor were removed, a fitted model value near zero would
produce a very large inverse-square-root weight. A negative background-tail
model could otherwise make the Poisson term invalid; if merely the maximum
with the floor were removed while retaining the later maximum with 1e-9, it
would still clamp the total variance at 1e-9, yielding weights up to about
31,623 rather than a controlled 1-DN-scale weight. The fit would be dominated
by a few tail pixels, becoming more sensitive to background-model error and
optimizer noise. The floor is therefore a robustness choice, not an additional
measurement model.

The IRLS loop does not recompute weights continuously within one SciPy solve.
It computes fixed weights from the current parameters, solves one weighted
least-squares problem, updates parameters, and repeats a configured number of
times (two in the final profile). This makes the cost bounded and predictable,
but it is only an approximation to a fully iterated likelihood optimization.

## 6. Fisher-information reference (the requested theory/crlb.py)

There is no theory/crlb.py in this repository. The actual implementation is
experiments/theory.py, imported by run_all.py:62. The public function is:

    def compute_position_crlb(
        config: ScenarioConfig,
        *,
        snr_values: Iterable[float] | None = None,
        phase_grid_size: int = 8,
        roi_size_px: int | None = None,
    ) -> dict[str, np.ndarray | int | float]

at experiments/theory.py:19-114.

For each requested SNR and each of an 8 x 8 grid of sub-pixel phases, it:

1. chooses a pixel-centered image location plus phase offsets
   (theory.py:48-60);
2. maps SNR to total flux with the same reference calibration used by the
   simulator (lines 61-63);
3. renders a clean expected pixel-integrated image and extracts the estimator's
   Gaussian ROI (lines 65-73);
4. forms the parameter vector [x0, y0, flux_dn, sigma, 0, 0, 0] and calls
   the Gaussian model/Jacobian (lines 74-84);
5. constructs diagonal approximate variance
   max(model_dn, 0)/electrons_per_dn + read_noise_dn^2 (lines 85-87);
6. divides every Jacobian row by the square root of its variance and assembles
   the Fisher matrix as weighted_jacobian.T @ weighted_jacobian (lines 88-93);
7. uses np.linalg.pinv(fisher, hermitian=True) and takes covariance entries
   [0,0] and [1,1] as x/y position variances (lines 93-95);
8. averages those variances over valid phase points and returns their square
   roots plus radial standard deviation (lines 97-113).

This is a clean diagonal Poisson/read-noise Fisher approximation, often called
a quasi-CRLB in the report. It includes nuisance parameters because the full
7-column Jacobian is inverted, but it is not the exact Fisher information for
the full Poisson-Gaussian convolution with parameter-dependent variance, nor a
bound for clipping, quantization, hot pixels, or model mismatch.

save_theory_results() (theory.py:117-151) writes theoretical_precision.csv
with requested SNR, x/y/radial standard deviations, and valid phase count. The
JSON adds the model description, full configuration, phase-grid size, ROI size,
and rows.

## 7. Failure modes: invalid versus poor-but-valid

| Component | Invalid/failure condition in code | Poor-but-valid condition |
|---|---|---|
| Shared peak/ROI/background | Non-2-D or all-nonfinite image; invalid odd ROI; no finite ROI border; linear algebra error. find_peak, extract_roi, and fit_background_plane raise; estimators convert selected ValueError/LinAlgError to failure_result. | Brightest finite pixel is a noise/hot-pixel peak; ROI is valid but centered on the wrong object; background fit is biased by contamination. |
| Centroid | Total clipped/non-finite weight <= minimum_signal_dn at centroid.py:38-48; setup exception at lines 23-32. | Any positive weighted signal returns success=True, even if the weighted mean is many pixels from truth. Saturation is only diagnosed, not intrinsically rejected. |
| Gaussian | Setup exception; fewer than 8 finite unsaturated pixels (gaussian_fit.py:118-128); non-positive initialization signal (lines 140-149); optimizer exception (lines 225-226); non-converged/non-finite final fit (lines 228-236). | least_squares converges to a bounded position with a high radial error; the optimizer can fit a noise peak or a mismatched PSF and still report success. The experiment later labels such a result lock loss if error exceeds 1 px. |
| Static experiment | Estimator result is unsuccessful or position is absent, leaving NaN. summarize_errors() counts it as failure_count. | Finite error above 1 px is valid for RMSE/bias but counted as outlier and lock_failure. |
| Dynamic experiment | Estimator failures become NaN rows in the time series. Spectral recovery raises if fewer than 4 samples are finite. | Invalid samples are linearly interpolated for spectral analysis, so the spectrum can still be produced even when the estimator had gaps. A finite estimate above 1 px is dynamic lock loss. |
| Disturbance recovery | No valid peak search can occur only if the signal has too few valid samples or the search mask is unusable. | The largest 20-Hz-to-Nyquist peak may be unrelated to the injected disturbance; the current code has no confidence gate beyond reporting the peak-to-noise ratio. |

The 1-pixel threshold is passed explicitly by run_all.py and saved in the
manifests/JSON metadata. It is an evaluation convention, not a hidden internal
quality threshold. This is why invalid result and out of lock must be reported
separately in a technical review.

## 8. Saved-artifacts map

Every file in the final run_all.py manifest is below. <run> means
results/runs/<run_id>/; an identical relative tree is copied to results/latest/
by run_all.py:629-634.

| Artifact | Contents | Main consumer |
|---|---|---|
| results/final_ideal/static_characterization.csv/json | One row per ideal estimator/SNR; bias, x/y standard deviation, RMSE, standard errors, covariance/correlation, invalid count/rate, lock-loss count/rate, conditional locked metrics, configuration/seed metadata. | Report Section 5 and Figure 3; ideal comparison. |
| results/final_ideal/theoretical_precision.csv/json | Phase-averaged Fisher reference: requested SNR, x/y/radial standard deviations, valid phase count, model/configuration metadata. | Report Section 6 and Figure 3. |
| figures/final_ideal_characterization.png | Four-panel ideal bias/precision/RMSE/failure plot with Fisher overlay. | Report Figure 3. |
| results/estimator_ablation/estimator_ablation.csv/json | Five sensor conditions x two SNRs x two estimators; lock-loss uncertainty, RMSE uncertainty, conditional RMSE, estimator disagreement, Gaussian cost and function evaluations. | Report Section 5.1 and Figure 2. |
| figures/estimator_ablation.png | Controlled lock-loss and RMSE comparison. | Report Figure 2. |
| results/final_sensor/static_characterization.csv/json | Same static metric schema for the stressed profile. | Report Section 5 and Figure 1. |
| figures/final_sensor_characterization.png | Stressed bias, precision, radial RMSE, and invalid/lock-loss panels. | Report Figure 1. |
| results/dynamic_repeats/dynamic_amplitude_repeats.csv/json | Per-seed estimator amplitude/frequency/noise-floor rows plus JSON aggregates with between-seed mean, standard deviation, bias, and standard error. | Report Section 8 and Figure 6. |
| figures/dynamic_amplitude_repeats.png | Recovered amplitude means with between-seed error bars and injected-amplitude line. | Report Figure 6. |
| results/final_roi_sensitivity/roi_sensitivity.csv/json | Static metric rows augmented with estimator and ROI size for SNR 3, 10, and 30. | Report Section 5 and ROI figure. |
| figures/final_roi_sensitivity.png | Radial RMSE and lock loss versus ROI size. | Report Section 5. |
| results/runtime_ideal/runtime_benchmark.csv/json | Centroid cold, Gaussian cold, Gaussian warm; median/p95/p99/worst/mean latency, success/failure, deadline misses, and nfev. JSON adds frame interval/software/config. | Report Section 7; ideal runtime figure. |
| figures/runtime_ideal.png | Median/p99/worst latency and fraction over 1 ms. | Runtime evidence. |
| results/runtime_sensor/runtime_benchmark.csv/json | Same runtime schema for the stressed profile. | Report Section 7 and Figure 4. |
| figures/runtime_sensor.png | Stressed runtime comparison. | Report Figure 4. |
| results/dynamic/dynamic_trajectory.csv | Per-frame time, truth, saturation count, each estimator's x/y, and radial error. | Report Section 8; recovered trajectory data. |
| results/dynamic/dynamic_summary.csv/json | Per-estimator dynamic RMS/p95/error and frequency/amplitude recovery metrics, plus configuration/threshold metadata. | Report Section 8 and Figure 7. |
| figures/dynamic_tracking.png | x/y trajectories, radial error, and 20-150 Hz spectrum. | Report Figure 7. |
| results/real_world/real_world_conditions.csv/json | Nominal/scintillation/bright-background/defocus/saturation metrics: effective-SNR range, saturation, success/lock loss, RMSE, p95, plus condition definitions. | Report Section 9 and Figure 8. |
| figures/real_world_conditions.png | Lock loss and conditional error for controlled outdoor stress cases. | Report Figure 8. |
| run_manifest.json | Workflow/version, run IDs, paths, software, master seed, all counts/thresholds, YAML overrides, and runtime-load caveat. | Report Section 11 and reproducibility. |

The final run contains 30 generated result/figure files listed in
results/latest/run_manifest.json. Standalone module commands may write to
different defaults such as results/static/, results/dynamic/, or
results/standalone/figures/; those are not the production tree unless the caller
chooses them. The optional fast-Gaussian and Jetson validation artifacts under
results/performance/ are also outside the final run_all.py manifest.

## 9. Suspicious spots and judgment calls to remember

These are the places most likely to generate a good technical-review question.

### Modeling and calibration

- sigma_from_1e2_diameter() in config.py:17-27 maps the assignment's
  approximately 7-pixel 1/e² diameter to sigma = 1.75 pixels. The simulator
  uses the equivalent Gaussian convention, not a literal 7-pixel hard disk.
- Target SNR is a reference peak SNR for a pixel-centered spot. It is not the
  exact realized SNR of every sub-pixel phase, sigma draw, PRNU realization,
  or hot-pixel map. The metadata records effective SNR for that reason.
- _reference_components() intentionally uses nominal dark current instead of a
  randomly located hot pixel for the SNR label (simulator.py:80-116). This
  makes the x-axis stable, but means the requested SNR does not fully describe
  a stressed frame's information content.
- The reference flux calculation accepts maps for interface consistency, but
  does not include PRNU or hot-pixel realization in the reference equation.

### Estimator behavior

- The centroid has only a positive-signal guard. It does not know the truth and
  does not automatically reject a plausible false peak.
- Gaussian saturation handling masks individual ADC-limit pixels, but it does
  not model blooming, clipping-induced shape changes, or a saturated ROI with
  fewer than eight remaining pixels.
- Gaussian position bounds are ROI-local bounds, not a field-wide physical
  prior. A wrong peak can therefore produce a bounded, finite, successful fit.
- gaussian_irls_iterations, gaussian_variance_floor_dn2, ROI sizes, sigma
  bounds, and optimizer_max_nfev are configuration choices, not derived from a
  camera calibration. The 8-pixel minimum is the notable nearby magic number
  that is not configurable.
- The variance floor is in DN². Applying it before converting the model from
  electrons would be a unit error; the current code converts first.

### Theory and signal processing

- The so-called CRLB is explicitly an approximation. It uses a diagonal
  variance-weighted Jacobian and pseudoinverse, not the exact likelihood Fisher
  matrix for a Poisson-plus-Gaussian ADC model.
- The phase average is over an 8 x 8 grid, not a continuous integral over phase.
- Dynamic spectral recovery interpolates missing samples before FFT, removes a
  linear trend, applies a Hann window, excludes frequencies below 20 Hz, and
  then refines only around the largest FFT peak. Those choices trade leakage and
  drift rejection against the risk of excluding a real low-frequency disturbance.
- _fit_sinusoid() includes an intercept and a centered linear-time term even
  after _filled_detrended_signal() has removed a linear trend. This is
  conservative for the local refinement but means the refinement model has
  nuisance trend columns.
- true_frequency_hz is not used to guide the search. It is used after recovery
  to compute the reported absolute error.

### Runtime interpretation

- Runtime benchmarks pre-generate frames (benchmark_runtime.py:50-68 and
  lines 71-92) and time only estimator calls. They do not include acquisition,
  transfer, ADC, filtering, synchronization, or actuator output.
- The warm-up calls are excluded from the reported timing, but the warm-start
  Gaussian state is reset before the measured sequence. Warm means previous
  position is used during measured tracking; it is not a claim that the whole
  application has been warmed for deployment.
- The 1 ms deadline is derived from config.sensor.frame_rate_hz in
  benchmark_runtime.py:124, while plotting labels explicitly describe the
  1-ms comparison. Python timing is load-dependent and is not a deterministic
  flight guarantee.
- The Gaussian speed profile changes ROI size, IRLS passes, optimizer limit, and
  initialization policy. It is a separate configured operating point, not a
  silent replacement of the full-quality estimator.

### Reproducibility and output behavior

- Every run gets a UTC timestamped directory. Reusing a run ID does not
  overwrite an old run; _make_run_id() adds _01, _02, and so on.
- Refreshing results/latest/ uses shutil.rmtree() and copytree() in the
  controlled output directory. This is intentional snapshot replacement, not
  part of the estimator algorithm.
- The final config enables the ablation and dynamic repeat stages. A quick run
  or a caller setting their counts to zero changes the stage count and omits
  those artifacts.
- results/latest is a copy, not a symlink. The timestamped archive remains the
  provenance record if a later run refreshes the latest snapshot.
- The GUI uses the production simulator and estimators but adds display-only
  interventions such as manual position offsets and a forced demo hot pixel. It
  is useful for review, but its timer and interactive state are not part of the
  authoritative quantitative run.

## 10. Short oral-defense checklist

Be able to answer these without opening the code:

1. What is the difference between requested/reference SNR and effective SNR?
2. Why does a pixel-integrated PSF use CDF differences rather than a sampled
   Gaussian value?
3. Why can failure_rate be zero while lock_failure_rate is nonzero?
4. Where is the centroid's negative-residual clipping performed?
5. Why does the Gaussian fit need a variance floor, and what are its units?
6. Which seven parameters are in the Gaussian Jacobian, and why are nuisance
   parameters retained in the Fisher inversion?
7. What exactly is included in the measured 1-ms runtime budget?
8. Why is the FFT search restricted to 20 Hz and why is a Hann window used?
9. How can the dynamic spectral analysis proceed when some estimates are NaN?
10. Which files prove the result, which files provide provenance, and which GUI
    artifacts are intentionally non-authoritative?
