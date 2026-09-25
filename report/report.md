# Sub-pixel spot localization under noise

> Working draft. This report will be completed as the simulator and experiments are implemented. No result should be stated as measured until it is reproduced by `run_all.py`.

## 1. Executive summary

TODO: Summarize the simulator, estimator comparison, dynamic result, real-time recommendation, and main limitations.

## 2. Problem framing and conventions

TODO: Describe the optical pointing problem, coordinate convention, pixel units, exposure, frame rate, and scope of the simplified model.

The complete methodology is shown in [methodology_flowchart.md](methodology_flowchart.md).

## 3. Simulator and sensor model

The Phase 1 simulator uses a pixel-integrated circular Gaussian PSF. Pixel
centers have integer coordinates, pixels are stored as `image[y, x]`, and a
spot position is represented as `(x, y)`. The assignment's approximately
7-pixel 1/e² diameter is converted to sigma using `sigma = diameter / 4`,
giving a nominal sigma of 1.75 pixels.

Spot brightness is represented internally by total expected spot electrons,
`N_total`. The experiment-facing `target_snr` control is converted to `N_total`
using a pixel-centered reference spot. Each frame records the reference SNR
and an effective SNR based on the actual position and sigma.

The implemented sensor chain is:

1. render spot, background, and dark-current expectations in electrons;
2. apply a persistent PRNU gain map;
3. Poisson-sample the total expected charge;
4. add Gaussian read noise in electrons;
5. convert to DN;
6. apply full-well/ADC clipping and round-to-nearest quantization.

Hot pixels are represented as persistent pixels with elevated dark-current
rates. The simulator exposes the noiseless expected components, sampled charge,
ADC image, saturation mask, and frame metadata so later experiments can
separate physical effects from estimator behavior.

All major effects are independently switchable through `NoiseConfig`, allowing
controlled ablation tests without changing the simulator or experiment code.

Cover explicitly:

- photon/Poisson noise;
- Gaussian read noise;
- dark current;
- hot pixels;
- non-uniform background gradient;
- pixel-gain non-uniformity;
- ADC bit depth, quantization, and saturation;
- target SNR control and the mapping from target SNR to total spot flux.

Phase 1 currently validates the PSF, fixed-pattern maps, SNR-to-flux
calibration, noise statistics, reproducibility, quantization, saturation,
individual noise switches, background-gradient orientation, exposure scaling,
edge behavior, ADC invariants, and frame-metadata consistency. The current
foundation and simulator test suite contains 20 passing tests.

## 4. Estimators

### 4.1 Windowed intensity-weighted centroid

TODO: Describe peak detection, ROI, background estimation, subtraction policy, quality flags, and expected failure modes.

### 4.2 Pixel-integrated 2D Gaussian fit

TODO: Describe the fitted parameters, pixel-integrated model, noise weighting, analytical Jacobian, initialization, bounds, convergence handling, and model mismatch.

## 5. Theoretical precision reference

TODO: Derive and explain the Fisher-information/CRLB reference, nuisance parameters, noise approximation, and why clipping, quantization, and unmodeled fixed-pattern effects are outside the clean bound.

## 6. Static characterization

TODO: Document Monte Carlo design, SNR sweep, trial counts, random-position and pixel-phase sampling, ROI sensitivity, error bars, and failure handling.

Include measured bias, standard deviation, RMSE, outliers/failures, and comparison with the theoretical reference.

## 7. Real-time performance

TODO: Report hardware/software environment, cold/warm timing, median, p95, p99, worst observed time, convergence failures, and the practical 1 kHz tradeoff.

## 8. Dynamic tracking and disturbance recovery

TODO: Describe the drift, jitter, periodic disturbance, SNR, frame count, persistent sensor maps, and why the scenario is challenging.

Report true versus recovered trajectory, position error, frequency, amplitude, uncertainty, detrending/windowing, and filtering effects.

## 9. Real-world conditions and robustness

TODO: For each condition, describe its image signature, effect on bias/variance/outliers/latency, detection signal, and mitigation.

At minimum discuss scintillation intensity fluctuations, beam wander/angle of arrival, thermal focus drift, vibration and motion blur, changing background, saturation/blooming, dust/defocus/misalignment, calibration drift, rolling shutter/timing, and end-to-end latency.

## 10. Assumptions and limitations

Every material assumption must appear in the register below and be traceable to `report/decisions.md`. Numerical values must include units and must be labelled as an assumption, calibration, measurement, or limitation.

| ID | Assumption or value | Units | Rationale | Expected effect | Validation or limitation |
|---|---|---|---|---|---|
| A-001 | Pixel-center coordinate convention | pixels | Keeps PSF, estimators, and theory consistent | Affects sub-pixel phase and boundary handling | Tested by PSF symmetry/shift tests |
| A-002 | Image array is indexed as `image[y, x]`; positions are `(x, y)` | — | Separates array and Cartesian conventions | Prevents axis swaps | Tested in foundation tests |
| A-003 | Pixel-integrated Gaussian PSF | — | Represents finite pixel area | Affects pixel-phase behavior and CRLB | Tested by flux/symmetry checks |
| A-004 | Total expected spot electrons control brightness | electrons | Makes photon budget physically meaningful | Target SNR is mapped to total flux | SNR calibration experiment |
| A-005 | 7 px 1/e² diameter implies sigma = 1.75 px | pixels | Conversion from assignment convention | Sets spot width, ROI, and precision | Algebra and simulator tests |
| A-006 | Named RNG streams from one master seed | — | Reproducible independent experiments | Affects repeatability, not expected means | Same-seed test |
| A-007 | 32×32 frame, 1 ms exposure, 1 kHz nominal rate | pixels, s, Hz | Simplified real-time scenario | Affects runtime and boundary margin | Configuration validation |
| A-008 | 73 Hz primary disturbance | Hz | Bin-aligned for 2,000 and 5,000 frames at 1 kHz | Simplifies primary spectral result | Confirmed from sampling/grid calculation |
| A-009 | PRNU multiplies expected total charge before Poisson sampling | dimensionless gain | Simple sensor-response convention | Changes mean and shot-noise statistics | Isolated simulator tests; model limitation |
| A-010 | Hot pixels are elevated persistent dark-current rates | electrons/s | Captures fixed locations with stochastic dark noise | Creates persistent outliers and extra shot noise | Fixed-map test; simplified hot-pixel model |
| A-011 | Read noise is additive Gaussian noise in electrons; ADC quantization rounds to nearest DN | electrons RMS, DN | Standard simplified readout model | Controls low-light floor and quantization behavior | Read-noise and ADC tests |
| A-012 | Reference SNR uses a spot centered on a pixel; actual effective SNR is recorded separately | dimensionless | Avoids making the SNR control depend on random sub-pixel phase | Actual localization information can vary by phase | SNR calibration test and metadata |
| A-013 | Stressed sigma variation is 5% RMS around 1.75 pixels | pixels | Represents small optics variation without changing the clean reference | Changes peak signal and estimator model match | Sigma metadata; dynamic/static experiments |
| A-014 | Default stressed sensor uses 1% PRNU, 50 electrons/s dark current, 4 hot pixels at 5,000 electrons/s, 3 electrons RMS read noise, 12-bit ADC, and 1 electron/DN | mixed | Explicit engineering assumptions for the simplified camera | Sets realistic stress level and saturation feasibility | Sensitivity analysis and report limitation |
| A-015 | Major noise/effect switches are controlled independently through `NoiseConfig` | — | Supports ablation tests and isolates causes of estimator error | Makes component-level validation possible | Toggle test; combinations still require experiment coverage |
| A-016 | Visual sanity check reuses the production simulator API | — | Prevents a separate visualization model from hiding simulator errors | Improves development/debugging confidence | Headless figure generation |
| — | Additional assumptions to be added during implementation | — | — | — | — |

## 11. Failure modes and future work

TODO: State where each estimator breaks, what was not implemented, and what would be done with more time. Separate completed optional extensions from proposed future work.

## 12. Reproduction

TODO: Document environment setup, `python run_all.py`, test command, configuration profile, seed, expected outputs, and approximate runtime.

## 13. Assignment traceability matrix

This matrix must be complete before submission. “Evidence” should identify the report section, implementation module, result file, or figure.

| Assignment requirement | Evidence | Status |
|---|---|---|
| Configurable Gaussian spot at known sub-pixel position | Section 3; `sim/psf.py`; simulator tests | Implemented in Phase 1 |
| Photon, read, dark, hot-pixel, gradient, PRNU, and quantization noise | Section 3; `sim/noise.py`; Phase 1 tests | Implemented in Phase 1 |
| Explicit SNR control and sweep | Section 3/6; `config.py`; SNR calibration tests | Implemented in Phase 1; sweep pending |
| At least two centroiding methods | Section 4; `estimators/` | Planned |
| Bias and standard deviation versus SNR | Section 6; characterization results | Planned |
| Theoretical precision reference | Section 5; `theory/crlb.py` | Planned |
| Per-frame compute cost and 1 kHz tradeoff | Section 7; timing results | Planned |
| Moving spot with drift, jitter, and periodic disturbance | Section 8; `sim/trajectory.py` | Planned |
| Recovered trajectory error | Section 8; dynamic results | Planned |
| Recovered disturbance frequency and amplitude | Section 8; spectral results | Planned |
| Challenging parameters and failure modes | Sections 6, 8, and 11 | Planned |
| Outdoor real-world conditions and handling | Section 9 | Planned |
| One-command reproducibility | Section 12; `run_all.py`; README | Planned |
| Written report | This document | In progress |
| Reproducible dynamic scenario and analysis | Section 8; saved results/configuration | Planned |
