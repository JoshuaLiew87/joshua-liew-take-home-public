# Sub-pixel laser spot localization

Simulation and Analysis of sub-pixel localization of a noisy laser beacon for a 
camera-guided pointing and tracking system. Evaluation done on a MacBook Pro and 
an NVIDIA Jetson Orin Nano Development Kit. A live GUI is included for visualization.

## Start here

Read the [technical report](report.md) or the [PDF version](report.pdf) first
for the methodology, assumptions, quantitative results, estimator comparison,
runtime analysis, dynamic recovery, and limitations.

## Setup

Create and activate a local Python virtual environment, then install the
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, create the environment with `py -m venv .venv`, then activate it
with `.venv\Scripts\activate.bat` in Command Prompt or
`.\.venv\Scripts\Activate.ps1` in PowerShell. On a Jetson or other Linux
system, use the commands above. The project uses CPU NumPy/SciPy execution;
CUDA or TensorRT is not required.

## Run the complete study

The recommended reproducible command is:

```bash
python run_all.py --config configs/final.yaml
```

This generates the static SNR characterization, theoretical precision
reference, ROI study, runtime benchmark, controlled estimator ablation,
dynamic trajectory/disturbance analysis, multi-seed amplitude repeat,
real-world stress analysis, figures, and a run manifest.

During execution, `run_all.py` prints the resolved seed and experiment sizes,
sanity checks for the sensor timing, estimator ROIs, and disturbance sampling,
stage-level progress, and a short summary of the saved RMSE, lock-loss,
runtime, and disturbance-recovery results. It also warns when the Gaussian
fit exceeds the nominal 1 ms estimator budget. The detailed values remain in
the CSV/JSON outputs and the run manifest.

The newest results are copied to `results/latest/`. Timestamped copies are
retained under `results/runs/`.

At a high level, the command runs ideal characterization, estimator ablation,
stressed-sensor characterization, dynamic repeatability, ROI sensitivity,
runtime benchmarking, dynamic disturbance recovery, outdoor-condition stress
analysis, and finally manifest/latest-snapshot generation. The first eight
stages produce scientific results; the last stage records provenance and
organizes the outputs.

Run the tests with:

```bash
python -m pytest -q
```

The test suite currently contains 44 tests. These are automated regression and
sanity checks for the simulator, pixel-integrated PSF, sensor-noise chain,
centroid and Gaussian estimators, analytical Jacobian, configuration
validation, trajectory generation, experiment outputs, and reproducibility
contracts. They are intended to catch implementation regressions such as
coordinate mistakes, incorrect noise ordering, invalid estimator results, or
broken output files.

`pytest` verifies that the implementation is behaving consistently; it does
not replace the Monte Carlo characterization or runtime benchmark. Use
`python run_all.py --config configs/final.yaml` to regenerate the quantitative
figures and results, and use `pytest` as the fast correctness check before or
after making changes.

## Design and implementation summary

- Pixel-integrated circular Gaussian spot model.
- Configurable photon, read, dark-current, background, PRNU, hot-pixel,
  quantization, and saturation effects.
- Background-subtracted windowed centroid.
- Bounded pixel-integrated Gaussian fit with an analytical Jacobian.
- Monte Carlo bias, precision, RMSE, failure, and lock-loss characterization
  versus SNR, with uncertainty bars.
- Controlled sensor-effect ablations explaining the clean/stressed estimator
  ranking reversal.
- Clean-model Fisher-information/CRLB reference.
- Runtime benchmark against a 1 ms / 1 kHz frame budget.
- Dynamic drift, correlated jitter, and periodic-disturbance recovery.
- Multi-seed disturbance-amplitude repeatability analysis.
- Controlled outdoor-condition stress analysis.
- Optional live GUI using the same simulator and estimators, including a
  day/night illumination sweep with bounded automatic exposure/gain control.

The centroid is the low-latency baseline. The Gaussian fit is generally more
accurate when the spot is observable, but is more computationally expensive
and can lose lock at very low SNR or under model mismatch.

## Key results

The values below are from the seeded Mac reference run in `results/latest/`.
The Jetson target-validation result is reported separately because runtime is
machine and load dependent.

| Experiment                  | Key result                                                                                                                                                                                    | Interpretation                                                                                                                   |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Static localization         | At SNR 10, radial RMSE was 0.161 px for the centroid and 0.081 px for the Gaussian fit in the stressed profile.                                                                               | The Gaussian fit improved accuracy when the spot remained observable.                                                            |
| Low-SNR lock loss           | At SNR 1, stressed-profile lock loss was 96.5% for the centroid and 90.1% for the Gaussian fit.                                                                                               | Both methods become unreliable; failure accounting is essential.                                                                 |
| Clean theoretical reference | The ideal-model radial precision reference was 0.0564 px at SNR 10.                                                                                                                           | The CRLB is a clean-model reference, not a universal bound for the stressed sensor.                                              |
| Dynamic recovery            | The 73 Hz disturbance was recovered within 0.007 Hz; 5,000-frame RMS tracking error was 0.164 px (centroid) and 0.082 px (Gaussian).                                                          | Both estimators recovered the injected periodic motion; the Gaussian fit gave lower error.                                       |
| Runtime (Mac reference)     | Centroid median latency was 0.055 ms; the full Gaussian fit was 1.819 ms warm on the stressed profile.                                                                                        | The centroid is the Python low-latency baseline; Gaussian fitting is better suited to supervisory or compiled paths.             |
| Jetson target validation    | On the Jetson target, centroid median latency was 0.485 ms and full Gaussian warm latency was 10.482 ms; seeded accuracy and dynamic metrics matched the Mac run to floating-point precision. | The centroid remains within the estimator-only 1 ms budget; the Gaussian fit remains a slower supervisory or compiled candidate. |
| Saturation stress           | At 100× nominal flux, centroid RMSE was 0.363 px versus 0.007 px for the clipped-pixel-masked Gaussian fit.                                                                                   | Saturation handling materially changes estimator behavior.                                                                       |

See the report for uncertainty estimates, controlled ablations, assumptions,
failure modes, and the exact output paths for every result.

The Jetson target-validation run used Ubuntu 22.04.5 on aarch64 with JetPack R36.4.3,
Python 3.13.13, NumPy 2.5.3, and SciPy 1.18.1. It used the CPU NumPy/SciPy
implementation; CUDA and TensorRT were not used. The saved runtime, dynamic
summary, and manifest artifacts are in `results/performance/jetson_orin/`.

![Static sensor characterization](docs/figures/final_sensor_characterization.png)

*Representative result: bias, precision, radial RMSE, and lock-loss behavior
across the SNR sweep for the stressed sensor profile.*

The primary `configs/final.yaml` keeps the full-quality Gaussian settings. To
experiment with a faster tracking configuration, use
[`configs/fast_gaussian.yaml`](configs/fast_gaussian.yaml) with the runtime
benchmark:

```bash
python -m experiments.benchmark_runtime \
  --profile dynamic \
  --config configs/fast_gaussian.yaml \
  --sequence dynamic \
  --frames 200 \
  --warmup 20 \
  --output-dir results/performance/gaussian_tracking_fast \
  --figure results/performance/gaussian_tracking_fast/runtime_benchmark.png
```

This optional study changes the Gaussian ROI, IRLS pass count, optimizer
limit, and tracking initialization policy. It does not replace the primary
results generated by `run_all.py`.

## Key assumptions

- Sensor: 32×32 pixels, 1 ms exposure, nominal 1 kHz frame rate.
- Spot: circular Gaussian with an approximately 7-pixel 1/e² diameter,
  corresponding to sigma = 1.75 pixels.
- Brightness: controlled through total expected spot electrons; target SNR is
  converted to flux using a pixel-centred reference spot.
- Sensor model: 12-bit ADC, 1 electron/DN, 3-electron RMS read noise, PRNU,
  dark current, background gradients, and persistent hot pixels.
- Dynamic drift: bounded Ornstein–Uhlenbeck process with correlated jitter and
  one 73 Hz sinusoidal disturbance.
- Lock-loss analysis: finite estimates with radial error greater than 1 pixel
  are classified as out of lock.
- CRLB: a clean Poisson Fisher-information reference for the ideal model
  (with a configurable read-noise term, set to zero in the ideal run), not a
  universal bound for saturated, quantized, or contaminated sensor frames.

The decision log is in
[`docs/decisions.md`](docs/decisions.md). The original assignment is preserved
in [`docs/assignment_spec.md`](docs/assignment_spec.md), and the development
plan is in [`docs/plan.md`](docs/plan.md).

## Live demonstration

Launch the optional review GUI with:

```bash
python -m experiments.live_gui \
  --profile sensor \
  --snr 10
```

The interactive window requires a graphical display. On a headless Jetson, set `MPLBACKEND=Agg` when using `--save-gif`; GIF export does not require a display.

The GUI shows the simulated sensor frame, true and estimated positions,
trajectory history, and position error. It provides controls for SNR, motion,
noise effects, sensor effects, and estimator selection.

The `Scene brightness` control sweeps from a low-background night scene through
the nominal midpoint to a bright daytime scene. Enabling `Auto exposure/gain`
lets the GUI respond to saturation and underexposure using bounded, hysteretic
updates. The status panel displays the scene level, effective SNR, exposure,
gain, saturation count, and lock status.

With the figure focused:

- Arrow keys: nudge the true position by 0.1 pixel.
- `Shift+S`: low-flux preset, SNR 1.
- `Shift+D`: high-flux preset, SNR 1000.
- `G`: toggle the background gradient.
- `H`: inject a persistent ROI-edge hot pixel for demonstration.
- `Q` or `Escape`: close the window.

Matplotlib's default keyboard shortcuts are disabled for this figure. The GUI
is a demonstration layer; its display timer is intentionally slower than 1
kHz. The offline benchmark and `run_all.py` outputs are the authoritative
results.

![Animated live simulator demonstration](docs/figures/live_demo.gif)

The animation uses the same production simulator and estimators as the batch
experiments. It also demonstrates the optional GUI-only illumination sweep and
automatic exposure/gain response. This is a qualitative review demonstration,
not a hardware-camera exposure controller or a replacement for the saved
quantitative results.

To generate the animated review asset locally:

```bash
python -m experiments.live_gui \
  --profile sensor \
  --snr 10 \
  --save-gif docs/figures/live_demo.gif \
  --scene-sweep \
  --auto-exposure
```

## Output locations

- Static characterization: `results/latest/results/final_ideal/` and
  `results/latest/results/final_sensor/`
- Theoretical reference: `results/latest/results/final_ideal/`
- Runtime benchmarks: `results/latest/results/runtime_ideal/` and
  `results/latest/results/runtime_sensor/`
- Dynamic trajectory and disturbance analysis:
  `results/latest/results/dynamic/`
- Outdoor stress analysis: `results/latest/results/real_world/`
- Estimator ablation: `results/latest/results/estimator_ablation/`
- Dynamic amplitude repeats: `results/latest/results/dynamic_repeats/`
- Optional Gaussian tracking speed study: `results/performance/`
- Jetson target validation: `results/performance/jetson_orin/`
- Figures: `results/latest/figures/`
- Standalone runtime command defaults: `results/runtime_sensor/` and
  `results/standalone/figures/` when `--output-dir` and `--figure` are omitted.
- Reproduction metadata: `results/latest/run_manifest.json`

The simulator and estimators use `image[y, x]` array indexing and `(x, y)`
pixel-centre coordinates. All experiments use deterministic named random
streams derived from a master seed.
