# Sub-pixel laser spot localization

Sub-pixel localization of a noisy laser beacon for a camera-guided pointing
and tracking system.

## Start here

Read the [technical report](report.md) first for the methodology, assumptions,
quantitative results, estimator comparison, runtime analysis, dynamic recovery,
and limitations.

## Setup

Create and activate a local Python virtual environment, then install the
dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

## Run the complete study

The recommended reproducible command is:

```bash
python run_all.py --config configs/final.yaml
```

This generates the static SNR characterization, theoretical precision
reference, ROI study, runtime benchmark, dynamic trajectory/disturbance
analysis, real-world stress analysis, figures, and a run manifest.

The newest results are copied to `results/latest/`. Timestamped copies are
retained under `results/runs/`.

Run the tests with:

```bash
python -m pytest -q
```

The current suite contains 38 tests.

## Design and implementation summary

- Pixel-integrated circular Gaussian spot model.
- Configurable photon, read, dark-current, background, PRNU, hot-pixel,
  quantization, and saturation effects.
- Background-subtracted windowed centroid.
- Bounded pixel-integrated Gaussian fit with an analytical Jacobian.
- Monte Carlo bias, precision, RMSE, failure, and lock-loss characterization
  versus SNR, with uncertainty bars.
- Clean-model Fisher-information/CRLB reference.
- Runtime benchmark against a 1 ms / 1 kHz frame budget.
- Dynamic drift, correlated jitter, and periodic-disturbance recovery.
- Controlled outdoor-condition stress analysis.
- Optional live GUI using the same simulator and estimators.

The centroid is the low-latency baseline. The Gaussian fit is generally more
accurate when the spot is observable, but is more computationally expensive
and can lose lock at very low SNR or under model mismatch.

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
- CRLB: a clean Poisson-plus-read-noise reference for the ideal model, not a
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

The GUI shows the simulated sensor frame, true and estimated positions,
trajectory history, and position error. It provides controls for SNR, motion,
noise effects, sensor effects, and estimator selection.

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

## Output locations

- Static characterization: `results/latest/results/final_ideal/` and
  `results/latest/results/final_sensor/`
- Theoretical reference: `results/latest/results/final_ideal/`
- Runtime benchmarks: `results/latest/results/runtime_ideal/` and
  `results/latest/results/runtime_sensor/`
- Dynamic trajectory and disturbance analysis:
  `results/latest/results/dynamic/`
- Outdoor stress analysis: `results/latest/results/real_world/`
- Figures: `results/latest/figures/`
- Reproduction metadata: `results/latest/run_manifest.json`

The simulator and estimators use `image[y, x]` array indexing and `(x, y)`
pixel-centre coordinates. All experiments use deterministic named random
streams derived from a master seed.
