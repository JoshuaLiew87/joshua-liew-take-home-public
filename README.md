# Transcelestial spot localization

This repository studies sub-pixel localization of a noisy Gaussian laser spot for a camera-guided pointing system.

## Environment

The project uses Python with NumPy, SciPy, Matplotlib, and pytest. The working Conda environment is named `transcelestial`.

```bash
conda activate transcelestial
python -m pytest -q
```

If the environment is not available, install the project dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Current status

Day 1 foundations and Phase 1 simulator foundations are implemented. The simulator currently provides:

- a pixel-integrated Gaussian PSF;
- configurable 32×32 sensor geometry and explicit electron/DN units;
- target-SNR-to-total-flux calibration;
- persistent PRNU, dark-current, and hot-pixel maps;
- independent toggles for photon, read, dark-current, background, PRNU, and hot-pixel effects;
- Poisson shot noise and Gaussian read noise;
- ADC conversion, quantization, and saturation diagnostics;
- deterministic named random-number-generator streams.

The Phase 1 validation suite currently contains 20 tests covering component
statistics, noise ablations, ADC invariants, edge behavior, metadata, and
reproducibility.

The estimator and experiment pipelines are still being implemented. The final one-command reproduction workflow will be added before submission.

## Simulator visual sanity check

The Phase 1 simulator can be inspected before the estimators are implemented:

```bash
python -m experiments.inspect_simulator --profile sensor --snr 10
```

To save a headless diagnostic figure:

```bash
python -m experiments.inspect_simulator \
  --profile sensor \
  --snr 10 \
  --x 15.25 \
  --y 16.40 \
  --output figures/simulator_sanity.png \
  --no-show
```

The four panels show the expected spot, expected background plus dark current,
sampled ADC output, and charge residual. Cyan marks the true sub-pixel position;
red marks the brightest expected spot pixel. The figure title reports target,
reference, and effective SNR, sigma, total flux, and saturation count.

## Reproducibility conventions

- Images use `image[y, x]` array indexing.
- Positions use `(x, y)` pixel-center coordinates.
- Pixel centers are integer coordinates and pixel boundaries are half a pixel away.
- Spot brightness is physically represented by total expected electrons.
- Experiments expose `target_snr`/`snr_sweep` and map them to total spot flux.
- A master seed creates independent named RNG streams.
