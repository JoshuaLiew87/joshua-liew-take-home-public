# Decision log

## Day 1 foundations

- **A-001 — Coordinate convention:** pixel centers are integer coordinates and pixel boundaries are half a pixel away. This keeps the simulator, estimators, and theory consistent.
- **A-002 — Array/coordinate order:** represent a spot position as `(x, y)` while storing images as `image[y, x]`. This matches the usual array convention without confusing coordinate order.
- **A-003 — Pixel integration:** render the Gaussian by integrating its probability mass over each pixel area rather than sampling only at pixel centers. This avoids making the simulator's sub-pixel behavior an artifact of point sampling.
- **A-004 — Brightness parameter:** use total expected spot electrons as the physical brightness parameter. The experiment layer exposes target SNR values and maps them to total flux.
- **A-005 — Spot width:** use a nominal sigma of 1.75 pixels because a 7-pixel 1/e² diameter implies `sigma = diameter / 4`.
- **A-006 — RNG isolation:** use named NumPy random-number-generator streams derived from one master seed. This prevents unrelated experiments from changing each other's random sequences.
- **A-007 — Scenario profiles:** keep ideal, sensor, and dynamic profiles in one configuration model. This avoids duplicating simulator logic while making assumptions visible.
- **A-008 — Initial sensor geometry:** use 32×32 pixels and 1 ms/1 kHz as initial sensor defaults. This is large enough to contain the spot and small enough for rapid experiments.
- **A-009 — PRNU convention:** apply a persistent multiplicative PRNU gain to expected total charge before Poisson sampling. This is a simple, explicit sensor-response convention; it affects both mean charge and shot-noise statistics.
- **A-010 — Hot-pixel model:** represent hot pixels as persistent locations with elevated dark-current rates. This preserves stochastic dark noise while creating fixed spatial outliers.
- **A-011 — Readout convention:** add Gaussian read noise in electrons, convert using electrons/DN, then round to nearest DN when quantization is enabled. This provides explicit units and a testable ADC model.
- **A-012 — SNR convention:** calibrate target SNR using a reference spot centered on a pixel and record the actual effective SNR separately. This avoids defining the sweep through a random sub-pixel phase.
- **A-013 — Sigma variation:** use 5% RMS sigma variation only in stressed scenarios; keep sigma fixed in the clean reference. This tests optics variation without obscuring the clean estimator comparison.
- **A-014 — Default stressed sensor parameters:** use 1% PRNU, 50 electrons/s baseline dark current, 4 hot pixels at 5,000 electrons/s, 3 electrons RMS read noise, 12-bit ADC, and 1 electron/DN. These are engineering assumptions for a controlled simplified scenario, not camera measurements.
- **A-015 — Noise ablations:** expose independent switches for photon shot noise, read noise, dark current, background, PRNU, and hot pixels. This makes it possible to attribute behavior to individual effects without duplicating the simulator.
- **A-016 — Visual sanity check:** use a small Matplotlib diagnostic tool that calls the same simulator API as later experiments. This helps detect PSF, noise, SNR, and saturation mistakes without creating a separate simulation implementation.

## Assumption-register requirements

Every later material choice must receive the next stable `A-###` identifier and include its value, units, rationale, expected effect, and validation or limitation. The final report must reproduce this information in its assumption register and map each entry to the relevant experiment or result.
