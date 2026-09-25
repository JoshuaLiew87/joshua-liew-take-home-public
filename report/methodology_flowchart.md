# Methodology flowchart

This is the high-level workflow for the project. The detailed implementation
choices, equations, assumptions, and validation checks are documented in the
main report and `plan.md`.

```mermaid
flowchart LR
    A[1. Define the problem<br/>Assumptions, units, coordinates,<br/>noise settings, and SNR range]
    B[2. Build the simulator<br/>Generate a noisy camera image<br/>with a known sub-pixel spot]
    C[3. Validate the simulator<br/>Check PSF shape, noise statistics,<br/>SNR behavior, and saturation]
    D[4. Estimate spot position<br/>Run two methods:<br/>centroid and Gaussian fit]
    E[5. Characterize performance<br/>Monte Carlo bias, precision,<br/>failure rate, and CRLB comparison]
    F[6. Measure real-time cost<br/>Timing, latency, and<br/>1 kHz tradeoff]
    G[7. Test dynamic tracking<br/>Drift, jitter, periodic disturbance,<br/>trajectory and frequency recovery]
    H[8. Analyze real-world limits<br/>Scintillation, beam wander,<br/>background, vibration, saturation]
    I[9. Produce reproducible submission<br/>Figures, results, report,<br/>assumptions, and traceability]

    A --> B --> C --> D
    D --> E
    D --> F
    D --> G
    E --> H
    F --> H
    G --> H
    H --> I
```

## What flows through the process

- **Input:** physical assumptions, sensor parameters, target SNR, and random seed.
- **Simulator output:** noisy frames plus the known true spot position.
- **Estimator output:** estimated sub-pixel position and quality information.
- **Static output:** bias, standard deviation, RMSE, failures, runtime, and comparison with the theoretical reference.
- **Dynamic output:** recovered trajectory and periodic disturbance frequency/amplitude.
- **Final output:** reproducible figures, machine-readable results, documented assumptions, and engineering conclusions.

