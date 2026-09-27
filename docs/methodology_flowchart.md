# Methodology flowchart

This is the high-level methodology for the project, rather than a literal
representation of every `run_all.py` stage. The exact reproducible execution
order and output paths are documented in Section 11.2 of the main report. The
detailed implementation choices, equations, assumptions, and validation checks
are documented in the main report and `plan.md`.

```mermaid
flowchart TD
    A[1. Define<br/>Units, coordinates,<br/>noise, and SNR]
    B[2. Simulate<br/>Noisy pixel-integrated spot]
    C[3. Validate<br/>PSF, noise, SNR,<br/>and saturation]
    D[4. Estimate<br/>Centroid and Gaussian fit]
    E[5. Static tests<br/>Monte Carlo, CRLB,<br/>ROI, and ablations]
    F[6. Runtime<br/>Fixed/sequential timing,<br/>tail latency, 1 kHz]
    G[7. Dynamic tests<br/>OU drift, jitter,<br/>periodic recovery, repeats]
    H[8. Robustness<br/>Scintillation, background,<br/>defocus, vibration, saturation]
    I[9. Reproduce<br/>Figures, results, report,<br/>and traceability]
    J[Optional GUI<br/>Interactive review aid]

    A --> B --> C --> D
    D --> E --> F --> G --> H
    H --> I
    D -.-> J
    H -.-> J
```

## What flows through the process

- **Input:** physical assumptions, sensor parameters, target SNR, and random seed.
- **Simulator output:** noisy frames plus the known true spot position.
- **Estimator output:** estimated sub-pixel position and quality information.
- **Static output:** bias, standard deviation, RMSE, failures, runtime, and comparison with the theoretical reference.
- **Dynamic output:** recovered trajectory and periodic disturbance frequency/amplitude.
- **Optional GUI output:** qualitative visual feedback for motion, noise, saturation,
  background, hot-pixel, and scene-brightness demonstrations. It is not an
  authoritative quantitative experiment.
- **Final output:** reproducible figures, machine-readable results, documented assumptions, and engineering conclusions.
