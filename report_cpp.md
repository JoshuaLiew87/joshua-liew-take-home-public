# Optional C++ Gaussian backend comparison

This document supplements the main [technical report](report.md) for the
`feature/cpp-gaussian-accelerator` branch. The main report and `results/`
remain the authoritative Python reference. This branch adds an optional C++
implementation of the pixel-integrated Gaussian model and analytical
Jacobian, with separate outputs under `results_cpp/`. The checked-in runtime
comparison is from the Mac reference system; a matched-contract C++ benchmark
was also run on the Jetson target.

## Reproduction

Build the optional extension from the repository root:

```bash
python -m pip install -r requirements-cpp.txt
python cpp/build_extension.py
```

Run the full-quality C++ comparison:

```bash
python run_all.py \
  --config configs/cpp_gaussian.yaml \
  --results-dir results_cpp
```

Run the additional fast-mode accuracy comparison:

```bash
python run_all.py \
  --config configs/fast_gaussian.yaml \
  --results-dir results_fast

python run_all.py \
  --config configs/fast_cpp_gaussian.yaml \
  --results-dir results_cpp_fast
```

The checked-in full comparison snapshot is under
[`results_cpp/latest/`](results_cpp/latest/). The fast-mode snapshots are
under `results_fast/latest/` and `results_cpp_fast/latest/`. The Jetson full
and fast target-validation artifacts are preserved under
`results/performance/jetson_cpp/`.

## What changed

The C++ extension implements the same seven-parameter pixel-integrated
Gaussian model and analytical Jacobian as the Python reference:

```text
(x0, y0, total_flux, sigma, b0, bx, by)
```

The Python estimator still performs ROI selection, background initialization,
variance-floor handling, IRLS, and bounded SciPy Trust Region Reflective
optimization. The C++ branch therefore changes the numerical kernel, not the
estimation algorithm or the authoritative full-quality settings.

The optional fast profile additionally uses a 13 × 13 ROI, one IRLS pass, a
30-evaluation optimizer limit, previous-position ROI placement, and
previous-position initialization.

## Consolidated results

All rows below use the same seeded experiment definitions. RMSE and lock loss
are from the stressed-sensor static characterization. Dynamic results use the
5,000-frame trajectory with the injected 73 Hz disturbance. Accuracy rows are
Mac reference measurements; the runtime table includes both Mac reference and
Jetson target-validation rows. All runtime values are estimator-only and
machine-load dependent.

### Accuracy and recovery

| Configuration | SNR 1 RMSE / lock loss | SNR 3 RMSE / lock loss | SNR 10 RMSE / lock loss | Dynamic RMS / p95 [px] | Disturbance frequency [Hz] | Disturbance amplitude [px] |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full Python | 14.4379 / 90.1% | 4.8114 / 7.3% | 0.0807 / 0.0% | 0.081718 / 0.140450 | 72.997420 | 0.056530 |
| Full C++ model/Jacobian | 14.4379 / 90.1% | 4.8114 / 7.3% | 0.0807 / 0.0% | 0.081718 / 0.140450 | 72.997420 | 0.056530 |
| Fast Python | 13.7566 / 93.0% | 3.9140 / 7.5% | 0.0827 / 0.0% | 0.083876 / 0.143205 | 72.998214 | 0.056507 |
| Fast C++ model/Jacobian | 13.7566 / 93.0% | 3.9140 / 7.5% | 0.0827 / 0.0% | 0.083876 / 0.143205 | 72.998214 | 0.056507 |

Fast Python and fast C++ match through SNR 300 to numerical precision. At SNR
1000, fast Python gives `0.017019 px` RMSE and `9.2%` lock loss, while fast C++
gives `0.017077 px` and `10.7%` lock loss. This small difference is a
convergence-boundary effect from the reduced optimizer budget, not a broad
change in the C++ model.

### Runtime

| Target | Configuration | Sequence | Warm median | Warm p95 | Warm p99 | Worst | >1 ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mac reference | Full Python | Fixed stressed | 1.819 ms | 2.059 ms | 2.218 ms | 2.246 ms | 100% |
| Mac reference | Full C++ model/Jacobian | Fixed stressed | 0.960 ms | 1.369 ms | 2.290 ms | 2.705 ms | 33.5% |
| Mac reference | Fast Python | Dynamic | 0.972 ms | 1.147 ms | 1.260 ms | 1.300 ms | 46% |
| Mac reference | Fast C++ model/Jacobian | Dynamic | 0.506 ms | 0.590 ms | 0.656 ms | 0.697 ms | 0% |
| Jetson Orin Nano | Full Python | Fixed | 10.477 ms | 12.361 ms | 13.358 ms | 13.405 ms | 100% |
| Jetson Orin Nano | Full C++ model/Jacobian | Fixed | 5.743 ms | 6.668 ms | 7.220 ms | 7.286 ms | 100% |
| Jetson Orin Nano | Fast Python | Dynamic | 5.918 ms | 6.938 ms | 7.834 ms | 7.897 ms | 100% |
| Jetson Orin Nano | Fast C++ model/Jacobian | Dynamic | 3.384 ms | 3.948 ms | 4.372 ms | 4.419 ms | 100% |

The full C++ backend reduces median latency by approximately 1.9× without
changing full-quality accuracy. Combining it with the fast tracking settings
reduces median latency by approximately 1.92× relative to fast Python. The
full and fast runtime rows use fixed and dynamic sequences respectively, so
their latency values are not a single like-for-like ranking.

### Jetson Orin Nano validation

The extension was rebuilt natively on an NVIDIA Jetson Orin Nano Engineering
Reference Developer Kit running JetPack R36.4.3, Ubuntu 22.04.5, aarch64,
Python 3.13.13, NumPy 2.5.3, SciPy 1.18.1, pybind11 3.1.0, and GCC 11.4.0.
CUDA and TensorRT were not used; this is a CPU C++ kernel benchmark. The full
and fast C++ pipelines completed their seeded studies with 30 and 24 output
files respectively, and the full-profile non-runtime accuracy/dynamic results matched the Python
reference. The fast profile retained the documented SNR-1000
convergence-boundary difference.

The Jetson rows in the runtime table above are the matched estimator benchmark
for the warm Gaussian path; the worst-case values are included there as well.

The C++ kernel therefore reduced Gaussian median latency by approximately
1.82× in the full configuration and 1.75× in the fast configuration. The
centroid remained approximately 0.48 ms with no deadline misses. The C++
backend is a meaningful embedded optimization, but the Python optimizer and
IRLS loop still keep Gaussian tracking above the 1 ms budget. The saved
matched-contract timing files are `comparison_runtime_benchmark.csv` under
`results/performance/jetson_cpp/full/` and `/fast/`; complete pipeline runtime,
dynamic, and manifest files are in the same directories.

![C++ backend runtime benchmark](results_cpp/latest/figures/runtime_sensor.png)

*Estimator-only runtime benchmark for the full C++ backend.*

## Validation and interpretation

Across the full non-runtime comparison, the largest numeric difference was
approximately `1.3e-8` in a derived correlation statistic. All non-runtime
figures were pixel-identical between the Python and full C++ snapshots. A
50-frame paired check for fast Python versus fast C++ produced no success-status
mismatches and a maximum coordinate difference of `1.8e-15 px`.

The full-quality C++ path is therefore a contained numerical-kernel
optimization. The fast path is a separate speed/robustness trade-off: it is
close to full-quality performance at SNR 10, has higher low-SNR lock loss, and
has a small SNR-1000 convergence-boundary difference. The Jetson benchmark
shows that the C++ kernel improves the embedded timing margin, but neither
Gaussian configuration is a deterministic 1 kHz estimator because p99 remains
above 1 ms.

The branch passes 48 tests, including direct model/Jacobian equivalence and
Python-versus-C++ estimator comparisons. The next escalation would move the
complete IRLS and nonlinear optimization loop into C++, which would require a
new numerical equivalence review and a fresh target-device benchmark.
