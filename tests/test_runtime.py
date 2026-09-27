from __future__ import annotations

import numpy as np

from config import scenario_config
from experiments.benchmark_runtime import run_runtime_benchmark


def test_runtime_benchmark_reports_cold_and_warm_paths() -> None:
    summaries = run_runtime_benchmark(
        scenario_config("ideal"),
        target_snr=10.0,
        frame_count=5,
        warmup_count=1,
    )

    assert [(item.estimator, item.initialization) for item in summaries] == [
        ("centroid", "cold"),
        ("gaussian", "cold"),
        ("gaussian", "warm"),
    ]
    for summary in summaries:
        assert summary.frame_count == 5
        assert summary.median_ms >= 0.0
        assert summary.p99_ms >= summary.median_ms
        assert 0.0 <= summary.failure_rate <= 1.0
        assert 0.0 <= summary.deadline_miss_rate <= 1.0
        assert np.isfinite(summary.median_nfev) or summary.estimator == "centroid"


def test_runtime_benchmark_supports_sequential_dynamic_frames() -> None:
    summaries = run_runtime_benchmark(
        scenario_config("dynamic"),
        target_snr=10.0,
        frame_count=5,
        warmup_count=1,
        sequence="dynamic",
    )

    assert all(summary.sequence == "dynamic" for summary in summaries)
