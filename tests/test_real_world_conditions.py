from __future__ import annotations

import numpy as np

from config import scenario_config
from experiments.real_world_conditions import run_condition_analysis


def test_real_world_condition_analysis_is_reproducible_and_diagnostic() -> None:
    config = scenario_config("sensor")
    first = run_condition_analysis(config, trial_count=24, target_snr=10.0)
    second = run_condition_analysis(config, trial_count=24, target_snr=10.0)

    assert len(first) == 10
    assert [metric.__dict__ for metric in first] == [metric.__dict__ for metric in second]
    assert {metric.condition for metric in first} == {
        "nominal",
        "scintillation",
        "bright_background",
        "defocus",
        "saturation",
    }
    assert {metric.estimator for metric in first} == {"centroid", "gaussian"}
    assert all(np.isfinite(metric.mean_effective_snr) for metric in first)
    assert all(0.0 <= metric.lock_failure_rate <= 1.0 for metric in first)

    nominal = next(
        metric for metric in first
        if metric.condition == "nominal" and metric.estimator == "gaussian"
    )
    bright_background = next(
        metric for metric in first
        if metric.condition == "bright_background" and metric.estimator == "gaussian"
    )
    saturation = next(
        metric for metric in first
        if metric.condition == "saturation" and metric.estimator == "centroid"
    )
    assert nominal.radial_rmse_px < 0.3
    assert bright_background.mean_effective_snr < nominal.mean_effective_snr
    assert saturation.saturated_frame_rate > 0.9
