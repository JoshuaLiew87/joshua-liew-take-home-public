from __future__ import annotations

from experiments.dynamic_repeats import run_dynamic_repeats
from experiments.estimator_ablation import run_ablation
from config import scenario_config


def test_controlled_ablation_reports_both_estimators() -> None:
    metrics = run_ablation(
        scenario_config("sensor"),
        snr_values=(1.0,),
        trial_count=4,
    )

    assert len(metrics) == 10
    assert {metric.estimator for metric in metrics} == {"centroid", "gaussian"}
    assert {metric.condition for metric in metrics} == {
        "ideal",
        "background_gradient",
        "prnu_hot_pixels",
        "quantization_saturation",
        "full_stressed",
    }


def test_dynamic_repeat_returns_per_seed_and_aggregate_rows() -> None:
    rows, aggregates = run_dynamic_repeats(
        scenario_config("dynamic"),
        repeat_count=2,
        frame_count=100,
    )

    assert len(rows) == 4
    assert len(aggregates) == 2
    assert {row.estimator for row in aggregates} == {"centroid", "gaussian"}
    assert all(row.repeat_count == 2 for row in aggregates)
