from __future__ import annotations

import numpy as np

from config import scenario_config
from experiments.characterize_snr import run_snr_sweep, summarize_errors
from experiments.theory import compute_position_crlb


def test_summarize_errors_reports_failures_and_outliers() -> None:
    errors = np.array(
        [
            [0.1, -0.1],
            [0.2, 0.0],
            [np.nan, np.nan],
            [2.0, 0.0],
        ]
    )
    summary = summarize_errors(
        errors,
        estimator="test",
        requested_snr=10.0,
        trial_count=4,
        outlier_threshold_px=1.0,
    )

    assert summary.success_count == 3
    assert summary.failure_count == 1
    assert summary.failure_rate == 0.25
    assert summary.outlier_count == 1
    assert summary.outlier_rate == 0.25
    assert summary.locked_count == 2
    assert summary.lock_failure_count == 2
    assert summary.lock_failure_rate == 0.5
    assert np.isfinite(summary.conditional_radial_rmse_px)
    assert summary.rmse_x_px > 0.0


def test_small_snr_sweep_is_reproducible_and_contains_both_estimators() -> None:
    config = scenario_config("ideal")
    first = run_snr_sweep(config, snr_values=(3.0, 10.0), trial_count=4)
    second = run_snr_sweep(config, snr_values=(3.0, 10.0), trial_count=4)

    assert len(first) == 4
    assert {item.estimator for item in first} == {"centroid", "gaussian"}
    np.testing.assert_equal(
        [item.__dict__ for item in first],
        [item.__dict__ for item in second],
    )


def test_clean_position_reference_is_finite_and_improves_with_snr() -> None:
    theory = compute_position_crlb(
        scenario_config("ideal"),
        snr_values=(3.0, 10.0),
        phase_grid_size=2,
    )

    assert np.all(np.isfinite(theory["std_x"]))
    assert np.all(np.isfinite(theory["std_y"]))
    assert theory["std_x"][1] < theory["std_x"][0]
    assert theory["std_y"][1] < theory["std_y"][0]
