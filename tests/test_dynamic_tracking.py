from __future__ import annotations

import numpy as np

from config import scenario_config
from experiments.dynamic_tracking import run_dynamic_experiment


def test_dynamic_experiment_recovers_injected_disturbance() -> None:
    result = run_dynamic_experiment(
        scenario_config("dynamic"),
        target_snr=10.0,
        frame_count=500,
    )

    assert result.positions_xy.shape == (500, 2)
    assert set(result.estimates_xy) == {"centroid", "gaussian"}
    assert np.all(np.isfinite(result.positions_xy))
    for summary in result.summaries:
        assert summary.valid_count > 0
        assert summary.estimated_frequency_hz > 20.0
        assert summary.frequency_abs_error_hz < 2.0
        assert summary.estimated_amplitude_px > 0.0
