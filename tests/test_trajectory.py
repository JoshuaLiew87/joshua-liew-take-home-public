from __future__ import annotations

import numpy as np

from common.rng import make_rng_streams
from config import scenario_config
from sim.trajectory import TrajectoryGenerator


def test_trajectory_contains_bounded_positions_and_is_reproducible() -> None:
    config = scenario_config("dynamic")
    assert config.trajectory.drift_model == "ou"
    first = TrajectoryGenerator(
        config.trajectory,
        sensor_shape=(config.sensor.height_px, config.sensor.width_px),
        position_margin_px=config.spot.position_margin_px,
        rng=make_rng_streams(config.experiment.master_seed).trajectory,
    ).generate(100)
    second = TrajectoryGenerator(
        config.trajectory,
        sensor_shape=(config.sensor.height_px, config.sensor.width_px),
        position_margin_px=config.spot.position_margin_px,
        rng=make_rng_streams(config.experiment.master_seed).trajectory,
    ).generate(100)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (100, 2)
    assert np.all(first[:, 0] >= config.spot.position_margin_px)
    assert np.all(first[:, 0] <= config.sensor.width_px - 1 - config.spot.position_margin_px)
    assert np.all(first[:, 1] >= config.spot.position_margin_px)
    assert np.all(first[:, 1] <= config.sensor.height_px - 1 - config.spot.position_margin_px)
    assert np.ptp(first[:, 0]) > 0.0
