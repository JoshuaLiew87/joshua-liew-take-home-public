from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from config import scenario_config
from experiments.live_gui import LiveLocalizationDemo


def test_live_gui_can_render_one_headless_frame() -> None:
    demo = LiveLocalizationDemo(
        scenario_config("sensor"),
        snr=10.0,
        display_interval_ms=50,
    )
    try:
        demo._update_frame()
        assert demo.frame_index == demo.sensor_frames_per_update
        assert len(demo.true_history) == demo.sensor_frames_per_update
        assert len(demo.error_history) == demo.sensor_frames_per_update
    finally:
        plt.close(demo.figure)
