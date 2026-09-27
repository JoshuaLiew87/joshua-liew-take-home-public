from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from types import SimpleNamespace

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


def test_live_gui_demo_shortcuts_control_review_scenarios() -> None:
    demo = LiveLocalizationDemo(
        scenario_config("sensor"),
        snr=10.0,
        display_interval_ms=50,
    )
    try:
        demo._on_key_press(SimpleNamespace(key="up"))
        assert demo.manual_offset_xy[1] == demo.manual_step_px
        demo._on_key_press(SimpleNamespace(key="down"))
        assert demo.manual_offset_xy[1] == 0.0
        demo._on_key_press(SimpleNamespace(key="s"))
        assert demo.manual_offset_xy[1] == 0.0

        demo._on_key_press(SimpleNamespace(key="g"))
        assert not demo.gradient_enabled
        assert demo.config.background.gradient_x_e_per_px == 0.0
        demo._on_key_press(SimpleNamespace(key="g"))
        assert demo.gradient_enabled

        demo._on_key_press(SimpleNamespace(key="h"))
        demo._update_frame()
        assert demo.hot_pixel_demo_enabled
        assert demo.forced_hot_pixel_xy is not None

        demo._on_key_press(SimpleNamespace(key="shift+s"))
        assert demo.snr_slider.val == 1.0
        demo._on_key_press(SimpleNamespace(key="shift+d"))
        assert demo.snr_slider.val == 1000.0
    finally:
        plt.close(demo.figure)


def test_live_gui_q_shortcut_closes_the_figure() -> None:
    demo = LiveLocalizationDemo(scenario_config("sensor"), snr=10.0)
    demo._on_key_press(SimpleNamespace(key="q"))
    assert not plt.fignum_exists(demo.figure.number)


def test_live_gui_scene_sweep_and_auto_exposure_are_display_controls() -> None:
    demo = LiveLocalizationDemo(
        scenario_config("sensor"),
        snr=10.0,
        display_interval_ms=50,
    )
    try:
        nominal_background = demo._base_background.offset_e
        demo.scene_slider.set_val(100.0)
        assert demo.config.background.offset_e > nominal_background

        demo.exposure_check.set_active(0)
        assert demo.auto_exposure_enabled
        for _ in range(4):
            demo._update_frame()

        assert demo.exposure_scale < 1.0
        assert demo.config.sensor.exposure_s < demo._base_sensor.exposure_s
    finally:
        plt.close(demo.figure)
