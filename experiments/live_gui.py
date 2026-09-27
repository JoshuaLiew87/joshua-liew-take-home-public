"""Interactive live simulator demo.

This is a post-core demonstration tool, not the authoritative Monte Carlo or
1 kHz benchmark. The GUI timer is intentionally slower than the sensor rate
so the display remains usable; estimator timing is measured separately.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import TimerBase
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

from common.rng import make_rng_streams
from config import ScenarioConfig, scenario_config
from estimators.centroid import estimate_centroid
from estimators.gaussian_fit import estimate_gaussian
from sim.simulator import SpotSimulator, total_flux_for_target_snr
from sim.trajectory import TrajectoryGenerator


class LiveLocalizationDemo:
    """Interactive moving-spot display backed by the production simulator."""

    _noise_labels = (
        "Photon shot",
        "Read noise",
        "Dark current",
        "Background",
        "PRNU",
        "Hot pixels",
        "Quantization",
        "Saturation",
    )

    def __init__(
        self,
        config: ScenarioConfig,
        *,
        snr: float | None = None,
        estimator_name: str = "Gaussian fit",
        display_interval_ms: int = 50,
        history_length: int = 2500,
    ) -> None:
        config.validate()
        self.config = config
        self._base_config = config
        self._base_sensor = config.sensor
        self._base_background = config.background
        self.snr = float(config.experiment.target_snr if snr is None else snr)
        self.estimator_name = estimator_name
        self.display_interval_ms = int(display_interval_ms)
        self.lock_threshold_px = 1.0
        self.sensor_frames_per_update = max(
            1,
            round(self.config.sensor.frame_rate_hz * self.display_interval_ms / 1000.0),
        )
        self.history_length = int(history_length)
        self.frame_index = 0
        self.paused = False
        self.manual_step_px = 0.1
        self.manual_offset_xy = np.zeros(2, dtype=float)
        self.scene_level = 0.5
        self.auto_exposure_enabled = False
        self.exposure_scale = 1.0
        self.analog_gain = 1.0
        self._exposure_scale_limits = (0.125, 4.0)
        self._analog_gain_limits = (0.5, 8.0)
        self._auto_target_peak_fraction = 0.65
        self._base_gradients = (
            float(config.background.gradient_x_e_per_px),
            float(config.background.gradient_y_e_per_px),
        )
        self.gradient_enabled = any(abs(value) > 0.0 for value in self._base_gradients)
        self.hot_pixel_demo_enabled = False
        self.forced_hot_pixel_xy: tuple[int, int] | None = None
        self.true_history: list[tuple[float, float]] = []
        self.estimated_history: list[tuple[float, float]] = []
        self.error_history: list[float] = []
        self._build_simulator_and_trajectory()
        self._build_figure()
        self._timer: TimerBase = self.figure.canvas.new_timer(interval=self.display_interval_ms)
        self._timer.add_callback(self._update_frame)

    def _build_simulator_and_trajectory(self) -> None:
        self._apply_runtime_config()
        streams = make_rng_streams(self.config.experiment.master_seed + self.frame_index)
        old_maps = getattr(self, "simulator", None)
        self.simulator = SpotSimulator(
            self.config,
            rng_streams=streams,
            fixed_pattern_maps=None if old_maps is None else old_maps.maps,
        )
        self.trajectory = TrajectoryGenerator(
            self.config.trajectory,
            sensor_shape=(self.config.sensor.height_px, self.config.sensor.width_px),
            position_margin_px=self.config.spot.position_margin_px,
            rng=streams.trajectory,
        )

    def _build_figure(self) -> None:
        self.figure = plt.figure(figsize=(15, 8))
        grid = self.figure.add_gridspec(
            2, 4, width_ratios=(1.35, 1.0, 1.0, 0.9), height_ratios=(1.0, 1.0)
        )
        self.ax_image = self.figure.add_subplot(grid[:, 0])
        self.ax_trace = self.figure.add_subplot(grid[0, 1:3])
        self.ax_error = self.figure.add_subplot(grid[1, 1:3])
        self.ax_controls = self.figure.add_subplot(grid[:, 3])
        # Controls are positioned explicitly below. Keep this GridSpec cell
        # invisible so its opaque face does not cover the widget axes.
        self.ax_controls.set_visible(False)

        blank = np.zeros((self.config.sensor.height_px, self.config.sensor.width_px))
        self.image_artist = self.ax_image.imshow(blank, origin="lower", cmap="magma")
        self.ax_image.set_title("Live simulated sensor frame")
        self.ax_image.set_xlabel("x [pixel]")
        self.ax_image.set_ylabel("y [pixel]")
        (self.true_marker,) = self.ax_image.plot([], [], "c+", ms=12, mew=2, label="true")
        (self.estimated_marker,) = self.ax_image.plot([], [], "wo", ms=7, mfc="none", label="estimated")
        (self.hot_pixel_marker,) = self.ax_image.plot([], [], "rx", ms=9, mew=2, label="demo hot pixel")
        self.ax_image.legend(loc="upper right", fontsize="small")

        self.ax_trace.set_title("Trajectory (x coordinate)")
        self.ax_trace.set_xlabel("frame")
        self.ax_trace.set_ylabel("x [pixel]")
        (self.true_trace,) = self.ax_trace.plot([], [], "c-", label="true")
        (self.estimated_trace,) = self.ax_trace.plot([], [], "w-", label="estimated")
        # Keep the plot legend away from the slider labels in the control column.
        self.ax_trace.legend(loc="upper left")
        self.ax_trace.set_facecolor("#202020")

        self.ax_error.set_title("Position error")
        self.ax_error.set_xlabel("frame")
        self.ax_error.set_ylabel("radial error [pixel]")
        (self.error_line,) = self.ax_error.plot([], [], "y-")
        self.ax_error.set_facecolor("#202020")

        self._make_controls()
        self.figure.subplots_adjust(left=0.05, right=0.98, bottom=0.07, top=0.96, wspace=0.35, hspace=0.35)

    def _make_controls(self) -> None:
        left = 0.765
        width = 0.18

        def make_slider(y: float, label: str, minimum: float, maximum: float, value: float) -> Slider:
            # Keep labels inside the control column instead of using Slider's
            # default left-side label, which overlaps the trajectory plot.
            self.figure.text(left, y + 0.032, label, ha="left", va="bottom", fontsize="small")
            return Slider(
                self.figure.add_axes([left, y, width, 0.027]),
                "",
                minimum,
                maximum,
                valinit=value,
            )

        self.snr_slider = make_slider(0.91, "Reference SNR", 1.0, 1000.0, self.snr)
        self.scene_slider = make_slider(0.86, "Scene brightness [%]", 0.0, 100.0, 50.0)
        self.drift_slider = make_slider(
            0.81,
            "Drift amplitude [px]",
            0.0,
            4.0,
            self.config.trajectory.drift_amplitude_px,
        )
        self.jitter_slider = make_slider(
            0.76,
            "Jitter RMS [px]",
            0.0,
            0.25,
            self.config.trajectory.jitter_rms_px,
        )
        self.disturbance_slider = make_slider(
            0.71,
            "Disturbance amplitude [px]",
            0.0,
            0.3,
            self.config.trajectory.disturbance_amplitude_px,
        )
        self.frequency_slider = make_slider(
            0.66,
            "Disturbance frequency [Hz]",
            1.0,
            200.0,
            self.config.trajectory.disturbance_frequency_hz,
        )

        self.noise_checks = CheckButtons(
            self.figure.add_axes([left, 0.18, width, 0.20]),
            self._noise_labels,
            [
                self.config.noise.photon_shot_enabled,
                self.config.noise.read_noise_enabled,
                self.config.noise.dark_current_enabled,
                self.config.noise.background_enabled,
                self.config.noise.prnu_enabled,
                self.config.noise.hot_pixels_enabled,
                self.config.sensor.quantization_enabled,
                self.config.sensor.saturation_enabled,
            ],
        )
        self.exposure_check = CheckButtons(
            self.figure.add_axes([left, 0.42, width, 0.055]),
            ("Auto exposure/gain",),
            (self.auto_exposure_enabled,),
        )
        self.estimator_radio = RadioButtons(
            self.figure.add_axes([left, 0.10, width, 0.09]),
            ("Gaussian fit", "Centroid"),
            active=0 if self.estimator_name == "Gaussian fit" else 1,
        )
        self.reset_button = Button(self.figure.add_axes([left, 0.06, 0.10, 0.04]), "Reset")
        self.pause_button = Button(self.figure.add_axes([left + 0.115, 0.06, 0.10, 0.04]), "Pause")
        self.status_text = self.figure.text(0.765, 0.595, "", va="top", fontsize="x-small")
        self.help_text = self.figure.text(
            0.765,
            0.008,
            "Keys: arrows = move 0.1 px\nShift+S/D = flux | G = gradient | H = hot pixel | Q/Esc = close",
            va="bottom",
            fontsize="x-small",
        )

        # Matplotlib's default key handler maps ``s`` to save-figure. The
        # live demo reserves lowercase ``s`` for downward target nudges, so
        # disconnect the default handler for this figure only.
        default_key_handler_id = getattr(self.figure.canvas.manager, "key_press_handler_id", None)
        if default_key_handler_id is not None:
            self.figure.canvas.mpl_disconnect(default_key_handler_id)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.snr_slider.on_changed(self._on_motion_control)
        self.scene_slider.on_changed(self._on_scene_control)
        self.drift_slider.on_changed(self._on_motion_control)
        self.jitter_slider.on_changed(self._on_motion_control)
        self.disturbance_slider.on_changed(self._on_motion_control)
        self.frequency_slider.on_changed(self._on_motion_control)
        self.noise_checks.on_clicked(self._on_noise_toggle)
        self.exposure_check.on_clicked(self._on_exposure_toggle)
        self.estimator_radio.on_clicked(self._on_estimator_change)
        self.reset_button.on_clicked(self.reset)
        self.pause_button.on_clicked(self.toggle_pause)

    def _on_key_press(self, event) -> None:
        """Handle review-demo shortcuts while the figure has focus."""

        key = str(event.key)
        if key.lower() in {"q", "escape"}:
            self._timer.stop()
            plt.close(self.figure)
            return
        if key in {"S", "shift+s"}:
            self.snr_slider.set_val(1.0)
            return
        if key in {"D", "shift+d"}:
            self.snr_slider.set_val(1000.0)
            return

        normalized_key = key.lower()
        nudge = {
            "left": (-self.manual_step_px, 0.0),
            "right": (self.manual_step_px, 0.0),
            "up": (0.0, self.manual_step_px),
            "down": (0.0, -self.manual_step_px),
        }.get(normalized_key)
        if nudge is not None:
            self.manual_offset_xy += np.asarray(nudge, dtype=float)
            self.manual_offset_xy = np.clip(self.manual_offset_xy, -4.0, 4.0)
            return
        if normalized_key == "g":
            self._toggle_gradient()
        elif normalized_key == "h":
            self._toggle_hot_pixel()

    def _toggle_gradient(self) -> None:
        self.gradient_enabled = not self.gradient_enabled
        gradient_x, gradient_y = self._base_gradients if self.gradient_enabled else (0.0, 0.0)
        self.config = replace(
            self.config,
            background=replace(
                self.config.background,
                gradient_x_e_per_px=gradient_x,
                gradient_y_e_per_px=gradient_y,
            ),
        )
        self.reset(None)

    def _toggle_hot_pixel(self) -> None:
        self.hot_pixel_demo_enabled = not self.hot_pixel_demo_enabled
        self.forced_hot_pixel_xy = None
        self.figure.canvas.draw_idle()

    def _on_motion_control(self, _value: float) -> None:
        trajectory = replace(
            self.config.trajectory,
            drift_amplitude_px=float(self.drift_slider.val),
            jitter_rms_px=float(self.jitter_slider.val),
            disturbance_amplitude_px=float(self.disturbance_slider.val),
            disturbance_frequency_hz=float(self.frequency_slider.val),
        )
        self.config = replace(self.config, trajectory=trajectory)
        self.reset(None)

    def _on_scene_control(self, value: float) -> None:
        """Change the physical scene without changing the motion history."""

        self.scene_level = float(value) / 100.0
        self._apply_runtime_config()

    def _on_exposure_toggle(self, _label: str) -> None:
        self.auto_exposure_enabled = not self.auto_exposure_enabled

    def _scene_multipliers(self) -> tuple[float, float]:
        """Return beacon and ambient-light multipliers for the GUI scene.

        The midpoint is the nominal scenario. The beacon spans roughly four
        orders of magnitude across the control, while the ambient background
        spans a smaller range. This deliberately represents a review-demo
        illumination sweep, not a calibrated day/night radiometry model.
        """

        level = float(self.scene_slider.val) / 100.0 if hasattr(self, "scene_slider") else self.scene_level
        beacon_multiplier = 10.0 ** (4.0 * (level - 0.5))
        ambient_multiplier = 10.0 ** (1.2 * (level - 0.5))
        return beacon_multiplier, ambient_multiplier

    def _runtime_config(self) -> ScenarioConfig:
        """Build the current GUI-only sensor state from nominal parameters."""

        _, ambient_multiplier = self._scene_multipliers()
        base_gradients = self._base_gradients if self.gradient_enabled else (0.0, 0.0)
        background = replace(
            self._base_background,
            offset_e=self._base_background.offset_e * ambient_multiplier,
            gradient_x_e_per_px=base_gradients[0] * ambient_multiplier,
            gradient_y_e_per_px=base_gradients[1] * ambient_multiplier,
        )
        sensor = replace(
            self._base_sensor,
            exposure_s=self._base_sensor.exposure_s * self.exposure_scale,
            electrons_per_dn=self._base_sensor.electrons_per_dn / self.analog_gain,
        )
        return replace(self.config, sensor=sensor, background=background)

    def _apply_runtime_config(self) -> None:
        """Apply GUI exposure, gain, and scene state to the simulator."""

        self.config = self._runtime_config()
        if hasattr(self, "simulator"):
            self.simulator.config = self.config

    def _update_auto_exposure(self, frame) -> None:
        """Apply a bounded, hysteretic exposure/gain update for the next frame."""

        if not self.auto_exposure_enabled or not self.config.sensor.saturation_enabled:
            return
        adc_max = self.config.sensor.adc_max_dn
        observed_peak = float(np.nanmax(frame.image_dn))
        saturated = frame.metadata.saturated_pixel_count > 0 or observed_peak >= (
            self._auto_target_peak_fraction + 0.30
        ) * adc_max
        too_dim = observed_peak < (self._auto_target_peak_fraction - 0.50) * adc_max
        changed = False
        if saturated:
            if self.exposure_scale > self._exposure_scale_limits[0] + 1e-9:
                self.exposure_scale = max(self._exposure_scale_limits[0], self.exposure_scale * 0.70)
                changed = True
            elif self.analog_gain > self._analog_gain_limits[0] + 1e-9:
                self.analog_gain = max(self._analog_gain_limits[0], self.analog_gain * 0.70)
                changed = True
        elif too_dim:
            if self.analog_gain < self._analog_gain_limits[1] - 1e-9:
                self.analog_gain = min(self._analog_gain_limits[1], self.analog_gain * 1.35)
                changed = True
            elif self.exposure_scale < self._exposure_scale_limits[1] - 1e-9:
                self.exposure_scale = min(self._exposure_scale_limits[1], self.exposure_scale * 1.35)
                changed = True
        if changed:
            self._apply_runtime_config()

    def _position_with_manual_offset(self, position: tuple[float, float]) -> tuple[float, float]:
        margin = self.config.spot.position_margin_px
        height = self.config.sensor.height_px
        width = self.config.sensor.width_px
        x = np.clip(position[0] + self.manual_offset_xy[0], margin, width - 1.0 - margin)
        y = np.clip(position[1] + self.manual_offset_xy[1], margin, height - 1.0 - margin)
        return float(x), float(y)

    def _inject_demo_hot_pixel(
        self,
        image_dn: np.ndarray,
        true_position: tuple[float, float],
    ) -> np.ndarray:
        """Add one visible, persistent ROI-edge pixel for the GUI demonstration.

        This is intentionally a display-only intervention. It does not alter
        the production simulator or any offline experiment output.
        """

        image = np.asarray(image_dn, dtype=float).copy()
        if not self.hot_pixel_demo_enabled:
            return image
        if self.forced_hot_pixel_xy is None:
            half = self.config.estimator.gaussian_roi_size_px // 2
            peak_x = int(round(true_position[0]))
            peak_y = int(round(true_position[1]))
            hot_x = int(np.clip(peak_x + half, 0, image.shape[1] - 1))
            hot_y = int(np.clip(peak_y, 0, image.shape[0] - 1))
            self.forced_hot_pixel_xy = (hot_x, hot_y)

        background = float(np.nanmedian(image))
        peak = float(np.nanmax(image))
        hot_value = background + 0.5 * max(peak - background, 20.0)
        if self.config.sensor.saturation_enabled:
            hot_value = min(hot_value, self.config.sensor.adc_max_dn)
        x, y = self.forced_hot_pixel_xy
        image[y, x] = max(image[y, x], hot_value)
        return image

    def _on_noise_toggle(self, label: str) -> None:
        index = self._noise_labels.index(label)
        if index < 6:
            field = (
                "photon_shot_enabled",
                "read_noise_enabled",
                "dark_current_enabled",
                "background_enabled",
                "prnu_enabled",
                "hot_pixels_enabled",
            )[index]
            noise = replace(self.config.noise, **{field: not getattr(self.config.noise, field)})
            self.config = replace(self.config, noise=noise)
        elif index == 6:
            self.config = replace(self.config, sensor=replace(self.config.sensor, quantization_enabled=not self.config.sensor.quantization_enabled))
        else:
            self.config = replace(self.config, sensor=replace(self.config.sensor, saturation_enabled=not self.config.sensor.saturation_enabled))
        self.reset(None)

    def _on_estimator_change(self, label: str) -> None:
        self.estimator_name = label

    def _update_frame(self) -> None:
        if self.paused:
            return
        frame = None
        result = None
        for _ in range(self.sensor_frames_per_update):
            true_position = self._position_with_manual_offset(self.trajectory.next_position())
            beacon_multiplier, _ = self._scene_multipliers()
            nominal_flux = total_flux_for_target_snr(
                self._base_config,
                self.simulator.maps,
                float(self.snr_slider.val),
                sigma_px=self._base_config.spot.sigma_px,
            )
            frame = self.simulator.render(
                true_position,
                total_flux_e=nominal_flux * beacon_multiplier * self.exposure_scale,
            )
            image_dn = self._inject_demo_hot_pixel(frame.image_dn, true_position)
            if self.estimator_name == "Centroid":
                result = estimate_centroid(image_dn, self.config)
            else:
                result = estimate_gaussian(image_dn, self.config)

            self.true_history.append(true_position)
            if result.success and result.position_xy is not None:
                estimated_position = result.position_xy
                error = float(np.linalg.norm(np.subtract(estimated_position, true_position)))
                self.estimated_history.append(estimated_position)
                self.error_history.append(error)
            else:
                self.estimated_history.append((np.nan, np.nan))
                self.error_history.append(np.nan)
            self._update_auto_exposure(frame)
        assert frame is not None and result is not None
        self.frame_index = self.trajectory.frame_index
        self.true_history = self.true_history[-self.history_length :]
        self.estimated_history = self.estimated_history[-self.history_length :]
        self.error_history = self.error_history[-self.history_length :]
        true_position = self.true_history[-1]
        estimated_position = self.estimated_history[-1]
        self.image_artist.set_data(image_dn)
        self.image_artist.set_clim(float(np.nanmin(image_dn)), float(np.nanmax(image_dn) + 1e-9))
        self.true_marker.set_data([true_position[0]], [true_position[1]])
        if not np.all(np.isfinite(estimated_position)):
            self.estimated_marker.set_data([], [])
        else:
            self.estimated_marker.set_data([estimated_position[0]], [estimated_position[1]])
        if self.hot_pixel_demo_enabled and self.forced_hot_pixel_xy is not None:
            hot_x, hot_y = self.forced_hot_pixel_xy
            self.hot_pixel_marker.set_data([hot_x], [hot_y])
        else:
            self.hot_pixel_marker.set_data([], [])
        frames = np.arange(self.frame_index - len(self.true_history), self.frame_index)
        true_array = np.asarray(self.true_history)
        estimated_array = np.asarray(self.estimated_history)
        self.true_trace.set_data(frames, true_array[:, 0])
        self.estimated_trace.set_data(frames, estimated_array[:, 0])
        self.error_line.set_data(frames, self.error_history)
        for axis in (self.ax_trace, self.ax_error):
            axis.relim()
            axis.autoscale_view()
        self.status_text.set_text(
            f"frame {self.frame_index} | SNR {self.snr_slider.val:.1f} | eff {frame.metadata.effective_snr:.2f}\n"
            f"scene {self.scene_slider.val:.0f}% | beacon x{self._scene_multipliers()[0]:.2f} | "
            f"{'AUTO' if self.auto_exposure_enabled else 'FIXED'}\n"
            f"exp {self.config.sensor.exposure_s * 1e3:.3f} ms | gain {self.analog_gain:.2f}x\n"
            f"{self.estimator_name} | {self._lock_status(result, self.error_history[-1])} | "
            f"err {self.error_history[-1]:.2f} px\n"
            f"gradient {'ON' if self.gradient_enabled else 'OFF'} | "
            f"hot {'ON' if self.hot_pixel_demo_enabled else 'OFF'} | "
            f"sat {int(np.count_nonzero(image_dn >= self.config.sensor.adc_max_dn)) if self.config.sensor.saturation_enabled else 0}\n"
            f"display {self.display_interval_ms} ms | display-only"
        )
        self.figure.canvas.draw_idle()

    def _lock_status(self, result, radial_error: float) -> str:
        """Return a demo status separating fit convergence from truth-based lock."""

        if not result.success:
            return "FAILED"
        if not np.isfinite(radial_error) or radial_error > self.lock_threshold_px:
            return "FIT OK / OUT OF LOCK"
        return "LOCKED"

    def reset(self, _event) -> None:
        self.frame_index = 0
        self.true_history.clear()
        self.estimated_history.clear()
        self.error_history.clear()
        self.manual_offset_xy.fill(0.0)
        self.forced_hot_pixel_xy = None
        self.exposure_scale = 1.0
        self.analog_gain = 1.0
        self._build_simulator_and_trajectory()
        self.status_text.set_text("Reset")
        self.figure.canvas.draw_idle()

    def toggle_pause(self, _event) -> None:
        self.paused = not self.paused
        self.pause_button.label.set_text("Resume" if self.paused else "Pause")

    def show(self) -> None:
        self._timer.start()
        plt.show()


def save_demo_gif(
    config: ScenarioConfig,
    output_path: str | Path,
    *,
    snr: float | None = None,
    estimator_name: str = "Gaussian fit",
    frame_count: int = 24,
    fps: int = 8,
    dpi: int = 80,
    auto_exposure: bool = False,
    scene_sweep: bool = False,
) -> Path:
    """Render a short animated review clip from the live-demo figure.

    The animation uses the same simulator, trajectory generator, controls,
    and estimator as the interactive GUI. It is intended for Markdown/README
    previews; PDF export uses the first-frame PNG fallback. ``scene_sweep``
    creates a triangular night-to-day-to-night illumination transition, and
    ``auto_exposure`` enables the optional GUI-only exposure/gain controller.
    """

    from matplotlib.animation import FuncAnimation, PillowWriter

    if frame_count <= 0 or fps <= 0 or dpi <= 0:
        raise ValueError("frame_count, fps, and dpi must be positive")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    demo = LiveLocalizationDemo(
        config,
        snr=snr,
        estimator_name=estimator_name,
        display_interval_ms=50,
    )
    demo.figure.set_facecolor("white")
    if auto_exposure:
        demo.exposure_check.set_active(0)

    def update(frame_index: int) -> tuple[()]:
        if scene_sweep:
            phase = frame_index / max(frame_count - 1, 1)
            triangle = 2.0 * phase if phase <= 0.5 else 2.0 * (1.0 - phase)
            demo.scene_slider.set_val(100.0 * triangle)
        demo._update_frame()
        return ()

    animation = FuncAnimation(
        demo.figure,
        update,
        frames=frame_count,
        interval=1000.0 / fps,
        blit=False,
    )
    animation.save(output, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(demo.figure)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("sensor", "ideal", "dynamic"), default="sensor")
    parser.add_argument("--snr", type=float, default=None)
    parser.add_argument("--estimator", choices=("centroid", "gaussian"), default="gaussian")
    parser.add_argument("--interval-ms", type=int, default=50)
    parser.add_argument("--save-gif", type=Path, default=None)
    parser.add_argument("--gif-frames", type=int, default=24)
    parser.add_argument("--gif-fps", type=int, default=8)
    parser.add_argument("--auto-exposure", action="store_true")
    parser.add_argument("--scene-sweep", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = scenario_config(args.profile)
    estimator_name = "Centroid" if args.estimator == "centroid" else "Gaussian fit"
    if args.save_gif is not None:
        save_demo_gif(
            config,
            args.save_gif,
            snr=args.snr,
            estimator_name=estimator_name,
            frame_count=args.gif_frames,
            fps=args.gif_fps,
            auto_exposure=args.auto_exposure,
            scene_sweep=args.scene_sweep,
        )
        print(f"Saved live-demo GIF to {args.save_gif}")
    else:
        LiveLocalizationDemo(
            config,
            snr=args.snr,
            estimator_name=estimator_name,
            display_interval_ms=args.interval_ms,
        ).show()


if __name__ == "__main__":
    main()
