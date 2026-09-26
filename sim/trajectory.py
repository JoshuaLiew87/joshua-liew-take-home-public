"""Reproducible moving-spot trajectories for dynamic experiments and demos."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import TrajectoryConfig


@dataclass
class TrajectoryGenerator:
    """Generate bounded drift, correlated jitter, and one sinusoidal disturbance."""

    config: TrajectoryConfig
    sensor_shape: tuple[int, int]
    position_margin_px: float
    rng: np.random.Generator

    def __post_init__(self) -> None:
        self.config.validate()
        self._frame_index = 0
        self._jitter = np.zeros(2, dtype=float)
        self._reset_jitter_scale()

    def _reset_jitter_scale(self) -> None:
        rho = self.config.jitter_correlation
        self._jitter_innovation_rms = self.config.jitter_rms_px * np.sqrt(1.0 - rho * rho)

    @property
    def frame_index(self) -> int:
        return self._frame_index

    def reset(self) -> None:
        """Restart the trajectory and its correlated-jitter state."""

        self._frame_index = 0
        self._jitter.fill(0.0)

    def _position_at(self, frame_index: int) -> tuple[float, float]:
        t = frame_index / self.config.frame_rate_hz
        drift_phase = 2.0 * np.pi * self.config.drift_frequency_hz * t
        x = self.config.center_x_px + self.config.drift_amplitude_px * np.sin(drift_phase)
        y = self.config.center_y_px + 0.6 * self.config.drift_amplitude_px * np.cos(drift_phase)

        rho = self.config.jitter_correlation
        if frame_index == 0:
            self._jitter.fill(0.0)
        else:
            innovation = self.rng.normal(0.0, self._jitter_innovation_rms, size=2)
            self._jitter = rho * self._jitter + innovation
        x += self._jitter[0]
        y += self._jitter[1]

        disturbance = self.config.disturbance_amplitude_px * np.sin(
            2.0 * np.pi * self.config.disturbance_frequency_hz * t
        )
        if self.config.disturbance_axis == "x":
            x += disturbance
        else:
            y += disturbance

        height, width = self.sensor_shape
        x = float(np.clip(x, self.position_margin_px, width - 1.0 - self.position_margin_px))
        y = float(np.clip(y, self.position_margin_px, height - 1.0 - self.position_margin_px))
        return x, y

    def next_position(self) -> tuple[float, float]:
        """Advance by one frame and return the true `(x, y)` position."""

        position = self._position_at(self._frame_index)
        self._frame_index += 1
        return position

    def generate(self, frame_count: int) -> np.ndarray:
        """Generate a reproducible `(frame, x/y)` position array."""

        if frame_count <= 0:
            raise ValueError("frame_count must be positive")
        positions = np.empty((frame_count, 2), dtype=float)
        for index in range(frame_count):
            positions[index] = self.next_position()
        return positions
