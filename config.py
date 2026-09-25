"""Configuration and scenario profiles for the spot-localization study.

The configuration deliberately separates physical sensor parameters from
experiment parameters.  The simulator will use total spot flux in electrons
as its physical brightness input; experiments expose target SNR values and
will calibrate them to flux in a later phase.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal


def sigma_from_1e2_diameter(diameter_px: float) -> float:
    """Convert a 1/e² intensity diameter to Gaussian sigma in pixels.

    For ``I(r) = I0 * exp(-2 r² / w²)``, ``w`` is the 1/e² radius and
    ``sigma = w / 2``.  A diameter is ``2*w``, therefore sigma is
    ``diameter / 4``.
    """

    if diameter_px <= 0:
        raise ValueError("1/e² diameter must be positive")
    return diameter_px / 4.0


@dataclass(frozen=True)
class SensorConfig:
    """Camera and ADC conventions; charge values are in electrons."""

    width_px: int = 32
    height_px: int = 32
    exposure_s: float = 1e-3
    frame_rate_hz: float = 1000.0
    bit_depth: int = 12
    electrons_per_dn: float = 1.0
    full_well_e: float = 4095.0
    read_noise_e_rms: float = 3.0
    quantization_enabled: bool = True
    saturation_enabled: bool = True

    @property
    def adc_max_dn(self) -> float:
        return float((1 << self.bit_depth) - 1)

    def validate(self) -> None:
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("sensor dimensions must be positive")
        if self.exposure_s <= 0 or self.frame_rate_hz <= 0:
            raise ValueError("exposure and frame rate must be positive")
        if self.bit_depth <= 0:
            raise ValueError("bit depth must be positive")
        if self.electrons_per_dn <= 0 or self.full_well_e <= 0:
            raise ValueError("conversion gain and full well must be positive")
        if self.read_noise_e_rms < 0:
            raise ValueError("read noise cannot be negative")


@dataclass(frozen=True)
class SpotConfig:
    """Laser spot parameters; total flux is expressed in expected electrons."""

    total_flux_e: float = 1000.0
    sigma_px: float = sigma_from_1e2_diameter(7.0)
    sigma_rms_fraction: float = 0.05
    position_margin_px: float = 8.0

    def validate(self) -> None:
        if self.total_flux_e <= 0 or self.sigma_px <= 0:
            raise ValueError("spot flux and sigma must be positive")
        if self.sigma_rms_fraction < 0:
            raise ValueError("sigma variation cannot be negative")
        if self.position_margin_px < 0:
            raise ValueError("position margin cannot be negative")


@dataclass(frozen=True)
class BackgroundConfig:
    """Expected background charge in electrons per pixel."""

    offset_e: float = 20.0
    gradient_x_e_per_px: float = 0.25
    gradient_y_e_per_px: float = -0.15

    def validate(self) -> None:
        if self.offset_e < 0:
            raise ValueError("background offset cannot be negative")


@dataclass(frozen=True)
class PatternConfig:
    """Persistent sensor non-uniformity settings."""

    prnu_std_fraction: float = 0.01
    dark_current_e_per_s: float = 50.0
    hot_pixel_count: int = 4
    hot_pixel_dark_current_e_per_s: float = 5000.0

    def validate(self) -> None:
        if self.prnu_std_fraction < 0:
            raise ValueError("PRNU standard deviation cannot be negative")
        if self.dark_current_e_per_s < 0:
            raise ValueError("dark current cannot be negative")
        if self.hot_pixel_count < 0:
            raise ValueError("hot-pixel count cannot be negative")
        if self.hot_pixel_dark_current_e_per_s < 0:
            raise ValueError("hot-pixel dark current cannot be negative")


@dataclass(frozen=True)
class NoiseConfig:
    """Independent switches for simulator ablation experiments."""

    photon_shot_enabled: bool = True
    read_noise_enabled: bool = True
    dark_current_enabled: bool = True
    background_enabled: bool = True
    prnu_enabled: bool = True
    hot_pixels_enabled: bool = True


@dataclass(frozen=True)
class EstimatorConfig:
    """Shared estimator defaults; implementations arrive in Phase 2."""

    centroid_roi_size_px: int = 13
    gaussian_roi_size_px: int = 15
    optimizer_max_nfev: int = 100

    def validate(self) -> None:
        for name, size in (
            ("centroid_roi_size_px", self.centroid_roi_size_px),
            ("gaussian_roi_size_px", self.gaussian_roi_size_px),
        ):
            if size <= 0 or size % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer")
        if self.optimizer_max_nfev <= 0:
            raise ValueError("optimizer_max_nfev must be positive")


@dataclass(frozen=True)
class ExperimentConfig:
    """Reproducible experiment controls."""

    target_snr: float = 10.0
    snr_sweep: tuple[float, ...] = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
    monte_carlo_trials: int = 1000
    development_trials: int = 200
    master_seed: int = 20260925

    def validate(self) -> None:
        if self.target_snr <= 0:
            raise ValueError("target_snr must be positive")
        if not self.snr_sweep or any(snr <= 0 for snr in self.snr_sweep):
            raise ValueError("snr_sweep must contain only positive values")
        if self.monte_carlo_trials <= 0 or self.development_trials <= 0:
            raise ValueError("trial counts must be positive")


@dataclass(frozen=True)
class TrajectoryConfig:
    """Dynamic experiment defaults; implemented in the dynamic phase."""

    development_frame_count: int = 2000
    final_frame_count: int = 5000
    frame_rate_hz: float = 1000.0
    # 73 Hz is aligned to both 0.5 Hz bins (2,000 frames) and 0.2 Hz bins
    # (5,000 frames) at a 1 kHz sample rate.
    disturbance_frequency_hz: float = 73.0
    disturbance_amplitude_px: float = 0.05

    def validate(self) -> None:
        if self.development_frame_count <= 0 or self.final_frame_count <= 0:
            raise ValueError("trajectory frame counts must be positive")
        if not 0 < self.disturbance_frequency_hz < self.frame_rate_hz / 2:
            raise ValueError("disturbance must be below Nyquist")
        if self.disturbance_amplitude_px < 0:
            raise ValueError("disturbance amplitude cannot be negative")


@dataclass(frozen=True)
class ScenarioConfig:
    """Complete immutable configuration for one experiment scenario."""

    name: str
    sensor: SensorConfig
    spot: SpotConfig
    background: BackgroundConfig
    pattern: PatternConfig
    noise: NoiseConfig
    estimator: EstimatorConfig
    experiment: ExperimentConfig
    trajectory: TrajectoryConfig

    def validate(self) -> None:
        self.sensor.validate()
        self.spot.validate()
        self.background.validate()
        self.pattern.validate()
        self.estimator.validate()
        self.experiment.validate()
        self.trajectory.validate()
        if 2 * self.spot.position_margin_px >= min(
            self.sensor.width_px, self.sensor.height_px
        ):
            raise ValueError("position margin leaves no valid interior region")


def scenario_config(profile: Literal["ideal", "sensor", "dynamic"] = "sensor") -> ScenarioConfig:
    """Return a validated named scenario profile.

    Profiles share the same model and differ only in explicitly documented
    assumptions.  ``ideal`` is intended for clean estimator/theory checks;
    ``sensor`` enables the stressed sensor effects; ``dynamic`` uses the
    sensor profile with dynamic-experiment defaults.
    """

    sensor = SensorConfig()
    spot = SpotConfig()
    background = BackgroundConfig()
    pattern = PatternConfig()
    noise = NoiseConfig()
    estimator = EstimatorConfig()
    experiment = ExperimentConfig()
    trajectory = TrajectoryConfig()

    if profile == "ideal":
        sensor = replace(
            sensor,
            quantization_enabled=False,
            saturation_enabled=False,
            read_noise_e_rms=0.0,
        )
        background = replace(background, offset_e=0.0, gradient_x_e_per_px=0.0, gradient_y_e_per_px=0.0)
        pattern = replace(pattern, prnu_std_fraction=0.0, dark_current_e_per_s=0.0, hot_pixel_count=0)
        noise = NoiseConfig(
            photon_shot_enabled=True,
            read_noise_enabled=False,
            dark_current_enabled=False,
            background_enabled=False,
            prnu_enabled=False,
            hot_pixels_enabled=False,
        )
        spot = replace(spot, sigma_rms_fraction=0.0)
    elif profile == "dynamic":
        trajectory = replace(trajectory, development_frame_count=2000)
    elif profile != "sensor":
        raise ValueError(f"unknown scenario profile: {profile}")

    config = ScenarioConfig(
        name=profile,
        sensor=sensor,
        spot=spot,
        background=background,
        pattern=pattern,
        noise=noise,
        estimator=estimator,
        experiment=experiment,
        trajectory=trajectory,
    )
    config.validate()
    return config
