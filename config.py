"""Configuration and scenario profiles for the spot-localization study.

The configuration deliberately separates physical sensor parameters from
experiment parameters. The simulator uses total spot flux in electrons as
its physical brightness input; experiments expose target SNR values and
calibrate them to flux through the reference-spot convention.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from collections.abc import Mapping
from pathlib import Path
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
    """Shared estimator defaults and configurable fit safeguards."""

    centroid_roi_size_px: int = 13
    gaussian_roi_size_px: int = 15
    background_border_width_px: int = 2
    background_fit_plane: bool = True
    centroid_clip_negative: bool = True
    minimum_signal_dn: float = 1e-9
    gaussian_sigma_min_px: float = 0.5
    gaussian_sigma_max_px: float = 5.0
    gaussian_irls_iterations: int = 2
    gaussian_variance_floor_dn2: float = 1.0
    optimizer_max_nfev: int = 100
    gaussian_tracking_use_previous_roi: bool = False
    gaussian_skip_centroid_with_previous: bool = True

    def validate(self) -> None:
        for name, size in (
            ("centroid_roi_size_px", self.centroid_roi_size_px),
            ("gaussian_roi_size_px", self.gaussian_roi_size_px),
        ):
            if size <= 0 or size % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer")
        if self.background_border_width_px <= 0:
            raise ValueError("background_border_width_px must be positive")
        if self.minimum_signal_dn < 0:
            raise ValueError("minimum_signal_dn cannot be negative")
        if not 0 < self.gaussian_sigma_min_px < self.gaussian_sigma_max_px:
            raise ValueError("Gaussian sigma bounds are invalid")
        if self.gaussian_irls_iterations <= 0:
            raise ValueError("gaussian_irls_iterations must be positive")
        if self.gaussian_variance_floor_dn2 <= 0:
            raise ValueError("gaussian_variance_floor_dn2 must be positive")
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
    """Dynamic motion defaults shared by experiments and the live demo."""

    development_frame_count: int = 2000
    final_frame_count: int = 5000
    frame_rate_hz: float = 1000.0
    center_x_px: float = 15.5
    center_y_px: float = 15.5
    drift_model: Literal["sinusoidal", "ou"] = "sinusoidal"
    drift_amplitude_px: float = 1.5
    drift_frequency_hz: float = 0.4
    ou_drift_rms_px: float = 0.6
    ou_correlation_time_s: float = 0.25
    ou_bound_px: float = 2.0
    jitter_rms_px: float = 0.04
    jitter_correlation: float = 0.95
    # 73 Hz is aligned to both 0.5 Hz bins (2,000 frames) and 0.2 Hz bins
    # (5,000 frames) at a 1 kHz sample rate.
    disturbance_frequency_hz: float = 73.0
    disturbance_amplitude_px: float = 0.05
    disturbance_axis: Literal["x", "y"] = "x"

    def validate(self) -> None:
        if self.development_frame_count <= 0 or self.final_frame_count <= 0:
            raise ValueError("trajectory frame counts must be positive")
        if self.frame_rate_hz <= 0:
            raise ValueError("trajectory frame rate must be positive")
        if not all(
            value == value and abs(value) != float("inf")
            for value in (self.center_x_px, self.center_y_px)
        ):
            raise ValueError("trajectory center must be finite")
        if self.drift_amplitude_px < 0 or self.drift_frequency_hz < 0:
            raise ValueError("drift parameters cannot be negative")
        if self.drift_model not in ("sinusoidal", "ou"):
            raise ValueError("drift_model must be 'sinusoidal' or 'ou'")
        if self.ou_drift_rms_px < 0 or self.ou_correlation_time_s <= 0 or self.ou_bound_px <= 0:
            raise ValueError("OU drift parameters are invalid")
        if self.jitter_rms_px < 0 or not 0 <= self.jitter_correlation < 1:
            raise ValueError("jitter parameters are invalid")
        if not 0 < self.disturbance_frequency_hz < self.frame_rate_hz / 2:
            raise ValueError("disturbance must be below Nyquist")
        if self.disturbance_amplitude_px < 0:
            raise ValueError("disturbance amplitude cannot be negative")
        if self.disturbance_axis not in ("x", "y"):
            raise ValueError("disturbance_axis must be 'x' or 'y'")


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


_CONFIG_SECTIONS = {
    "sensor",
    "spot",
    "background",
    "pattern",
    "noise",
    "estimator",
    "experiment",
    "trajectory",
}
_PROFILE_NAMES = {"ideal", "sensor", "dynamic"}


def apply_scenario_overrides(
    config: ScenarioConfig,
    overrides: Mapping[str, Mapping[str, object]],
) -> ScenarioConfig:
    """Apply strict, nested overrides to one validated scenario.

    YAML is intentionally an override layer.  Dataclass defaults and named
    scenario profiles remain the source of truth for omitted values, while
    unknown sections or fields fail early instead of being silently ignored.
    """

    if not isinstance(overrides, Mapping):
        raise ValueError("scenario overrides must be a mapping")
    updates: dict[str, object] = {}
    for section, values in overrides.items():
        if section not in _CONFIG_SECTIONS:
            raise ValueError(
                f"unknown scenario section {section!r}; expected one of "
                f"{sorted(_CONFIG_SECTIONS)}"
            )
        if not isinstance(values, Mapping):
            raise ValueError(f"overrides for {section!r} must be a mapping")
        current = getattr(config, section)
        field_names = {field.name for field in fields(current)}
        unknown = set(values) - field_names
        if unknown:
            raise ValueError(
                f"unknown fields in {section!r}: {sorted(unknown)}; "
                f"expected one of {sorted(field_names)}"
            )
        section_values = dict(values)
        if section == "experiment" and "snr_sweep" in section_values:
            sweep = section_values["snr_sweep"]
            if isinstance(sweep, (str, bytes)) or not hasattr(sweep, "__iter__"):
                raise ValueError("experiment.snr_sweep must be a sequence")
            section_values["snr_sweep"] = tuple(float(value) for value in sweep)
        try:
            updates[section] = replace(current, **section_values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid overrides for {section!r}: {exc}") from exc
    resolved = replace(config, **updates)
    resolved.validate()
    return resolved


def load_yaml_overrides(path: str | Path) -> dict[str, dict[str, object]]:
    """Load the supported YAML override document.

    The document may contain ``pipeline`` controls for ``run_all.py`` and a
    ``profiles`` mapping whose keys are ``ideal``, ``sensor``, or ``dynamic``.
    """

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise RuntimeError("PyYAML is required for YAML configuration overrides") from exc

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"YAML override file does not exist: {source}")
    document = yaml.safe_load(source.read_text(encoding="utf-8"))
    if document is None:
        return {"pipeline": {}, "profiles": {}}
    if not isinstance(document, Mapping):
        raise ValueError("YAML override document must be a mapping")
    unknown_top_level = set(document) - {"pipeline", "profiles"}
    if unknown_top_level:
        raise ValueError(
            f"unknown YAML top-level keys: {sorted(unknown_top_level)}; "
            "expected 'pipeline' and/or 'profiles'"
        )
    pipeline = document.get("pipeline", {})
    profiles = document.get("profiles", {})
    if not isinstance(pipeline, Mapping):
        raise ValueError("YAML 'pipeline' must be a mapping")
    if not isinstance(profiles, Mapping):
        raise ValueError("YAML 'profiles' must be a mapping")
    unknown_profiles = set(profiles) - _PROFILE_NAMES
    if unknown_profiles:
        raise ValueError(f"unknown YAML profiles: {sorted(unknown_profiles)}")
    normalized_profiles: dict[str, object] = {}
    for profile, overrides in profiles.items():
        if not isinstance(overrides, Mapping):
            raise ValueError(f"YAML profile {profile!r} must be a mapping")
        normalized_profiles[profile] = dict(overrides)
    return {"pipeline": dict(pipeline), "profiles": normalized_profiles}


def scenario_config(
    profile: Literal["ideal", "sensor", "dynamic"] = "sensor",
    *,
    overrides: Mapping[str, Mapping[str, object]] | None = None,
) -> ScenarioConfig:
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
        trajectory = replace(
            trajectory,
            development_frame_count=2000,
            drift_model="ou",
            ou_drift_rms_px=0.6,
            ou_correlation_time_s=0.25,
            ou_bound_px=2.0,
        )
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
    if overrides:
        config = apply_scenario_overrides(config, overrides)
    config.validate()
    return config
