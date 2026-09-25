"""Composable camera simulator for the spot-localization experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

from common.rng import RNGStreams, make_rng_streams
from config import ScenarioConfig
from sim.noise import (
    ADCResult,
    FixedPatternMaps,
    convert_to_adc,
    make_fixed_pattern_maps,
    sample_poisson,
    sample_read_noise,
)
from sim.psf import render_pixel_integrated_gaussian


@dataclass(frozen=True)
class FrameMetadata:
    """Diagnostics needed to interpret one simulated sensor frame."""

    position_xy: tuple[float, float]
    target_snr: float | None
    reference_snr: float
    effective_snr: float
    total_flux_e: float
    sigma_px: float
    expected_peak_signal_e: float
    expected_peak_background_e: float
    expected_peak_dark_e: float
    peak_pixel_xy: tuple[int, int]
    saturated_pixel_count: int


@dataclass(frozen=True)
class FrameResult:
    """Noisy frame plus intermediate values useful for tests and analysis."""

    image_dn: np.ndarray
    expected_signal_e: np.ndarray
    expected_background_e: np.ndarray
    expected_dark_e: np.ndarray
    expected_charge_after_prnu_e: np.ndarray
    sampled_charge_e: np.ndarray
    saturated_mask: np.ndarray
    metadata: FrameMetadata


def render_background(shape: tuple[int, int], config: ScenarioConfig) -> np.ndarray:
    """Render the configured planar background in expected electrons/pixel."""

    height, width = shape
    y_indices, x_indices = np.indices(shape, dtype=float)
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    background = (
        config.background.offset_e
        + config.background.gradient_x_e_per_px * (x_indices - center_x)
        + config.background.gradient_y_e_per_px * (y_indices - center_y)
    )
    if np.any(background < 0):
        raise ValueError("background configuration produces negative expected charge")
    return background


def _reference_position(config: ScenarioConfig) -> tuple[float, float]:
    """Return a pixel-centered position used to label reference SNR."""

    return (
        float(config.sensor.width_px // 2),
        float(config.sensor.height_px // 2),
    )


def _reference_components(
    config: ScenarioConfig,
    maps: FixedPatternMaps,
    sigma_px: float,
) -> tuple[float, float, float]:
    """Return unit-flux peak coefficient, background, and dark charge."""

    position = _reference_position(config)
    unit_signal = render_pixel_integrated_gaussian(
        (config.sensor.height_px, config.sensor.width_px),
        position,
        total_flux_e=1.0,
        sigma_px=sigma_px,
    )
    peak_y = config.sensor.height_px // 2
    peak_x = config.sensor.width_px // 2
    background = (
        render_background(unit_signal.shape, config)
        if config.noise.background_enabled
        else np.zeros_like(unit_signal)
    )
    if config.noise.dark_current_enabled:
        # Reference SNR uses the nominal dark-current rate rather than a
        # randomly located hot pixel, so the sweep label is reproducible
        # across fixed-pattern realizations.
        dark = np.full(
            unit_signal.shape,
            config.pattern.dark_current_e_per_s * config.sensor.exposure_s,
            dtype=float,
        )
    else:
        dark = np.zeros_like(unit_signal)
    return (
        float(unit_signal[peak_y, peak_x]),
        float(background[peak_y, peak_x]),
        float(dark[peak_y, peak_x]),
    )


def total_flux_for_target_snr(
    config: ScenarioConfig,
    maps: FixedPatternMaps,
    target_snr: float,
    sigma_px: float | None = None,
) -> float:
    """Convert reference peak SNR to total expected spot electrons.

    The conversion uses a spot centered on a pixel and the pre-PRNU expected
    signal model.  The actual effective SNR is recorded separately for each
    frame because sub-pixel phase and sigma variation can change the peak
    pixel value.
    """

    if target_snr <= 0 or not np.isfinite(target_snr):
        raise ValueError("target_snr must be finite and positive")
    sigma = config.spot.sigma_px if sigma_px is None else float(sigma_px)
    unit_peak, background_peak, dark_peak = _reference_components(config, maps, sigma)
    if unit_peak <= 0:
        raise ValueError("reference PSF has no signal in its center pixel")

    noise_constant = background_peak + dark_peak + config.sensor.read_noise_e_rms**2
    snr_squared = float(target_snr) ** 2
    peak_signal = 0.5 * (
        snr_squared + sqrt(snr_squared**2 + 4.0 * snr_squared * noise_constant)
    )
    return peak_signal / unit_peak


def reference_snr_for_total_flux(
    config: ScenarioConfig,
    maps: FixedPatternMaps,
    total_flux_e: float,
    sigma_px: float,
) -> float:
    """Calculate the reference peak SNR for a given total spot flux."""

    if total_flux_e < 0 or not np.isfinite(total_flux_e):
        raise ValueError("total_flux_e must be finite and non-negative")
    unit_peak, background_peak, dark_peak = _reference_components(config, maps, sigma_px)
    peak_signal = total_flux_e * unit_peak
    denominator = sqrt(
        peak_signal
        + background_peak
        + dark_peak
        + config.sensor.read_noise_e_rms**2
    )
    return 0.0 if denominator == 0 else peak_signal / denominator


class SpotSimulator:
    """Generate reproducible noisy frames from one scenario configuration."""

    def __init__(
        self,
        config: ScenarioConfig,
        rng_streams: RNGStreams | None = None,
        fixed_pattern_maps: FixedPatternMaps | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.rngs = rng_streams or make_rng_streams(config.experiment.master_seed)
        shape = (config.sensor.height_px, config.sensor.width_px)
        self.maps = fixed_pattern_maps or make_fixed_pattern_maps(
            shape=shape,
            prnu_std_fraction=config.pattern.prnu_std_fraction,
            dark_current_e_per_s=config.pattern.dark_current_e_per_s,
            hot_pixel_count=config.pattern.hot_pixel_count,
            hot_pixel_dark_current_e_per_s=config.pattern.hot_pixel_dark_current_e_per_s,
            rng=self.rngs.fixed_pattern,
        )
        self._validate_maps()

    def _validate_maps(self) -> None:
        shape = (self.config.sensor.height_px, self.config.sensor.width_px)
        if self.maps.prnu_gain.shape != shape:
            raise ValueError("PRNU map shape does not match sensor")
        if self.maps.dark_current_e_per_s.shape != shape:
            raise ValueError("dark-current map shape does not match sensor")
        if self.maps.hot_pixel_mask.shape != shape:
            raise ValueError("hot-pixel map shape does not match sensor")

    def _choose_sigma(self, sigma_px: float | None) -> float:
        if sigma_px is not None:
            if sigma_px <= 0 or not np.isfinite(sigma_px):
                raise ValueError("sigma_px must be finite and positive")
            return float(sigma_px)
        fraction = self.config.spot.sigma_rms_fraction
        sampled_sigma = self.config.spot.sigma_px * self.rngs.spot.normal(1.0, fraction)
        if sampled_sigma <= 0:
            # This is only reachable for an extreme, invalid-by-assumption
            # sigma variation; keep the simulator safe and deterministic.
            raise ValueError("sampled sigma is non-positive")
        return float(sampled_sigma)

    def render(
        self,
        position_xy: tuple[float, float],
        *,
        target_snr: float | None = None,
        total_flux_e: float | None = None,
        sigma_px: float | None = None,
    ) -> FrameResult:
        """Render one frame at a known position.

        Exactly one brightness control is selected: ``target_snr`` maps to
        total flux using the reference pixel-centered model, while
        ``total_flux_e`` bypasses that calibration for direct physical tests.
        If neither is supplied, the scenario's default target SNR is used.
        """

        if len(position_xy) != 2 or not all(np.isfinite(position_xy)):
            raise ValueError("position_xy must contain two finite values")
        if target_snr is not None and total_flux_e is not None:
            raise ValueError("specify either target_snr or total_flux_e, not both")

        sigma = self._choose_sigma(sigma_px)
        requested_snr = (
            self.config.experiment.target_snr if target_snr is None and total_flux_e is None else target_snr
        )
        if total_flux_e is None:
            assert requested_snr is not None
            total_flux = total_flux_for_target_snr(
                self.config, self.maps, float(requested_snr), sigma_px=sigma
            )
        else:
            if total_flux_e < 0 or not np.isfinite(total_flux_e):
                raise ValueError("total_flux_e must be finite and non-negative")
            total_flux = float(total_flux_e)

        shape = (self.config.sensor.height_px, self.config.sensor.width_px)
        expected_signal = render_pixel_integrated_gaussian(
            shape, position_xy, total_flux, sigma
        )
        expected_background = (
            render_background(shape, self.config)
            if self.config.noise.background_enabled
            else np.zeros(shape, dtype=float)
        )
        if self.config.noise.dark_current_enabled:
            dark_rate = (
                self.maps.dark_current_e_per_s
                if self.config.noise.hot_pixels_enabled
                else np.full(
                    shape,
                    self.config.pattern.dark_current_e_per_s,
                    dtype=float,
                )
            )
            expected_dark = dark_rate * self.config.sensor.exposure_s
        else:
            expected_dark = np.zeros(shape, dtype=float)
        expected_before_prnu = expected_signal + expected_background + expected_dark
        if np.any(expected_before_prnu < 0):
            raise ValueError("sensor chain contains negative expected charge")
        prnu_gain = self.maps.prnu_gain if self.config.noise.prnu_enabled else 1.0
        expected_after_prnu = expected_before_prnu * prnu_gain

        if self.config.noise.photon_shot_enabled:
            photon_charge = sample_poisson(expected_after_prnu, self.rngs.shot_noise)
        else:
            photon_charge = expected_after_prnu.copy()
        if self.config.noise.read_noise_enabled:
            read_noise = sample_read_noise(
                shape, self.config.sensor.read_noise_e_rms, self.rngs.read_noise
            )
        else:
            read_noise = np.zeros(shape, dtype=float)
        sampled_charge = photon_charge + read_noise
        adc_result: ADCResult = convert_to_adc(sampled_charge, self.config.sensor)

        peak_y, peak_x = np.unravel_index(np.argmax(expected_signal), expected_signal.shape)
        peak_signal = float(expected_signal[peak_y, peak_x])
        peak_background = float(expected_background[peak_y, peak_x])
        peak_dark = float(expected_dark[peak_y, peak_x])
        peak_denominator = sqrt(
            peak_signal
            + peak_background
            + peak_dark
            + self.config.sensor.read_noise_e_rms**2
        )
        effective_snr = 0.0 if peak_denominator == 0 else peak_signal / peak_denominator
        reference_snr = reference_snr_for_total_flux(
            self.config, self.maps, total_flux, sigma
        )
        metadata = FrameMetadata(
            position_xy=(float(position_xy[0]), float(position_xy[1])),
            target_snr=None if requested_snr is None else float(requested_snr),
            reference_snr=float(reference_snr),
            effective_snr=float(effective_snr),
            total_flux_e=float(total_flux),
            sigma_px=float(sigma),
            expected_peak_signal_e=peak_signal,
            expected_peak_background_e=peak_background,
            expected_peak_dark_e=peak_dark,
            peak_pixel_xy=(int(peak_x), int(peak_y)),
            saturated_pixel_count=int(np.count_nonzero(adc_result.saturated_mask)),
        )
        return FrameResult(
            image_dn=adc_result.image_dn,
            expected_signal_e=expected_signal,
            expected_background_e=expected_background,
            expected_dark_e=expected_dark,
            expected_charge_after_prnu_e=expected_after_prnu,
            sampled_charge_e=sampled_charge,
            saturated_mask=adc_result.saturated_mask,
            metadata=metadata,
        )
