"""Independent sensor-noise and ADC operations.

The functions in this module are intentionally small and composable.  The
simulator controls their physical ordering; this module does not hide whether
an effect is persistent across frames or resampled for each frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import SensorConfig


@dataclass(frozen=True)
class FixedPatternMaps:
    """Sensor maps that persist for the lifetime of a simulator."""

    prnu_gain: np.ndarray
    dark_current_e_per_s: np.ndarray
    hot_pixel_mask: np.ndarray


@dataclass(frozen=True)
class ADCResult:
    """ADC output and information about clipping before quantization."""

    image_dn: np.ndarray
    saturated_mask: np.ndarray


def make_fixed_pattern_maps(
    shape: tuple[int, int],
    prnu_std_fraction: float,
    dark_current_e_per_s: float,
    hot_pixel_count: int,
    hot_pixel_dark_current_e_per_s: float,
    rng: np.random.Generator,
) -> FixedPatternMaps:
    """Create persistent PRNU, dark-current, and hot-pixel maps.

    PRNU is represented as a multiplicative gain on expected charge.  Hot
    pixels are represented as pixels with elevated dark-current rates, so they
    remain stochastic but retain their locations across frames.
    """

    height, width = shape
    if len(shape) != 2 or height <= 0 or width <= 0:
        raise ValueError("shape must contain two positive dimensions")
    if prnu_std_fraction < 0:
        raise ValueError("PRNU standard deviation cannot be negative")
    if dark_current_e_per_s < 0 or hot_pixel_dark_current_e_per_s < 0:
        raise ValueError("dark-current rates cannot be negative")
    if not 0 <= hot_pixel_count <= height * width:
        raise ValueError("hot-pixel count must fit within the image")

    prnu_gain = rng.normal(1.0, prnu_std_fraction, size=shape)
    # The normal approximation is appropriate for the small default PRNU;
    # clipping prevents an invalid negative photo-response in extreme configs.
    prnu_gain = np.maximum(prnu_gain, np.finfo(float).eps)

    dark_current = np.full(shape, dark_current_e_per_s, dtype=float)
    hot_pixel_mask = np.zeros(shape, dtype=bool)
    if hot_pixel_count:
        flat_indices = rng.choice(height * width, size=hot_pixel_count, replace=False)
        hot_pixel_mask.flat[flat_indices] = True
        dark_current[hot_pixel_mask] = hot_pixel_dark_current_e_per_s

    return FixedPatternMaps(
        prnu_gain=prnu_gain,
        dark_current_e_per_s=dark_current,
        hot_pixel_mask=hot_pixel_mask,
    )


def sample_poisson(expected_e: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample independent Poisson charge from a non-negative expectation."""

    expected = np.asarray(expected_e, dtype=float)
    if np.any(~np.isfinite(expected)) or np.any(expected < 0):
        raise ValueError("Poisson expectation must be finite and non-negative")
    return rng.poisson(expected).astype(float)


def sample_read_noise(
    shape: tuple[int, int],
    sigma_e_rms: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate independent additive Gaussian read noise in electrons."""

    if sigma_e_rms < 0:
        raise ValueError("read-noise RMS cannot be negative")
    return rng.normal(0.0, sigma_e_rms, size=shape)


def convert_to_adc(charge_e: np.ndarray, sensor: SensorConfig) -> ADCResult:
    """Convert charge in electrons to ADC output using sensor settings.

    Read noise is already included in ``charge_e`` when this function is
    called.  Saturation is applied before conversion and quantization is
    round-to-nearest.  When saturation is disabled, the function leaves the
    floating-point signal unclipped unless quantization is enabled.
    """

    sensor.validate()
    charge = np.asarray(charge_e, dtype=float)
    if np.any(~np.isfinite(charge)):
        raise ValueError("charge must be finite")

    if sensor.saturation_enabled:
        max_charge_e = min(sensor.full_well_e, sensor.adc_max_dn * sensor.electrons_per_dn)
        saturated_mask = (charge < 0.0) | (charge > max_charge_e)
        charge_for_adc = np.clip(charge, 0.0, max_charge_e)
    else:
        saturated_mask = np.zeros(charge.shape, dtype=bool)
        charge_for_adc = charge

    image_dn = charge_for_adc / sensor.electrons_per_dn
    if sensor.quantization_enabled:
        image_dn = np.rint(image_dn).astype(np.int64)
        if sensor.saturation_enabled:
            image_dn = np.clip(
                image_dn, 0, int(sensor.adc_max_dn)
            ).astype(np.int64)

    return ADCResult(image_dn=image_dn, saturated_mask=saturated_mask)
