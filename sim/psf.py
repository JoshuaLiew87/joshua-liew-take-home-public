"""Pixel-integrated Gaussian point-spread functions.

Coordinate convention:

* arrays are indexed as ``image[y, x]``;
* pixel centers have integer coordinates;
* pixel ``i`` spans ``[i - 0.5, i + 0.5]``;
* a position is supplied as ``(x, y)`` in pixel-center coordinates.
"""

from __future__ import annotations

from math import isfinite

import numpy as np
from scipy.special import ndtr


def sigma_from_1e2_diameter(diameter_px: float) -> float:
    """Return sigma for an intensity diameter measured at 1/e²."""

    if diameter_px <= 0:
        raise ValueError("1/e² diameter must be positive")
    return diameter_px / 4.0


def _pixel_probability(length: int, center: float, sigma: float) -> np.ndarray:
    """Integrate a normalized 1-D Gaussian over all pixels in one axis."""

    edges = np.arange(length + 1, dtype=float) - 0.5
    upper = ndtr((edges[1:] - center) / sigma)
    lower = ndtr((edges[:-1] - center) / sigma)
    return upper - lower


def render_pixel_integrated_gaussian(
    shape: tuple[int, int],
    position_xy: tuple[float, float],
    total_flux_e: float,
    sigma_px: float,
) -> np.ndarray:
    """Render a circular Gaussian integrated over each pixel area.

    Parameters
    ----------
    shape:
        Image shape as ``(height, width)``.
    position_xy:
        Spot center as ``(x, y)`` in pixel-center coordinates.
    total_flux_e:
        Total expected spot flux in electrons before finite-frame cropping.
    sigma_px:
        Gaussian sigma in pixels.

    Returns
    -------
    numpy.ndarray
        Expected spot electrons per pixel.  The sum can be slightly less than
        ``total_flux_e`` when the Gaussian tail extends outside the frame.
    """

    if len(shape) != 2 or any(int(size) != size or size <= 0 for size in shape):
        raise ValueError("shape must contain two positive integer dimensions")
    if len(position_xy) != 2 or not all(isfinite(float(value)) for value in position_xy):
        raise ValueError("position_xy must contain two finite values")
    if not isfinite(float(total_flux_e)) or total_flux_e < 0:
        raise ValueError("total_flux_e must be finite and non-negative")
    if not isfinite(float(sigma_px)) or sigma_px <= 0:
        raise ValueError("sigma_px must be finite and positive")

    height, width = shape
    x0, y0 = position_xy
    x_probability = _pixel_probability(width, float(x0), float(sigma_px))
    y_probability = _pixel_probability(height, float(y0), float(sigma_px))
    return float(total_flux_e) * np.outer(y_probability, x_probability)

