from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from config import scenario_config
from estimators.centroid import estimate_centroid
from estimators.gaussian_fit import (
    estimate_gaussian,
    pixel_integrated_gaussian_model_and_jacobian,
)
from sim.simulator import SpotSimulator


def _clean_config():
    config = scenario_config("ideal")
    return replace(
        config,
        noise=replace(config.noise, photon_shot_enabled=False),
    )


def test_centroid_localizes_clean_subpixel_spot() -> None:
    config = _clean_config()
    true_position = (15.25, 16.4)
    frame = SpotSimulator(config).render(
        true_position,
        total_flux_e=10000.0,
        sigma_px=1.75,
    )

    result = estimate_centroid(frame.image_dn, config)

    assert result.success
    assert result.position_xy is not None
    np.testing.assert_allclose(result.position_xy, true_position, atol=0.02)
    assert result.diagnostics["roi_width_px"] == 13


def test_gaussian_fit_localizes_clean_subpixel_spot() -> None:
    config = _clean_config()
    true_position = (15.25, 16.4)
    frame = SpotSimulator(config).render(
        true_position,
        total_flux_e=10000.0,
        sigma_px=1.75,
    )

    result = estimate_gaussian(frame.image_dn, config)

    assert result.success, result.message
    assert result.position_xy is not None
    np.testing.assert_allclose(result.position_xy, true_position, atol=1e-5)
    assert result.diagnostics["fitted_sigma_px"] == pytest.approx(1.75)


def test_gaussian_analytic_jacobian_matches_finite_difference() -> None:
    parameters = np.array([15.2, 16.4, 500.0, 1.75, 20.0, 0.2, -0.1])
    x_coords = np.arange(12.0, 20.0)
    y_coords = np.arange(13.0, 21.0)
    center = (15.5, 16.5)
    _, analytic = pixel_integrated_gaussian_model_and_jacobian(
        parameters, x_coords, y_coords, center
    )

    numeric = np.empty_like(analytic)
    for index in range(parameters.size):
        step = 1e-5 * max(1.0, abs(parameters[index]))
        plus = parameters.copy()
        minus = parameters.copy()
        plus[index] += step
        minus[index] -= step
        plus_model, _ = pixel_integrated_gaussian_model_and_jacobian(
            plus, x_coords, y_coords, center
        )
        minus_model, _ = pixel_integrated_gaussian_model_and_jacobian(
            minus, x_coords, y_coords, center
        )
        numeric[:, index] = (plus_model - minus_model).ravel() / (2.0 * step)

    np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-6)


def test_estimators_report_failure_without_signal() -> None:
    config = _clean_config()
    image = np.zeros((config.sensor.height_px, config.sensor.width_px), dtype=float)

    centroid = estimate_centroid(image, config)
    gaussian = estimate_gaussian(image, config)

    assert not centroid.success
    assert not gaussian.success


def test_gaussian_fit_excludes_saturated_pixels() -> None:
    config = scenario_config("sensor")
    frame = SpotSimulator(config).render(
        (15.25, 16.4), total_flux_e=2e6, sigma_px=1.75
    )

    result = estimate_gaussian(frame.image_dn, config)

    assert result.peak_pixel_xy is not None
    assert result.diagnostics.get("saturated_pixel_count", 0) > 0
