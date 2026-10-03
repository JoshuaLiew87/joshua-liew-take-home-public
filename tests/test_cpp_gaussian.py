from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from config import scenario_config
from estimators.gaussian_fit import (
    estimate_gaussian,
    pixel_integrated_gaussian_model_and_jacobian,
)
from sim.simulator import SpotSimulator


cpp_backend = pytest.importorskip(
    "estimators._gaussian_model",
    reason="build the optional C++ extension to run C++ backend tests",
)


def _parameters() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    parameters = np.array([15.2, 16.4, 500.0, 1.75, 20.0, 0.2, -0.1])
    x_coords = np.arange(12.0, 20.0)
    y_coords = np.arange(13.0, 21.0)
    center = (15.5, 16.5)
    return parameters, x_coords, y_coords, center


def test_cpp_model_and_jacobian_match_python_reference() -> None:
    parameters, x_coords, y_coords, center = _parameters()
    python_model, python_jacobian = pixel_integrated_gaussian_model_and_jacobian(
        parameters, x_coords, y_coords, center, backend="python"
    )
    cpp_model, cpp_jacobian = pixel_integrated_gaussian_model_and_jacobian(
        parameters, x_coords, y_coords, center, backend="cpp"
    )

    np.testing.assert_allclose(cpp_model, python_model, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cpp_jacobian, python_jacobian, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("profile", ["ideal", "sensor"])
def test_cpp_estimator_matches_python_reference(profile: str) -> None:
    python_config = scenario_config(profile)
    cpp_config = replace(
        python_config,
        estimator=replace(python_config.estimator, gaussian_backend="cpp"),
    )
    frame = SpotSimulator(python_config).render(
        (15.25, 16.4), total_flux_e=10000.0, sigma_px=1.75
    )

    python_result = estimate_gaussian(frame.image_dn, python_config)
    cpp_result = estimate_gaussian(frame.image_dn, cpp_config)

    assert python_result.success == cpp_result.success
    assert python_result.success
    assert python_result.position_xy is not None
    assert cpp_result.position_xy is not None
    np.testing.assert_allclose(cpp_result.position_xy, python_result.position_xy, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(
        cpp_result.diagnostics["fitted_sigma_px"],
        python_result.diagnostics["fitted_sigma_px"],
        rtol=0.0,
        atol=1e-7,
    )
