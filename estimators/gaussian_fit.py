"""Pixel-integrated 2-D Gaussian least-squares estimator."""

from __future__ import annotations

from math import sqrt

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr

from config import ScenarioConfig
from estimators.centroid import estimate_centroid
from estimators.common import (
    EstimateResult,
    extract_roi,
    failure_result,
    find_peak,
    find_peak_in_roi,
    fit_background_plane,
)

try:
    from estimators._gaussian_model import model_and_jacobian as _cpp_model_and_jacobian
except ImportError:  # Optional branch-local acceleration extension.
    _cpp_model_and_jacobian = None

_SQRT_2PI = sqrt(2.0 * np.pi)


def _normal_pdf(z: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * z * z) / _SQRT_2PI


def _integrated_probability_and_derivatives(
    coordinates: np.ndarray,
    center: float,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges_low = coordinates - 0.5
    edges_high = coordinates + 0.5
    z_low = (edges_low - center) / sigma
    z_high = (edges_high - center) / sigma
    probability = ndtr(z_high) - ndtr(z_low)
    d_center = (_normal_pdf(z_low) - _normal_pdf(z_high)) / sigma
    d_sigma = (z_low * _normal_pdf(z_low) - z_high * _normal_pdf(z_high)) / sigma
    return probability, d_center, d_sigma


def pixel_integrated_gaussian_model_and_jacobian(
    parameters: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    background_center_xy: tuple[float, float],
    backend: str = "python",
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the Gaussian model and its analytical parameter Jacobian.

    Parameters are ``[x0, y0, total_flux_dn, sigma_px, b0, bx, by]`` and the
    background is ``b0 + bx*(x-cx) + by*(y-cy)``.
    """

    if backend not in {"python", "cpp"}:
        raise ValueError("backend must be 'python' or 'cpp'")
    if backend == "cpp":
        if _cpp_model_and_jacobian is None:
            raise RuntimeError(
                "C++ Gaussian backend is unavailable; build the optional extension first"
            )
        center_x, center_y = background_center_xy
        model, jacobian = _cpp_model_and_jacobian(
            np.asarray(parameters, dtype=float),
            np.asarray(x_coords, dtype=float),
            np.asarray(y_coords, dtype=float),
            float(center_x),
            float(center_y),
        )
        return np.asarray(model, dtype=float), np.asarray(jacobian, dtype=float)

    x0, y0, total_flux, sigma, b0, bx, by = np.asarray(parameters, dtype=float)
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    x_values = np.asarray(x_coords, dtype=float)
    y_values = np.asarray(y_coords, dtype=float)
    px, dpx_dx, dpx_dsigma = _integrated_probability_and_derivatives(x_values, x0, sigma)
    py, dpy_dy, dpy_dsigma = _integrated_probability_and_derivatives(y_values, y0, sigma)
    spot = total_flux * np.outer(py, px)
    center_x, center_y = background_center_xy
    x_grid = np.broadcast_to(x_values[None, :], spot.shape)
    y_grid = np.broadcast_to(y_values[:, None], spot.shape)
    model = spot + b0 + bx * (x_grid - center_x) + by * (y_grid - center_y)

    jacobian = np.column_stack(
        (
            (total_flux * np.outer(py, dpx_dx)).ravel(),
            (total_flux * np.outer(dpy_dy, px)).ravel(),
            np.outer(py, px).ravel(),
            (total_flux * (
                np.outer(dpy_dsigma, px) + np.outer(py, dpx_dsigma)
            )).ravel(),
            np.ones(model.size),
            (x_grid - center_x).ravel(),
            (y_grid - center_y).ravel(),
        )
    )
    return model, jacobian


def estimate_gaussian(
    image_dn: np.ndarray,
    config: ScenarioConfig,
    previous_position: tuple[float, float] | None = None,
) -> EstimateResult:
    """Fit a pixel-integrated Gaussian with fixed-iteration IRLS weighting."""

    method = "pixel_integrated_gaussian"
    image = np.asarray(image_dn, dtype=float)
    backend = config.estimator.gaussian_backend
    saturation_limit = config.sensor.adc_max_dn if config.sensor.saturation_enabled else None
    previous_is_valid = previous_position is not None and len(previous_position) == 2 and all(
        np.isfinite(previous_position)
    )
    tracking_roi_used = previous_is_valid and config.estimator.gaussian_tracking_use_previous_roi
    try:
        if tracking_roi_used:
            tracking_center = (
                int(round(float(previous_position[0]))),
                int(round(float(previous_position[1]))),
            )
            roi = extract_roi(image, tracking_center, config.estimator.gaussian_roi_size_px)
            peak = find_peak_in_roi(roi, saturation_limit)
        else:
            peak = find_peak(image, saturation_limit)
            roi = extract_roi(image, peak.pixel_xy, config.estimator.gaussian_roi_size_px)
        background_coefficients, initial_background = fit_background_plane(
            roi,
            config.estimator.background_border_width_px,
            config.estimator.background_fit_plane,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        return failure_result(method, str(exc))

    valid = np.isfinite(roi.image_dn)
    if config.sensor.saturation_enabled:
        valid &= roi.image_dn < config.sensor.adc_max_dn
    if np.count_nonzero(valid) < 8:
        return failure_result(
            method,
            "too few unsaturated finite pixels for Gaussian fit",
            peak.pixel_xy,
            roi.bounds,
            valid_pixel_count=int(np.count_nonzero(valid)),
        )

    centroid = None
    if previous_is_valid and config.estimator.gaussian_skip_centroid_with_previous:
        initial_position = (float(previous_position[0]), float(previous_position[1]))
    else:
        centroid = estimate_centroid(image, config)
        if centroid.success and centroid.position_xy is not None:
            initial_position = centroid.position_xy
        else:
            initial_position = peak.pixel_xy

    signal_residual = np.maximum(roi.image_dn - initial_background, 0.0)
    initial_flux = float(np.sum(signal_residual))
    if initial_flux <= config.estimator.minimum_signal_dn:
        return failure_result(
            method,
            "ROI contains insufficient positive signal for Gaussian initialization",
            peak.pixel_xy,
            roi.bounds,
            valid_pixel_count=int(np.count_nonzero(valid)),
        )

    center_xy = (float(np.mean(roi.x_coords)), float(np.mean(roi.y_coords)))
    parameters = np.array(
        [
            initial_position[0],
            initial_position[1],
            initial_flux,
            config.spot.sigma_px,
            background_coefficients[0],
            background_coefficients[1],
            background_coefficients[2],
        ],
        dtype=float,
    )
    lower_bounds = np.array(
        [
            float(roi.x_coords.min() - 0.5),
            float(roi.y_coords.min() - 0.5),
            0.0,
            config.estimator.gaussian_sigma_min_px,
            -np.inf,
            -np.inf,
            -np.inf,
        ]
    )
    upper_bounds = np.array(
        [
            float(roi.x_coords.max() + 0.5),
            float(roi.y_coords.max() + 0.5),
            np.inf,
            config.estimator.gaussian_sigma_max_px,
            np.inf,
            np.inf,
            np.inf,
        ]
    )
    parameter_scale = np.maximum(np.abs(parameters), 1.0)
    data = roi.image_dn.ravel()[valid.ravel()]
    electrons_per_dn = config.sensor.electrons_per_dn
    read_noise_dn = config.sensor.read_noise_e_rms / electrons_per_dn
    last_fit = None
    try:
        for _ in range(config.estimator.gaussian_irls_iterations):
            model, _ = pixel_integrated_gaussian_model_and_jacobian(
                parameters, roi.x_coords, roi.y_coords, center_xy, backend=backend
            )
            variance_dn = np.maximum(
                model.ravel() / electrons_per_dn,
                config.estimator.gaussian_variance_floor_dn2,
            )
            variance_dn += read_noise_dn * read_noise_dn
            weights = 1.0 / np.sqrt(np.maximum(variance_dn, 1e-9))[valid.ravel()]

            def residual(candidate: np.ndarray) -> np.ndarray:
                candidate_model, _ = pixel_integrated_gaussian_model_and_jacobian(
                    candidate, roi.x_coords, roi.y_coords, center_xy, backend=backend
                )
                return (candidate_model.ravel()[valid.ravel()] - data) * weights

            def jacobian(candidate: np.ndarray) -> np.ndarray:
                _, candidate_jacobian = pixel_integrated_gaussian_model_and_jacobian(
                    candidate, roi.x_coords, roi.y_coords, center_xy, backend=backend
                )
                return candidate_jacobian[valid.ravel(), :] * weights[:, None]

            last_fit = least_squares(
                residual,
                parameters,
                jac=jacobian,
                bounds=(lower_bounds, upper_bounds),
                max_nfev=config.estimator.optimizer_max_nfev,
                method="trf",
                x_scale=parameter_scale,
            )
            parameters = last_fit.x
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        return failure_result(method, f"Gaussian optimization failed: {exc}", peak.pixel_xy, roi.bounds)

    if last_fit is None or not last_fit.success or not np.all(np.isfinite(parameters)):
        message = "Gaussian optimization did not converge" if last_fit is None else last_fit.message
        return failure_result(
            method,
            str(message),
            peak.pixel_xy,
            roi.bounds,
            optimizer_success=False if last_fit is None else bool(last_fit.success),
        )

    return EstimateResult(
        method=method,
        position_xy=(float(parameters[0]), float(parameters[1])),
        success=True,
        message=str(last_fit.message),
        peak_pixel_xy=peak.pixel_xy,
        roi_bounds=roi.bounds,
        diagnostics={
            "peak_value_dn": peak.value_dn,
            "peak_saturated": peak.saturated,
            "fitted_flux_dn": float(parameters[2]),
            "fitted_sigma_px": float(parameters[3]),
            "background_b0_dn": float(parameters[4]),
            "background_bx_dn_per_px": float(parameters[5]),
            "background_by_dn_per_px": float(parameters[6]),
            "cost": float(last_fit.cost),
            "nfev": int(last_fit.nfev),
            "optimality": float(last_fit.optimality),
            "valid_pixel_count": int(np.count_nonzero(valid)),
            "saturated_pixel_count": int(np.count_nonzero(~valid & np.isfinite(roi.image_dn))),
            "tracking_roi_used": bool(tracking_roi_used),
            "centroid_initialization_used": bool(centroid is not None),
        },
    )


def estimate(
    image_dn: np.ndarray,
    config: ScenarioConfig,
    previous_position: tuple[float, float] | None = None,
) -> EstimateResult:
    """Short alias used by experiment code and examples."""

    return estimate_gaussian(image_dn, config, previous_position=previous_position)
