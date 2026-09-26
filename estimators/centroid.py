"""Windowed intensity-weighted centroid estimator."""

from __future__ import annotations

import numpy as np

from config import ScenarioConfig
from estimators.common import (
    EstimateResult,
    extract_roi,
    failure_result,
    find_peak,
    fit_background_plane,
)


def estimate_centroid(image_dn: np.ndarray, config: ScenarioConfig) -> EstimateResult:
    """Estimate position from positive, background-subtracted pixel weights."""

    method = "windowed_centroid"
    image = np.asarray(image_dn, dtype=float)
    saturation_limit = config.sensor.adc_max_dn if config.sensor.saturation_enabled else None
    try:
        peak = find_peak(image, saturation_limit)
        roi = extract_roi(image, peak.pixel_xy, config.estimator.centroid_roi_size_px)
        coefficients, background = fit_background_plane(
            roi,
            config.estimator.background_border_width_px,
            config.estimator.background_fit_plane,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        return failure_result(method, str(exc))

    residual = roi.image_dn - background
    weights = np.maximum(residual, 0.0) if config.estimator.centroid_clip_negative else residual
    finite = np.isfinite(weights)
    weights = np.where(finite, weights, 0.0)
    total_signal = float(np.sum(weights))
    if total_signal <= config.estimator.minimum_signal_dn:
        return failure_result(
            method,
            "background-subtracted ROI contains insufficient positive signal",
            peak.pixel_xy,
            roi.bounds,
            peak_value_dn=peak.value_dn,
            peak_saturated=peak.saturated,
            total_signal_dn=total_signal,
        )

    x_position = float(np.sum(weights * roi.x_grid) / total_signal)
    y_position = float(np.sum(weights * roi.y_grid) / total_signal)
    background_at_peak = float(
        coefficients[0]
        + coefficients[1] * (peak.pixel_xy[0] - np.mean(roi.x_coords))
        + coefficients[2] * (peak.pixel_xy[1] - np.mean(roi.y_coords))
    )
    return EstimateResult(
        method=method,
        position_xy=(x_position, y_position),
        success=True,
        message="centroid computed",
        peak_pixel_xy=peak.pixel_xy,
        roi_bounds=roi.bounds,
        diagnostics={
            "peak_value_dn": peak.value_dn,
            "peak_saturated": peak.saturated,
            "total_signal_dn": total_signal,
            "background_at_peak_dn": background_at_peak,
            "background_b0_dn": float(coefficients[0]),
            "background_bx_dn_per_px": float(coefficients[1]),
            "background_by_dn_per_px": float(coefficients[2]),
            "roi_width_px": int(roi.image_dn.shape[1]),
            "roi_height_px": int(roi.image_dn.shape[0]),
        },
    )


def estimate(image_dn: np.ndarray, config: ScenarioConfig) -> EstimateResult:
    """Short alias used by experiment code and examples."""

    return estimate_centroid(image_dn, config)
