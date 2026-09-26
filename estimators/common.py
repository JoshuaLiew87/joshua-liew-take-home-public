"""Shared image, ROI, and background helpers for estimator implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PeakDetection:
    """Result of finding the brightest finite pixel in an image."""

    pixel_xy: tuple[int, int]
    value_dn: float
    saturated: bool


@dataclass(frozen=True)
class ROI:
    """A rectangular image crop together with its global pixel coordinates."""

    image_dn: np.ndarray
    x_coords: np.ndarray
    y_coords: np.ndarray
    bounds: tuple[int, int, int, int]

    @property
    def x_grid(self) -> np.ndarray:
        return np.broadcast_to(self.x_coords[None, :], self.image_dn.shape)

    @property
    def y_grid(self) -> np.ndarray:
        return np.broadcast_to(self.y_coords[:, None], self.image_dn.shape)


@dataclass(frozen=True)
class EstimateResult:
    """Common output contract for all position estimators."""

    method: str
    position_xy: tuple[float, float] | None
    success: bool
    message: str
    peak_pixel_xy: tuple[int, int] | None
    roi_bounds: tuple[int, int, int, int] | None
    diagnostics: dict[str, float | int | bool | str]


def find_peak(image_dn: np.ndarray, saturation_limit_dn: float | None = None) -> PeakDetection:
    """Find the brightest finite pixel without using any ground-truth position."""

    image = np.asarray(image_dn, dtype=float)
    if image.ndim != 2:
        raise ValueError("image_dn must be a two-dimensional array")
    finite = np.isfinite(image)
    if not np.any(finite):
        raise ValueError("image_dn contains no finite pixels")
    safe_image = np.where(finite, image, -np.inf)
    y_index, x_index = np.unravel_index(int(np.argmax(safe_image)), image.shape)
    value = float(image[y_index, x_index])
    saturated = saturation_limit_dn is not None and value >= float(saturation_limit_dn)
    return PeakDetection((int(x_index), int(y_index)), value, bool(saturated))


def extract_roi(image_dn: np.ndarray, center_xy: tuple[int, int], size_px: int) -> ROI:
    """Extract the largest centered odd-sized ROI that fits in the frame."""

    image = np.asarray(image_dn, dtype=float)
    if image.ndim != 2:
        raise ValueError("image_dn must be a two-dimensional array")
    if size_px <= 0 or size_px % 2 == 0:
        raise ValueError("ROI size must be a positive odd integer")

    height, width = image.shape
    requested_size = int(size_px)
    half = requested_size // 2
    center_x = int(np.clip(round(center_xy[0]), 0, width - 1))
    center_y = int(np.clip(round(center_xy[1]), 0, height - 1))
    x_size = min(requested_size, width)
    y_size = min(requested_size, height)

    x_start = max(0, min(center_x - half, width - x_size))
    y_start = max(0, min(center_y - half, height - y_size))
    x_stop = x_start + x_size
    y_stop = y_start + y_size
    return ROI(
        image_dn=image[y_start:y_stop, x_start:x_stop],
        x_coords=np.arange(x_start, x_stop, dtype=float),
        y_coords=np.arange(y_start, y_stop, dtype=float),
        bounds=(int(x_start), int(x_stop), int(y_start), int(y_stop)),
    )


def fit_background_plane(
    roi: ROI,
    border_width_px: int,
    fit_plane: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a constant or planar background to the ROI border.

    The returned coefficients use the ROI center as their origin:
    ``background = b0 + bx * (x - cx) + by * (y - cy)``.
    """

    if border_width_px <= 0:
        raise ValueError("border_width_px must be positive")
    image = np.asarray(roi.image_dn, dtype=float)
    height, width = image.shape
    border = np.zeros_like(image, dtype=bool)
    border[:border_width_px, :] = True
    border[-border_width_px:, :] = True
    border[:, :border_width_px] = True
    border[:, -border_width_px:] = True
    valid = border & np.isfinite(image)
    if not np.any(valid):
        raise ValueError("ROI border contains no finite pixels")

    center_x = float(np.mean(roi.x_coords))
    center_y = float(np.mean(roi.y_coords))
    x_delta = roi.x_grid - center_x
    y_delta = roi.y_grid - center_y
    if fit_plane:
        design = np.column_stack((
            np.ones(np.count_nonzero(valid)),
            x_delta[valid],
            y_delta[valid],
        ))
        coefficients, *_ = np.linalg.lstsq(design, image[valid], rcond=None)
        coefficients = np.asarray(coefficients, dtype=float)
    else:
        coefficients = np.array([float(np.median(image[valid])), 0.0, 0.0])
    background = (
        coefficients[0]
        + coefficients[1] * x_delta
        + coefficients[2] * y_delta
    )
    return coefficients, background


def failure_result(
    method: str,
    message: str,
    peak_pixel_xy: tuple[int, int] | None = None,
    roi_bounds: tuple[int, int, int, int] | None = None,
    **diagnostics: float | int | bool | str,
) -> EstimateResult:
    """Create a consistent unsuccessful estimate result."""

    return EstimateResult(
        method=method,
        position_xy=None,
        success=False,
        message=message,
        peak_pixel_xy=peak_pixel_xy,
        roi_bounds=roi_bounds,
        diagnostics=diagnostics,
    )
