"""Clean-model Fisher-information reference for spot-position precision."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np

from common.io import save_json
from config import ScenarioConfig
from estimators.common import extract_roi
from estimators.gaussian_fit import pixel_integrated_gaussian_model_and_jacobian
from sim.psf import render_pixel_integrated_gaussian
from sim.simulator import SpotSimulator, total_flux_for_target_snr


def compute_position_crlb(
    config: ScenarioConfig,
    *,
    snr_values: Iterable[float] | None = None,
    phase_grid_size: int = 8,
    roi_size_px: int | None = None,
) -> dict[str, np.ndarray | int | float]:
    """Compute a phase-averaged Poisson Fisher-information reference.

    The parameter vector contains position, total flux, width, and a planar
    background.  This is a clean-model precision reference, not a bound for
    clipped, quantized, or contaminated stressed-sensor frames.
    """

    config.validate()
    if phase_grid_size <= 0:
        raise ValueError("phase_grid_size must be positive")
    requested_snrs = np.asarray(
        tuple(config.experiment.snr_sweep if snr_values is None else snr_values),
        dtype=float,
    )
    if requested_snrs.ndim != 1 or not requested_snrs.size or np.any(requested_snrs <= 0):
        raise ValueError("snr_values must contain positive values")
    roi_size = config.estimator.gaussian_roi_size_px if roi_size_px is None else int(roi_size_px)
    if roi_size <= 0 or roi_size % 2 == 0:
        raise ValueError("roi_size_px must be a positive odd integer")

    simulator = SpotSimulator(config)
    center_x = config.sensor.width_px // 2
    center_y = config.sensor.height_px // 2
    phase_offsets = (np.arange(phase_grid_size, dtype=float) + 0.5) / phase_grid_size - 0.5
    variance_x = np.zeros(requested_snrs.size, dtype=float)
    variance_y = np.zeros(requested_snrs.size, dtype=float)
    valid_phase_counts = np.zeros(requested_snrs.size, dtype=int)

    for snr_index, requested_snr in enumerate(requested_snrs):
        phase_variances: list[tuple[float, float]] = []
        for offset_x in phase_offsets:
            for offset_y in phase_offsets:
                position = (center_x + offset_x, center_y + offset_y)
                sigma_px = config.spot.sigma_px
                total_flux_e = total_flux_for_target_snr(
                    config, simulator.maps, float(requested_snr), sigma_px=sigma_px
                )
                # Reuse the estimator's ROI convention while evaluating the
                # expected clean image at the known phase.
                expected_e = render_pixel_integrated_gaussian(
                    (config.sensor.height_px, config.sensor.width_px),
                    position,
                    total_flux_e,
                    sigma_px,
                )
                roi = extract_roi(expected_e, (round(position[0]), round(position[1])), roi_size)
                total_flux_dn = total_flux_e / config.sensor.electrons_per_dn
                parameters = np.array(
                    [position[0], position[1], total_flux_dn, sigma_px, 0.0, 0.0, 0.0],
                    dtype=float,
                )
                model_dn, jacobian = pixel_integrated_gaussian_model_and_jacobian(
                    parameters,
                    roi.x_coords,
                    roi.y_coords,
                    (float(np.mean(roi.x_coords)), float(np.mean(roi.y_coords))),
                )
                variance_dn = np.maximum(model_dn.ravel(), 0.0) / config.sensor.electrons_per_dn
                read_noise_dn = config.sensor.read_noise_e_rms / config.sensor.electrons_per_dn
                variance_dn += read_noise_dn * read_noise_dn
                valid = variance_dn > 0.0
                if np.count_nonzero(valid) < 8:
                    continue
                weighted_jacobian = jacobian[valid] / np.sqrt(variance_dn[valid, None])
                fisher = weighted_jacobian.T @ weighted_jacobian
                covariance = np.linalg.pinv(fisher, hermitian=True)
                if np.all(np.isfinite(covariance)):
                    phase_variances.append((float(covariance[0, 0]), float(covariance[1, 1])))

        if not phase_variances:
            variance_x[snr_index] = np.nan
            variance_y[snr_index] = np.nan
        else:
            phase_array = np.asarray(phase_variances, dtype=float)
            variance_x[snr_index] = float(np.mean(phase_array[:, 0]))
            variance_y[snr_index] = float(np.mean(phase_array[:, 1]))
            valid_phase_counts[snr_index] = phase_array.shape[0]

    return {
        "snr": requested_snrs,
        "std_x": np.sqrt(variance_x),
        "std_y": np.sqrt(variance_y),
        "radial_std": np.sqrt(variance_x + variance_y),
        "phase_grid_size": int(phase_grid_size),
        "roi_size_px": int(roi_size),
        "valid_phase_counts": valid_phase_counts,
    }


def save_theory_results(
    output_dir: str | Path,
    config: ScenarioConfig,
    theory: dict[str, np.ndarray | int | float],
) -> tuple[Path, Path]:
    """Save the theoretical reference in JSON and CSV form."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    snr = np.asarray(theory["snr"], dtype=float)
    rows = [
        {
            "requested_snr": float(snr[index]),
            "crlb_std_x_px": float(np.asarray(theory["std_x"])[index]),
            "crlb_std_y_px": float(np.asarray(theory["std_y"])[index]),
            "crlb_radial_std_px": float(np.asarray(theory["radial_std"])[index]),
            "valid_phase_count": int(np.asarray(theory["valid_phase_counts"])[index]),
        }
        for index in range(snr.size)
    ]
    metadata = {
        "model": "pixel-integrated Gaussian with Poisson/read-noise Fisher information",
        "configuration": config,
        "phase_grid_size": theory["phase_grid_size"],
        "roi_size_px": theory["roi_size_px"],
        "rows": rows,
    }
    json_path = destination / "theoretical_precision.json"
    csv_path = destination / "theoretical_precision.csv"
    save_json(json_path, metadata)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path
