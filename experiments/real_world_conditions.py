"""Controlled stress analysis for outdoor operating conditions.

The assignment asks primarily for an analysis of uncontrolled outdoor effects.
This module adds a small reproducible experiment for the effects that can be
represented without creating a second sensor model: temporal scintillation,
stronger background illumination, PSF broadening, and saturation.  The
remaining effects are documented as engineering failure modes in the report.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from common.io import save_json
from config import BackgroundConfig, ScenarioConfig, scenario_config
from experiments.characterize_snr import summarize_errors
from estimators.centroid import estimate_centroid
from estimators.gaussian_fit import estimate_gaussian
from sim.simulator import SpotSimulator, total_flux_for_target_snr


@dataclass(frozen=True)
class ConditionSpec:
    """One controlled deviation from the nominal sensor scenario."""

    name: str
    description: str
    dominant_effect: str
    flux_multiplier: float = 1.0
    sigma_px: float | None = None
    background: BackgroundConfig | None = None
    scintillation_log_rms: float = 0.0
    scintillation_correlation: float = 0.0


@dataclass(frozen=True)
class ConditionMetric:
    """Estimator metrics and diagnostics for one stress condition."""

    condition: str
    estimator: str
    description: str
    dominant_effect: str
    trial_count: int
    nominal_target_snr: float
    mean_effective_snr: float
    p05_effective_snr: float
    p95_effective_snr: float
    mean_saturated_pixels: float
    saturated_frame_rate: float
    mean_scintillation_factor: float
    success_count: int
    failure_count: int
    failure_rate: float
    lock_failure_count: int
    lock_failure_rate: float
    radial_rmse_px: float
    conditional_radial_rmse_px: float
    p95_radial_error_px: float


def default_conditions() -> tuple[ConditionSpec, ...]:
    """Return the controlled Phase 6 scenarios.

    The nominal spot flux is held fixed across the non-saturation cases.  This
    makes a background or width change visible as a real loss of information,
    rather than silently recalibrating each condition back to SNR 10.
    """

    return (
        ConditionSpec(
            name="nominal",
            description="Nominal stressed sensor profile at reference SNR 10",
            dominant_effect="reference",
        ),
        ConditionSpec(
            name="scintillation",
            description="Correlated log-normal intensity modulation",
            dominant_effect="effective SNR variation and temporal amplitude modulation",
            scintillation_log_rms=0.5,
            scintillation_correlation=0.95,
        ),
        ConditionSpec(
            name="bright_background",
            description="Elevated sky/background illumination and gradient",
            dominant_effect="increased variance and background-model sensitivity",
            background=BackgroundConfig(
                offset_e=200.0,
                gradient_x_e_per_px=1.5,
                gradient_y_e_per_px=-1.0,
            ),
        ),
        ConditionSpec(
            name="defocus",
            description="Broadened PSF from focus or thermal-optical drift",
            dominant_effect="reduced peak SNR and possible model mismatch",
            sigma_px=2.4,
        ),
        ConditionSpec(
            name="saturation",
            description="Large brightness increase with the nominal ADC enabled",
            dominant_effect="clipping, centroid bias, or estimator dropout",
            flux_multiplier=100.0,
        ),
    )


def _sample_positions(config: ScenarioConfig, trial_count: int) -> np.ndarray:
    margin = config.spot.position_margin_px
    rng = np.random.default_rng(np.random.SeedSequence([config.experiment.master_seed, 602]))
    x = rng.uniform(
        margin,
        config.sensor.width_px - 1.0 - margin,
        size=trial_count,
    )
    y = rng.uniform(
        margin,
        config.sensor.height_px - 1.0 - margin,
        size=trial_count,
    )
    return np.column_stack((x, y))


def _scintillation_factors(
    frame_count: int,
    *,
    log_rms: float,
    correlation: float,
    seed: int,
) -> np.ndarray:
    """Generate stationary, mean-one correlated log-normal intensity factors."""

    if log_rms <= 0:
        return np.ones(frame_count, dtype=float)
    if not 0 <= correlation < 1:
        raise ValueError("scintillation correlation must lie in [0, 1)")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 603]))
    innovation_rms = log_rms * np.sqrt(1.0 - correlation * correlation)
    log_intensity = np.empty(frame_count, dtype=float)
    log_intensity[0] = rng.normal(0.0, log_rms)
    for index in range(1, frame_count):
        log_intensity[index] = (
            correlation * log_intensity[index - 1]
            + rng.normal(0.0, innovation_rms)
        )
    return np.exp(log_intensity - 0.5 * log_rms * log_rms)


def run_condition_analysis(
    config: ScenarioConfig,
    *,
    trial_count: int = 300,
    target_snr: float = 10.0,
    outlier_threshold_px: float = 1.0,
    conditions: tuple[ConditionSpec, ...] | None = None,
) -> list[ConditionMetric]:
    """Evaluate both estimators under the controlled outdoor stress cases."""

    config.validate()
    if trial_count <= 0:
        raise ValueError("trial_count must be positive")
    if target_snr <= 0 or not np.isfinite(target_snr):
        raise ValueError("target_snr must be finite and positive")

    condition_specs = default_conditions() if conditions is None else conditions
    nominal_config = scenario_config("sensor")
    nominal_simulator = SpotSimulator(nominal_config)
    nominal_flux = total_flux_for_target_snr(
        nominal_config,
        nominal_simulator.maps,
        target_snr,
        sigma_px=nominal_config.spot.sigma_px,
    )
    positions = _sample_positions(config, trial_count)
    metrics: list[ConditionMetric] = []

    for condition_index, spec in enumerate(condition_specs):
        condition_config = config
        if spec.background is not None:
            condition_config = replace(condition_config, background=spec.background)
        simulator = SpotSimulator(condition_config)
        factors = _scintillation_factors(
            trial_count,
            log_rms=spec.scintillation_log_rms,
            correlation=spec.scintillation_correlation,
            seed=config.experiment.master_seed + condition_index,
        )
        errors = {
            "centroid": np.full((trial_count, 2), np.nan, dtype=float),
            "gaussian": np.full((trial_count, 2), np.nan, dtype=float),
        }
        effective_snrs = np.empty(trial_count, dtype=float)
        saturated_counts = np.empty(trial_count, dtype=float)

        for index, position in enumerate(positions):
            frame = simulator.render(
                (float(position[0]), float(position[1])),
                total_flux_e=nominal_flux * spec.flux_multiplier * factors[index],
                sigma_px=spec.sigma_px,
            )
            effective_snrs[index] = frame.metadata.effective_snr
            saturated_counts[index] = frame.metadata.saturated_pixel_count
            centroid_result = estimate_centroid(frame.image_dn, condition_config)
            gaussian_result = estimate_gaussian(frame.image_dn, condition_config)
            if centroid_result.success and centroid_result.position_xy is not None:
                errors["centroid"][index] = np.subtract(
                    centroid_result.position_xy, position
                )
            if gaussian_result.success and gaussian_result.position_xy is not None:
                errors["gaussian"][index] = np.subtract(
                    gaussian_result.position_xy, position
                )

        for estimator in ("centroid", "gaussian"):
            summary = summarize_errors(
                errors[estimator],
                estimator=estimator,
                requested_snr=target_snr,
                trial_count=trial_count,
                outlier_threshold_px=outlier_threshold_px,
            )
            metrics.append(
                ConditionMetric(
                    condition=spec.name,
                    estimator=estimator,
                    description=spec.description,
                    dominant_effect=spec.dominant_effect,
                    trial_count=trial_count,
                    nominal_target_snr=float(target_snr),
                    mean_effective_snr=float(np.mean(effective_snrs)),
                    p05_effective_snr=float(np.percentile(effective_snrs, 5.0)),
                    p95_effective_snr=float(np.percentile(effective_snrs, 95.0)),
                    mean_saturated_pixels=float(np.mean(saturated_counts)),
                    saturated_frame_rate=float(np.mean(saturated_counts > 0)),
                    mean_scintillation_factor=float(np.mean(factors)),
                    success_count=summary.success_count,
                    failure_count=summary.failure_count,
                    failure_rate=summary.failure_rate,
                    lock_failure_count=summary.lock_failure_count,
                    lock_failure_rate=summary.lock_failure_rate,
                    radial_rmse_px=summary.radial_rmse_px,
                    conditional_radial_rmse_px=summary.conditional_radial_rmse_px,
                    p95_radial_error_px=summary.p95_radial_error_px,
                )
            )
    return metrics


def save_condition_results(
    output_dir: str | Path,
    config: ScenarioConfig,
    metrics: list[ConditionMetric],
    *,
    trial_count: int,
    target_snr: float,
    outlier_threshold_px: float,
) -> tuple[Path, Path]:
    """Save machine-readable Phase 6 metrics."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = [asdict(metric) for metric in metrics]
    metadata = {
        "profile": config.name,
        "master_seed": config.experiment.master_seed,
        "trial_count": trial_count,
        "target_snr": target_snr,
        "outlier_threshold_px": outlier_threshold_px,
        "conditions": [asdict(condition) for condition in default_conditions()],
        "metrics": rows,
    }
    json_path = destination / "real_world_conditions.json"
    csv_path = destination / "real_world_conditions.csv"
    save_json(json_path, metadata)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def plot_condition_results(
    metrics: list[ConditionMetric],
    output_path: str | Path,
    *,
    show: bool = False,
) -> Path:
    """Plot lock loss and conditional error for the controlled stress cases."""

    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    conditions = list(dict.fromkeys(metric.condition for metric in metrics))
    labels = {"centroid": "Windowed centroid", "gaussian": "Gaussian fit"}
    colors = {"centroid": "tab:blue", "gaussian": "tab:orange"}
    x = np.arange(len(conditions))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for offset, estimator in ((-width / 2, "centroid"), (width / 2, "gaussian")):
        values = [
            next(
                metric.lock_failure_rate
                for metric in metrics
                if metric.condition == condition and metric.estimator == estimator
            )
            for condition in conditions
        ]
        axes[0].bar(x + offset, values, width, label=labels[estimator], color=colors[estimator])
        errors = [
            next(
                metric.conditional_radial_rmse_px
                for metric in metrics
                if metric.condition == condition and metric.estimator == estimator
            )
            for condition in conditions
        ]
        axes[1].bar(x + offset, errors, width, label=labels[estimator], color=colors[estimator])
    axes[0].set_ylabel("lock-loss rate")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Outdoor stress: lock loss")
    axes[1].set_ylabel("conditional radial RMSE [pixel]")
    axes[1].set_title("Outdoor stress: error when locked")
    for axis in axes:
        axis.set_xticks(x, conditions, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.3)
        axis.legend()
    figure.suptitle("Controlled real-world condition analysis (SNR 10 reference)")
    figure.savefig(output, dpi=160)
    if show:
        plt.show()
    plt.close(figure)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--target-snr", type=float, default=10.0)
    parser.add_argument("--output-dir", default="results/real_world")
    parser.add_argument("--figure", default="results/standalone/figures/real_world_conditions.png")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
    config = scenario_config("sensor")
    metrics = run_condition_analysis(
        config,
        trial_count=args.trials,
        target_snr=args.target_snr,
    )
    save_condition_results(
        args.output_dir,
        config,
        metrics,
        trial_count=args.trials,
        target_snr=args.target_snr,
        outlier_threshold_px=1.0,
    )
    if not args.no_plot:
        plot_condition_results(metrics, args.figure, show=args.show)


if __name__ == "__main__":
    main()
