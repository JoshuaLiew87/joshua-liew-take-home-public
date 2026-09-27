"""Controlled sensor-effect ablations for estimator comparison."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from common.io import save_json
from config import NoiseConfig, ScenarioConfig, scenario_config
from experiments.characterize_snr import summarize_errors
from estimators.centroid import estimate_centroid
from estimators.gaussian_fit import estimate_gaussian
from sim.simulator import SpotSimulator


@dataclass(frozen=True)
class AblationSpec:
    """One deliberately isolated sensor configuration."""

    name: str
    description: str


@dataclass(frozen=True)
class AblationMetric:
    """Comparison metrics for one effect, SNR, and estimator."""

    condition: str
    description: str
    estimator: str
    requested_snr: float
    trial_count: int
    lock_failure_rate: float
    lock_failure_rate_stderr: float
    radial_rmse_px: float
    radial_rmse_stderr_px: float
    conditional_radial_rmse_px: float
    mean_estimator_disagreement_px: float
    mean_gaussian_cost: float
    mean_gaussian_nfev: float


def default_ablation_specs() -> tuple[AblationSpec, ...]:
    """Return the clean-to-stressed progression used in the report."""

    return (
        AblationSpec("ideal", "Photon shot noise only; no background, read noise, or sensor clipping"),
        AblationSpec("background_gradient", "Planar background and gradient added to the clean model"),
        AblationSpec("prnu_hot_pixels", "Persistent PRNU and elevated dark-current hot pixels"),
        AblationSpec("quantization_saturation", "12-bit quantization and clipping without other stressed effects"),
        AblationSpec("full_stressed", "Nominal stressed sensor profile with all configured effects"),
    )


def _ablation_config(base: ScenarioConfig, condition: str) -> ScenarioConfig:
    """Build an ablation without duplicating simulator or estimator code."""

    clean_sensor = replace(
        base.sensor,
        read_noise_e_rms=0.0,
        quantization_enabled=False,
        saturation_enabled=False,
    )
    clean_spot = replace(base.spot, sigma_rms_fraction=0.0)
    no_pattern = replace(
        base.pattern,
        prnu_std_fraction=0.0,
        dark_current_e_per_s=0.0,
        hot_pixel_count=0,
    )
    no_background = replace(
        base.background,
        offset_e=0.0,
        gradient_x_e_per_px=0.0,
        gradient_y_e_per_px=0.0,
    )
    photon_only = NoiseConfig(
        photon_shot_enabled=True,
        read_noise_enabled=False,
        dark_current_enabled=False,
        background_enabled=False,
        prnu_enabled=False,
        hot_pixels_enabled=False,
    )
    if condition == "ideal":
        resolved = replace(
            base,
            sensor=clean_sensor,
            spot=clean_spot,
            background=no_background,
            pattern=no_pattern,
            noise=photon_only,
        )
    elif condition == "background_gradient":
        resolved = replace(
            base,
            sensor=clean_sensor,
            spot=clean_spot,
            pattern=no_pattern,
            noise=replace(photon_only, background_enabled=True),
        )
    elif condition == "prnu_hot_pixels":
        resolved = replace(
            base,
            sensor=clean_sensor,
            spot=clean_spot,
            background=no_background,
            noise=replace(photon_only, dark_current_enabled=True, prnu_enabled=True, hot_pixels_enabled=True),
        )
    elif condition == "quantization_saturation":
        resolved = replace(
            base,
            sensor=replace(base.sensor, read_noise_e_rms=0.0),
            spot=clean_spot,
            background=no_background,
            pattern=no_pattern,
            noise=photon_only,
        )
    elif condition == "full_stressed":
        resolved = base
    else:
        raise ValueError(f"unknown ablation condition: {condition}")
    resolved.validate()
    return resolved


def _sample_positions(config: ScenarioConfig, trial_count: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([config.experiment.master_seed, 711]))
    margin = config.spot.position_margin_px
    return np.column_stack(
        (
            rng.uniform(margin, config.sensor.width_px - 1.0 - margin, size=trial_count),
            rng.uniform(margin, config.sensor.height_px - 1.0 - margin, size=trial_count),
        )
    )


def run_ablation(
    config: ScenarioConfig,
    *,
    snr_values: tuple[float, ...] = (1.0, 3.0),
    trial_count: int = 250,
    outlier_threshold_px: float = 1.0,
    specs: tuple[AblationSpec, ...] | None = None,
) -> list[AblationMetric]:
    """Measure lock loss, RMSE, fit cost, and estimator disagreement."""

    config.validate()
    if trial_count <= 0 or outlier_threshold_px <= 0:
        raise ValueError("trial_count and outlier threshold must be positive")
    if not snr_values or any(value <= 0 or not np.isfinite(value) for value in snr_values):
        raise ValueError("snr_values must be finite and positive")
    selected_specs = default_ablation_specs() if specs is None else specs
    base_positions = _sample_positions(config, trial_count)
    metrics: list[AblationMetric] = []

    for spec in selected_specs:
        condition_config = _ablation_config(config, spec.name)
        simulator = SpotSimulator(condition_config)
        for requested_snr in snr_values:
            errors = {name: np.full((trial_count, 2), np.nan) for name in ("centroid", "gaussian")}
            gaussian_costs: list[float] = []
            gaussian_nfev: list[float] = []
            disagreements: list[float] = []
            for index, position in enumerate(base_positions):
                frame = simulator.render(tuple(position), target_snr=float(requested_snr))
                centroid = estimate_centroid(frame.image_dn, condition_config)
                gaussian = estimate_gaussian(frame.image_dn, condition_config)
                if centroid.success and centroid.position_xy is not None:
                    errors["centroid"][index] = np.subtract(centroid.position_xy, position)
                if gaussian.success and gaussian.position_xy is not None:
                    errors["gaussian"][index] = np.subtract(gaussian.position_xy, position)
                    cost = gaussian.diagnostics.get("cost")
                    nfev = gaussian.diagnostics.get("nfev")
                    if isinstance(cost, (int, float)) and np.isfinite(cost):
                        gaussian_costs.append(float(cost))
                    if isinstance(nfev, (int, float)) and np.isfinite(nfev):
                        gaussian_nfev.append(float(nfev))
                if centroid.success and gaussian.success and centroid.position_xy and gaussian.position_xy:
                    disagreements.append(float(np.linalg.norm(np.subtract(centroid.position_xy, gaussian.position_xy))))

            for estimator in ("centroid", "gaussian"):
                summary = summarize_errors(
                    errors[estimator],
                    estimator=estimator,
                    requested_snr=float(requested_snr),
                    trial_count=trial_count,
                    outlier_threshold_px=outlier_threshold_px,
                )
                metrics.append(
                    AblationMetric(
                        condition=spec.name,
                        description=spec.description,
                        estimator=estimator,
                        requested_snr=float(requested_snr),
                        trial_count=trial_count,
                        lock_failure_rate=summary.lock_failure_rate,
                        lock_failure_rate_stderr=summary.lock_failure_rate_stderr,
                        radial_rmse_px=summary.radial_rmse_px,
                        radial_rmse_stderr_px=summary.radial_rmse_stderr_px,
                        conditional_radial_rmse_px=summary.conditional_radial_rmse_px,
                        mean_estimator_disagreement_px=float(np.mean(disagreements)) if disagreements else float("nan"),
                        mean_gaussian_cost=float(np.mean(gaussian_costs)) if gaussian_costs else float("nan"),
                        mean_gaussian_nfev=float(np.mean(gaussian_nfev)) if gaussian_nfev else float("nan"),
                    )
                )
    return metrics


def save_ablation_results(
    output_dir: str | Path,
    config: ScenarioConfig,
    metrics: list[AblationMetric],
    *,
    snr_values: tuple[float, ...],
    trial_count: int,
    outlier_threshold_px: float,
) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = [asdict(metric) for metric in metrics]
    payload = {
        "configuration": config,
        "snr_values": snr_values,
        "trial_count": trial_count,
        "outlier_threshold_px": outlier_threshold_px,
        "metrics": rows,
    }
    json_path = destination / "estimator_ablation.json"
    csv_path = destination / "estimator_ablation.csv"
    save_json(json_path, payload)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def plot_ablation(metrics: list[AblationMetric], output_path: str | Path) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    conditions = list(dict.fromkeys(metric.condition for metric in metrics))
    snrs = sorted({metric.requested_snr for metric in metrics})
    colors = dict(zip(conditions, plt.cm.viridis(np.linspace(0.1, 0.9, len(conditions)))))
    markers = {"centroid": "o", "gaussian": "s"}
    labels = {"centroid": "Windowed centroid", "gaussian": "Gaussian fit"}
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for condition in conditions:
        for estimator in ("centroid", "gaussian"):
            selected = [
                metric for metric in metrics
                if metric.condition == condition and metric.estimator == estimator
            ]
            selected.sort(key=lambda metric: metric.requested_snr)
            axes[0].errorbar(
                [metric.requested_snr for metric in selected],
                [metric.lock_failure_rate for metric in selected],
                yerr=[metric.lock_failure_rate_stderr for metric in selected],
                color=colors[condition], marker=markers[estimator], linestyle="-",
                capsize=3,
            )
            axes[1].plot(
                [metric.requested_snr for metric in selected],
                [metric.radial_rmse_px for metric in selected],
                color=colors[condition], marker=markers[estimator], linestyle="-",
            )
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xlabel("reference SNR")
    axes[1].set_xlabel("reference SNR")
    axes[0].set_ylabel("lock-loss rate")
    axes[1].set_ylabel("radial RMSE [pixel]")
    axes[0].set_title("Controlled lock-loss ablation")
    axes[1].set_title("Controlled RMSE ablation")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    condition_handles = [
        Line2D(
            [0], [0], color=colors[condition], marker="o", linestyle="-",
            label=condition,
        )
        for condition in conditions
    ]
    estimator_handles = [
        Line2D(
            [0], [0], color="0.25", marker=markers[estimator], linestyle="-",
            label=labels[estimator],
        )
        for estimator in ("centroid", "gaussian")
    ]
    figure.legend(
        handles=condition_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.125),
        ncol=len(condition_handles),
        fontsize="small",
        title="Sensor condition",
    )
    figure.legend(
        handles=estimator_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=2,
        fontsize="small",
        title="Estimator",
    )
    figure.suptitle("Estimator ranking under isolated sensor effects", y=0.98)
    figure.subplots_adjust(top=0.86, bottom=0.31, wspace=0.24)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=250)
    parser.add_argument("--snr-values", type=float, nargs="+", default=[1.0, 3.0])
    parser.add_argument("--output-dir", default="results/ablation")
    parser.add_argument("--figure", default="figures/estimator_ablation.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    config = scenario_config("sensor")
    snr_values = tuple(float(value) for value in args.snr_values)
    metrics = run_ablation(config, snr_values=snr_values, trial_count=args.trials)
    save_ablation_results(
        args.output_dir,
        config,
        metrics,
        snr_values=snr_values,
        trial_count=args.trials,
        outlier_threshold_px=1.0,
    )
    plot_ablation(metrics, args.figure)


if __name__ == "__main__":
    main()
