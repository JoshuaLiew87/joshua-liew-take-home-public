"""Monte Carlo characterization of localization error versus SNR."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from common.io import save_json
from config import ScenarioConfig, scenario_config
from estimators.centroid import estimate_centroid
from estimators.gaussian_fit import estimate_gaussian
from sim.simulator import SpotSimulator


@dataclass(frozen=True)
class MetricSummary:
    """Summary statistics for one estimator at one requested SNR."""

    estimator: str
    requested_snr: float
    trial_count: int
    success_count: int
    failure_count: int
    failure_rate: float
    outlier_count: int
    outlier_rate: float
    bias_x_px: float
    bias_y_px: float
    bias_x_stderr_px: float
    bias_y_stderr_px: float
    std_x_px: float
    std_y_px: float
    std_x_stderr_px: float
    std_y_stderr_px: float
    rmse_x_px: float
    rmse_y_px: float
    radial_rmse_px: float
    radial_rmse_stderr_px: float
    median_radial_error_px: float
    p95_radial_error_px: float
    covariance_xy_px2: float
    correlation_xy: float
    locked_count: int
    lock_failure_count: int
    lock_failure_rate: float
    failure_rate_stderr: float
    lock_failure_rate_stderr: float
    conditional_bias_x_px: float
    conditional_bias_y_px: float
    conditional_std_x_px: float
    conditional_std_y_px: float
    conditional_radial_rmse_px: float


def _finite_std(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if values.size > 1 else float("nan")


def summarize_errors(
    errors_xy: np.ndarray,
    *,
    estimator: str,
    requested_snr: float,
    trial_count: int,
    outlier_threshold_px: float,
) -> MetricSummary:
    """Calculate success, bias, precision, and outlier metrics."""

    errors = np.asarray(errors_xy, dtype=float)
    if errors.ndim != 2 or errors.shape[1] != 2 or errors.shape[0] != trial_count:
        raise ValueError("errors_xy must have shape (trial_count, 2)")
    if outlier_threshold_px <= 0:
        raise ValueError("outlier_threshold_px must be positive")

    successful = np.all(np.isfinite(errors), axis=1)
    valid_errors = errors[successful]
    radial_errors = np.linalg.norm(valid_errors, axis=1)
    success_count = int(valid_errors.shape[0])
    failure_count = trial_count - success_count
    outlier_count = int(np.count_nonzero(radial_errors > outlier_threshold_px))
    locked = valid_errors[radial_errors <= outlier_threshold_px]
    locked_count = int(locked.shape[0])
    lock_failure_count = trial_count - locked_count

    if success_count:
        bias_x = float(np.mean(valid_errors[:, 0]))
        bias_y = float(np.mean(valid_errors[:, 1]))
        std_x = _finite_std(valid_errors[:, 0])
        std_y = _finite_std(valid_errors[:, 1])
        bias_x_stderr = float(std_x / np.sqrt(success_count)) if np.isfinite(std_x) else 0.0
        bias_y_stderr = float(std_y / np.sqrt(success_count)) if np.isfinite(std_y) else 0.0
        std_x_stderr = (
            float(std_x / np.sqrt(2.0 * (success_count - 1)))
            if success_count > 1 and np.isfinite(std_x)
            else float("nan")
        )
        std_y_stderr = (
            float(std_y / np.sqrt(2.0 * (success_count - 1)))
            if success_count > 1 and np.isfinite(std_y)
            else float("nan")
        )
        rmse_x = float(np.sqrt(np.mean(valid_errors[:, 0] ** 2)))
        rmse_y = float(np.sqrt(np.mean(valid_errors[:, 1] ** 2)))
        radial_rmse = float(np.sqrt(np.mean(radial_errors**2)))
        squared_radial_errors = radial_errors**2
        radial_rmse_stderr = (
            float(
                np.std(squared_radial_errors, ddof=1)
                / (2.0 * np.sqrt(success_count) * radial_rmse)
            )
            if success_count > 1 and radial_rmse > 0
            else 0.0
        )
        median_radial = float(np.median(radial_errors))
        p95_radial = float(np.percentile(radial_errors, 95.0))
        if success_count > 1:
            covariance_xy = float(np.cov(valid_errors[:, 0], valid_errors[:, 1], ddof=1)[0, 1])
            if np.isfinite(std_x) and np.isfinite(std_y) and std_x > 0 and std_y > 0:
                correlation_xy = float(covariance_xy / (std_x * std_y))
            else:
                correlation_xy = float("nan")
        else:
            covariance_xy = correlation_xy = float("nan")
    else:
        bias_x = bias_y = bias_x_stderr = bias_y_stderr = float("nan")
        std_x = std_y = std_x_stderr = std_y_stderr = float("nan")
        rmse_x = rmse_y = radial_rmse = radial_rmse_stderr = median_radial = p95_radial = float("nan")
        covariance_xy = correlation_xy = float("nan")

    if locked_count:
        conditional_bias_x = float(np.mean(locked[:, 0]))
        conditional_bias_y = float(np.mean(locked[:, 1]))
        conditional_std_x = _finite_std(locked[:, 0])
        conditional_std_y = _finite_std(locked[:, 1])
        conditional_radial_rmse = float(
            np.sqrt(np.mean(np.linalg.norm(locked, axis=1) ** 2))
        )
    else:
        conditional_bias_x = conditional_bias_y = float("nan")
        conditional_std_x = conditional_std_y = float("nan")
        conditional_radial_rmse = float("nan")

    return MetricSummary(
        estimator=estimator,
        requested_snr=float(requested_snr),
        trial_count=int(trial_count),
        success_count=success_count,
        failure_count=failure_count,
        failure_rate=float(failure_count / trial_count),
        outlier_count=outlier_count,
        outlier_rate=float(outlier_count / trial_count),
        bias_x_px=bias_x,
        bias_y_px=bias_y,
        bias_x_stderr_px=bias_x_stderr,
        bias_y_stderr_px=bias_y_stderr,
        std_x_px=std_x,
        std_y_px=std_y,
        std_x_stderr_px=std_x_stderr,
        std_y_stderr_px=std_y_stderr,
        rmse_x_px=rmse_x,
        rmse_y_px=rmse_y,
        radial_rmse_px=radial_rmse,
        radial_rmse_stderr_px=radial_rmse_stderr,
        median_radial_error_px=median_radial,
        p95_radial_error_px=p95_radial,
        covariance_xy_px2=covariance_xy,
        correlation_xy=correlation_xy,
        locked_count=locked_count,
        lock_failure_count=lock_failure_count,
        lock_failure_rate=float(lock_failure_count / trial_count),
        failure_rate_stderr=float(
            np.sqrt((failure_count / trial_count) * (1.0 - failure_count / trial_count) / trial_count)
        ),
        lock_failure_rate_stderr=float(
            np.sqrt(
                (lock_failure_count / trial_count)
                * (1.0 - lock_failure_count / trial_count)
                / trial_count
            )
        ),
        conditional_bias_x_px=conditional_bias_x,
        conditional_bias_y_px=conditional_bias_y,
        conditional_std_x_px=conditional_std_x,
        conditional_std_y_px=conditional_std_y,
        conditional_radial_rmse_px=conditional_radial_rmse,
    )


def _sample_positions(
    simulator: SpotSimulator,
    trial_count: int,
) -> np.ndarray:
    """Sample positions once so all SNR points use the same phases."""

    config = simulator.config
    margin = config.spot.position_margin_px
    x = simulator.rngs.positions.uniform(
        margin,
        config.sensor.width_px - 1.0 - margin,
        size=trial_count,
    )
    y = simulator.rngs.positions.uniform(
        margin,
        config.sensor.height_px - 1.0 - margin,
        size=trial_count,
    )
    return np.column_stack((x, y))


def run_snr_sweep(
    config: ScenarioConfig,
    *,
    snr_values: Iterable[float] | None = None,
    trial_count: int | None = None,
    outlier_threshold_px: float = 1.0,
) -> list[MetricSummary]:
    """Run a reproducible SNR sweep for both estimators."""

    config.validate()
    requested_snrs = tuple(
        float(snr) for snr in (config.experiment.snr_sweep if snr_values is None else snr_values)
    )
    if not requested_snrs or any(snr <= 0 or not np.isfinite(snr) for snr in requested_snrs):
        raise ValueError("snr_values must contain finite positive values")
    trials = config.experiment.monte_carlo_trials if trial_count is None else int(trial_count)
    if trials <= 0:
        raise ValueError("trial_count must be positive")

    simulator = SpotSimulator(config)
    positions = _sample_positions(simulator, trials)
    errors = {
        "centroid": np.full((trials, 2), np.nan, dtype=float),
        "gaussian": np.full((trials, 2), np.nan, dtype=float),
    }
    summaries: list[MetricSummary] = []

    for requested_snr in requested_snrs:
        errors["centroid"].fill(np.nan)
        errors["gaussian"].fill(np.nan)
        for trial_index, position in enumerate(positions):
            frame = simulator.render(
                (float(position[0]), float(position[1])),
                target_snr=requested_snr,
            )
            centroid_result = estimate_centroid(frame.image_dn, config)
            gaussian_result = estimate_gaussian(frame.image_dn, config)
            if centroid_result.success and centroid_result.position_xy is not None:
                errors["centroid"][trial_index] = np.subtract(
                    centroid_result.position_xy, position
                )
            if gaussian_result.success and gaussian_result.position_xy is not None:
                errors["gaussian"][trial_index] = np.subtract(
                    gaussian_result.position_xy, position
                )

        summaries.append(
            summarize_errors(
                errors["centroid"].copy(),
                estimator="centroid",
                requested_snr=requested_snr,
                trial_count=trials,
                outlier_threshold_px=outlier_threshold_px,
            )
        )
        summaries.append(
            summarize_errors(
                errors["gaussian"].copy(),
                estimator="gaussian",
                requested_snr=requested_snr,
                trial_count=trials,
                outlier_threshold_px=outlier_threshold_px,
            )
        )
    return summaries


def _summary_rows(summaries: Iterable[MetricSummary]) -> list[dict[str, float | int | str]]:
    return [
        {
            key: value
            for key, value in summary.__dict__.items()
        }
        for summary in summaries
    ]


def save_sweep_results(
    output_dir: str | Path,
    config: ScenarioConfig,
    summaries: list[MetricSummary],
    *,
    profile: str,
    outlier_threshold_px: float,
) -> tuple[Path, Path]:
    """Save JSON metadata and a flat CSV table for downstream analysis."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = _summary_rows(summaries)
    actual_trial_count = summaries[0].trial_count if summaries else 0
    metadata = {
        "profile": profile,
        "master_seed": config.experiment.master_seed,
        "trial_count": actual_trial_count,
        "outlier_threshold_px": float(outlier_threshold_px),
        "configuration": config,
        "summaries": rows,
    }
    json_path = destination / "static_characterization.json"
    csv_path = destination / "static_characterization.csv"
    save_json(json_path, metadata)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def plot_sweep(
    summaries: list[MetricSummary],
    output_path: str | Path,
    *,
    theory: dict[str, np.ndarray] | None = None,
    show: bool = False,
) -> Path:
    """Create bias, precision, RMSE, and failure-rate versus SNR plots."""

    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    by_estimator = {
        estimator: [summary for summary in summaries if summary.estimator == estimator]
        for estimator in ("centroid", "gaussian")
    }
    colors = {"centroid": "tab:blue", "gaussian": "tab:orange"}
    labels = {"centroid": "Windowed centroid", "gaussian": "Gaussian fit"}

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    panels = (
        (
            axes[0, 0],
            (("bias_x_px", "bias_x_stderr_px", "x"), ("bias_y_px", "bias_y_stderr_px", "y")),
            "bias [pixel]",
            "Position bias",
        ),
        (
            axes[0, 1],
            (("std_x_px", "std_x_stderr_px", "x"), ("std_y_px", "std_y_stderr_px", "y")),
            "standard deviation [pixel]",
            "Position precision",
        ),
        (
            axes[1, 0],
            (("radial_rmse_px", "radial_rmse_stderr_px", "radial"),),
            "radial RMSE [pixel]",
            "Radial RMSE",
        ),
    )
    for estimator, estimator_summaries in by_estimator.items():
        if not estimator_summaries:
            continue
        snrs = np.asarray([item.requested_snr for item in estimator_summaries])
        color = colors[estimator]
        label = labels[estimator]
        for axis, quantities, ylabel, title in panels:
            for value_name, error_name, component in quantities:
                values = np.asarray([getattr(item, value_name) for item in estimator_summaries])
                errors = None if error_name is None else np.asarray(
                    [getattr(item, error_name) for item in estimator_summaries]
                )
                line_style = "-" if component in ("x", "radial") else "--"
                component_label = "" if component == "radial" or component == "failure" else f" ({component})"
                axis.errorbar(
                    snrs,
                    values,
                    yerr=errors,
                    marker="o",
                    linestyle=line_style,
                    linewidth=1.5,
                    capsize=3,
                    color=color,
                    label=f"{label}{component_label}",
                )
            axis.set_xscale("log")
            axis.set_xlabel("reference SNR")
            axis.set_ylabel(ylabel)
            axis.set_title(title)
            axis.grid(True, alpha=0.3)
    if theory is not None:
        axes[0, 1].plot(
            theory["snr"],
            theory["std_x"],
            color="black",
            linestyle=":",
            linewidth=1.5,
            label="Poisson reference (x)",
        )
        axes[0, 1].plot(
            theory["snr"],
            theory["std_y"],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="Poisson reference (y)",
        )

    failure_axis = axes[1, 1]
    for estimator, estimator_summaries in by_estimator.items():
        if not estimator_summaries:
            continue
        snrs = np.asarray([item.requested_snr for item in estimator_summaries])
        color = colors[estimator]
        label = labels[estimator]
        failure_axis.errorbar(
            snrs,
            [item.failure_rate for item in estimator_summaries],
            yerr=[item.failure_rate_stderr for item in estimator_summaries],
            fmt="o",
            linestyle=":",
            color=color,
            label=f"{label} invalid",
        )
        failure_axis.errorbar(
            snrs,
            [item.lock_failure_rate for item in estimator_summaries],
            yerr=[item.lock_failure_rate_stderr for item in estimator_summaries],
            fmt="s",
            linestyle="-",
            color=color,
            label=f"{label} lock loss",
        )
    failure_axis.set_xscale("log")
    failure_axis.set_xlabel("reference SNR")
    failure_axis.set_ylabel("rate")
    failure_axis.set_title("Invalid results and lock loss")
    failure_axis.grid(True, alpha=0.3)
    failure_axis.set_ylim(bottom=0.0)
    axes[0, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 0].legend()
    axes[0, 1].legend()
    failure_axis.legend(fontsize="small")
    figure.suptitle("Static sub-pixel localization characterization")
    figure.savefig(output, dpi=160)
    if show:
        plt.show()
    plt.close(figure)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("ideal", "sensor"), default="ideal")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--snr-values", type=float, nargs="+")
    parser.add_argument("--output-dir", default="results/static")
    parser.add_argument("--figure", default="figures/static_characterization.png")
    parser.add_argument("--outlier-threshold-px", type=float, default=1.0)
    parser.add_argument(
        "--no-theory",
        action="store_true",
        help="do not compute or overlay the clean-model precision reference",
    )
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
    config = scenario_config(args.profile)
    summaries = run_snr_sweep(
        config,
        snr_values=args.snr_values,
        trial_count=args.trials,
        outlier_threshold_px=args.outlier_threshold_px,
    )
    save_sweep_results(
        args.output_dir,
        config,
        summaries,
        profile=args.profile,
        outlier_threshold_px=args.outlier_threshold_px,
    )
    theory = None
    if args.profile == "ideal" and not args.no_theory:
        from experiments.theory import compute_position_crlb, save_theory_results

        theory = compute_position_crlb(
            config,
            snr_values=args.snr_values,
        )
        save_theory_results(args.output_dir, config, theory)
    if not args.no_plot:
        plot_sweep(summaries, args.figure, theory=theory, show=args.show)


if __name__ == "__main__":
    main()
