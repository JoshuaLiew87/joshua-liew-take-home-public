"""ROI-size sensitivity study for the two localization estimators."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from common.io import save_json
from config import scenario_config
from experiments.characterize_snr import MetricSummary, run_snr_sweep


def run_roi_sensitivity(
    config,
    *,
    snr_values: tuple[float, ...] = (3.0, 10.0, 30.0),
    trial_count: int | None = None,
    roi_sizes: tuple[int, ...] = (11, 13, 15),
    outlier_threshold_px: float = 1.0,
) -> list[dict[str, float | int | str]]:
    """Evaluate centroid and Gaussian ROI choices using paired frames."""

    if not roi_sizes or any(size <= 0 or size % 2 == 0 for size in roi_sizes):
        raise ValueError("roi_sizes must contain positive odd integers")
    rows: list[dict[str, float | int | str]] = []
    for estimator in ("centroid", "gaussian"):
        for roi_size in roi_sizes:
            estimator_config = replace(
                config.estimator,
                centroid_roi_size_px=roi_size
                if estimator == "centroid"
                else config.estimator.centroid_roi_size_px,
                gaussian_roi_size_px=roi_size
                if estimator == "gaussian"
                else config.estimator.gaussian_roi_size_px,
            )
            scenario = replace(config, estimator=estimator_config)
            summaries = run_snr_sweep(
                scenario,
                snr_values=snr_values,
                trial_count=trial_count,
                outlier_threshold_px=outlier_threshold_px,
            )
            for summary in summaries:
                if summary.estimator != estimator:
                    continue
                row = dict(summary.__dict__)
                row.update({"estimator": estimator, "roi_size_px": roi_size})
                rows.append(row)
    return rows


def save_roi_results(
    output_dir: str | Path,
    config,
    rows: list[dict[str, float | int | str]],
    *,
    snr_values: tuple[float, ...],
    outlier_threshold_px: float,
) -> tuple[Path, Path]:
    """Save ROI sensitivity data for report tables and plotting."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {
        "configuration": config,
        "snr_values": snr_values,
        "outlier_threshold_px": outlier_threshold_px,
        "rows": rows,
    }
    json_path = destination / "roi_sensitivity.json"
    csv_path = destination / "roi_sensitivity.csv"
    save_json(json_path, metadata)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def plot_roi_sensitivity(
    rows: list[dict[str, float | int | str]],
    output_path: str | Path,
    *,
    show: bool = False,
) -> Path:
    """Plot radial RMSE and lock-loss rate against ROI size."""

    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    colors = {"centroid": "tab:blue", "gaussian": "tab:orange"}
    labels = {"centroid": "Windowed centroid", "gaussian": "Gaussian fit"}
    for estimator in ("centroid", "gaussian"):
        estimator_rows = [row for row in rows if row["estimator"] == estimator]
        snrs = sorted({float(row["requested_snr"]) for row in estimator_rows})
        for snr in snrs:
            selected = sorted(
                (row for row in estimator_rows if float(row["requested_snr"]) == snr),
                key=lambda row: int(row["roi_size_px"]),
            )
            sizes = [row["roi_size_px"] for row in selected]
            suffix = f"{labels[estimator]} SNR {snr:g}"
            axes[0].plot(
                sizes,
                [row["radial_rmse_px"] for row in selected],
                marker="o",
                color=colors[estimator],
                alpha=0.55 + 0.1 * snrs.index(snr),
                label=suffix,
            )
            axes[1].plot(
                sizes,
                [row["lock_failure_rate"] for row in selected],
                marker="o",
                color=colors[estimator],
                alpha=0.55 + 0.1 * snrs.index(snr),
                label=suffix,
            )
    axes[0].set_title("ROI sensitivity: radial RMSE")
    axes[0].set_ylabel("radial RMSE [pixel]")
    axes[1].set_title("ROI sensitivity: lock loss")
    axes[1].set_ylabel("lock-loss rate")
    for axis in axes:
        axis.set_xlabel("ROI size [pixel]")
        axis.set_xticks(sorted({int(row["roi_size_px"]) for row in rows}))
        axis.grid(True, alpha=0.3)
    axes[1].set_ylim(bottom=0.0)
    axes[1].legend(fontsize="small")
    figure.savefig(output, dpi=160)
    if show:
        plt.show()
    plt.close(figure)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("ideal", "sensor"), default="ideal")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--snr-values", type=float, nargs="+", default=(3.0, 10.0, 30.0))
    parser.add_argument("--roi-sizes", type=int, nargs="+", default=(11, 13, 15))
    parser.add_argument("--output-dir", default="results/roi_sensitivity")
    parser.add_argument("--figure", default="results/standalone/figures/roi_sensitivity.png")
    parser.add_argument("--outlier-threshold-px", type=float, default=1.0)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
    config = scenario_config(args.profile)
    trials = config.experiment.development_trials if args.trials is None else args.trials
    snr_values = tuple(float(value) for value in args.snr_values)
    rows = run_roi_sensitivity(
        config,
        snr_values=snr_values,
        trial_count=trials,
        roi_sizes=tuple(args.roi_sizes),
        outlier_threshold_px=args.outlier_threshold_px,
    )
    save_roi_results(
        args.output_dir,
        config,
        rows,
        snr_values=snr_values,
        outlier_threshold_px=args.outlier_threshold_px,
    )
    plot_roi_sensitivity(rows, args.figure, show=args.show)


if __name__ == "__main__":
    main()
