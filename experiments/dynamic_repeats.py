"""Small multi-seed repeat for disturbance-amplitude uncertainty."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from common.io import save_json
from config import ScenarioConfig, scenario_config
from experiments.dynamic_tracking import run_dynamic_experiment


@dataclass(frozen=True)
class RepeatMetric:
    seed: int
    estimator: str
    frame_count: int
    target_snr: float
    true_amplitude_px: float
    estimated_amplitude_px: float
    amplitude_bias_px: float
    estimated_frequency_hz: float
    spectral_noise_floor_px: float
    peak_to_noise_ratio: float


@dataclass(frozen=True)
class RepeatAggregate:
    estimator: str
    repeat_count: int
    frame_count: int
    target_snr: float
    true_amplitude_px: float
    mean_estimated_amplitude_px: float
    std_estimated_amplitude_px: float
    amplitude_bias_px: float
    amplitude_bias_stderr_px: float
    mean_estimated_frequency_hz: float
    std_estimated_frequency_hz: float
    mean_spectral_noise_floor_px: float
    mean_peak_to_noise_ratio: float


def run_dynamic_repeats(
    config: ScenarioConfig,
    *,
    repeat_count: int = 5,
    frame_count: int = 2500,
    target_snr: float | None = None,
    seed_stride: int = 1009,
) -> tuple[list[RepeatMetric], list[RepeatAggregate]]:
    """Repeat dynamic recovery with independent master seeds."""

    config.validate()
    if repeat_count <= 0 or frame_count <= 0 or seed_stride <= 0:
        raise ValueError("repeat count, frame count, and seed stride must be positive")
    snr = config.experiment.target_snr if target_snr is None else float(target_snr)
    if snr <= 0 or not np.isfinite(snr):
        raise ValueError("target_snr must be finite and positive")

    rows: list[RepeatMetric] = []
    for repeat_index in range(repeat_count):
        seed = int(config.experiment.master_seed + repeat_index * seed_stride)
        repeat_config = replace(
            config,
            experiment=replace(config.experiment, master_seed=seed),
        )
        result = run_dynamic_experiment(
            repeat_config,
            target_snr=snr,
            frame_count=frame_count,
        )
        for summary in result.summaries:
            rows.append(
                RepeatMetric(
                    seed=seed,
                    estimator=summary.estimator,
                    frame_count=frame_count,
                    target_snr=snr,
                    true_amplitude_px=summary.true_amplitude_px,
                    estimated_amplitude_px=summary.estimated_amplitude_px,
                    amplitude_bias_px=summary.estimated_amplitude_px - summary.true_amplitude_px,
                    estimated_frequency_hz=summary.estimated_frequency_hz,
                    spectral_noise_floor_px=summary.spectral_noise_floor_px,
                    peak_to_noise_ratio=summary.peak_to_noise_ratio,
                )
            )

    aggregates: list[RepeatAggregate] = []
    for estimator in ("centroid", "gaussian"):
        selected = [row for row in rows if row.estimator == estimator]
        amplitudes = np.asarray([row.estimated_amplitude_px for row in selected], dtype=float)
        frequencies = np.asarray([row.estimated_frequency_hz for row in selected], dtype=float)
        floors = np.asarray([row.spectral_noise_floor_px for row in selected], dtype=float)
        ratios = np.asarray([row.peak_to_noise_ratio for row in selected], dtype=float)
        true_amplitude = selected[0].true_amplitude_px
        aggregates.append(
            RepeatAggregate(
                estimator=estimator,
                repeat_count=len(selected),
                frame_count=frame_count,
                target_snr=snr,
                true_amplitude_px=true_amplitude,
                mean_estimated_amplitude_px=float(np.mean(amplitudes)),
                std_estimated_amplitude_px=float(np.std(amplitudes, ddof=1)) if len(amplitudes) > 1 else 0.0,
                amplitude_bias_px=float(np.mean(amplitudes) - true_amplitude),
                amplitude_bias_stderr_px=float(np.std(amplitudes, ddof=1) / np.sqrt(len(amplitudes))) if len(amplitudes) > 1 else 0.0,
                mean_estimated_frequency_hz=float(np.mean(frequencies)),
                std_estimated_frequency_hz=float(np.std(frequencies, ddof=1)) if len(frequencies) > 1 else 0.0,
                mean_spectral_noise_floor_px=float(np.mean(floors)),
                mean_peak_to_noise_ratio=float(np.mean(ratios)),
            )
        )
    return rows, aggregates


def save_repeat_results(
    output_dir: str | Path,
    config: ScenarioConfig,
    rows: list[RepeatMetric],
    aggregates: list[RepeatAggregate],
) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "dynamic_amplitude_repeats.json"
    csv_path = destination / "dynamic_amplitude_repeats.csv"
    save_json(
        json_path,
        {
            "configuration": config,
            "repeats": [asdict(row) for row in rows],
            "aggregates": [asdict(row) for row in aggregates],
        },
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(asdict(rows[0])),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return json_path, csv_path


def plot_repeat_results(aggregates: list[RepeatAggregate], output_path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = {"centroid": "Windowed centroid", "gaussian": "Gaussian fit"}
    colors = {"centroid": "tab:blue", "gaussian": "tab:orange"}
    figure, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    x = np.arange(len(aggregates))
    means = [row.mean_estimated_amplitude_px for row in aggregates]
    errors = [row.std_estimated_amplitude_px for row in aggregates]
    axis.errorbar(x, means, yerr=errors, fmt="o", capsize=4, color="0.25")
    for index, row in enumerate(aggregates):
        axis.scatter(index, row.mean_estimated_amplitude_px, color=colors[row.estimator], label=labels[row.estimator], zorder=3)
        axis.annotate(
            f"bias={row.amplitude_bias_px:+.4f} px",
            (index, row.mean_estimated_amplitude_px),
            xytext=(0, 12), textcoords="offset points", ha="center", fontsize="small",
        )
    true_amplitude = aggregates[0].true_amplitude_px
    axis.axhline(true_amplitude, color="black", linestyle="--", label="injected amplitude")
    axis.set_xticks(x, [labels[row.estimator] for row in aggregates])
    axis.set_ylabel("recovered disturbance amplitude [pixel]")
    axis.set_title("Multi-seed amplitude-recovery repeat")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--frames", type=int, default=2500)
    parser.add_argument("--snr", type=float, default=10.0)
    parser.add_argument("--output-dir", default="results/dynamic_repeats")
    parser.add_argument("--figure", default="figures/dynamic_amplitude_repeats.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import matplotlib

    matplotlib.use("Agg")
    config = scenario_config("dynamic")
    rows, aggregates = run_dynamic_repeats(
        config,
        repeat_count=args.repeats,
        frame_count=args.frames,
        target_snr=args.snr,
    )
    save_repeat_results(args.output_dir, config, rows, aggregates)
    plot_repeat_results(aggregates, args.figure)


if __name__ == "__main__":
    main()
