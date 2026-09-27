"""Benchmark estimator latency on pre-generated sensor frames."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from common.io import save_json
from config import ScenarioConfig, scenario_config
from estimators.centroid import estimate_centroid
from estimators.gaussian_fit import estimate_gaussian
from sim.simulator import SpotSimulator
from sim.trajectory import TrajectoryGenerator
from common.rng import make_rng_streams


@dataclass(frozen=True)
class RuntimeSummary:
    """Latency and convergence statistics for one benchmark configuration."""

    profile: str
    estimator: str
    initialization: str
    sequence: str
    target_snr: float
    frame_count: int
    warmup_count: int
    median_ms: float
    p95_ms: float
    p99_ms: float
    worst_ms: float
    mean_ms: float
    success_count: int
    failure_count: int
    failure_rate: float
    deadline_miss_count: int
    deadline_miss_rate: float
    median_nfev: float
    p95_nfev: float


def generate_benchmark_frames(
    config: ScenarioConfig,
    *,
    target_snr: float,
    frame_count: int,
) -> list[np.ndarray]:
    """Generate frames before timing so simulator cost is excluded."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    simulator = SpotSimulator(config)
    position = (
        (config.sensor.width_px - 1) / 2.0 + 0.23,
        (config.sensor.height_px - 1) / 2.0 - 0.31,
    )
    return [
        simulator.render(position, target_snr=target_snr).image_dn.copy()
        for _ in range(frame_count)
    ]


def generate_tracking_benchmark_frames(
    config: ScenarioConfig,
    *,
    target_snr: float,
    frame_count: int,
) -> list[np.ndarray]:
    """Generate sequential frames from the configured dynamic trajectory."""

    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    streams = make_rng_streams(config.experiment.master_seed)
    simulator = SpotSimulator(config, rng_streams=streams)
    trajectory = TrajectoryGenerator(
        config.trajectory,
        sensor_shape=(config.sensor.height_px, config.sensor.width_px),
        position_margin_px=config.spot.position_margin_px,
        rng=streams.trajectory,
    ).generate(frame_count)
    return [
        simulator.render(tuple(position), target_snr=target_snr).image_dn.copy()
        for position in trajectory
    ]


def _run_timed_sequence(
    frames: list[np.ndarray],
    config: ScenarioConfig,
    *,
    estimator: str,
    warm_start: bool,
    sequence: str,
    warmup_count: int,
    target_snr: float,
) -> RuntimeSummary:
    if estimator not in {"centroid", "gaussian"}:
        raise ValueError("estimator must be centroid or gaussian")
    if warm_start and estimator != "gaussian":
        raise ValueError("warm_start is only supported for the Gaussian fit")

    def estimate(image: np.ndarray, previous: tuple[float, float] | None):
        if estimator == "centroid":
            return estimate_centroid(image, config)
        return estimate_gaussian(image, config, previous_position=previous)

    previous_position = None
    for index in range(min(warmup_count, len(frames))):
        result = estimate(frames[index], previous_position if warm_start else None)
        if warm_start and result.success and result.position_xy is not None:
            previous_position = result.position_xy

    timings_ms: list[float] = []
    nfev_values: list[float] = []
    success_count = 0
    deadline_ms = 1000.0 / config.sensor.frame_rate_hz
    deadline_miss_count = 0
    previous_position = None
    for frame in frames:
        start_ns = time.perf_counter_ns()
        result = estimate(frame, previous_position if warm_start else None)
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
        timings_ms.append(elapsed_ms)
        if result.success:
            success_count += 1
            if warm_start and result.position_xy is not None:
                previous_position = result.position_xy
        if elapsed_ms > deadline_ms:
            deadline_miss_count += 1
        nfev = result.diagnostics.get("nfev")
        if isinstance(nfev, (int, float)) and np.isfinite(nfev):
            nfev_values.append(float(nfev))

    timings = np.asarray(timings_ms, dtype=float)
    nfev_array = np.asarray(nfev_values, dtype=float)
    failure_count = len(frames) - success_count
    return RuntimeSummary(
        profile=config.name,
        estimator=estimator,
        initialization="warm" if warm_start else "cold",
        sequence=sequence,
        target_snr=float(target_snr),
        frame_count=len(frames),
        warmup_count=min(warmup_count, len(frames)),
        median_ms=float(np.median(timings)),
        p95_ms=float(np.percentile(timings, 95.0)),
        p99_ms=float(np.percentile(timings, 99.0)),
        worst_ms=float(np.max(timings)),
        mean_ms=float(np.mean(timings)),
        success_count=success_count,
        failure_count=failure_count,
        failure_rate=float(failure_count / len(frames)),
        deadline_miss_count=deadline_miss_count,
        deadline_miss_rate=float(deadline_miss_count / len(frames)),
        median_nfev=float(np.median(nfev_array)) if nfev_array.size else float("nan"),
        p95_nfev=float(np.percentile(nfev_array, 95.0)) if nfev_array.size else float("nan"),
    )


def run_runtime_benchmark(
    config: ScenarioConfig,
    *,
    target_snr: float | None = None,
    frame_count: int = 200,
    warmup_count: int = 20,
    sequence: str = "fixed",
) -> list[RuntimeSummary]:
    """Benchmark centroid and Gaussian latency on fixed or sequential frames."""

    config.validate()
    snr = config.experiment.target_snr if target_snr is None else float(target_snr)
    if snr <= 0 or not np.isfinite(snr):
        raise ValueError("target_snr must be finite and positive")
    if sequence == "fixed":
        frames = generate_benchmark_frames(config, target_snr=snr, frame_count=frame_count)
    elif sequence == "dynamic":
        frames = generate_tracking_benchmark_frames(config, target_snr=snr, frame_count=frame_count)
    else:
        raise ValueError("sequence must be fixed or dynamic")
    summaries = [
        _run_timed_sequence(
            frames,
            config,
            estimator="centroid",
            warm_start=False,
            warmup_count=warmup_count,
            target_snr=snr,
            sequence=sequence,
        ),
        _run_timed_sequence(
            frames,
            config,
            estimator="gaussian",
            warm_start=False,
            warmup_count=warmup_count,
            target_snr=snr,
            sequence=sequence,
        ),
        _run_timed_sequence(
            frames,
            config,
            estimator="gaussian",
            warm_start=True,
            warmup_count=warmup_count,
            target_snr=snr,
            sequence=sequence,
        ),
    ]
    return summaries


def _software_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": importlib.metadata.version("numpy"),
        "scipy": importlib.metadata.version("scipy"),
    }


def save_runtime_results(
    output_dir: str | Path,
    config: ScenarioConfig,
    summaries: list[RuntimeSummary],
) -> tuple[Path, Path]:
    """Save benchmark summaries and software metadata."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = [asdict(summary) for summary in summaries]
    metadata = {
        "configuration": config,
        "frame_interval_ms": 1000.0 / config.sensor.frame_rate_hz,
        "software": _software_versions(),
        "summaries": rows,
    }
    json_path = destination / "runtime_benchmark.json"
    csv_path = destination / "runtime_benchmark.csv"
    save_json(json_path, metadata)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def plot_runtime_results(
    summaries: list[RuntimeSummary],
    output_path: str | Path,
    *,
    show: bool = False,
) -> Path:
    """Plot median and tail latency against the 1 ms frame interval."""

    import matplotlib.pyplot as plt

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = [
        f"{item.estimator}\n{item.initialization}"
        for item in summaries
    ]
    median = np.asarray([item.median_ms for item in summaries])
    p99 = np.asarray([item.p99_ms for item in summaries])
    worst = np.asarray([item.worst_ms for item in summaries])
    x = np.arange(len(summaries))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].bar(x, median, color="tab:blue", label="median")
    axes[0].scatter(x, p99, color="tab:orange", marker="o", label="p99")
    axes[0].scatter(x, worst, color="tab:red", marker="x", label="worst")
    axes[0].axhline(1.0, color="black", linestyle="--", label="1 ms budget")
    axes[0].set_ylabel("latency [ms]")
    axes[0].set_title("Estimator latency")
    axes[0].set_xticks(x, labels)
    axes[0].grid(True, axis="y", alpha=0.3)
    axes[0].legend(fontsize="small")

    axes[1].bar(x, [item.deadline_miss_rate for item in summaries], color="tab:purple")
    axes[1].set_ylabel("fraction exceeding 1 ms")
    axes[1].set_title("Frame-budget misses")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(True, axis="y", alpha=0.3)
    figure.savefig(output, dpi=160)
    if show:
        plt.show()
    plt.close(figure)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("ideal", "sensor", "dynamic"), default="sensor")
    parser.add_argument("--config", default=None, help="optional YAML profile override file")
    parser.add_argument("--snr", type=float, default=10.0)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--sequence", choices=("fixed", "dynamic"), default="fixed")
    parser.add_argument("--output-dir", default="results/runtime_sensor")
    parser.add_argument("--figure", default="results/standalone/figures/runtime_sensor.png")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
    profile_overrides = None
    if args.config is not None:
        from config import load_yaml_overrides

        yaml_document = load_yaml_overrides(args.config)
        profile_overrides = {
            profile: dict(overrides)
            for profile, overrides in yaml_document["profiles"].items()
        }
        if "master_seed" in yaml_document["pipeline"]:
            profile_override = dict(profile_overrides.get(args.profile, {}))
            experiment_override = dict(profile_override.get("experiment", {}))
            experiment_override["master_seed"] = int(yaml_document["pipeline"]["master_seed"])
            profile_override["experiment"] = experiment_override
            profile_overrides[args.profile] = profile_override
    config = scenario_config(args.profile, overrides=profile_overrides.get(args.profile) if profile_overrides else None)
    summaries = run_runtime_benchmark(
        config,
        target_snr=args.snr,
        frame_count=args.frames,
        warmup_count=args.warmup,
        sequence=args.sequence,
    )
    save_runtime_results(args.output_dir, config, summaries)
    if not args.no_plot:
        plot_runtime_results(summaries, args.figure, show=args.show)


if __name__ == "__main__":
    main()
