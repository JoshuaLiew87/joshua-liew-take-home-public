"""Regenerate the complete take-home study from one command.

This module intentionally contains orchestration only.  The scientific work
stays in the experiment modules so that each stage can also be run and tested
independently.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import math
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

from common.io import save_json
from config import load_yaml_overrides, scenario_config
from experiments.benchmark_runtime import (
    plot_runtime_results,
    run_runtime_benchmark,
    save_runtime_results,
)
from experiments.characterize_snr import (
    plot_sweep,
    run_snr_sweep,
    save_sweep_results,
)
from experiments.dynamic_tracking import (
    plot_dynamic_results,
    run_dynamic_experiment,
    save_dynamic_results,
)
from experiments.dynamic_repeats import (
    plot_repeat_results,
    run_dynamic_repeats,
    save_repeat_results,
)
from experiments.estimator_ablation import (
    plot_ablation,
    run_ablation,
    save_ablation_results,
)
from experiments.real_world_conditions import (
    plot_condition_results,
    run_condition_analysis,
    save_condition_results,
)
from experiments.roi_sensitivity import (
    plot_roi_sensitivity,
    run_roi_sensitivity,
    save_roi_results,
)
from experiments.theory import compute_position_crlb, save_theory_results


def _software_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for package in ("numpy", "scipy", "matplotlib"):
        versions[package] = importlib.metadata.version(package)
    return versions


def _relative_paths(root: Path, paths: list[Path]) -> list[str]:
    return [str(path.relative_to(root)) for path in paths]


def _print_stage(stage: int, total: int, description: str) -> None:
    """Print a compact, flush-on-write pipeline progress message."""

    print(f"[{stage:02d}/{total:02d}] {description}", flush=True)


def _print_check(name: str, passed: bool, detail: str) -> None:
    """Print a human-readable configuration or output sanity check."""

    status = "PASS" if passed else "WARN"
    print(f"    {status:<4} {name}: {detail}", flush=True)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _find_csv_row(path: Path, **criteria: str) -> dict[str, str] | None:
    for row in _read_csv_rows(path):
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    return None


def _format_csv_value(row: dict[str, str] | None, key: str, digits: int = 3) -> str:
    if row is None or key not in row:
        return "n/a"
    try:
        return f"{float(row[key]):.{digits}f}"
    except (TypeError, ValueError):
        return row[key]


def _format_rate(row: dict[str, str] | None, key: str) -> str:
    if row is None or key not in row:
        return "n/a"
    return f"{100 * float(row[key]):.1f}%"


def _print_configuration_summary(
    ideal: Any,
    sensor: Any,
    dynamic: Any,
    *,
    static_count: int,
    roi_count: int,
    runtime_frames: int,
    runtime_warmup: int,
    dynamic_count: int,
    real_world_trials: int,
    ablation_trials: int,
    dynamic_repeat_count: int,
    dynamic_repeat_frames: int,
    target_snr: float,
) -> None:
    """Print the resolved controls and checks before expensive work starts."""

    print("run_all: resolved configuration", flush=True)
    print(
        f"  sensor={sensor.sensor.width_px}x{sensor.sensor.height_px}, "
        f"ADC={sensor.sensor.bit_depth}-bit, "
        f"frame_rate={sensor.sensor.frame_rate_hz:g} Hz, "
        f"exposure={sensor.sensor.exposure_s:g} s",
        flush=True,
    )
    print(
        f"  SNR={target_snr:g}, seed={sensor.experiment.master_seed}, "
        f"static_trials={static_count}, roi_trials={roi_count}",
        flush=True,
    )
    print(
        f"  runtime_frames={runtime_frames}, warmup={runtime_warmup}, "
        f"dynamic_frames={dynamic_count}, outdoor_trials={real_world_trials}",
        flush=True,
    )
    print(
        f"  ablation_trials={ablation_trials}, "
        f"dynamic_repeats={dynamic_repeat_count} x {dynamic_repeat_frames} frames",
        flush=True,
    )
    print("run_all: configuration sanity checks", flush=True)

    shared_seed = (
        ideal.experiment.master_seed
        == sensor.experiment.master_seed
        == dynamic.experiment.master_seed
    )
    _print_check(
        "shared master seed",
        shared_seed,
        str(sensor.experiment.master_seed),
    )
    _print_check(
        "estimator ROIs",
        sensor.estimator.centroid_roi_size_px % 2 == 1
        and sensor.estimator.gaussian_roi_size_px % 2 == 1
        and sensor.estimator.gaussian_roi_size_px <= min(
            sensor.sensor.width_px, sensor.sensor.height_px
        ),
        f"centroid={sensor.estimator.centroid_roi_size_px} px, "
        f"Gaussian={sensor.estimator.gaussian_roi_size_px} px",
    )
    one_frame_period = math.isclose(
        sensor.sensor.exposure_s * sensor.sensor.frame_rate_hz,
        1.0,
        rel_tol=1e-6,
        abs_tol=1e-9,
    )
    _print_check(
        "nominal frame timing",
        one_frame_period,
        f"exposure x frame rate = "
        f"{sensor.sensor.exposure_s * sensor.sensor.frame_rate_hz:g}",
    )
    below_nyquist = dynamic.trajectory.disturbance_frequency_hz < (
        dynamic.trajectory.frame_rate_hz / 2.0
    )
    _print_check(
        "periodic disturbance below Nyquist",
        below_nyquist,
        f"{dynamic.trajectory.disturbance_frequency_hz:g} Hz < "
        f"{dynamic.trajectory.frame_rate_hz / 2.0:g} Hz",
    )
    _print_check(
        "validated scenario profiles",
        True,
        "ideal, sensor, and dynamic profiles passed validation",
    )


def _print_result_summary(run_root: Path, manifest: dict[str, Any]) -> None:
    """Print key saved results and output-integrity checks after completion."""

    missing = [
        run_root / relative_path
        for relative_path in manifest["generated_files"]
        if not (run_root / relative_path).is_file()
    ]
    _print_check(
        "generated outputs",
        not missing,
        f"{len(manifest['generated_files'])} files recorded in the manifest"
        if not missing
        else f"missing {len(missing)} recorded files",
    )
    _print_check(
        "run manifest",
        (run_root / "run_manifest.json").is_file(),
        str(run_root / "run_manifest.json"),
    )

    sensor_csv = run_root / "results" / "final_sensor" / "static_characterization.csv"
    runtime_csv = run_root / "results" / "runtime_sensor" / "runtime_benchmark.csv"
    dynamic_csv = run_root / "results" / "dynamic" / "dynamic_summary.csv"
    snr_one_centroid = _find_csv_row(
        sensor_csv, estimator="centroid", requested_snr="1.0"
    )
    snr_one_gaussian = _find_csv_row(
        sensor_csv, estimator="gaussian", requested_snr="1.0"
    )
    snr_ten_centroid = _find_csv_row(
        sensor_csv, estimator="centroid", requested_snr="10.0"
    )
    snr_ten_gaussian = _find_csv_row(
        sensor_csv, estimator="gaussian", requested_snr="10.0"
    )
    centroid_runtime = _find_csv_row(
        runtime_csv, estimator="centroid", initialization="cold"
    )
    gaussian_runtime = _find_csv_row(
        runtime_csv, estimator="gaussian", initialization="warm"
    )
    dynamic_centroid = _find_csv_row(dynamic_csv, estimator="centroid")
    dynamic_gaussian = _find_csv_row(dynamic_csv, estimator="gaussian")

    print("run_all: key results", flush=True)
    print(
        "  stressed SNR=10 RMSE: "
        f"centroid={_format_csv_value(snr_ten_centroid, 'radial_rmse_px')} px, "
        f"Gaussian={_format_csv_value(snr_ten_gaussian, 'radial_rmse_px')} px",
        flush=True,
    )
    print(
        "  stressed SNR=1 lock loss: "
        f"centroid={_format_rate(snr_one_centroid, 'lock_failure_rate')}, "
        f"Gaussian={_format_rate(snr_one_gaussian, 'lock_failure_rate')}",
        flush=True,
    )
    print(
        "  runtime median: "
        f"centroid={_format_csv_value(centroid_runtime, 'median_ms')} ms, "
        f"Gaussian warm={_format_csv_value(gaussian_runtime, 'median_ms')} ms",
        flush=True,
    )
    if gaussian_runtime is not None:
        gaussian_p99 = float(gaussian_runtime["p99_ms"])
        if gaussian_p99 > 1.0:
            print(
                f"  WARNING: Gaussian warm p99={gaussian_p99:.3f} ms exceeds "
                "the nominal 1 ms estimator budget",
                flush=True,
            )
    print(
        "  dynamic RMS error: "
        f"centroid={_format_csv_value(dynamic_centroid, 'rms_error_px', 4)} px, "
        f"Gaussian={_format_csv_value(dynamic_gaussian, 'rms_error_px', 4)} px",
        flush=True,
    )
    print(
        "  disturbance frequency: "
        f"centroid={_format_csv_value(dynamic_centroid, 'estimated_frequency_hz', 4)} Hz, "
        f"Gaussian={_format_csv_value(dynamic_gaussian, 'estimated_frequency_hz', 4)} Hz",
        flush=True,
    )
    print(
        f"  results: {run_root}"
        f"\n  latest: {run_root.parent.parent / 'latest'}"
        f"\n  manifest: {run_root / 'run_manifest.json'}",
        flush=True,
    )


def _make_run_id(results_root: Path, requested_run_id: str | None) -> str:
    """Return a safe, non-colliding UTC run-directory name."""

    if requested_run_id is None:
        base = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    else:
        base = requested_run_id
    if not base or Path(base).name != base or base in {".", ".."}:
        raise ValueError("run_id must be a non-empty directory name")
    candidate = base
    suffix = 1
    while (results_root / "runs" / candidate).exists():
        candidate = f"{base}_{suffix:02d}"
        suffix += 1
    return candidate


def run_pipeline(
    output_root: str | Path = ".",
    *,
    results_dir: str | Path = "results",
    static_trials: int | None = None,
    roi_trials: int | None = None,
    runtime_frames: int = 200,
    runtime_warmup: int = 20,
    dynamic_frames: int | None = None,
    real_world_trials: int = 300,
    ablation_trials: int = 0,
    dynamic_repeat_count: int = 0,
    dynamic_repeat_frames: int = 2500,
    run_id: str | None = None,
    target_snr: float | None = None,
    master_seed: int | None = None,
    profile_overrides: dict[str, dict[str, object]] | None = None,
    yaml_overrides: dict[str, object] | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run every required experiment and save its final outputs.

    Each invocation receives a timestamped directory under
    ``results_dir/runs/``.  ``results_dir/latest/`` is refreshed as a stable
    snapshot of that run for report links. Runtime measurements are expected
    to vary slightly with machine load, while seeded simulation products are
    deterministic.
    """

    root = Path(output_root).resolve()
    results_path = Path(results_dir)
    if results_path.is_absolute() or not results_path.parts or ".." in results_path.parts:
        raise ValueError("results_dir must be a non-empty relative directory")
    results_root = root / results_path
    results_label = results_path.as_posix()
    run_id = _make_run_id(results_root, run_id)
    run_root = results_root / "runs" / run_id
    run_results_root = run_root / "results"
    run_figures_root = run_root / "figures"
    run_results_root.mkdir(parents=True, exist_ok=True)
    run_figures_root.mkdir(parents=True, exist_ok=True)

    profile_overrides = {} if profile_overrides is None else profile_overrides
    if master_seed is not None:
        resolved_overrides = {
            profile: dict(overrides) for profile, overrides in profile_overrides.items()
        }
        for profile in ("ideal", "sensor", "dynamic"):
            profile_override = dict(resolved_overrides.get(profile, {}))
            experiment_override = dict(profile_override.get("experiment", {}))
            experiment_override["master_seed"] = int(master_seed)
            profile_override["experiment"] = experiment_override
            resolved_overrides[profile] = profile_override
        profile_overrides = resolved_overrides
    ideal = scenario_config("ideal", overrides=profile_overrides.get("ideal"))
    sensor = scenario_config("sensor", overrides=profile_overrides.get("sensor"))
    dynamic = scenario_config("dynamic", overrides=profile_overrides.get("dynamic"))
    static_count = ideal.experiment.monte_carlo_trials if static_trials is None else int(static_trials)
    roi_count = sensor.experiment.development_trials if roi_trials is None else int(roi_trials)
    dynamic_count = dynamic.trajectory.final_frame_count if dynamic_frames is None else int(dynamic_frames)
    run_snr = dynamic.experiment.target_snr if target_snr is None else float(target_snr)
    if (
        static_count <= 0
        or roi_count <= 0
        or runtime_frames <= 0
        or runtime_warmup < 0
        or ablation_trials < 0
        or dynamic_repeat_count < 0
        or dynamic_repeat_frames <= 0
    ):
        raise ValueError("trial and runtime counts must be positive")
    if dynamic_count <= 0 or real_world_trials <= 0 or run_snr <= 0:
        raise ValueError("dynamic and real-world trial counts must be positive")

    if verbose:
        print(f"run_all: starting run {run_id}", flush=True)
        print(f"  output root: {root}", flush=True)
        _print_configuration_summary(
            ideal,
            sensor,
            dynamic,
            static_count=static_count,
            roi_count=roi_count,
            runtime_frames=runtime_frames,
            runtime_warmup=runtime_warmup,
            dynamic_count=dynamic_count,
            real_world_trials=real_world_trials,
            ablation_trials=ablation_trials,
            dynamic_repeat_count=dynamic_repeat_count,
            dynamic_repeat_frames=dynamic_repeat_frames,
            target_snr=run_snr,
        )

    total_stages = 7 + int(ablation_trials > 0) + int(dynamic_repeat_count > 0)
    current_stage = 0

    def advance_stage(description: str) -> None:
        nonlocal current_stage
        current_stage += 1
        if verbose:
            _print_stage(current_stage, total_stages, description)

    generated: list[Path] = []

    advance_stage("ideal SNR characterization and Fisher reference")
    ideal_summaries = run_snr_sweep(ideal, trial_count=static_count)
    ideal_dir = run_results_root / "final_ideal"
    save_sweep_results(
        ideal_dir,
        ideal,
        ideal_summaries,
        profile="ideal",
        outlier_threshold_px=1.0,
    )
    ideal_theory = compute_position_crlb(ideal)
    save_theory_results(ideal_dir, ideal, ideal_theory)
    plot_sweep(
        ideal_summaries,
        run_figures_root / "final_ideal_characterization.png",
        theory=ideal_theory,
    )
    generated.extend(
        [
            ideal_dir / "static_characterization.json",
            ideal_dir / "static_characterization.csv",
            ideal_dir / "theoretical_precision.json",
            ideal_dir / "theoretical_precision.csv",
            run_figures_root / "final_ideal_characterization.png",
        ]
    )

    if ablation_trials > 0:
        advance_stage("controlled estimator ablation")
        ablation_metrics = run_ablation(
            sensor,
            snr_values=(1.0, 3.0),
            trial_count=ablation_trials,
            outlier_threshold_px=1.0,
        )
        ablation_dir = run_results_root / "estimator_ablation"
        save_ablation_results(
            ablation_dir,
            sensor,
            ablation_metrics,
            snr_values=(1.0, 3.0),
            trial_count=ablation_trials,
            outlier_threshold_px=1.0,
        )
        plot_ablation(ablation_metrics, run_figures_root / "estimator_ablation.png")
        generated.extend(
            [
                ablation_dir / "estimator_ablation.json",
                ablation_dir / "estimator_ablation.csv",
                run_figures_root / "estimator_ablation.png",
            ]
        )

    advance_stage("stressed-sensor SNR characterization")
    sensor_summaries = run_snr_sweep(sensor, trial_count=static_count)
    sensor_dir = run_results_root / "final_sensor"
    save_sweep_results(
        sensor_dir,
        sensor,
        sensor_summaries,
        profile="sensor",
        outlier_threshold_px=1.0,
    )
    plot_sweep(
        sensor_summaries,
        run_figures_root / "final_sensor_characterization.png",
    )
    generated.extend(
        [
            sensor_dir / "static_characterization.json",
            sensor_dir / "static_characterization.csv",
            run_figures_root / "final_sensor_characterization.png",
        ]
    )

    if dynamic_repeat_count > 0:
        advance_stage("multi-seed dynamic amplitude repeat")
        repeat_rows, repeat_aggregates = run_dynamic_repeats(
            dynamic,
            repeat_count=dynamic_repeat_count,
            frame_count=dynamic_repeat_frames,
            target_snr=run_snr,
        )
        repeat_dir = run_results_root / "dynamic_repeats"
        save_repeat_results(repeat_dir, dynamic, repeat_rows, repeat_aggregates)
        plot_repeat_results(repeat_aggregates, run_figures_root / "dynamic_amplitude_repeats.png")
        generated.extend(
            [
                repeat_dir / "dynamic_amplitude_repeats.json",
                repeat_dir / "dynamic_amplitude_repeats.csv",
                run_figures_root / "dynamic_amplitude_repeats.png",
            ]
        )

    advance_stage("ROI sensitivity study")
    roi_rows = run_roi_sensitivity(sensor, trial_count=roi_count)
    roi_dir = run_results_root / "final_roi_sensitivity"
    save_roi_results(
        roi_dir,
        sensor,
        roi_rows,
        snr_values=(3.0, 10.0, 30.0),
        outlier_threshold_px=1.0,
    )
    plot_roi_sensitivity(roi_rows, run_figures_root / "final_roi_sensitivity.png")
    generated.extend(
        [
            roi_dir / "roi_sensitivity.json",
            roi_dir / "roi_sensitivity.csv",
            run_figures_root / "final_roi_sensitivity.png",
        ]
    )

    advance_stage("estimator runtime benchmarks")
    for profile, config in (("ideal", ideal), ("sensor", sensor)):
        runtime_summaries = run_runtime_benchmark(
            config,
            target_snr=run_snr,
            frame_count=runtime_frames,
            warmup_count=runtime_warmup,
        )
        runtime_dir = run_results_root / f"runtime_{profile}"
        save_runtime_results(runtime_dir, config, runtime_summaries)
        plot_runtime_results(
            runtime_summaries,
            run_figures_root / f"runtime_{profile}.png",
        )
        generated.extend(
            [
                runtime_dir / "runtime_benchmark.json",
                runtime_dir / "runtime_benchmark.csv",
                run_figures_root / f"runtime_{profile}.png",
            ]
        )

    advance_stage("dynamic trajectory and disturbance recovery")
    dynamic_result = run_dynamic_experiment(
        dynamic,
        target_snr=run_snr,
        frame_count=dynamic_count,
        lock_threshold_px=1.0,
    )
    dynamic_dir = run_results_root / "dynamic"
    save_dynamic_results(dynamic_dir, dynamic, dynamic_result, lock_threshold_px=1.0)
    plot_dynamic_results(dynamic, dynamic_result, run_figures_root / "dynamic_tracking.png")
    generated.extend(
        [
            dynamic_dir / "dynamic_trajectory.csv",
            dynamic_dir / "dynamic_summary.json",
            dynamic_dir / "dynamic_summary.csv",
            run_figures_root / "dynamic_tracking.png",
        ]
    )

    advance_stage("controlled outdoor-condition stress analysis")
    real_world_metrics = run_condition_analysis(
        sensor,
        trial_count=real_world_trials,
        target_snr=run_snr,
    )
    real_world_dir = run_results_root / "real_world"
    save_condition_results(
        real_world_dir,
        sensor,
        real_world_metrics,
        trial_count=real_world_trials,
        target_snr=run_snr,
        outlier_threshold_px=1.0,
    )
    plot_condition_results(
        real_world_metrics,
        run_figures_root / "real_world_conditions.png",
    )
    generated.extend(
        [
            real_world_dir / "real_world_conditions.json",
            real_world_dir / "real_world_conditions.csv",
            run_figures_root / "real_world_conditions.png",
        ]
    )

    advance_stage("write manifest and refresh latest snapshot")
    manifest = {
        "workflow": "run_all",
        "workflow_version": 2,
        "run_id": run_id,
        "output_root": str(root),
        "run_directory": str(run_root.relative_to(root)),
        "latest_directory": f"{results_label}/latest",
        "software": _software_versions(),
        "seed": sensor.experiment.master_seed,
        "parameters": {
            "static_trials": static_count,
            "roi_trials": roi_count,
            "runtime_frames": runtime_frames,
            "runtime_warmup": runtime_warmup,
            "dynamic_frames": dynamic_count,
            "real_world_trials": real_world_trials,
            "ablation_trials": ablation_trials,
            "dynamic_repeat_count": dynamic_repeat_count,
            "dynamic_repeat_frames": dynamic_repeat_frames,
            "target_snr": run_snr,
            "master_seed": sensor.experiment.master_seed,
            "lock_threshold_px": 1.0,
        },
        "generated_files": _relative_paths(run_root, generated),
        "yaml_overrides": {} if yaml_overrides is None else yaml_overrides,
        "runtime_measurements_are_machine_load_dependent": True,
    }
    save_json(run_root / "run_manifest.json", manifest)

    latest_dir = results_root / "latest"
    if latest_dir.is_symlink() or latest_dir.is_file():
        latest_dir.unlink()
    elif latest_dir.exists():
        shutil.rmtree(latest_dir)
    shutil.copytree(run_root, latest_dir)
    if verbose:
        print("run_all: post-run sanity checks", flush=True)
        _print_result_summary(run_root, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=".",
        help="repository or staging directory receiving results/runs/",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="relative results directory below the output root (default: results)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="optional run-directory name; defaults to the current UTC timestamp",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="optional YAML file containing pipeline and profile overrides",
    )
    parser.add_argument("--static-trials", type=int, default=None)
    parser.add_argument("--roi-trials", type=int, default=None)
    parser.add_argument("--runtime-frames", type=int, default=None)
    parser.add_argument("--runtime-warmup", type=int, default=None)
    parser.add_argument("--dynamic-frames", type=int, default=None)
    parser.add_argument("--real-world-trials", type=int, default=None)
    parser.add_argument("--ablation-trials", type=int, default=None)
    parser.add_argument("--dynamic-repeat-count", type=int, default=None)
    parser.add_argument("--dynamic-repeat-frames", type=int, default=None)
    parser.add_argument("--target-snr", type=float, default=None)
    parser.add_argument("--master-seed", type=int, default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use a small deterministic smoke configuration",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    yaml_document: dict[str, object] = {"pipeline": {}, "profiles": {}}
    if args.config is not None:
        yaml_document = load_yaml_overrides(args.config)
    pipeline_overrides = dict(yaml_document["pipeline"])
    allowed_pipeline_keys = {
        "master_seed",
        "static_trials",
        "roi_trials",
        "runtime_frames",
        "runtime_warmup",
        "dynamic_frames",
        "real_world_trials",
        "ablation_trials",
        "dynamic_repeat_count",
        "dynamic_repeat_frames",
        "target_snr",
    }
    unknown_pipeline_keys = set(pipeline_overrides) - allowed_pipeline_keys
    if unknown_pipeline_keys:
        raise ValueError(
            f"unknown YAML pipeline keys: {sorted(unknown_pipeline_keys)}; "
            f"expected one of {sorted(allowed_pipeline_keys)}"
        )
    pipeline_args: dict[str, object] = {}
    for key in allowed_pipeline_keys:
        cli_value = getattr(args, key)
        if cli_value is not None:
            pipeline_args[key] = cli_value
        elif key in pipeline_overrides:
            pipeline_args[key] = pipeline_overrides[key]
    if args.quick:
        pipeline_args.update(
            {
                "static_trials": 20,
                "roi_trials": 20,
                "runtime_frames": 20,
                "runtime_warmup": 5,
                "dynamic_frames": 500,
                "real_world_trials": 20,
                "ablation_trials": 0,
                "dynamic_repeat_count": 0,
            }
        )
    manifest = run_pipeline(
        args.output_root,
        results_dir=args.results_dir,
        run_id=args.run_id,
        profile_overrides=yaml_document["profiles"],
        yaml_overrides=yaml_document if args.config is not None else None,
        **pipeline_args,
    )
    print(
        f"Generated {len(manifest['generated_files'])} result/figure files in "
        f"{manifest['run_directory']}; updated {manifest['latest_directory']}"
    )


if __name__ == "__main__":
    main()
