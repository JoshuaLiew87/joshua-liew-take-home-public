"""Regenerate the complete take-home study from one command.

This module intentionally contains orchestration only.  The scientific work
stays in the experiment modules so that each stage can also be run and tested
independently.
"""

from __future__ import annotations

import argparse
import importlib.metadata
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
    static_trials: int | None = None,
    roi_trials: int | None = None,
    runtime_frames: int = 200,
    runtime_warmup: int = 20,
    dynamic_frames: int | None = None,
    real_world_trials: int = 300,
    run_id: str | None = None,
    target_snr: float | None = None,
    profile_overrides: dict[str, dict[str, object]] | None = None,
    yaml_overrides: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Run every required experiment and save its final outputs.

    Each invocation receives a timestamped directory under
    ``results/runs/``.  ``results/latest/`` is refreshed as a stable snapshot
    of that run for report links.  Runtime measurements are expected to vary
    slightly with machine load, while seeded simulation products are
    deterministic.
    """

    root = Path(output_root).resolve()
    results_root = root / "results"
    run_id = _make_run_id(results_root, run_id)
    run_root = results_root / "runs" / run_id
    run_results_root = run_root / "results"
    run_figures_root = run_root / "figures"
    run_results_root.mkdir(parents=True, exist_ok=True)
    run_figures_root.mkdir(parents=True, exist_ok=True)

    profile_overrides = {} if profile_overrides is None else profile_overrides
    ideal = scenario_config("ideal", overrides=profile_overrides.get("ideal"))
    sensor = scenario_config("sensor", overrides=profile_overrides.get("sensor"))
    dynamic = scenario_config("dynamic", overrides=profile_overrides.get("dynamic"))
    static_count = ideal.experiment.monte_carlo_trials if static_trials is None else int(static_trials)
    roi_count = sensor.experiment.development_trials if roi_trials is None else int(roi_trials)
    dynamic_count = dynamic.trajectory.final_frame_count if dynamic_frames is None else int(dynamic_frames)
    run_snr = dynamic.experiment.target_snr if target_snr is None else float(target_snr)
    if static_count <= 0 or roi_count <= 0 or runtime_frames <= 0 or runtime_warmup < 0:
        raise ValueError("trial and runtime counts must be positive")
    if dynamic_count <= 0 or real_world_trials <= 0 or run_snr <= 0:
        raise ValueError("dynamic and real-world trial counts must be positive")

    generated: list[Path] = []

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

    manifest = {
        "workflow": "run_all",
        "workflow_version": 2,
        "run_id": run_id,
        "output_root": str(root),
        "run_directory": str(run_root.relative_to(root)),
        "latest_directory": "results/latest",
        "software": _software_versions(),
        "seed": sensor.experiment.master_seed,
        "parameters": {
            "static_trials": static_count,
            "roi_trials": roi_count,
            "runtime_frames": runtime_frames,
            "runtime_warmup": runtime_warmup,
            "dynamic_frames": dynamic_count,
            "real_world_trials": real_world_trials,
            "target_snr": run_snr,
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
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=".",
        help="repository or staging directory receiving results/runs/",
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
    parser.add_argument("--target-snr", type=float, default=None)
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
        "static_trials",
        "roi_trials",
        "runtime_frames",
        "runtime_warmup",
        "dynamic_frames",
        "real_world_trials",
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
            }
        )
    manifest = run_pipeline(
        args.output_root,
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
