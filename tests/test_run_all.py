from __future__ import annotations

from pathlib import Path

from run_all import run_pipeline


def test_run_pipeline_writes_declared_outputs(tmp_path: Path) -> None:
    manifest = run_pipeline(
        tmp_path,
        static_trials=4,
        roi_trials=4,
        runtime_frames=5,
        runtime_warmup=1,
        dynamic_frames=200,
        real_world_trials=4,
        run_id="test-run",
        master_seed=12345,
    )

    assert manifest["workflow"] == "run_all"
    assert manifest["parameters"]["static_trials"] == 4
    assert manifest["parameters"]["dynamic_frames"] == 200
    assert manifest["seed"] == 12345
    assert manifest["parameters"]["master_seed"] == 12345
    run_root = tmp_path / "results" / "runs" / "test-run"
    assert (run_root / "run_manifest.json").is_file()
    assert (tmp_path / "results" / "latest" / "run_manifest.json").is_file()
    assert manifest["generated_files"]
    assert all((run_root / relative_path).is_file() for relative_path in manifest["generated_files"])
    assert all(
        (tmp_path / "results" / "latest" / relative_path).is_file()
        for relative_path in manifest["generated_files"]
    )


def test_run_pipeline_supports_separate_results_directory(tmp_path: Path) -> None:
    manifest = run_pipeline(
        tmp_path,
        results_dir="results_cpp",
        static_trials=4,
        roi_trials=4,
        runtime_frames=5,
        runtime_warmup=1,
        dynamic_frames=200,
        real_world_trials=4,
        run_id="cpp-run",
        master_seed=12345,
    )

    assert manifest["latest_directory"] == "results_cpp/latest"
    run_root = tmp_path / "results_cpp" / "runs" / "cpp-run"
    assert (run_root / "run_manifest.json").is_file()
    assert (tmp_path / "results_cpp" / "latest" / "run_manifest.json").is_file()
    assert not (tmp_path / "results" / "latest").exists()
