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
    )

    assert manifest["workflow"] == "run_all"
    assert manifest["parameters"]["static_trials"] == 4
    assert manifest["parameters"]["dynamic_frames"] == 200
    run_root = tmp_path / "results" / "runs" / "test-run"
    assert (run_root / "run_manifest.json").is_file()
    assert (tmp_path / "results" / "latest" / "run_manifest.json").is_file()
    assert manifest["generated_files"]
    assert all((run_root / relative_path).is_file() for relative_path in manifest["generated_files"])
    assert all(
        (tmp_path / "results" / "latest" / relative_path).is_file()
        for relative_path in manifest["generated_files"]
    )
