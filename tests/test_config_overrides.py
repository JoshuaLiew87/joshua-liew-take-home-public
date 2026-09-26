from __future__ import annotations

import pytest

from config import load_yaml_overrides, scenario_config


def test_final_yaml_overrides_resolve_through_dataclass_validation() -> None:
    document = load_yaml_overrides("configs/final.yaml")
    dynamic = scenario_config("dynamic", overrides=document["profiles"]["dynamic"])
    sensor = scenario_config("sensor", overrides=document["profiles"]["sensor"])

    assert dynamic.trajectory.drift_model == "ou"
    assert dynamic.trajectory.disturbance_frequency_hz == 73.0
    assert sensor.estimator.centroid_roi_size_px == 13
    assert sensor.estimator.gaussian_roi_size_px == 15


def test_yaml_override_rejects_unknown_fields(tmp_path) -> None:
    source = tmp_path / "invalid.yaml"
    source.write_text(
        "profiles:\n  sensor:\n    estimator:\n      made_up_field: 1\n",
        encoding="utf-8",
    )
    document = load_yaml_overrides(source)
    with pytest.raises(ValueError, match="unknown fields"):
        scenario_config("sensor", overrides=document["profiles"]["sensor"])
