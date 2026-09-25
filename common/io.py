"""Small helpers for reproducible experiment output."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def jsonable(value: Any) -> Any:
    """Convert dataclasses and tuples into JSON-compatible values."""

    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    return value


def save_json(path: str | Path, value: Any) -> None:
    """Write deterministic, human-readable JSON metadata."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

