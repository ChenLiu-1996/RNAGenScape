"""Save / load structured run metrics (JSON). No log scraping.

Training scripts call ``write_metrics``; analysis scripts call ``read_metrics``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping, Optional


METRICS_FILENAME = "metrics.json"


def metrics_path(run_dir: str) -> str:
    """``<run_dir>/metrics.json`` (e.g. an OAE ``seed_*`` directory)."""
    return os.path.join(run_dir, METRICS_FILENAME)


def write_metrics(run_dir: str, payload: Mapping[str, Any]) -> str:
    """Write metrics JSON under ``run_dir``. Returns the written path."""
    os.makedirs(run_dir, exist_ok=True)
    path = metrics_path(run_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dict(payload), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def read_metrics(run_dir: str) -> Optional[Dict[str, Any]]:
    """Load metrics JSON, or ``None`` if missing."""
    path = metrics_path(run_dir)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(data)}")
    return data
