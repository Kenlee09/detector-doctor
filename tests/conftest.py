"""Shared test fixtures."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from dd.data.store import MetricsStore  # noqa: E402


@pytest.fixture(scope="session")
def metrics_path() -> Path:
    return REPO_ROOT / "fixtures" / "metrics.json"


@pytest.fixture(scope="session")
def store(metrics_path: Path) -> MetricsStore:
    return MetricsStore.load(metrics_path)


@pytest.fixture
def tiny_store() -> MetricsStore:
    """Hand-built table with a *rank inversion* the main fixture deliberately lacks."""
    payload = {
        "meta": {"dataset": "tiny", "attr": "pitch"},
        "detectors": ["a", "b"],
        "slices": ["low", "high"],
        "values": {
            "a": {
                "low": {"map50": 60.0, "per_class": {"car": 60.0, "truck": 40.0}},
                "high": {"map50": 40.0, "per_class": {"car": 40.0, "truck": 35.0}},
            },
            "b": {
                "low": {"map50": 50.0, "per_class": {"car": 50.0, "truck": 45.0}},
                "high": {"map50": 70.0, "per_class": {"car": 70.0, "truck": 48.0}},
            },
        },
    }
    return MetricsStore(payload, source="<tiny>")


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    d = tmp_path / "run"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
