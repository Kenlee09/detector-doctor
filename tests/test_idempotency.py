"""Idempotency and budget interaction of the experiment tool.

The invariant worth protecting: retrying a run must never re-spend quota. If this breaks,
the whole "safe to resume" story breaks with it.
"""

from __future__ import annotations

import json

import pytest

from conftest import REPO_ROOT  # noqa: F401

from dd.budget import Budget
from dd.errors import BudgetExceeded
from dd.tools.run_experiment import ExperimentTool


def _tool(store, out_dir, budget=None):
    return ExperimentTool(store=store, out_dir=out_dir, budget=budget or Budget())


def test_second_call_is_a_cache_hit(store, out_dir):
    budget = Budget()
    tool = _tool(store, out_dir, budget)

    first = tool("yolov8n", {"postprocess": "whitelist"}, gpu_minutes=2.0)
    assert first["status"] == "ran"
    spent = budget.gpu_minutes_used
    assert spent == 2.0

    second = tool("yolov8n", {"postprocess": "whitelist"}, gpu_minutes=2.0)
    assert second["status"] == "cached"
    assert budget.gpu_minutes_used == spent, "a cache hit must not spend GPU-minutes"
    assert budget.experiments_used == 1


def test_override_order_does_not_change_the_hash(store, out_dir):
    tool = _tool(store, out_dir)
    a = tool("yolov8n", {"postprocess": "whitelist", "inference": "tta"})
    b = tool("yolov8n", {"inference": "tta", "postprocess": "whitelist"})
    assert a["config_hash"] == b["config_hash"]
    assert b["status"] == "cached"


def test_different_config_is_a_different_key(store, out_dir):
    tool = _tool(store, out_dir)
    a = tool("yolov8n", {"postprocess": "whitelist"})
    b = tool("yolov8n", {"postprocess": "track_filter"})
    assert a["config_hash"] != b["config_hash"]
    assert b["status"] == "ran"


def test_dry_run_never_spends(store, out_dir):
    budget = Budget()
    tool = _tool(store, out_dir, budget)
    result = tool("yolov8n", {"inference": "tta"}, gpu_minutes=1.0, dry_run=True)
    assert result["status"] == "dry_run"
    assert result["would_fit"] is True
    assert budget.gpu_minutes_used == 0.0
    assert not (out_dir / "experiments").joinpath(f"{result['config_hash']}.json").exists()


def test_rejected_budget_leaves_no_artifact(store, out_dir):
    budget = Budget(gpu_minutes_max=1.0)
    tool = _tool(store, out_dir, budget)
    with pytest.raises(BudgetExceeded):
        tool("yolov8n", {"train": "oblique_aug"}, gpu_minutes=18.0)
    assert not list((out_dir / "experiments").glob("*.json"))


def test_artifact_is_self_describing(store, out_dir):
    tool = _tool(store, out_dir)
    result = tool("yolov8n", {"inference": "tiling"})
    payload = json.loads((out_dir / result["artifact"]).read_text(encoding="utf-8"))
    assert payload["config"]["config_hash"] == result["config_hash"]
    assert payload["detail"]["simulated"] is True
    assert payload["detail"]["applied_overrides"] == ["inference=tiling"]


def test_simulated_gain_comes_from_the_playbook(store, out_dir):
    """The runner must not invent an improvement: it replays the playbook's number."""
    tool = _tool(store, out_dir)
    baseline = min(store.table("yolov8n").values())
    result = tool("yolov8n", {"inference": "tiling"})
    assert result["map50"] == pytest.approx(baseline + 3.4, abs=0.01)
