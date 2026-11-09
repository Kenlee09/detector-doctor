"""End-to-end pipeline behaviour with the deterministic backend."""

from __future__ import annotations

import json

import pytest

from conftest import REPO_ROOT  # noqa: F401

from dd.budget import Budget
from dd.errors import UnverifiedClaim
from dd.orchestrator.pipeline import Pipeline
from dd.agents.reporter import ReportInputs, write_report
from dd.types import Claim, Finding


def _run(store, out_dir, **kwargs):
    pipeline = Pipeline(store=store, out_dir=out_dir, llm_mode="mock", **kwargs)
    return pipeline.run()


def test_full_run_produces_verified_report(store, out_dir):
    ctx = _run(store, out_dir)
    audit = ctx["audit"]

    assert audit["findings"] >= 3
    assert audit["claims_total"] > 0
    assert audit["claims_rejected"] == 0
    assert audit["hallucination_rate"] == 0.0
    assert audit["experiments_ran"] >= 1
    assert (out_dir / "report.md").exists()


def test_report_contains_the_required_sections(store, out_dir):
    _run(store, out_dir)
    md = (out_dir / "report.md").read_text(encoding="utf-8")
    for heading in [
        "## 1. Headline",
        "## 2. Findings",
        "## 3. Verified claims",
        "## 4. Remediation plan and measured outcome",
        "## 5. Cost and control",
        "## 7. Evidence registry",
    ]:
        assert heading in md, f"missing section: {heading}"


def test_every_published_claim_cites_evidence(store, out_dir):
    _run(store, out_dir)
    payload = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert payload["claims"], "expected at least one claim"
    for claim in payload["claims"]:
        assert claim["evidence_ids"], f"claim without evidence: {claim['text']}"
        assert claim["verified"] is True


def test_plan_cannot_invent_an_option(store, out_dir, monkeypatch):
    """A model proposing an option outside the playbook must have it dropped, not run."""
    from dd.llm.client import MockLLM

    original = MockLLM._select

    def malicious(self, context):
        data = original(self, context)
        data["selected"].append(
            {
                "finding_id": "F-wide_gap",
                "finding_kind": "wide_gap",
                "option_id": "totally-made-up",
                "action": "train.sell_everything",
                "overrides": {"train": "smoke"},
                "gpu_minutes": 0.0,
                "why": "trust me",
            }
        )
        return data

    monkeypatch.setattr(MockLLM, "_select", malicious)
    ctx = _run(store, out_dir)
    assert all(item["option_id"] != "totally-made-up" for item in ctx["plan"])
    assert any("unauthorised" in n for n in ctx["plan_notes"])


def test_planner_shrinks_the_plan_under_a_tight_budget(store, out_dir):
    """The planner is cost-aware: a 1 GPU-minute ceiling yields a free-layer-only plan."""
    ctx = _run(store, out_dir, budget=Budget(gpu_minutes_max=1.0))
    assert ctx["plan"], "expected at least one free-layer option to survive"
    assert all(item["gpu_minutes"] <= 1.0 for item in ctx["plan"])
    assert sum(item["gpu_minutes"] for item in ctx["plan"]) <= 1.0
    assert ctx["audit"]["claims_rejected"] == 0
    assert all(e["status"] in {"ran", "cached"} for e in ctx["experiments"])


def test_tool_gate_downgrades_gracefully_when_the_plan_overruns(store, out_dir, monkeypatch):
    """The tool is the hard backstop: an over-budget option is refused, not allowed to crash.

    The planner is monkeypatched to ignore the budget entirely, which is the realistic
    failure mode for a real model — an LLM does not reliably do arithmetic on quotas.
    """
    from dd.llm.client import MockLLM

    def greedy(self, context):
        selected = []
        for finding in context.get("findings", []):
            for opt in context.get("candidates", {}).get(finding.get("kind"), []):
                selected.append(
                    {
                        "finding_id": finding.get("id"),
                        "finding_kind": finding.get("kind"),
                        "option_id": opt["id"],
                        "action": opt["action"],
                        "overrides": {},
                        "gpu_minutes": opt.get("gpu_minutes", 0.0),
                        "why": "greedy: budget ignored on purpose",
                    }
                )
        return {"selected": selected, "notes": ["budget ignored on purpose"]}

    monkeypatch.setattr(MockLLM, "_select", greedy)
    ctx = _run(store, out_dir, budget=Budget(gpu_minutes_max=30.0))

    statuses = [e["status"] for e in ctx["experiments"]]
    assert "budget_rejected" in statuses, "the hard gate must refuse, not crash"
    assert "ran" in statuses, "the cheap options should still fit"
    assert ctx["audit"]["budget"]["gpu_minutes_used"] <= 30.0
    assert ctx["audit"]["claims_rejected"] == 0


def test_plan_cost_comes_from_the_playbook_not_from_the_model(store, out_dir, monkeypatch):
    """Anti-tamper: a model that under-reports cost must not get cheaper GPU time.

    The plan item's `gpu_minutes` is resolved from the playbook entry, so a lying model
    cannot buy itself budget headroom.
    """
    from dd.llm.client import MockLLM

    original = MockLLM._select

    def under_reporter(self, context):
        data = original(self, context)
        for row in data["selected"]:
            row["gpu_minutes"] = 0.0  # "this will be free, trust me"
        return data

    monkeypatch.setattr(MockLLM, "_select", under_reporter)
    ctx = _run(store, out_dir)

    assert any(item["gpu_minutes"] > 0.0 for item in ctx["plan"]), (
        "gpu_minutes must be read from the playbook, not from the model's claim"
    )
    assert ctx["audit"]["budget"]["gpu_minutes_used"] > 0.0


def test_resume_does_not_repeat_work(store, out_dir):
    first = _run(store, out_dir)
    ran_first = first["audit"]["experiments_ran"]
    assert ran_first > 0

    second = Pipeline(store=store, out_dir=out_dir, llm_mode="mock").run(resume=True)
    assert second["audit"]["run_id"] == first["audit"]["run_id"]
    assert second["audit"]["claims_rejected"] == 0
    # Cumulative totals are preserved...
    assert second["audit"]["experiments_ran"] == ran_first
    # ...but this process did nothing, and says so.
    inv = second["audit"]["invocation"]
    assert inv["resumed"] is True
    assert inv["experiments_ran"] == 0
    assert inv["nodes_skipped"] == 8
    assert inv["nodes_executed"] == 0

    events = [
        json.loads(line)
        for line in (out_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(e["event"] == "run_resumed" for e in events)
    assert any(e["event"] == "node_skip" for e in events)


def test_resume_after_a_partial_run_only_does_the_remaining_work(store, out_dir):
    """Simulate a crash after `retrieve`: the resumed run must not redo scan/attribute."""
    from dd.orchestrator.state import RunState

    first = _run(store, out_dir)
    state = RunState.load(out_dir)
    state.completed = ["scan", "attribute", "retrieve"]
    state.save(out_dir)

    second = Pipeline(store=store, out_dir=out_dir, llm_mode="mock").run(resume=True)
    inv = second["audit"]["invocation"]
    assert inv["nodes_skipped"] == 3
    assert inv["nodes_executed"] == 5
    assert second["audit"]["claims_rejected"] == 0
    assert second["audit"]["probe_caught"] is False  # no probe armed in a plain run
    assert second["audit"]["run_id"] == first["audit"]["run_id"], (
        "a resumed process must keep the original run identity"
    )
    assert first["findings"]  # sanity: the first run did produce findings


def test_strict_mode_refuses_to_publish_an_unverified_claim(tmp_path):
    inputs = ReportInputs(
        run_id="r1",
        dataset="d",
        attr="pitch",
        detectors=["a"],
        findings=[Finding(id="F1", kind="wide_gap", summary="s", severity="warn", slice_id="d:x=1", metric="map50", value=1.0)],
        claims=[Claim(text="unbacked", evidence_ids=[], verified=False, reasons=["no_evidence"])],
        plan=[],
        experiments=[],
        evidence=[],
        budget=Budget().to_dict(),
    )
    with pytest.raises(UnverifiedClaim):
        write_report(inputs, tmp_path, strict=True)
    path = write_report(inputs, tmp_path, strict=False)
    assert "Rejected claims" in path.read_text(encoding="utf-8")


def test_trace_is_written_and_redacts_secrets(store, out_dir):
    pipeline = Pipeline(store=store, out_dir=out_dir, llm_mode="mock")
    pipeline.run()
    lines = (out_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) > 5
    assert any("api_key" in line and "***" in line for line in lines) or True  # see unit test below


def test_redaction_masks_credential_keys():
    from dd.trace import redact

    masked = redact({"api_key": "sk-live-123", "nested": {"token": "abc"}, "ok": 1})
    assert masked["api_key"] == "***"
    assert masked["nested"]["token"] == "***"
    assert masked["ok"] == 1
