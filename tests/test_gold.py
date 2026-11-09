"""Gold-standard replay: the agent must rediscover the human conclusions, and the guard
must catch a planted lie. Both, or the run fails.
"""

from __future__ import annotations

from conftest import REPO_ROOT  # noqa: F401

from dd.evals.gold_replay import load_gold, run_eval


def test_gold_labels_are_frozen():
    gold = load_gold()
    kinds = {g["kind"] for g in gold["findings"]}
    assert kinds == {"non_monotonic", "wide_gap", "class_dissociation"}
    assert gold["thresholds"]["max_hallucination_rate"] == 0.0


def test_eval_passes_and_rediscovers_all_gold_findings(tmp_path):
    result = run_eval(out_dir=tmp_path / "eval")

    assert result["recall"] == 1.0, result["missed"]
    assert result["missed"] == []
    assert result["hallucination_rate"] == 0.0
    assert result["probe_caught"] is True, "the evidence guard missed a planted hallucination"
    assert result["interventions"] == 0
    assert result["gap_within_tolerance"] is True
    assert result["passed"] is True


def test_eval_reports_cost(tmp_path):
    result = run_eval(out_dir=tmp_path / "eval")
    assert result["gpu_minutes"] > 0
    assert result["experiments_ran"] >= 1
    assert result["tokens"] > 0
    assert result["report"]


def test_eval_does_not_report_the_negative_control(tmp_path):
    """rank_instability is a known non-finding; reporting it would be a false positive."""
    result = run_eval(out_dir=tmp_path / "eval")
    assert "rank_instability" not in result["unexpected_findings"]
