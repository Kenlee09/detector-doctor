"""Deterministic attribution: the analysis must stand on its own, without any model."""

from __future__ import annotations

from conftest import REPO_ROOT  # noqa: F401

from dd.tools.attribute_errors import AttributeErrorsTool


def _findings_by_kind(store):
    return {f.kind: f for f in AttributeErrorsTool(store)()}


def test_detects_non_monotonic_shape(store):
    findings = _findings_by_kind(store)
    assert "non_monotonic" in findings
    f = findings["non_monotonic"]
    assert f.severity == "critical"
    assert len(f.detail["offending_detectors"]) == len(store.detectors)
    # Peak must land in the mid band, not at either extreme.
    assert f.detail["offending_detectors"]["yolov8n"]["peak_slice"] == "45-60"


def test_detects_wide_gap_in_expected_range(store):
    f = _findings_by_kind(store)["wide_gap"]
    assert 24.5 <= f.detail["gap_min"] <= 25.5
    assert 31.5 <= f.detail["gap_max"] <= 32.5
    assert f.value == f.detail["gap_max"]


def test_detects_class_dissociation(store):
    f = _findings_by_kind(store)["class_dissociation"]
    case = f.detail["cases"]["yolov8n"]
    assert case["best_top_class"] == "car"
    assert case["worst_top_class"] == "truck"


def test_does_not_report_rank_instability_when_order_is_stable(store):
    """Negative control: the fixture has consistent ordering, so this must stay silent."""
    assert "rank_instability" not in _findings_by_kind(store)


def test_reports_rank_instability_when_order_flips(tiny_store):
    findings = {f.kind: f for f in AttributeErrorsTool(tiny_store)()}
    assert "rank_instability" in findings
    assert findings["rank_instability"].detail["reference_slice"] == "low"


def test_findings_are_json_round_trippable(store):
    from dd.types import Finding

    for f in AttributeErrorsTool(store)():
        assert Finding.from_dict(f.to_dict()) == f


def test_monotonic_series_is_not_flagged(tiny_store):
    """A strictly improving series must not be reported as non-monotonic."""
    kinds = {f.kind for f in AttributeErrorsTool(tiny_store)()}
    assert "non_monotonic" not in kinds
