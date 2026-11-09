"""The evidence guard.

These tests are the contract. If any of them goes green while a false claim can reach
the report, the product's central claim is a lie — so they cover the four ways a claim
goes wrong: no citation, dangling citation, drifted artifact, wrong number.
"""

from __future__ import annotations

import json
from pathlib import Path

from conftest import REPO_ROOT  # noqa: F401

from dd.data.store import MetricsStore
from dd.tools.verify_claim import ClaimVerifier, EvidenceRegistry
from dd.types import Claim


def _snapshot(out_dir: Path, store: MetricsStore) -> tuple[EvidenceRegistry, ClaimVerifier]:
    snap = out_dir / "snapshots" / "metrics.json"
    snap.parent.mkdir(parents=True, exist_ok=True)
    snap.write_text(json.dumps(store.snapshot()), encoding="utf-8")
    registry = EvidenceRegistry(out_dir)
    registry.add_file("metric_table#anglebench", "metric_table", snap)
    return registry, ClaimVerifier(registry, store, out_dir)


def test_claim_without_evidence_is_rejected(store, out_dir):
    _, verifier = _snapshot(out_dir, store)
    claim = verifier.verify(Claim(text="Something looks bad."))
    assert claim.verified is False
    assert claim.reasons[0].startswith("no_evidence")


def test_dangling_evidence_id_is_rejected(store, out_dir):
    _, verifier = _snapshot(out_dir, store)
    claim = verifier.verify(
        Claim(text="x", evidence_ids=["metric_table#nope"], metric="map50", value=50.0, subject="yolov8n")
    )
    assert claim.verified is False
    assert claim.reasons[0].startswith("evidence_missing")


def test_correct_number_is_verified(store, out_dir):
    _, verifier = _snapshot(out_dir, store)
    truth = store.get("yolov8n", "75-90").map50
    claim = verifier.verify(
        Claim(
            text=f"yolov8n reaches {truth} AP50 at 75-90 degrees.",
            evidence_ids=["metric_table#anglebench"],
            metric="map50",
            value=round(truth, 2),
            slice_id="anglebench:pitch=75-90",
            subject="yolov8n",
        )
    )
    assert claim.verified is True, claim.reasons
    assert claim.reasons == []


def test_wrong_number_is_rejected(store, out_dir):
    _, verifier = _snapshot(out_dir, store)
    claim = verifier.verify(
        Claim(
            text="yolov8n reaches 88.0 AP50 at 75-90 degrees.",  # the classic hallucination
            evidence_ids=["metric_table#anglebench"],
            metric="map50",
            value=88.0,
            slice_id="anglebench:pitch=75-90",
            subject="yolov8n",
        )
    )
    assert claim.verified is False
    assert claim.reasons[0].startswith("value_mismatch")


def test_artifact_drift_is_rejected(store, out_dir):
    registry, verifier = _snapshot(out_dir, store)
    truth = store.get("yolov8n", "75-90").map50
    # Mutate the artifact *after* registration: the sha256 no longer matches.
    snap = out_dir / "snapshots" / "metrics.json"
    payload = json.loads(snap.read_text(encoding="utf-8"))
    payload["values"]["yolov8n"]["75-90"]["map50"] = truth + 40
    snap.write_text(json.dumps(payload), encoding="utf-8")

    claim = verifier.verify(
        Claim(
            text="x",
            evidence_ids=["metric_table#anglebench"],
            metric="map50",
            value=round(truth, 2),
            slice_id="anglebench:pitch=75-90",
            subject="yolov8n",
        )
    )
    assert claim.verified is False
    assert claim.reasons[0].startswith("evidence_drift")


def test_derived_metric_is_rederived(store, out_dir):
    _, verifier = _snapshot(out_dir, store)
    series = [store.table("yolov8n")[s] for s in store.slices]
    tail_drop = round(max(series) - series[-1], 2)

    good = verifier.verify(
        Claim(
            text="tail drop",
            evidence_ids=["metric_table#anglebench"],
            metric="tail_drop",
            value=tail_drop,
            subject="yolov8n",
        )
    )
    assert good.verified is True, good.reasons

    bad = verifier.verify(
        Claim(
            text="tail drop",
            evidence_ids=["metric_table#anglebench"],
            metric="tail_drop",
            value=tail_drop + 3.0,
            subject="yolov8n",
        )
    )
    assert bad.verified is False


def test_prose_claim_with_valid_evidence_is_allowed(store, out_dir):
    _, verifier = _snapshot(out_dir, store)
    claim = verifier.verify(
        Claim(text="The audit found a wide domain gap.", evidence_ids=["metric_table#anglebench"])
    )
    assert claim.verified is True
