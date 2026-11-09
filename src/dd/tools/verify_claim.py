"""Tool 5/5 — the evidence guard.

A claim is a tuple (prose, evidence ids, optional metric/slice/subject). This module
decides whether it may be printed. It re-derives the number from the artifact on disk
instead of trusting the prose, and it fails closed: any missing pointer, hash drift or
value mismatch makes the claim unprintable.

This is the component that turns "please don't hallucinate" from a prompt request into
an enforced invariant, and it is the one to demo in an interview.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..data.store import MetricsStore
from ..errors import EvidenceMissing
from ..types import Claim, Evidence
from .base import ToolSpec

TOLERANCE = 0.05  # AP50 points; absorbs rounding in the report's 1-decimal display


class EvidenceRegistry:
    """Run-scoped registry of citable artifacts."""

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self._items: dict[str, Evidence] = {}

    def add(self, evidence: Evidence) -> Evidence:
        if evidence.id in self._items:
            return self._items[evidence.id]
        self._items[evidence.id] = evidence
        return evidence

    def add_file(
        self,
        evidence_id: str,
        kind: str,
        path: str | Path,
        summary: str = "",
    ) -> Evidence:
        """Register an artifact.

        `path` may be absolute, relative to the working directory, or relative to the run
        directory — resolved in that order. Ambiguity here silently breaks citations, so
        resolution is explicit and fails loudly.
        """
        p = Path(path)
        if not p.exists() and not p.is_absolute():
            alt = self.out_dir / p
            if alt.exists():
                p = alt
        if not p.exists():
            raise EvidenceMissing(f"cannot register missing artifact: {path} (resolved: {p})")
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        try:
            rel = str(p.resolve().relative_to(self.out_dir.resolve()))
        except ValueError:
            rel = str(p)
        return self.add(Evidence(id=evidence_id, kind=kind, path=rel, sha256=digest, summary=summary))

    def get(self, evidence_id: str) -> Evidence | None:
        return self._items.get(evidence_id)

    def ids(self) -> list[str]:
        return list(self._items)

    def all(self) -> list[Evidence]:
        return list(self._items.values())

    def to_list(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._items.values()]


class ClaimVerifier:
    """Verifies claims; returns the same Claim object with `verified` / `reasons` set."""

    def __init__(self, registry: EvidenceRegistry, store: MetricsStore, out_dir: str | Path) -> None:
        self.registry = registry
        self.store = store
        self.out_dir = Path(out_dir)

    spec = ToolSpec(
        name="verify_claim",
        description=(
            "Re-derive a claim's metric from its cited artifacts and mark the claim "
            "printable only if the evidence exists, has not drifted, and the number matches."
        ),
        params={"claim": "object"},
        returns="claim with verified flag and reasons",
        side_effect=False,
        cost="free",
    )

    def verify(self, claim: Claim) -> Claim:
        claim.reasons = []
        claim.verified = False

        if not claim.evidence_ids:
            claim.reasons.append("no_evidence: claim cites no artifact")
            return claim

        resolved: list[Evidence] = []
        for eid in claim.evidence_ids:
            ev = self.registry.get(eid)
            if ev is None:
                claim.reasons.append(f"evidence_missing: {eid} is not registered for this run")
                return claim
            resolved.append(ev)

        for ev in resolved:
            path = self.out_dir / ev.path
            if not path.exists():
                claim.reasons.append(f"evidence_missing: {ev.path} no longer exists")
                return claim
            if ev.sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != ev.sha256:
                claim.reasons.append(f"evidence_drift: {ev.path} changed after registration")
                return claim

        # A prose-only claim with valid evidence is acceptable; a numeric claim is not
        # until the number has been recomputed from the artifact.
        if claim.metric is None or claim.value is None:
            claim.verified = True
            return claim

        recomputed, why = self._recompute(resolved, claim)
        if recomputed is None:
            claim.reasons.append(f"metric_unrecoverable: {why}")
            return claim
        if abs(recomputed - claim.value) > TOLERANCE:
            claim.reasons.append(
                f"value_mismatch: claim says {claim.value:.2f}, artifact gives {recomputed:.2f}"
            )
            return claim

        claim.verified = True
        return claim

    def verify_all(self, claims: list[Claim]) -> list[Claim]:
        return [self.verify(c) for c in claims]

    # -- helpers -------------------------------------------------------------------
    def _recompute(
        self, evidence: list[Evidence], claim: Claim
    ) -> tuple[float | None, str]:
        for ev in evidence:
            path = self.out_dir / ev.path
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                return None, f"{ev.path} unreadable ({exc})"

            if ev.kind == "experiment":
                value = payload.get(claim.metric)
                if value is None:
                    return None, f"{ev.path} has no metric {claim.metric!r}"
                return float(value), ""

            if ev.kind == "metric_table":
                subject = claim.subject
                if not subject:
                    return None, "numeric claim on a metric_table needs a subject (detector)"
                series_map = payload.get("values", {}).get(subject) or {}
                if not series_map:
                    return None, f"no rows for subject={subject!r}"

                # Derived quantity: re-derive it from the whole series, not from a stored value.
                if claim.metric == "tail_drop":
                    ordered = [series_map[s]["map50"] for s in payload.get("slices", []) if s in series_map]
                    if len(ordered) < 2:
                        return None, "series too short to derive tail_drop"
                    return float(max(ordered) - ordered[-1]), ""

                try:
                    row = series_map[self._slice_value(claim.slice_id)]
                except (KeyError, TypeError):
                    return None, f"no row for subject={subject!r} slice={claim.slice_id!r}"
                if claim.metric not in row:
                    return None, f"row has no metric {claim.metric!r}"
                return float(row[claim.metric]), ""

        return None, "no evidence of a recomputable kind"

    def _slice_value(self, slice_id: str | None) -> str:
        """`anglebench:pitch=75-90` -> `75-90`."""
        if not slice_id:
            raise KeyError("missing slice_id")
        return slice_id.split("=", 1)[-1]


class VerifyClaimTool:
    """Thin callable wrapper so the verifier is registered like every other tool."""

    spec = ClaimVerifier.spec

    def __init__(self, verifier: ClaimVerifier) -> None:
        self.verifier = verifier

    def __call__(self, claim: Claim | dict[str, Any]) -> dict[str, Any]:
        obj = claim if isinstance(claim, Claim) else Claim.from_dict(claim)
        return self.verifier.verify(obj).to_dict()
