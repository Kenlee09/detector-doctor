"""The narrow agent: propose, then get checked.

Scope discipline is the whole design. The LLM is allowed to do exactly two things:

1. `propose`  — rank playbook options and emit a plan, as strict JSON.
2. `narrate`  — turn verified facts into prose.

It may not read images, compute a metric, invent an intervention, or write a number into
a claim. Everything it produces lands in a `Claim` and goes through `ClaimVerifier`
before the report writer will print it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..budget import Budget
from ..errors import BudgetExceeded, SchemaError
from ..llm.client import LLMClient, check_schema
from ..types import Claim, Finding

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["selected"],
    "properties": {
        "selected": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["finding_id", "option_id", "action", "overrides", "gpu_minutes", "why"],
                "properties": {
                    "finding_id": "string",
                    "finding_kind": "string",
                    "option_id": "string",
                    "action": "string",
                    "overrides": "object",
                    "gpu_minutes": "number",
                    "why": "string",
                    "source": "string",
                },
            },
        },
        "notes": {"type": "array"},
    },
}

NARRATIVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["narrative"],
    "properties": {"narrative": "string"},
}

PROPOSE_SYSTEM = """You are the remediation planner of Detector Doctor.

Hard rules:
- You may only choose `option_id` values present in the supplied candidates list.
- You never invent numbers. `expected_gain` values come from the playbook; quote them.
- Prefer free/cheap layers over retraining when the expected gain per GPU-minute is comparable.
- Stay within the stated GPU-minute budget. If an option does not fit, omit it and say so in notes.
- Output JSON only, matching the given schema.
"""


@dataclass
class PlanItem:
    finding_id: str
    finding_kind: str
    option_id: str
    action: str
    overrides: dict[str, str]
    gpu_minutes: float
    why: str
    source: str = ""
    expected_gain: float = 0.0
    result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "finding_kind": self.finding_kind,
            "option_id": self.option_id,
            "action": self.action,
            "overrides": dict(self.overrides),
            "gpu_minutes": self.gpu_minutes,
            "why": self.why,
            "source": self.source,
            "expected_gain": self.expected_gain,
            "result": self.result,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "PlanItem":
        return cls(
            finding_id=d["finding_id"],
            finding_kind=d.get("finding_kind", ""),
            option_id=d["option_id"],
            action=d["action"],
            overrides=dict(d.get("overrides") or {}),
            gpu_minutes=float(d.get("gpu_minutes", 0.0)),
            why=d.get("why", ""),
            source=d.get("source", ""),
            expected_gain=float(d.get("expected_gain", 0.0)),
            result=d.get("result"),
        )


@dataclass
class Proposal:
    items: list[PlanItem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [i.to_dict() for i in self.items],
            "notes": list(self.notes),
            "claims": [c.to_dict() for c in self.claims],
        }


class Diagnostician:
    """Owns the single LLM-shaped step of the pipeline."""

    def __init__(
        self,
        llm: LLMClient,
        budget: Budget,
        trace: Any | None = None,
        per_finding: int = 2,
    ) -> None:
        self.llm = llm
        self.budget = budget
        self.trace = trace
        self.per_finding = per_finding

    # -- propose -------------------------------------------------------------------
    def propose(
        self,
        findings: list[Finding],
        candidates: Mapping[str, list[dict[str, Any]]],
        evidence_id: str,
        table: Mapping[str, Mapping[str, float]],
    ) -> Proposal:
        context = {
            "per_finding": self.per_finding,
            "gpu_minutes_available": self.budget.gpu_minutes_remaining(),
            "findings": [f.to_dict() for f in findings],
            "candidates": {k: list(v) for k, v in candidates.items()},
        }
        system = PROPOSE_SYSTEM
        user = (
            "Rank the candidate remediations and return the plan.\n\n"
            "```json\n" + json.dumps(context, ensure_ascii=False, indent=1) + "\n```"
        )

        if self.trace is not None:
            self.trace.emit("llm_call", stage="propose", model=self.llm.name)
        resp = self.llm.complete(system, user, schema=PLAN_SCHEMA)
        self._charge(resp.tokens, "propose")

        data = resp.data if resp.data is not None else {}
        try:
            check_schema(data, PLAN_SCHEMA)
        except SchemaError:
            if self.trace is not None:
                self.trace.emit("llm_schema_error", stage="propose", raw=resp.text[:400])
            raise

        allowed = {o["id"]: o for opts in candidates.values() for o in opts}
        items: list[PlanItem] = []
        notes = list(data.get("notes") or [])
        for row in data["selected"]:
            opt = allowed.get(row.get("option_id"))
            if opt is None:
                # The model proposed something outside the playbook. Refuse, don't repair:
                # a hallucinated intervention must never reach an execution step.
                notes.append(f"rejected unauthorised option: {row.get('option_id')!r}")
                if self.trace is not None:
                    self.trace.emit("plan_rejected_option", option_id=row.get("option_id"))
                continue
            items.append(
                PlanItem(
                    finding_id=row["finding_id"],
                    finding_kind=row.get("finding_kind", ""),
                    option_id=opt["id"],
                    action=opt["action"],
                    overrides=dict(opt.get("overrides") or _overrides_from_source(opt)),
                    gpu_minutes=float(opt.get("gpu_minutes", 0.0)),
                    why=row.get("why", ""),
                    source=opt.get("source", ""),
                    expected_gain=float(opt.get("expected_gain", 0.0)),
                )
            )

        claims = self._diagnosis_claims(findings, evidence_id, table)
        return Proposal(items=items, notes=notes, claims=claims)

    # -- narrate -------------------------------------------------------------------
    def narrate(self, facts: list[dict[str, Any]]) -> str:
        context = {"findings": facts}
        user = "Summarise the audit for a technical reader.\n\n```json\n" + json.dumps(context, ensure_ascii=False) + "\n```"
        resp = self.llm.complete(PROPOSE_SYSTEM, user, schema=NARRATIVE_SCHEMA)
        self._charge(resp.tokens, "narrate")
        if resp.data and "narrative" in resp.data:
            return str(resp.data["narrative"])
        return resp.text

    # -- internals -----------------------------------------------------------------
    def _charge(self, tokens: int, stage: str) -> None:
        try:
            self.budget.reserve_tokens(tokens, label=stage)
        except BudgetExceeded:
            if self.trace is not None:
                self.trace.emit("budget_rejected", stage=stage, tokens=tokens)
            raise
        if self.trace is not None:
            self.trace.emit("llm_result", stage=stage, tokens=tokens)

    @staticmethod
    def _diagnosis_claims(
        findings: list[Finding],
        evidence_id: str,
        table: Mapping[str, Mapping[str, float]],
    ) -> list[Claim]:
        """Numeric claims derived from findings — each one must survive verification."""
        claims: list[Claim] = []
        for f in findings:
            if f.kind == "wide_gap":
                worst = (f.detail or {}).get("worst_detector")
                if worst and worst in table:
                    slices = table[worst]
                    lo_slice = (f.detail or {}).get("per_detector", {}).get(worst, {}).get("worst_slice")
                    if lo_slice in slices:
                        claims.append(
                            Claim(
                                text=(
                                    f"{worst} drops to {slices[lo_slice]:.1f} AP50 at the "
                                    f"{lo_slice} degree pitch bin, the worst domain in the audit."
                                ),
                                evidence_ids=[evidence_id],
                                metric="map50",
                                value=round(slices[lo_slice], 2),
                                slice_id=f"anglebench:pitch={lo_slice}",
                                subject=worst,
                            )
                        )
            if f.kind == "non_monotonic":
                offenders = (f.detail or {}).get("offending_detectors", {})
                for det, info in list(offenders.items())[:1]:
                    claims.append(
                        Claim(
                            text=(
                                f"{det} peaks at the {info['peak_slice']} degree bin and then falls "
                                f"{info['tail_drop']:.1f} AP50, so accuracy is not monotonic in viewpoint."
                            ),
                            evidence_ids=[evidence_id],
                            metric="tail_drop",
                            value=round(float(info["tail_drop"]), 2),
                            slice_id=None,  # derived across the series; verifier re-derives it
                            subject=det,
                        )
                    )
        return claims


def _overrides_from_source(opt: Mapping[str, Any]) -> dict[str, str]:
    token = str(opt.get("source", "")).split("override", 1)[-1].strip()
    if "=" in token:
        key, value = token.split("=", 1)
        return {key.strip(): value.strip()}
    return {}
