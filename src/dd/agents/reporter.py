"""Report writer.

The report is a *view over verified state*, never a second opinion. It refuses to print
an unverified claim; rejected claims are listed explicitly in an appendix instead of
being quietly dropped, because "what the agent tried to say and could not prove" is
exactly the signal an operator needs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..errors import UnverifiedClaim
from ..types import Claim, Evidence, Finding


@dataclass
class ReportInputs:
    run_id: str
    dataset: str
    attr: str
    detectors: Sequence[str]
    findings: Sequence[Finding]
    claims: Sequence[Claim]
    plan: Sequence[Mapping[str, Any]]
    experiments: Sequence[Mapping[str, Any]]
    evidence: Sequence[Evidence]
    budget: Mapping[str, Any]
    notes: Sequence[str] = field(default_factory=list)
    probe_claim: Claim | None = None


def _fmt(value: Any, digits: int = 1) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _findings_md(findings: Sequence[Finding]) -> str:
    if not findings:
        return "_No anomaly exceeded the reporting threshold._\n"
    lines = ["| id | severity | finding | metric | value |", "|---|---|---|---|---|"]
    for f in findings:
        lines.append(
            f"| `{f.id}` | {f.severity} | {f.summary} | {f.metric} | {_fmt(f.value, 1)} |"
        )
    return "\n".join(lines) + "\n"


def _plan_md(plan: Sequence[Mapping[str, Any]]) -> str:
    if not plan:
        return "_No remediation was proposed (budget or playbook empty)._\n"
    lines = [
        "| action | layer override | expected | GPU-min | status | measured AP50 |",
        "|---|---|---|---|---|---|",
    ]
    for item in plan:
        result = item.get("result") or {}
        overrides = ", ".join(f"`{k}={v}`" for k, v in (item.get("overrides") or {}).items()) or "—"
        lines.append(
            "| {action} | {overrides} | +{gain} | {gpu} | {status} | {map50} |".format(
                action=item.get("action", "?"),
                overrides=overrides,
                gain=_fmt(item.get("expected_gain", 0.0), 1),
                gpu=_fmt(item.get("gpu_minutes", 0.0), 1),
                status=result.get("status", "not run"),
                map50=_fmt(result.get("map50"), 2) if result.get("map50") is not None else "—",
            )
        )
    return "\n".join(lines) + "\n"


def render(inputs: ReportInputs) -> str:
    verified = [c for c in inputs.claims if c.verified]
    rejected = [c for c in inputs.claims if not c.verified]

    out: list[str] = []
    out.append(f"# Detector Doctor report — {inputs.dataset} / {inputs.attr}\n")
    out.append(f"`run_id={inputs.run_id}` · detectors: {len(inputs.detectors)} · "
               f"slices attributed by `{inputs.attr}`\n")

    out.append("## 1. Headline\n")
    if inputs.findings:
        worst = max(inputs.findings, key=lambda f: f.value)
        out.append(f"- **{len(inputs.findings)}** robustness findings, highest severity "
                   f"`{worst.severity}`.\n")
    else:
        out.append("- No findings.\n")
    out.append(f"- **{len(verified)}** claims verified against artifacts, "
               f"**{len(rejected)}** rejected by the evidence guard.\n")
    ran = [e for e in inputs.experiments if e.get("status") == "ran"]
    cached = [e for e in inputs.experiments if e.get("status") == "cached"]
    out.append(f"- Experiments: {len(ran)} ran, {len(cached)} cache hits "
               f"({_fmt(inputs.budget.get('gpu_minutes_used', 0.0), 1)} GPU-min of "
               f"{_fmt(inputs.budget.get('gpu_minutes_max', 0.0), 0)}).\n")

    out.append("\n## 2. Findings\n")
    out.append(_findings_md(inputs.findings))

    out.append("\n## 3. Verified claims\n")
    if not verified:
        out.append("_None._\n")
    for c in verified:
        cite = ", ".join(f"`{eid}`" for eid in c.evidence_ids)
        out.append(f"- {c.text}  \n  <sub>evidence: {cite}</sub>\n")

    out.append("\n## 4. Remediation plan and measured outcome\n")
    out.append(_plan_md(inputs.plan))

    if inputs.notes:
        out.append("\n### Planner notes\n")
        for n in inputs.notes:
            out.append(f"- {n}\n")

    out.append("\n## 5. Cost and control\n")
    out.append(
        "| counter | used | cap |\n|---|---|---|\n"
        f"| GPU-minutes | {_fmt(inputs.budget.get('gpu_minutes_used', 0.0), 1)} | "
        f"{_fmt(inputs.budget.get('gpu_minutes_max', 0.0), 0)} |\n"
        f"| experiments | {inputs.budget.get('experiments_used', 0)} | "
        f"{inputs.budget.get('experiments_max', 0)} |\n"
        f"| LLM tokens | {inputs.budget.get('tokens_used', 0)} | "
        f"{inputs.budget.get('tokens_max', 0)} |\n"
    )
    rejections = inputs.budget.get("rejections") or []
    if rejections:
        out.append("\nBudget rejections (pre-flight, nothing was launched):\n")
        for r in rejections:
            out.append(f"- {r}\n")

    if rejected or inputs.probe_claim is not None:
        out.append("\n## 6. Rejected claims (evidence guard)\n")
        out.append("These statements were generated during the run and refused publication.\n\n")
        for c in rejected:
            out.append(f"- ~~{c.text}~~  \n  <sub>{'; '.join(c.reasons)}</sub>\n")
        if inputs.probe_claim is not None:
            out.append(
                f"- ~~{inputs.probe_claim.text}~~  \n  "
                f"<sub>injected probe — {'; '.join(inputs.probe_claim.reasons) or 'rejected'}</sub>\n"
            )

    out.append("\n## 7. Evidence registry\n")
    if inputs.evidence:
        out.append("| id | kind | path | sha256 |\n|---|---|---|---|\n")
        for ev in inputs.evidence:
            out.append(f"| `{ev.id}` | {ev.kind} | `{ev.path}` | `{ev.sha256[:12]}…` |\n")
    else:
        out.append("_empty_\n")

    return "".join(out)


def write_report(inputs: ReportInputs, out_dir: str | Path, strict: bool = True) -> Path:
    """Render and persist `report.md` + `report.json`.

    `strict=True` raises if any claim is unverified — used by CI so a regression in the
    guard fails the build instead of silently degrading report quality.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if strict and any(not c.verified for c in inputs.claims):
        bad = [c for c in inputs.claims if not c.verified]
        raise UnverifiedClaim(
            f"{len(bad)} unverified claim(s) reached the report writer: "
            + "; ".join(c.reasons[0] if c.reasons else "unknown" for c in bad)
        )

    md = render(inputs)
    (out_path / "report.md").write_text(md, encoding="utf-8")
    (out_path / "report.json").write_text(
        json.dumps(
            {
                "run_id": inputs.run_id,
                "dataset": inputs.dataset,
                "attr": inputs.attr,
                "detectors": list(inputs.detectors),
                "findings": [f.to_dict() for f in inputs.findings],
                "claims": [c.to_dict() for c in inputs.claims],
                "plan": [dict(p) for p in inputs.plan],
                "experiments": [dict(e) for e in inputs.experiments],
                "evidence": [e.to_dict() for e in inputs.evidence],
                "budget": dict(inputs.budget),
                "notes": list(inputs.notes),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return out_path / "report.md"
