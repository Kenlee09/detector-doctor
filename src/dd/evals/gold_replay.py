"""Gold-standard replay evaluation.

The point of this module is to answer, with numbers, the only question an interviewer
(or an operator) should care about: **is the agent's output trustworthy, and at what
cost?**

Method: replay the full pipeline against a frozen run and score its conclusions against
human-authored gold findings, plus three reliability counters that no self-report can
fake:

* `recall`             — did the agent independently rediscover the known findings?
* `hallucination_rate` — fraction of published claims that failed the evidence guard
* `probe_caught`       — did the guard reject a deliberately injected false claim?
* `interventions`      — how many times a human/guard had to correct the agent

A run that reports high recall *and* a caught probe is meaningful. High recall alone is
not: it just means the fixture is easy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..budget import Budget
from ..data.store import MetricsStore
from ..orchestrator.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GOLD = REPO_ROOT / "eval" / "gold" / "gold_findings.json"
DEFAULT_METRICS = REPO_ROOT / "fixtures" / "metrics.json"


@dataclass
class GoldResult:
    recall: float
    matched: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    hallucination_rate: float = 0.0
    probe_caught: bool = False
    interventions: int = 0
    gpu_minutes: float = 0.0
    experiments_ran: int = 0
    experiments_cached: int = 0
    tokens: int = 0
    wall_seconds: float = 0.0
    passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "recall": round(self.recall, 4),
            "matched": self.matched,
            "missed": self.missed,
            "unexpected_findings": self.extra,
            "hallucination_rate": round(self.hallucination_rate, 4),
            "probe_caught": self.probe_caught,
            "interventions": self.interventions,
            "gpu_minutes": round(self.gpu_minutes, 2),
            "experiments_ran": self.experiments_ran,
            "experiments_cached": self.experiments_cached,
            "tokens": self.tokens,
            "wall_seconds": round(self.wall_seconds, 2),
            "passed": self.passed,
        }


def load_gold(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_GOLD
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def run_eval(
    out_dir: str | Path = "artifacts/eval",
    gold: str | Path | None = None,
    metrics: str | Path | None = None,
    per_finding: int = 2,
) -> dict[str, Any]:
    gold_payload = load_gold(gold)
    store = MetricsStore.load(metrics or DEFAULT_METRICS)

    # Probe armed: the negative control must fire on every eval run.
    pipeline = Pipeline(
        store=store,
        out_dir=out_dir,
        llm_mode="mock",
        budget=Budget(),
        per_finding=per_finding,
        probe_hallucination=True,
        strict=False,  # the probe is deliberately unverifiable
    )
    ctx = pipeline.run()
    audit = ctx["audit"]

    detected = {f["kind"] for f in ctx.get("findings", [])}
    expected = {g["kind"] for g in gold_payload["findings"]}
    matched = sorted(detected & expected)
    missed = sorted(expected - detected)
    extra = sorted(detected - expected)

    tolerance = float(gold_payload.get("gap_tolerance", 1.0))
    gap_ok = True
    for g in gold_payload["findings"]:
        if g["kind"] != "wide_gap":
            continue
        lo, hi = g.get("range", [None, None])
        actual = next((f for f in ctx["findings"] if f["kind"] == "wide_gap"), None)
        if actual and lo is not None and hi is not None:
            detail = actual.get("detail", {})
            gap_ok = (
                abs(detail.get("gap_min", -1) - lo) <= tolerance
                and abs(detail.get("gap_max", -1) - hi) <= tolerance
            )

    recall = len(matched) / len(expected) if expected else 1.0
    thresholds = gold_payload.get("thresholds", {})
    result = GoldResult(
        recall=recall,
        matched=matched,
        missed=missed,
        extra=extra,
        hallucination_rate=float(audit["hallucination_rate"]),
        probe_caught=bool(audit["probe_caught"]),
        interventions=int(audit["interventions"]),
        gpu_minutes=float(audit["budget"]["gpu_minutes_used"]),
        experiments_ran=int(audit["experiments_ran"]),
        experiments_cached=int(audit["experiments_cached"]),
        tokens=int(audit["budget"]["tokens_used"]),
    )
    result.passed = (
        recall >= float(thresholds.get("min_recall", 1.0))
        and result.hallucination_rate <= float(thresholds.get("max_hallucination_rate", 0.0))
        and result.probe_caught
        and gap_ok
        and result.interventions == 0
    )
    payload = result.to_dict()
    payload["gap_within_tolerance"] = gap_ok
    payload["report"] = audit["report"]
    payload["gold_version"] = gold_payload.get("version", "unknown")
    return payload
