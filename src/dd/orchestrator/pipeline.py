"""The pipeline: eight nodes, one point of LLM freedom, one guard.

    scan -> attribute -> retrieve -> propose -> verify_plan -> experiment
         -> verify_results -> report

Read it as a contract:

| stage          | who decides                    | can spend money | can produce prose |
|----------------|--------------------------------|-----------------|-------------------|
| scan           | code (metrics read-out)        | no              | no                |
| attribute      | code (deterministic analysis)  | no              | no                |
| retrieve       | code (playbook index)          | no              | no                |
| propose        | **LLM** (rank + justify)       | no              | yes (non-binding) |
| verify_plan    | code (evidence guard)          | no              | no                |
| experiment     | code (idempotent, budgeted)    | **yes**         | no                |
| verify_results | code (evidence guard)          | no              | no                |
| report         | code (renders verified state)  | no              | no                |

No single stage can both make a claim and validate it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..agents.diagnostician import Diagnostician, PlanItem, Proposal
from ..agents.reporter import ReportInputs, write_report
from ..budget import Budget
from ..data.store import MetricsStore
from ..errors import BudgetExceeded
from ..llm.client import LLMClient, build_llm
from ..tools.attribute_errors import AttributeErrorsTool
from ..tools.base import ToolRegistry
from ..tools.playbook import PlaybookTool
from ..tools.run_experiment import ExperimentTool
from ..tools.slice_eval import SliceEvalTool
from ..tools.verify_claim import ClaimVerifier, EvidenceRegistry
from ..trace import TraceWriter, new_run_id
from ..types import Claim, Evidence, Finding
from .dag import DAG, Node
from .state import RunState


@dataclass
class Runtime:
    store: MetricsStore
    out_dir: Path
    run_id: str
    budget: Budget
    llm: LLMClient
    trace: TraceWriter | None = None
    per_finding: int = 2
    probe_hallucination: bool = False
    strict: bool = True
    registry: EvidenceRegistry = field(init=False)

    def __post_init__(self) -> None:
        self.registry = EvidenceRegistry(self.out_dir)

    # -- helpers shared by nodes ---------------------------------------------------
    def evidence_id_metric(self) -> str:
        return f"metric_table#{self.store.dataset}"

    def restore_evidence(self, raw: list[dict[str, Any]]) -> None:
        for item in raw or []:
            self.registry.add(Evidence.from_dict(item))

    def verifier(self) -> ClaimVerifier:
        return ClaimVerifier(self.registry, self.store, self.out_dir)


class Pipeline:
    def __init__(
        self,
        store: MetricsStore,
        out_dir: str | Path,
        llm: LLMClient | None = None,
        llm_mode: str = "mock",
        run_id: str | None = None,
        budget: Budget | None = None,
        per_finding: int = 2,
        probe_hallucination: bool = False,
        strict: bool = True,
        trace_enabled: bool = True,
    ) -> None:
        self.out_dir = Path(out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or new_run_id()
        self.trace = TraceWriter(self.out_dir / "trace.jsonl", self.run_id) if trace_enabled else None
        self.rt = Runtime(
            store=store,
            out_dir=self.out_dir,
            run_id=self.run_id,
            budget=budget or Budget(),
            llm=llm or build_llm(llm_mode),
            trace=self.trace,
            per_finding=per_finding,
            probe_hallucination=probe_hallucination,
            strict=strict,
        )
        self.registry_tools = ToolRegistry(trace=self.trace)
        # Counters for THIS process, as opposed to the cumulative totals in `audit`.
        # They are what make `--resume` auditable: a resumed process that did nothing
        # must report zero experiments, not the previous process's total.
        self.invocation: dict[str, Any] = {
            "resumed": False,
            "experiments_ran": 0,
            "experiments_cached": 0,
            "experiments_rejected": 0,
            "nodes_executed": 0,
            "nodes_skipped": 0,
        }

    # -- tool wiring ---------------------------------------------------------------
    def _wire_tools(self) -> None:
        rt = self.rt
        self.slice_eval = SliceEvalTool(rt.store)
        self.attribute = AttributeErrorsTool(rt.store)
        self.playbook = PlaybookTool()
        self.experiment = ExperimentTool(
            store=rt.store, out_dir=rt.out_dir, budget=rt.budget, trace=rt.trace
        )
        for tool in (self.slice_eval, self.attribute, self.playbook, self.experiment):
            if tool.spec.name not in self.registry_tools.names():
                self.registry_tools.register(tool)

    # -- nodes ---------------------------------------------------------------------
    def _scan(self, ctx: dict[str, Any]) -> dict[str, Any]:
        rt = self.rt
        table: dict[str, dict[str, float]] = {}
        for det in rt.store.detectors:
            rows = self.registry_tools.call("slice_eval", detector=det)
            table[det] = {r["key"]["value"]: r["map50"] for r in rows}

        snapshot_dir = rt.out_dir / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / f"metrics_{rt.store.dataset}.json"
        snapshot_path.write_text(
            json.dumps(rt.store.snapshot(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        ev = rt.registry.add_file(
            rt.evidence_id_metric(),
            "metric_table",
            snapshot_path,
            summary=f"frozen {rt.store.dataset} metric table",
        )

        focus = min(table, key=lambda d: min(table[d].values())) if table else None
        return {
            "dataset": rt.store.dataset,
            "attr": rt.store.attr,
            "detectors": list(rt.store.detectors),
            "table": table,
            "focus_detector": focus,
            "evidence": rt.registry.to_list(),
            "evidence_metric": ev.id,
        }

    def _attribute(self, ctx: dict[str, Any]) -> dict[str, Any]:
        findings = self.registry_tools.call("attribute_errors", detectors=ctx["detectors"])
        return {"findings": [f.to_dict() for f in findings]}

    def _retrieve(self, ctx: dict[str, Any]) -> dict[str, Any]:
        findings = [Finding.from_dict(d) for d in ctx.get("findings", [])]
        candidates: dict[str, list[dict[str, Any]]] = {}
        for f in findings:
            if f.kind not in candidates:
                candidates[f.kind] = self.registry_tools.call("retrieve_playbook", symptom=f.kind)
        return {"candidates": candidates}

    def _propose(self, ctx: dict[str, Any]) -> dict[str, Any]:
        rt = self.rt
        diagnostician = Diagnostician(
            llm=rt.llm, budget=rt.budget, trace=rt.trace, per_finding=rt.per_finding
        )
        findings = [Finding.from_dict(d) for d in ctx.get("findings", [])]
        proposal: Proposal = diagnostician.propose(
            findings=findings,
            candidates=ctx.get("candidates", {}),
            evidence_id=ctx.get("evidence_metric", ""),
            table=ctx.get("table", {}),
        )
        if rt.trace is not None:
            rt.trace.emit(
                "plan_proposed",
                items=[i.option_id for i in proposal.items],
                notes=proposal.notes,
            )
        return {
            "plan": [i.to_dict() for i in proposal.items],
            "plan_notes": proposal.notes,
            "plan_claims": [c.to_dict() for c in proposal.claims],
        }

    def _verify_plan(self, ctx: dict[str, Any]) -> dict[str, Any]:
        verifier = self.rt.verifier()
        claims = [verifier.verify(Claim.from_dict(d)) for d in ctx.get("plan_claims", [])]
        rejected = [c for c in claims if not c.verified]
        if self.rt.trace is not None:
            self.rt.trace.emit(
                "claims_verified", stage="plan", total=len(claims), rejected=len(rejected)
            )
        return {
            "plan_claims": [c.to_dict() for c in claims],
            "interventions": len(rejected),
        }

    def _experiment(self, ctx: dict[str, Any]) -> dict[str, Any]:
        rt = self.rt
        detector = ctx.get("focus_detector") or (ctx.get("detectors") or [None])[0]
        results: list[dict[str, Any]] = []
        claims: list[Claim] = []
        plan = [PlanItem.from_dict(d) for d in ctx.get("plan", [])]

        for item in plan:
            try:
                result = self.registry_tools.call(
                    "run_experiment",
                    detector=detector,
                    overrides=item.overrides,
                    gpu_minutes=item.gpu_minutes,
                )
            except BudgetExceeded as exc:
                # A refused reservation is a normal outcome, not a crash: record it and go on.
                result = {"status": "budget_rejected", "reason": str(exc)}
                if rt.trace is not None:
                    rt.trace.emit("experiment_rejected", option=item.option_id, reason=str(exc))
            item.result = result
            results.append({"option_id": item.option_id, **result})
            status = result.get("status")
            if status == "ran":
                self.invocation["experiments_ran"] += 1
            elif status == "cached":
                self.invocation["experiments_cached"] += 1
            else:
                self.invocation["experiments_rejected"] += 1

            if result.get("status") in {"ran", "cached"}:
                ev = rt.registry.add_file(
                    f"experiment:{result['config_hash']}",
                    "experiment",
                    result["artifact"],
                    summary=f"{item.option_id} on {detector}",
                )
                baseline = (result.get("detail") or {}).get("baseline_map50")
                if baseline is None:
                    baseline = min((ctx.get("table", {}).get(detector) or {"": 0.0}).values())
                claims.append(
                    Claim(
                        text=(
                            f"`{item.action}` moves {detector}'s worst-domain AP50 from "
                            f"{baseline:.1f} to {result['map50']:.1f} "
                            f"({'+' if result['map50'] >= baseline else ''}"
                            f"{result['map50'] - baseline:.1f})."
                        ),
                        evidence_ids=[ev.id],
                        metric="map50",
                        value=result["map50"],
                        subject=detector,
                    )
                )

        return {
            "plan": [i.to_dict() for i in plan],
            "experiments": results,
            "result_claims": [c.to_dict() for c in claims],
            "evidence": rt.registry.to_list(),
        }

    def _verify_results(self, ctx: dict[str, Any]) -> dict[str, Any]:
        verifier = self.rt.verifier()
        claims = [verifier.verify(Claim.from_dict(d)) for d in ctx.get("result_claims", [])]
        rejected = [c for c in claims if not c.verified]
        probe = self._probe_claim(ctx, verifier) if self.rt.probe_hallucination else None
        if self.rt.trace is not None:
            self.rt.trace.emit(
                "claims_verified", stage="results", total=len(claims), rejected=len(rejected)
            )
        out = {
            "result_claims": [c.to_dict() for c in claims],
            "interventions": int(ctx.get("interventions", 0)) + len(rejected),
        }
        if probe is not None:
            out["probe_claim"] = probe.to_dict()
        return out

    def _probe_claim(self, ctx: dict[str, Any], verifier: ClaimVerifier) -> Claim:
        """Inject a plausible-looking but ungrounded claim.

        Its purpose is negative-control coverage: if the guard ever stops catching this,
        the eval suite fails. A verifier that is never tested against a lie is decoration.
        """
        detector = ctx.get("focus_detector") or "detector"
        probe = Claim(
            text=f"{detector} reaches 88.0 AP50 in the hardest pitch bin, outperforming every other model.",
            evidence_ids=[ctx.get("evidence_metric", "")],
            metric="map50",
            value=88.0,
            slice_id="anglebench:pitch=75-90",
            subject=detector,
        )
        verified = verifier.verify(probe)
        if self.rt.trace is not None:
            self.rt.trace.emit(
                "probe_claim_checked", verified=verified.verified, reasons=verified.reasons
            )
        return verified

    def _report(self, ctx: dict[str, Any]) -> dict[str, Any]:
        rt = self.rt
        claims = [Claim.from_dict(d) for d in ctx.get("plan_claims", [])] + [
            Claim.from_dict(d) for d in ctx.get("result_claims", [])
        ]
        inputs = ReportInputs(
            run_id=rt.run_id,
            dataset=ctx.get("dataset", rt.store.dataset),
            attr=ctx.get("attr", rt.store.attr),
            detectors=ctx.get("detectors", []),
            findings=[Finding.from_dict(d) for d in ctx.get("findings", [])],
            claims=claims,
            plan=ctx.get("plan", []),
            experiments=ctx.get("experiments", []),
            evidence=[Evidence.from_dict(d) for d in ctx.get("evidence", [])],
            budget=rt.budget.to_dict(),
            notes=ctx.get("plan_notes", []),
            probe_claim=(
                Claim.from_dict(ctx["probe_claim"]) if ctx.get("probe_claim") else None
            ),
        )
        path = write_report(inputs, rt.out_dir, strict=rt.strict)
        rejected = [c for c in claims if not c.verified]
        audit = {
            "run_id": rt.run_id,
            "report": str(path),
            "findings": len(inputs.findings),
            "claims_total": len(claims),
            "claims_verified": len(claims) - len(rejected),
            "claims_rejected": len(rejected),
            "hallucination_rate": round(len(rejected) / len(claims), 4) if claims else 0.0,
            "probe_caught": bool(ctx.get("probe_claim") and not ctx["probe_claim"]["verified"]),
            "interventions": int(ctx.get("interventions", 0)),
            "experiments_ran": len([e for e in ctx.get("experiments", []) if e.get("status") == "ran"]),
            "experiments_cached": len([e for e in ctx.get("experiments", []) if e.get("status") == "cached"]),
            "budget": rt.budget.to_dict(),
            "invocation": dict(self.invocation),
        }
        if rt.trace is not None:
            rt.trace.emit("run_finished", **audit)
        return {"audit": audit, "report_path": str(path)}

    # -- graph ---------------------------------------------------------------------
    def build(self) -> DAG:
        self._wire_tools()
        return DAG(
            [
                Node("scan", self._scan, (), "freeze the metric table and register it as evidence"),
                Node("attribute", self._attribute, ("scan",), "derive robustness findings"),
                Node("retrieve", self._retrieve, ("attribute",), "index the playbook by finding"),
                Node("propose", self._propose, ("retrieve",), "LLM ranks costed options"),
                Node("verify_plan", self._verify_plan, ("propose",), "evidence guard on diagnosis"),
                Node("experiment", self._experiment, ("verify_plan",), "budgeted idempotent runs"),
                Node("verify_results", self._verify_results, ("experiment",), "evidence guard on results"),
                Node("report", self._report, ("verify_results",), "render verified state"),
            ]
        )

    def run(self, resume: bool = False) -> dict[str, Any]:
        dag = self.build()
        state = RunState.load_or_new(self.out_dir, self.run_id) if resume else RunState(self.run_id)
        if resume and state.completed:
            self.invocation["resumed"] = True
            # Adopt the original identity: a resumed process is the *same run*, so the
            # checkpoint, the trace and the report must not disagree about which run it is.
            self.run_id = state.run_id
            self.rt.run_id = state.run_id
            if self.trace is not None:
                self.trace.run_id = state.run_id
            self.rt.restore_evidence(state.ctx.get("evidence", []))
            if self.rt.trace is not None:
                self.rt.trace.emit("run_resumed", completed=list(state.completed))

        def on_done(node: Node, ctx: dict[str, Any]) -> None:
            state.completed = list(dict.fromkeys(state.completed + [node.name]))
            state.ctx = ctx
            state.save(self.out_dir)

        ctx = dict(state.ctx)
        ctx.setdefault("run_id", self.run_id)
        dag.run(
            ctx,
            completed=state.completed,
            on_start=self._trace_node_start,
            on_done=on_done,
            on_skip=self._trace_node_skip,
        )
        # Refresh the invocation block even when every node was skipped, so a no-op
        # resume reports "0 experiments in this process" instead of echoing the
        # previous process's totals.
        if isinstance(ctx.get("audit"), dict):
            ctx["audit"]["invocation"] = dict(self.invocation)
        state.ctx = ctx
        state.save(self.out_dir)
        if self.trace is not None:
            self.trace.close()
        return ctx

    # -- callbacks -----------------------------------------------------------------
    def _trace_node_start(self, node: Node) -> None:
        self.invocation["nodes_executed"] += 1
        if self.trace is not None:
            self.trace.emit("node_start", node=node.name, description=node.description)

    def _trace_node_skip(self, node: Node) -> None:
        self.invocation["nodes_skipped"] += 1
        if self.trace is not None:
            self.trace.emit("node_skip", node=node.name, reason="already completed")
