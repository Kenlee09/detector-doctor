"""Command line entry point.

    dd run     --metrics fixtures/metrics.json --out artifacts/demo --llm mock
    dd replay  --trace artifacts/demo/trace.jsonl
    dd eval    --out artifacts/eval
    dd selftest

Exit codes: 0 ok, 2 verification/strict failure, 3 budget exhausted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .budget import Budget
from .data.store import MetricsStore
from .errors import BudgetExceeded, UnverifiedClaim
from .orchestrator.pipeline import Pipeline
from .trace import read_trace, summarize

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METRICS = REPO_ROOT / "fixtures" / "metrics.json"


def _load_budget(args: argparse.Namespace) -> Budget:
    return Budget(
        gpu_minutes_max=args.gpu_minutes,
        experiments_max=args.max_experiments,
        tokens_max=args.max_tokens,
    )


def cmd_run(args: argparse.Namespace) -> int:
    store = MetricsStore.load(args.metrics)
    pipeline = Pipeline(
        store=store,
        out_dir=args.out,
        llm_mode=args.llm,
        budget=_load_budget(args),
        per_finding=args.per_finding,
        probe_hallucination=args.probe,
        strict=args.strict,
        trace_enabled=not args.no_trace,
    )
    try:
        ctx = pipeline.run(resume=args.resume)
    except UnverifiedClaim as exc:
        print(f"[strict] {exc}", file=sys.stderr)
        return 2
    except BudgetExceeded as exc:
        print(f"[budget] {exc}", file=sys.stderr)
        return 3

    audit = ctx.get("audit", {})
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"\nreport: {audit.get('report')}")
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    events = read_trace(args.trace)
    print(json.dumps(summarize(args.trace), indent=2, ensure_ascii=False))
    if args.node:
        for ev in events:
            if ev.get("event") in {"node_start", "node_skip", "node_done"}:
                print(f"  {ev['event']:<12} {ev['payload'].get('node')}")
    if args.tool:
        for ev in events:
            if ev.get("event") == "tool_call":
                print(f"  tool {ev['payload'].get('tool')}({json.dumps(ev['payload'].get('args'), ensure_ascii=False)})")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .evals.gold_replay import run_eval

    result = run_eval(out_dir=args.out, gold=args.gold)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 2


def cmd_selftest(args: argparse.Namespace) -> int:
    from .orchestrator.pipeline import Pipeline as P

    store = MetricsStore.load(args.metrics)
    out = Path(args.out)
    ctx = P(store=store, out_dir=out, llm_mode="mock", probe_hallucination=True).run()
    audit = ctx["audit"]
    ok = (
        audit["claims_rejected"] == 0
        and audit["probe_caught"]
        and audit["findings"] >= 3
        and audit["experiments_ran"] >= 1
    )
    print(json.dumps({"selftest": "pass" if ok else "fail", **audit}, indent=2, ensure_ascii=False))
    return 0 if ok else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dd", description="Detector Doctor — controlled robustness-diagnosis agent")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--metrics", default=str(DEFAULT_METRICS), help="metrics table JSON")
        sp.add_argument("--out", default="artifacts/run", help="run directory")

    run = sub.add_parser("run", help="execute the diagnosis pipeline")
    common(run)
    run.add_argument("--llm", default="mock", choices=["mock", "openai"], help="reasoning backend")
    run.add_argument("--gpu-minutes", type=float, default=90.0)
    run.add_argument("--max-experiments", type=int, default=16)
    run.add_argument("--max-tokens", type=int, default=250_000)
    run.add_argument("--per-finding", type=int, default=2, help="max options per finding")
    run.add_argument("--probe", action="store_true", help="inject an ungrounded claim as a negative control")
    run.add_argument("--no-strict", dest="strict", action="store_false", help="allow unverified claims to render")
    run.add_argument("--no-trace", action="store_true", help="disable JSONL tracing")
    run.add_argument("--resume", action="store_true", help="resume from the last checkpoint")
    run.set_defaults(func=cmd_run, strict=True)

    replay = sub.add_parser("replay", help="summarise a trace")
    replay.add_argument("--trace", default="artifacts/run/trace.jsonl")
    replay.add_argument("--node", action="store_true", help="print node timeline")
    replay.add_argument("--tool", action="store_true", help="print tool calls")
    replay.set_defaults(func=cmd_replay)

    ev = sub.add_parser("eval", help="run the gold-standard replay evaluation")
    ev.add_argument("--out", default="artifacts/eval")
    ev.add_argument("--gold", default=None)
    ev.set_defaults(func=cmd_eval)

    st = sub.add_parser("selftest", help="end-to-end smoke test with the hallucination probe armed")
    common(st)
    st.set_defaults(func=cmd_selftest)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
