# Detector Doctor

**A controlled agent that diagnoses object-detector robustness and proposes evidence-backed fixes.**

Object detectors are evaluated on average accuracy, but they fail unevenly: accuracy across
viewpoint, illumination or scale is rarely flat, and the worst slice — not the mean — is what
breaks a deployment. Auditing that by hand means running tens of detector × slice × intervention
combinations, then writing up conclusions that a reviewer has to trust.

Detector Doctor automates the whole loop — **slice → attribute → propose → run → verify → report** —
under three constraints that a free-running LLM agent cannot give you:

| constraint | mechanism |
|---|---|
| it must not fabricate findings | the analysis is deterministic code; no metric is ever computed by a model |
| it must not publish unproven claims | every claim is re-derived from a hashed artifact before it can appear in the report |
| it must not burn unbounded GPU time | a pre-flight budget gate plus config-hash idempotency on every experiment |

---

## Quickstart

No GPU, no API key, no dataset. The shipped metrics table is a **synthetic fixture** shaped like a
real viewpoint-robustness benchmark, and the default reasoning backend is a deterministic stand-in.

```bash
git clone https://github.com/Kenlee09/detector-doctor.git
cd detector-doctor
pip install -e ".[dev]"

dd run --out artifacts/demo        # full pipeline → artifacts/demo/report.md
dd eval                            # gold-standard replay + reliability counters
dd selftest                        # smoke test with the hallucination probe armed
dd replay --trace artifacts/demo/trace.jsonl --node
```

To point it at a real model (any OpenAI-compatible endpoint):

```bash
pip install -e ".[llm]"
export DD_API_KEY=...            # or OPENAI_API_KEY
export DD_MODEL=gpt-4o-mini      # optional
export DD_BASE_URL=...           # optional, for non-OpenAI endpoints
dd run --llm openai --out artifacts/real
```

The model only ranks and narrates; every number in the report is still computed and verified by
code, so switching backends cannot change the measured findings — only the wording and the plan
ordering.

### What you get

```
artifacts/demo/
├── report.md            # the deliverable: findings, verified claims, plan, costs, evidence
├── report.json          # the same, machine-readable
├── trace.jsonl          # one JSON line per node / tool call / LLM turn — replayable
├── state.json           # checkpoint; `--resume` continues from here
├── snapshots/           # frozen metric table (the citable source of every number)
└── experiments/         # one JSON per config hash; re-running is a cache hit
```

---

## Architecture

Eight nodes in a fixed DAG. Exactly one node contains a model, and it cannot spend money.

```
             ┌───────────────────────────── deterministic ─────────────────────────────┐
metrics ──▶  scan ──▶ attribute ──▶ retrieve ──▶ propose ──▶ verify_plan ──▶ experiment
             │        │            │          │  (LLM)      │  (guard)     │  (budgeted,
             │        │            │          │             │              │   idempotent)
             │        │            │          │             │              ▼
             │        │            │          │             │        verify_results ──▶ report
             ▼        ▼            ▼          ▼             ▼              │            │
        frozen    findings     costed     plan claims   verified?          │        report.md
        table     (typed)      options    + notes       rejected?          │        report.json
```

| stage | decides | can spend GPU | can write prose |
|---|---|---|---|
| `scan` | code — metrics read-out | no | no |
| `attribute` | code — deterministic analysis | no | no |
| `retrieve` | code — playbook index | no | no |
| `propose` | **LLM** — rank + justify | no | yes (non-binding) |
| `verify_plan` | code — evidence guard | no | no |
| `experiment` | code — idempotent, budgeted | **yes** | no |
| `verify_results` | code — evidence guard | no | no |
| `report` | code — renders verified state | no | no |

No stage can both make a claim and validate it.

### The five tools

Every capability the agent can reach is a typed tool with an explicit schema, plus two policy flags
(`side_effect`, `cost`) that the registry enforces.

| tool | purpose | side effect |
|---|---|---|
| `slice_eval` | read per-slice mAP50 / per-class AP50 | no |
| `attribute_errors` | derive typed findings: `wide_gap`, `non_monotonic`, `class_dissociation`, `rank_instability` | no |
| `retrieve_playbook` | costed remediation options for a finding kind | no |
| `run_experiment` | idempotent, budget-gated execution | **yes (GPU)** |
| `verify_claim` | re-derive a claim's number from its artifact | no |

### The evidence guard

A claim is `(text, evidence_ids, metric, value, subject)`. `ClaimVerifier` re-derives the number
from the cited artifact and rejects the claim on any of:

- `no_evidence` — nothing cited
- `evidence_missing` — the artifact or its registry entry is gone
- `evidence_drift` — the file's sha256 changed after registration
- `value_mismatch` — the artifact disagrees with the prose
- `metric_unrecoverable` — the claimed quantity cannot be re-derived

This is the component to look at first (`src/dd/tools/verify_claim.py`), and it is exercised as a
negative control on every eval run: a deliberately false claim is injected and **must** be caught.

---

## Evaluation

`dd eval` replays the pipeline against conclusions written **before** the agent existed, and reports
counters that no self-report can fake:

```json
{
  "recall": 1.0,
  "matched": ["class_dissociation", "non_monotonic", "wide_gap"],
  "missed": [],
  "hallucination_rate": 0.0,
  "probe_caught": true,
  "interventions": 0,
  "gpu_minutes": 51.5,
  "experiments_ran": 6,
  "tokens": 3612,
  "passed": true
}
```

| counter | meaning |
|---|---|
| `recall` | of the human gold findings, how many did the agent rediscover unaided |
| `hallucination_rate` | fraction of published claims that failed the guard (target: 0) |
| `probe_caught` | did the guard reject a planted lie (target: true) |
| `interventions` | times a claim had to be corrected before publication |
| `gpu_minutes` | cost of the whole audit |

High recall alone is not evidence of quality — it can just mean the fixture is easy. The pairing of
`recall` with `probe_caught` is what makes the number meaningful.

The gold set also carries a **known non-finding** (`rank_instability`), so a false positive fails the
suite too.

---

## Repository layout

```
src/dd/
├── types.py              # value objects; JSON round-trippable; config_hash = idempotency key
├── trace.py              # append-only JSONL tracing, secret redaction
├── budget.py             # hard pre-flight cost gates (GPU-min / experiments / tokens)
├── data/store.py         # metrics table access (swap for a DB, nothing above changes)
├── tools/                # the agent's entire action surface (5 typed tools)
├── orchestrator/         # DAG executor, JSON checkpointing, pipeline wiring
├── agents/               # the narrow LLM layer: diagnostician + reporter
├── llm/client.py         # provider-agnostic boundary: MockLLM | OpenAI-compatible
└── evals/gold_replay.py  # gold-standard replay + reliability counters
playbook/playbook.json    # versioned, costed remediation options (the only legal actions)
fixtures/                 # synthetic metrics table + its generator
eval/gold/                # frozen human conclusions = the label set
tests/                    # 47 tests, all offline
```

## Design decisions

Three choices that shaped everything else, written up as ADRs in [`docs/decisions.md`](docs/decisions.md):

1. **A deterministic DAG, not a ReAct loop.** Stages have hard ordering constraints and expensive
   irreversible side effects; a free agent re-launches training on every retry. The graph is
   resumable and its freedom is scoped to one node.
2. **The LLM never does arithmetic.** It ranks playbook options and writes prose. Every number in
   the report is computed by code and re-derived by the guard before publication.
3. **Idempotency by config hash.** Every experiment's artifact path is its content hash, so retries
   are cache hits. This is what makes "safe to resume" true rather than aspirational.

## What is real and what is simulated

Honesty matters more than a demo that looks bigger than it is.

| component | status |
|---|---|
| pipeline, DAG, checkpointing, budget gate, evidence guard, tracing | **real and tested** |
| metric analysis (`attribute_errors`) | **real** — operates on any table in the documented layout |
| playbook, gold labels | **real** structure, hand-authored values |
| metrics fixture | **synthetic** — reproduces the *shape* of published viewpoint findings, not the numbers |
| experiment runner | **simulated** — replays playbook `expected_gain` instead of training |
| LLM | pluggable; `MockLLM` is a deterministic policy, not a model |

The swap points are single, named seams: `data/store.py` for data, `ExperimentTool.runner` for a
real trainer, `llm/client.py` for a real model. Nothing else needs to change.

## Roadmap

- [ ] Real Ultralytics/RT-DETR runner behind `ExperimentTool.runner`
- [ ] Per-class attribution beyond top-1 dissociation (confusion-flow analysis)
- [ ] Human-in-the-loop approval gate before any `gpu`-cost option executes
- [ ] Multi-tenant: several detectors/datasets audited in one run, results diffed across commits
- [ ] Cost model learned from the lab's own GPU history instead of playbook constants

## License

MIT — see [LICENSE](LICENSE).
