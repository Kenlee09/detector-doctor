# Architecture

## 1. Why this shape

Detector Doctor answers one question: *across the domain slices this detector will actually meet,
where does it break, and what is the cheapest change that fixes it?* Three properties of that
question decide the architecture.

- **The analysis is deterministic.** Wide-gap detection, non-monotonicity, class dissociation and
  rank instability are all arithmetic over a metrics table. Delegating them to a model would trade
  reproducibility for nothing.
- **The expensive step is irreversible.** Training or re-inference costs GPU-minutes. It must be
  gated before it starts, not after.
- **The output is a claim.** A report that says "+2.8 AP50" is either backed by an artifact or it is
  a liability. Verification is a first-class stage, not a review habit.

## 2. Data flow

```
                 ┌──────────────────────────── run directory ──────────────────────────┐
 metrics.json ──▶│ scan      freeze table → snapshots/, register evidence               │
 (fixture or    │    │                                                                │
  real export)  │    ▼                                                                │
                │ attribute 4 finding kinds → Finding[] (typed, JSON round-trippable)  │
                │    │                                                                │
                │    ▼                                                                │
                │ retrieve  playbook → costed RemediationOption[] per finding          │
                │    │                                                                │
                │    ▼                              ┌──── the only model in the system ─┐│
                │ propose  LLM ranks options ────────┘ schema-validated, non-binding   ││
                │    │                                                                │
                │    ▼                                                                │
                │ verify_plan  evidence guard on plan claims → verified / rejected    │
                │    │                                                                │
                │    ▼                                                                │
                │ experiment  per option: config_hash → cache hit? or budget gate → run│
                │    │                                                                │
                │    ▼                                                                │
                │ verify_results  evidence guard on measured outcomes                 │
                │    │                                                                │
                │    ▼                                                                │
                │ report  render.md + report.json from verified state only            │
                └─────────────────────────────────────────────────────────────────────┘
```

Each node returns a dict of context updates; the executor merges them, appends to `trace.jsonl` and
writes `state.json`. A resumed run loads the state and skips completed nodes.

## 3. The evidence guard (the load-bearing component)

```python
claim = Claim(
    text="yolov8n peaks at 45-60° then falls 25.0 AP50",
    evidence_ids=["metric_table#anglebench"],
    metric="tail_drop",      # what to re-derive
    value=25.0,              # what the prose asserts
    subject="yolov8n",       # which row
)
```

`ClaimVerifier.verify()` resolves each evidence id, re-reads the artifact, re-computes the number
from it, and compares with `TOLERANCE = 0.05`. Rejection reasons are explicit and returned to the
caller, never swallowed:

| reason | trigger |
|---|---|
| `no_evidence` | claim cites nothing |
| `evidence_missing` | id not registered, or file deleted |
| `evidence_drift` | sha256 differs from registration time |
| `value_mismatch` | artifact contradicts the prose |
| `metric_unrecoverable` | the metric cannot be re-derived from the artifact |

Two derived metrics exist because raw per-slice numbers are not enough to state a conclusion:
`tail_drop` (`max(series) − series[-1]`, the non-monotonicity magnitude) and the per-slice
`map50` lookup. Both are recomputed, never trusted from the caller.

## 4. The tool boundary

Tools are the *only* way the agent affects the world. Each carries a schema and two policy flags.

```python
ToolSpec(
    name="run_experiment",
    description="Run one playbook intervention and record its artifact.",
    params={"detector": "str", "overrides": "dict", "gpu_minutes": "float"},
    returns="dict",
    side_effect=True,        # registry refuses to call it in read-only runs
    cost="gpu",
)
```

The registry emits `tool_call` / `tool_result` / `tool_error` around every invocation, so the trace
is a complete record of what the agent touched. Adding a capability means adding a tool with a
schema, not editing a prompt.

## 5. Budget

Three ceilings, checked **before** the side effect, so a rejection leaves no artifact behind:

```python
budget.reserve_experiment(gpu_minutes=18.0)   # raises BudgetExceeded, atomically
```

`reserve_experiment` is all-or-nothing: either both the GPU-minutes counter and the experiment
counter advance, or neither does. Rejections are recorded with a reason, which the pipeline turns
into a `budget_rejected` row rather than a crash — the audit still completes and still reports.

## 6. Failure model

| failure | behaviour |
|---|---|
| malformed LLM JSON | schema check fails → run aborts before any GPU spend |
| LLM proposes an unlisted option | dropped with a `plan_rejected_option` trace event |
| LLM reports a false cost | ignored; cost is resolved from the playbook |
| claim cites a missing artifact | `EvidenceMissing`; claim cannot be published |
| artifact changed after citation | `evidence_drift`; claim rejected |
| budget exhausted mid-run | `budget_rejected` rows; report still produced and still honest |
| process killed | `--resume` continues from `state.json`; completed experiments are cache hits |

## 7. Extension seams

| to change | edit | nothing above it changes |
|---|---|---|
| the data source | `data/store.py` (`MetricsStore.load`) | analysis, guard, report |
| a real detector runner | `ExperimentTool.runner` | pipeline, budget, cache |
| the model | `llm/client.py` (`build_llm`) | everything |
| the remediation library | `playbook/playbook.json` | no code |
| the analysis | `tools/attribute_errors.py` | add a finding kind + a gold label |
