# Architecture Decision Records

Short, dated records of the choices that constrain everything else. Format: context →
decision → consequences → what would change my mind.

---

## ADR-001 — A deterministic DAG, not a ReAct loop

**Status:** accepted

**Context.** The tempting shape for this product is a ReAct agent with the five tools exposed and a
prompt that says "diagnose this model". It demos well in ten minutes and fails in production for
three specific reasons:

1. The stages have hard ordering constraints. You cannot attribute error to a slice before the
   slice has been evaluated; you cannot verify a claim before the artifact it cites exists.
2. Every stage that mixes them has an expensive, irreversible side effect — a GPU-minutes spend.
   A free-running loop that retries on a parse error re-launches training.
3. A loop that re-plans from scratch on resume cannot be resumed at all. After a crash the run
   starts over and the quota is spent twice.

**Decision.** The pipeline is a fixed DAG of eight nodes with explicit dependencies and per-node
JSON checkpoints. Freedom of *action* lives in the graph and is fixed at build time; freedom of
*judgement* lives inside exactly one node (`propose`) and is bounded by a schema.

**Consequences.**

- `--resume` is trustworthy: completed nodes are skipped, `state.json` is the durable record.
- A tool error cannot change the shape of the run; it either fails the node or is handled in it.
- The cost profile is knowable in advance from the graph, not discovered at runtime.
- Adding a genuinely new capability means editing the graph — a deliberate friction.

**What would change my mind.** A task whose next step is unpredictable *and* cheap *and* reversible
(e.g. open-ended document research). None of those hold here: the next step is predictable and the
expensive step is irreversible.

---

## ADR-002 — The LLM never does arithmetic

**Status:** accepted

**Context.** The failure mode that kills trust in an analysis agent is not a wrong plan; it is a
fluent, plausible, wrong number. An LLM asked to summarise a metrics table will interpolate values
it was never given, and it will do so in prose that reads better than the correct answer.

**Decision.** A strict split of labour:

| who | does |
|---|---|
| code (tools) | computes every metric, derives every finding, writes every artifact |
| LLM | ranks playbook options, writes connective prose and justification |
| guard (code) | re-derives each claim's number from a hashed artifact before publication |

The LLM's output is validated against a JSON schema and every proposed option is checked against
the playbook: an intervention that is not in the playbook is dropped, not repaired.

**Consequences.**

- Report numbers are traceable to a file and a sha256.
- The `expected_gain` a model reports is ignored; the plan's cost is resolved from the playbook, so
  a model that under-reports cost cannot buy budget headroom (there is a test for this).
- The model cannot contribute a novel intervention. That is a real capability ceiling, accepted
  deliberately: an unauditable action is worse than a missing one.

**What would change my mind.** A domain where the model must reason over unstructured evidence with
no ground-truth table available. Then the guard becomes "quote the span" rather than
"recompute the number".

---

## ADR-003 — Idempotency by config hash, not by run id

**Status:** accepted

**Context.** Retries happen: a schema error, a rate limit, a killed process. If an experiment's
identity is tied to a run id, the retry produces a second artifact for the same configuration —
two numbers for one experiment, and no way to tell which the report used.

**Decision.** `ExperimentConfig.config_hash()` — a sha256 over the normalised configuration
(detector, dataset, imgsz, conf, iou, sorted overrides) — names the artifact. If
`experiments/<hash>.json` exists, the call is a cache hit: status `cached`, zero GPU-minutes
reserved, zero experiments counted.

**Consequences.**

- Retries are free and safe; `--resume` cannot double-spend.
- Override ordering cannot fork the cache (sorted tuple of pairs, tested).
- The artifact is self-describing: it embeds its own config and hash.
- Storage grows monotonically with distinct configs; needs a retention policy at scale.

**What would change my mind.** Non-deterministic experiments (stochastic training where two runs of
the same config legitimately differ). Then the key becomes `(hash, seed, attempt)` and the report
must aggregate across seeds rather than dedupe them.

---

## ADR-004 — Ship a negative control, not just a benchmark

**Status:** accepted

**Context.** An eval suite that only measures "did it get the right answer" cannot distinguish an
agent that is careful from one that is lucky. Recall on an easy fixture is not evidence.

**Decision.** Every eval run plants a deliberately false claim in the pipeline and asserts that the
evidence guard rejects it (`probe_caught`). The gold set also records a **known non-finding**
(`rank_instability`) that the analysis must *not* report.

**Consequences.**

- A regression that weakens the guard fails CI instead of silently improving the demo.
- False positives are scored, not ignored.
- The honest headline metric is the pair `(recall, probe_caught)`, which is harder to game than
  either alone.
