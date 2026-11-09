# Evaluation protocol

How this project decides whether the agent is *good*, and why those numbers can be trusted.

## 1. The problem with "it looks right"

An agent that produces a fluent robustness report is indistinguishable from one that produces a
correct one, to a human reading it once. Three failure modes need separate measurement, because a
single "accuracy" number hides all of them:

| failure mode | what it looks like | measured by |
|---|---|---|
| **miss** — the agent fails to find a real problem | report is clean, but a real gap exists | `recall` |
| **hallucination** — the agent asserts a number no artifact supports | "+2.8 AP50" with nothing behind it | `hallucination_rate`, `probe_caught` |
| **false positive** — the agent reports a problem that is not there | a finding no human agrees with | `unexpected_findings` |
| **cost blowout** — the audit is correct but unaffordable | 6 GPU-hours for a 10-minute question | `gpu_minutes`, `experiments_ran` |

## 2. The label set

`eval/gold/gold_findings.json` holds conclusions written **before** the agent existed, from reading
the metrics table by hand:

```json
{
  "version": "1.0.0",
  "findings": [
    {"id": "G-1", "kind": "non_monotonic", "note": "all detectors peak mid-band then fall"},
    {"id": "G-2", "kind": "wide_gap", "range": [25.0, 32.0]},
    {"id": "G-3", "kind": "class_dissociation"}
  ],
  "known_non_findings": [
    {"kind": "rank_instability", "note": "detector ordering is consistent across slices"}
  ],
  "thresholds": {"min_recall": 1.0, "max_hallucination_rate": 0.0},
  "gap_tolerance": 1.0
}
```

Two design points matter here:

- **`known_non_findings` is part of the label set.** `rank_instability` is a real detector of the
  analysis code, and on this fixture it must stay silent. A run that reports it fails. Without this,
  the eval only rewards finding more things.
- **`range` + `gap_tolerance` instead of an exact number.** The gold asserts the *magnitude* is
  right to ±1.0 AP50, which is the precision a human reading the table can honestly claim. Tighter
  would be false authority.

## 3. The negative control

Recall alone is gameable: a suite that only checks "did it find the three things" can be passed by a
system that finds everything, including things that are not there.

So every eval run injects a **probe** — a fabricated claim, emitted in-process before the report:

> "fasterrcnn_r50 reaches 88.0 AP50 in the hardest pitch bin, outperforming every other model."

The actual value in the artifact is 41.23. The evidence guard must reject it:
`value_mismatch: claim says 88.00, artifact gives 41.23`. The claim is rendered in the report's
rejected section, with strikethrough, as a visible receipt.

```
probe_caught: true   →  the guard is alive
probe_caught: false  →  the "0 hallucinations" number means nothing
```

This is the single most valuable assertion in the repo. A refactor that weakens
`ClaimVerifier` fails CI immediately, instead of producing a demo that looks better than before.

## 4. Counters and thresholds

`dd eval` exits non-zero unless **all** of these hold:

| counter | threshold | why |
|---|---|---|
| `recall` | `>= 1.0` | three gold findings, all must be found; the fixture is small enough that missing one is a bug, not bad luck |
| `hallucination_rate` | `== 0.0` | a claim that fails verification must never be published, so the published set must be perfect |
| `probe_caught` | `true` | the negative control must fire |
| `unexpected_findings` | `== []` | no false positives, including the known non-finding |
| `gap_within_tolerance` | `true` | the reported magnitude is within ±1.0 AP50 of gold |
| `interventions` | `== 0` | no human had to correct a claim before publication |

`gpu_minutes`, `experiments_ran`, `tokens` and `wall_seconds` are reported but not gated: they are
the cost side of the ledger, and the right ceiling depends on the deployment.

## 5. Cost accounting

Cost is part of correctness here, not an operational afterthought. Every run reports:

```
gpu_minutes: 51.5        (of a 90 cap)
experiments_ran: 6
experiments_cached: 0
tokens: 3612
```

The 51.5 GPU-minutes is the plan's own estimate — 0.2 + 0.3 + 1.0 + 14.0 + 16.0 + 20.0 — so the
number in the report is derived from the playbook rather than measured from a real GPU. That gap is
deliberate and documented (see README "What is real and what is simulated"); the accounting,
gating and cache-hit logic around it are real and tested.

`experiments_cached` is the idempotency receipt: on a second run of the same config, it must be 6
and `experiments_ran` must be 0, with no additional GPU-minutes reserved.

## 6. What this protocol does not establish

Being explicit about the boundary is part of the design:

- **The fixture is synthetic.** It reproduces the *shape* of a published viewpoint-robustness
  result (mid-band peak, ~25–32 AP50 spread, class dissociation), not its numbers. Recall on it
  demonstrates the pipeline works, not that the analysis is novel.
- **The experiment runner simulates.** Gains are replayed from the playbook's `expected_gain`, so
  "+4.6 AP50" verifies that the *plumbing* carries numbers correctly end to end. It does not
  establish that class-balanced loss actually buys 4.6 points.
- **The default LLM is a deterministic stand-in.** `MockLLM` exercises the schema, the option
  whitelist and the failure paths — not model quality.

Each of these is a named seam with a single swap point (`data/store.py`, `ExperimentTool.runner`,
`llm/client.py`), and the eval keeps working unchanged after each swap. That is the point of
measuring plumbing and judgement separately.

## 7. Running it

```bash
dd eval                          # writes artifacts/eval/, exits 0 on pass, 2 on failure
python eval/run_gold_replay.py    # same thing, without installing
python -m pytest -q               # 47 offline tests
```

CI runs all three in `.github/workflows/ci.yml`, on every push and pull request.
