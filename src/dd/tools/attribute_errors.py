"""Tool 2/5 — deterministic error attribution.

This is the analytical core of the agent and, on purpose, it contains **no LLM call**.
Slice metrics go in; typed `Finding` objects come out. Every number in the final
report originates here (or in an experiment artifact), which is what lets
`verify_claim` recompute it later.

The four detectors implemented here mirror what a robustness audit actually needs:

* `wide_gap`            — the spread between the best and worst slice domain
* `non_monotonic`       — accuracy that rises then falls as the viewpoint degrades
* `class_dissociation`  — the top-1 class at the best slice is not the top-1 at the worst
* `rank_instability`    — detector ordering flips between slices (i.e. no global winner)
"""

from __future__ import annotations

from typing import Any, Sequence

from ..data.store import MetricsStore
from ..types import Finding
from .base import ToolSpec

EPS = 1e-9


def _is_monotonic(values: Sequence[float]) -> bool:
    if len(values) < 2:
        return True
    non_decreasing = all(b >= a - EPS for a, b in zip(values, values[1:]))
    non_increasing = all(b <= a + EPS for a, b in zip(values, values[1:]))
    return non_decreasing or non_increasing


def _ranks(values: Sequence[float]) -> list[int]:
    """1 = best (highest AP50). Ties fall back to position, keeping the result stable."""
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    ranks = [0] * len(values)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def _top_class(per_class: dict[str, float]) -> str | None:
    if not per_class:
        return None
    return max(per_class.items(), key=lambda kv: kv[1])[0]


class AttributeErrorsTool:
    spec = ToolSpec(
        name="attribute_errors",
        description=(
            "Analyse slice metrics of one or more detectors and return typed robustness "
            "findings (wide_gap, non_monotonic, class_dissociation, rank_instability)."
        ),
        params={"detectors": "array"},
        returns="array of Finding objects",
        side_effect=False,
        cost="free",
    )

    def __init__(self, store: MetricsStore) -> None:
        self.store = store

    def __call__(self, detectors: list[str] | None = None) -> list[Finding]:
        dets = detectors or self.store.detectors
        findings: list[Finding] = []
        findings.extend(self._wide_gap(dets))
        findings.extend(self._non_monotonic(dets))
        findings.extend(self._class_dissociation(dets))
        findings.extend(self._rank_instability(dets))
        return findings

    # -- individual detectors ------------------------------------------------------
    def _wide_gap(self, dets: list[str]) -> list[Finding]:
        per_det: dict[str, dict[str, Any]] = {}
        for det in dets:
            table = self.store.table(det)
            if len(table) < 2:
                continue
            worst_slice = min(table, key=lambda s: table[s])
            best_slice = max(table, key=lambda s: table[s])
            gap = table[best_slice] - table[worst_slice]
            per_det[det] = {
                "best_slice": best_slice,
                "best_map50": round(table[best_slice], 2),
                "worst_slice": worst_slice,
                "worst_map50": round(table[worst_slice], 2),
                "gap": round(gap, 2),
            }
        if not per_det:
            return []
        gaps = [v["gap"] for v in per_det.values()]
        lo, hi = min(gaps), max(gaps)
        worst_det = max(per_det, key=lambda d: per_det[d]["gap"])
        return [
            Finding(
                id="F-wide_gap",
                kind="wide_gap",
                summary=(
                    f"Best-to-worst viewpoint-domain gap spans {lo:.1f}-{hi:.1f} AP50 points "
                    f"across {len(per_det)} detectors, far exceeding the spread between detectors."
                ),
                severity="critical",
                slice_id=f"{self.store.dataset}:{self.store.attr}=worst",
                metric="map50",
                value=hi,
                refs=[f"metric_table#{self.store.dataset}"],
                detail={"per_detector": per_det, "gap_min": lo, "gap_max": hi, "worst_detector": worst_det},
            )
        ]

    def _non_monotonic(self, dets: list[str]) -> list[Finding]:
        offenders: dict[str, dict[str, Any]] = {}
        for det in dets:
            table = self.store.table(det)
            series = [table[s] for s in self.store.slices if s in table]
            if not _is_monotonic(series):
                peak_idx = series.index(max(series))
                offenders[det] = {
                    "series": [round(v, 2) for v in series],
                    "peak_slice": self.store.slices[peak_idx],
                    "tail_drop": round(max(series) - series[-1], 2),
                }
        if not offenders:
            return []
        worst = max(offenders.items(), key=lambda kv: kv[1]["tail_drop"])
        return [
            Finding(
                id="F-non_monotonic",
                kind="non_monotonic",
                summary=(
                    f"Accuracy does not degrade monotonically with viewpoint: {len(offenders)}/"
                    f"{len(dets)} detectors peak in a middle slice domain and then fall, "
                    f"worst tail drop {worst[1]['tail_drop']:.1f} AP50."
                ),
                severity="critical",
                slice_id=f"{self.store.dataset}:{self.store.attr}={worst[1]['peak_slice']}",
                metric="map50",
                value=float(worst[1]["tail_drop"]),
                refs=[f"metric_table#{self.store.dataset}"],
                detail={"offending_detectors": offenders},
            )
        ]

    def _class_dissociation(self, dets: list[str]) -> list[Finding]:
        cases: dict[str, dict[str, Any]] = {}
        for det in dets:
            rows = self.store.rows(det)
            if not rows:
                continue
            best = max(rows, key=lambda r: r.map50)
            worst = min(rows, key=lambda r: r.map50)
            best_top = _top_class(dict(best.per_class))
            worst_top = _top_class(dict(worst.per_class))
            if best_top and worst_top and best_top != worst_top:
                cases[det] = {
                    "best_slice": best.key.value,
                    "best_top_class": best_top,
                    "worst_slice": worst.key.value,
                    "worst_top_class": worst_top,
                    "best_per_class": dict(best.per_class),
                    "worst_per_class": dict(worst.per_class),
                }
        if not cases:
            return []
        return [
            Finding(
                id="F-class_dissociation",
                kind="class_dissociation",
                summary=(
                    f"Class dissociation: for {len(cases)}/{len(dets)} detectors the strongest "
                    f"class at the best slice is not the strongest class at the worst slice."
                ),
                severity="warn",
                slice_id=f"{self.store.dataset}:{self.store.attr}=mixed",
                metric="affected_detectors",
                value=float(len(cases)),
                refs=[f"metric_table#{self.store.dataset}"],
                detail={"cases": cases},
            )
        ]

    def _rank_instability(self, dets: list[str]) -> list[Finding]:
        if len(dets) < 2:
            return []
        inversions: list[dict[str, Any]] = []
        ref_slice = self.store.slices[0]
        ref_ranks = _ranks([self.store.table(d)[ref_slice] for d in dets if ref_slice in self.store.table(d)])
        ref_dets = [d for d in dets if ref_slice in self.store.table(d)]
        for s in self.store.slices[1:]:
            present = [d for d in dets if s in self.store.table(d)]
            if [d for d in present if d in ref_dets] != ref_dets:
                continue
            ranks = _ranks([self.store.table(d)[s] for d in present])
            if ranks != ref_ranks:
                inversions.append(
                    {
                        "slice": s,
                        "ranks": {d: r for d, r in zip(present, ranks)},
                        "reference_slice": ref_slice,
                    }
                )
        if not inversions:
            return []
        return [
            Finding(
                id="F-rank_instability",
                kind="rank_instability",
                summary=(
                    f"Detector ordering is not stable across slices: {len(inversions)} slice(s) "
                    f"reverse the ranking observed at {ref_slice}."
                ),
                severity="warn",
                slice_id=f"{self.store.dataset}:{self.store.attr}=multi",
                metric="rank_inversions",
                value=float(len(inversions)),
                refs=[f"metric_table#{self.store.dataset}"],
                detail={"inversions": inversions, "reference_slice": ref_slice},
            )
        ]
