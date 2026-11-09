"""Core value objects shared by tools, the orchestrator and the report writer.

Design rule: every object that can appear inside a claim is JSON round-trippable via
`to_dict` / `from_dict`, and every side-effecting experiment has a stable
`config_hash` used as its idempotency key.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

# --------------------------------------------------------------------------------------
# Slicing
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SliceKey:
    """One evaluation slice, e.g. dataset=anglebench, attr=pitch, value=75-90."""

    dataset: str
    attr: str
    value: str

    @property
    def id(self) -> str:
        return f"{self.dataset}:{self.attr}={self.value}"

    def to_dict(self) -> dict[str, Any]:
        return {"dataset": self.dataset, "attr": self.attr, "value": self.value}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SliceKey":
        return cls(dataset=d["dataset"], attr=d["attr"], value=str(d["value"]))


@dataclass(frozen=True)
class SliceMetrics:
    """Metrics for one (detector, slice) pair.

    `map50` is expressed on the 0-100 scale on purpose: this is how the source paper
    reports it, and mixing 0-1 and 0-100 in one report is a real source of bad claims.
    """

    key: SliceKey
    map50: float
    precision: float
    recall: float
    n_gt: int
    n_pred: int
    per_class: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "map50": self.map50,
            "precision": self.precision,
            "recall": self.recall,
            "n_gt": self.n_gt,
            "n_pred": self.n_pred,
            "per_class": dict(self.per_class),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SliceMetrics":
        return cls(
            key=SliceKey.from_dict(d["key"]),
            map50=float(d["map50"]),
            precision=float(d["precision"]),
            recall=float(d["recall"]),
            n_gt=int(d["n_gt"]),
            n_pred=int(d["n_pred"]),
            per_class={k: float(v) for k, v in (d.get("per_class") or {}).items()},
        )


# --------------------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable experiment spec. `config_hash()` is the idempotency key."""

    detector: str
    dataset: str
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.60
    # Sorted tuple-of-pairs rather than a dict so the object stays hashable.
    overrides: tuple[tuple[str, str], ...] = ()

    @staticmethod
    def _freeze(overrides: Mapping[str, str] | Iterable[tuple[str, str]] | None) -> tuple[tuple[str, str], ...]:
        if not overrides:
            return ()
        items = overrides.items() if isinstance(overrides, Mapping) else overrides
        return tuple(sorted((str(k), str(v)) for k, v in items))

    @classmethod
    def build(
        cls,
        detector: str,
        dataset: str,
        overrides: Mapping[str, str] | Iterable[tuple[str, str]] | None = None,
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.60,
    ) -> "ExperimentConfig":
        return cls(
            detector=detector,
            dataset=dataset,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            overrides=cls._freeze(overrides),
        )

    def config_hash(self) -> str:
        payload = {
            "detector": self.detector,
            "dataset": self.dataset,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "overrides": dict(self.overrides),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "dataset": self.dataset,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "overrides": dict(self.overrides),
            "config_hash": self.config_hash(),
        }


# --------------------------------------------------------------------------------------
# Findings, evidence, claims
# --------------------------------------------------------------------------------------


@dataclass
class Finding:
    """A machine-detected anomaly. Produced only by deterministic tools, never by an LLM."""

    id: str
    kind: str  # non_monotonic | wide_gap | class_dissociation | rank_instability
    summary: str
    severity: str  # info | warn | critical
    slice_id: str
    metric: str
    value: float
    refs: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "summary": self.summary,
            "severity": self.severity,
            "slice_id": self.slice_id,
            "metric": self.metric,
            "value": self.value,
            "refs": list(self.refs),
            "detail": dict(self.detail),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Finding":
        return cls(
            id=d["id"],
            kind=d["kind"],
            summary=d["summary"],
            severity=d.get("severity", "info"),
            slice_id=d["slice_id"],
            metric=d.get("metric", "map50"),
            value=float(d["value"]),
            refs=list(d.get("refs") or []),
            detail=dict(d.get("detail") or {}),
        )


@dataclass
class Evidence:
    """A pointer to a file on disk that a claim is allowed to cite."""

    id: str
    kind: str  # metric_table | experiment | sample_set
    path: str  # relative to the run directory
    sha256: str
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "path": self.path,
            "sha256": self.sha256,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Evidence":
        return cls(
            id=d["id"],
            kind=d["kind"],
            path=d["path"],
            sha256=d.get("sha256", ""),
            summary=d.get("summary", ""),
        )


@dataclass
class Claim:
    """A falsifiable statement.

    A claim is only printable once `verified` is True. If it names a metric and a
    slice, `ClaimVerifier` recomputes that metric from the underlying artifact and
    compares — the LLM's prose can therefore never introduce a number.
    """

    text: str
    evidence_ids: list[str] = field(default_factory=list)
    metric: str | None = None
    value: float | None = None
    slice_id: str | None = None
    subject: str | None = None  # detector the claim is about, if any
    verified: bool = False
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "evidence_ids": list(self.evidence_ids),
            "metric": self.metric,
            "value": self.value,
            "slice_id": self.slice_id,
            "subject": self.subject,
            "verified": self.verified,
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Claim":
        return cls(
            text=d["text"],
            evidence_ids=list(d.get("evidence_ids") or []),
            metric=d.get("metric"),
            value=None if d.get("value") is None else float(d["value"]),
            slice_id=d.get("slice_id"),
            subject=d.get("subject"),
            verified=bool(d.get("verified", False)),
            reasons=list(d.get("reasons") or []),
        )


# --------------------------------------------------------------------------------------
# Remediation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RemediationOption:
    """One entry from the playbook: a concrete, costed action for a symptom."""

    id: str
    symptom: str
    action: str
    layer: str  # postprocess | data | train | inference
    expected_gain: float  # AP50 points, from the playbook's own record
    cost: str  # free | cheap | gpu
    gpu_minutes: float
    rationale: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "symptom": self.symptom,
            "action": self.action,
            "layer": self.layer,
            "expected_gain": self.expected_gain,
            "cost": self.cost,
            "gpu_minutes": self.gpu_minutes,
            "rationale": self.rationale,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "RemediationOption":
        return cls(
            id=d["id"],
            symptom=d["symptom"],
            action=d["action"],
            layer=d.get("layer", "postprocess"),
            expected_gain=float(d.get("expected_gain", 0.0)),
            cost=d.get("cost", "free"),
            gpu_minutes=float(d.get("gpu_minutes", 0.0)),
            rationale=d.get("rationale", ""),
            source=d.get("source", ""),
        )
