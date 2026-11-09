"""Read-only access to a metrics table.

In production this reads the JSON that `eval_suite` emits after a real Ultralytics /
RT-DETR evaluation pass. In this repo it reads a synthetic fixture that reproduces the
shape and magnitude of the published AngleDrone findings, so the whole agent is
runnable and testable with no GPU. The seam is this class: swap the backing file for a
database and nothing above it changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..errors import ConfigError
from ..types import SliceKey, SliceMetrics


class MetricsStore:
    """Loads a metrics table and serves slices.

    Expected JSON layout::

        {
          "meta":       {"dataset": "...", "attr": "pitch", "scale": "AP50 0-100"},
          "detectors":  ["yolov8n", ...],
          "slices":     ["0-15", "15-30", ...],
          "values":     {"<detector>": {"<slice>": {"map50": .., "precision": ..,
                                                    "recall": .., "n_gt": ..,
                                                    "n_pred": .., "per_class": {..}}}}
        }
    """

    def __init__(self, payload: Mapping[str, Any], source: str = "<memory>") -> None:
        self._payload = dict(payload)
        self.source = source
        self.dataset: str = self._payload.get("meta", {}).get("dataset", "unknown")
        self.attr: str = self._payload.get("meta", {}).get("attr", "pitch")
        self.detectors: list[str] = list(self._payload.get("detectors", []))
        self.slices: list[str] = [str(s) for s in self._payload.get("slices", [])]
        self._values: dict[str, dict[str, Any]] = self._payload.get("values", {})
        if not self.detectors or not self.slices:
            raise ConfigError(f"metrics store {source} has no detectors or slices")

    # -- construction --------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "MetricsStore":
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"metrics file not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            return cls(json.load(fh), source=str(p))

    # -- queries -------------------------------------------------------------------
    def slice_keys(self) -> list[SliceKey]:
        return [SliceKey(self.dataset, self.attr, s) for s in self.slices]

    def has(self, detector: str, slice_value: str) -> bool:
        return slice_value in self._values.get(detector, {})

    def get(self, detector: str, slice_value: str) -> SliceMetrics:
        try:
            row = self._values[detector][slice_value]
        except KeyError as exc:
            raise ConfigError(f"no metrics for detector={detector!r} slice={slice_value!r}") from exc
        return SliceMetrics(
            key=SliceKey(self.dataset, self.attr, slice_value),
            map50=float(row["map50"]),
            precision=float(row.get("precision", 0.0)),
            recall=float(row.get("recall", 0.0)),
            n_gt=int(row.get("n_gt", 0)),
            n_pred=int(row.get("n_pred", 0)),
            per_class={k: float(v) for k, v in (row.get("per_class") or {}).items()},
        )

    def rows(self, detector: str) -> list[SliceMetrics]:
        return [self.get(detector, s) for s in self.slices if self.has(detector, s)]

    def table(self, detector: str) -> dict[str, float]:
        """Compact {slice: map50} view — this, not the full rows, is what reaches the LLM."""
        return {s: self.get(detector, s).map50 for s in self.slices if self.has(detector, s)}

    def classes(self) -> list[str]:
        seen: list[str] = []
        for det in self.detectors:
            for sl in self.slices:
                if self.has(det, sl):
                    for cls in (self._values[det][sl].get("per_class") or {}):
                        if cls not in seen:
                            seen.append(cls)
        return seen

    def iter_slices(self) -> Iterable[tuple[str, SliceKey]]:
        for s in self.slices:
            yield s, SliceKey(self.dataset, self.attr, s)

    def snapshot(self) -> dict[str, Any]:
        """Frozen copy of the full table.

        This is what gets written into the run directory and registered as citable
        evidence: claims are checked against *this* byte-for-byte, not against whatever
        the store points at later.
        """
        return json.loads(json.dumps(self._payload))
