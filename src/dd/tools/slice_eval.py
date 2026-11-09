"""Tool 1/5 — sliced evaluation read-out.

Wraps a metrics table as a callable. Deliberately narrow: it filters and returns
numbers. It never samples images, never calls a model, never renders text. Keeping the
deterministic part deterministic is what makes the agent's claims checkable.
"""

from __future__ import annotations

from typing import Any

from ..data.store import MetricsStore
from .base import ToolSpec


class SliceEvalTool:
    spec = ToolSpec(
        name="slice_eval",
        description=(
            "Read per-slice detection metrics (mAP50, precision, recall, per-class AP50) "
            "for one detector over an attribute such as pitch angle."
        ),
        params={"detector": "string", "slices": "array"},
        returns="array of slice metric objects",
        side_effect=False,
        cost="free",
    )

    def __init__(self, store: MetricsStore) -> None:
        self.store = store

    def __call__(self, detector: str, slices: list[str] | None = None) -> list[dict[str, Any]]:
        wanted = slices or self.store.slices
        out: list[dict[str, Any]] = []
        for s in wanted:
            if not self.store.has(detector, s):
                continue
            out.append(self.store.get(detector, s).to_dict())
        return out
