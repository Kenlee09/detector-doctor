"""Tool 3/5 — remediation playbook lookup.

A curated, versioned table of "symptom -> costed options". This is the agent's only
source of proposed actions: the LLM may *rank* and *explain* entries, but it cannot
invent an intervention that is not in the playbook. That restriction is what makes the
"propose" stage auditable.

Each entry carries `expected_gain` in AP50 points and a `source`, so a proposal is a
citation rather than an opinion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..errors import ConfigError
from ..types import RemediationOption
from .base import ToolSpec

DEFAULT_PLAYBOOK = Path(__file__).resolve().parents[3] / "playbook" / "playbook.json"


class Playbook:
    """Symptom-indexed option store."""

    def __init__(self, options: list[RemediationOption], version: str = "0") -> None:
        self.version = version
        self.options = options
        self._by_symptom: dict[str, list[RemediationOption]] = {}
        for opt in options:
            self._by_symptom.setdefault(opt.symptom, []).append(opt)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Playbook":
        p = Path(path) if path else DEFAULT_PLAYBOOK
        if not p.exists():
            raise ConfigError(f"playbook not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return cls(
            [RemediationOption.from_dict(o) for o in payload.get("options", [])],
            version=str(payload.get("version", "0")),
        )

    def symptoms(self) -> list[str]:
        return list(self._by_symptom)

    def for_symptom(self, symptom: str) -> list[RemediationOption]:
        return list(self._by_symptom.get(symptom, []))

    def sorted_for_symptom(self, symptom: str) -> list[RemediationOption]:
        """Best value first: highest expected gain per GPU-minute, free options first."""
        def key(o: RemediationOption) -> tuple[int, float]:
            return (0 if o.cost == "free" else 1, -o.expected_gain / max(o.gpu_minutes, 0.1))

        return sorted(self.for_symptom(symptom), key=key)


class PlaybookTool:
    spec = ToolSpec(
        name="retrieve_playbook",
        description=(
            "Look up costed remediation options recorded for a given finding kind "
            "(e.g. non_monotonic, wide_gap, class_dissociation)."
        ),
        params={"symptom": "string"},
        returns="array of remediation options",
        side_effect=False,
        cost="free",
    )

    def __init__(self, playbook: Playbook | None = None) -> None:
        self.playbook = playbook or Playbook.load()

    def __call__(self, symptom: str, limit: int = 4) -> list[dict[str, Any]]:
        return [o.to_dict() for o in self.playbook.sorted_for_symptom(symptom)[:limit]]
