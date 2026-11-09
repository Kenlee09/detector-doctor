"""Run state and checkpoints.

Everything that survives a process must be JSON. That constraint is deliberate: it
forces the run to be inspectable with `cat`, diffable between versions, and replayable
from the trace. Objects are rebuilt at node boundaries instead of being pickled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STATE_FILE = "state.json"


@dataclass
class RunState:
    run_id: str
    completed: list[str] = field(default_factory=list)
    ctx: dict[str, Any] = field(default_factory=dict)

    # -- persistence ---------------------------------------------------------------
    def save(self, out_dir: str | Path) -> Path:
        path = Path(out_dir) / STATE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"run_id": self.run_id, "completed": list(self.completed), "ctx": self.ctx},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, out_dir: str | Path) -> "RunState":
        path = Path(out_dir) / STATE_FILE
        if not path.exists():
            raise FileNotFoundError(f"no run state at {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            run_id=payload["run_id"],
            completed=list(payload.get("completed") or []),
            ctx=dict(payload.get("ctx") or {}),
        )

    @classmethod
    def load_or_new(cls, out_dir: str | Path, run_id: str) -> "RunState":
        try:
            return cls.load(out_dir)
        except FileNotFoundError:
            return cls(run_id=run_id)
