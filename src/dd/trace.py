"""Append-only JSONL tracing.

Every tool call, door decision and LLM exchange is written as one JSON line. The trace
is the audit surface: a run can be replayed from it without touching a GPU, and any
claim in the final report can be walked back to the call that produced it.

Secrets are redacted on the way in, not on the way out.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

SECRET_KEY = re.compile(r"(token|secret|password|api[_-]?key|authorization)", re.IGNORECASE)


def new_run_id() -> str:
    """Sortable, collision-safe run id."""
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:6]


def redact(payload: Any) -> Any:
    """Recursively mask values whose key looks like a credential."""
    if isinstance(payload, dict):
        out = {}
        for k, v in payload.items():
            if isinstance(k, str) and SECRET_KEY.search(k):
                out[k] = "***"
            else:
                out[k] = redact(v)
        return out
    if isinstance(payload, (list, tuple)):
        return [redact(v) for v in payload]
    return payload


class TraceWriter:
    """Writes `<run_dir>/trace.jsonl`, flush-per-event so a crash still leaves evidence."""

    def __init__(self, path: str | os.PathLike[str], run_id: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._fh = self.path.open("a", encoding="utf-8")

    def emit(self, event: str, **payload: Any) -> dict[str, Any]:
        record = {
            "ts": time.time(),
            "run_id": self.run_id,
            "event": event,
            "payload": redact(payload),
        }
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()
        return record

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_trace(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Load a trace file into memory (used by `dd replay` and by tests)."""
    events: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def iter_events(path: str | os.PathLike[str], event: str | None = None) -> Iterator[dict[str, Any]]:
    for record in read_trace(path):
        if event is None or record.get("event") == event:
            yield record


def summarize(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Cheap post-run summary: event counts, tool calls, LLM turns, wall time."""
    events = read_trace(path)
    counts: dict[str, int] = {}
    for e in events:
        counts[e.get("event", "?")] = counts.get(e.get("event", "?"), 0) + 1
    if not events:
        return {"events": 0, "counts": {}, "wall_seconds": 0.0}
    return {
        "events": len(events),
        "counts": counts,
        "wall_seconds": round(events[-1]["ts"] - events[0]["ts"], 3),
    }
