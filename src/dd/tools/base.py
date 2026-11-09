"""Typed tool layer.

Every capability the agent can reach is declared here with an explicit parameter
schema, an explicit return description, and two policy flags: `side_effect` and `cost`.
The registry can therefore export OpenAI-style function schemas *and* answer the
question "is this call allowed to touch a GPU?" without inspecting the call site.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from ..errors import ToolError


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    params: dict[str, str] = field(default_factory=dict)  # name -> json type
    returns: str = "object"
    side_effect: bool = False
    cost: str = "free"  # free | llm | gpu

    def to_json_schema(self) -> dict[str, Any]:
        """OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        k: {"type": v} for k, v in self.params.items()
                    },
                    "required": list(self.params),
                    "additionalProperties": False,
                },
            },
        }


@runtime_checkable
class Tool(Protocol):
    spec: ToolSpec

    def __call__(self, **kwargs: Any) -> Any: ...


class ToolRegistry:
    """Name -> callable, with tracing and policy in one place."""

    def __init__(self, trace: Any | None = None) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}
        self._specs: dict[str, ToolSpec] = {}
        self.trace = trace

    def register(self, tool: Any) -> None:
        spec = getattr(tool, "spec", None)
        if not isinstance(spec, ToolSpec):
            raise ToolError(f"{tool!r} has no ToolSpec")
        if spec.name in self._tools:
            raise ToolError(f"duplicate tool name: {spec.name}")
        self._tools[spec.name] = tool
        self._specs[spec.name] = spec

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolError(f"unknown tool: {name}") from exc

    def spec(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise ToolError(f"unknown tool: {name}") from exc

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def json_schemas(self) -> list[dict[str, Any]]:
        return [s.to_json_schema() for s in self._specs.values()]

    def call(self, name: str, /, **kwargs: Any) -> Any:
        """Invoke a tool, emitting `tool_call` / `tool_result` (or `tool_error`) events."""
        spec = self.spec(name)
        if self.trace is not None:
            self.trace.emit("tool_call", tool=name, cost=spec.cost, args=kwargs)
        try:
            result = self.get(name)(**kwargs)
        except Exception as exc:  # noqa: BLE001 - traced, then re-raised unchanged
            if self.trace is not None:
                self.trace.emit(
                    "tool_error", tool=name, error=f"{type(exc).__name__}: {exc}"
                )
            raise
        if self.trace is not None:
            self.trace.emit("tool_result", tool=name, ok=True)
        return result
