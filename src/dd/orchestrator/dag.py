"""Deterministic DAG executor.

This is the "controlled" half of "controlled agent". The pipeline is a fixed graph with
explicit dependencies, not a free-form loop that decides its own next step. Reasons:

* the stages have hard ordering constraints (you cannot attribute before you evaluate);
* every stage that mixes them has an expensive, irreversible side effect (a GPU run);
* a fixed graph is resumable: a crash mid-run resumes at the first unfinished node
  instead of re-planning from scratch and re-spending quota.

The LLM still gets freedom — but only *inside* a node, over a bounded decision, with its
output schema-checked. Freedom of action lives in the graph; freedom of judgement lives
in the node.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from ..errors import ConfigError

NodeFn = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class Node:
    name: str
    fn: NodeFn
    deps: tuple[str, ...] = ()
    description: str = ""

    def __call__(self, ctx: dict[str, Any]) -> dict[str, Any]:
        return self.fn(ctx)


@dataclass
class DAG:
    """A validated, topologically ordered pipeline."""

    nodes: list[Node] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._by_name: dict[str, Node] = {}
        for node in self.nodes:
            if node.name in self._by_name:
                raise ConfigError(f"duplicate node name: {node.name}")
            self._by_name[node.name] = node
        for node in self.nodes:
            for dep in node.deps:
                if dep not in self._by_name:
                    raise ConfigError(f"node {node.name} depends on unknown node {dep}")
        self.order()  # raises on cycles

    def order(self) -> list[Node]:
        """Stable topological order (declaration order breaks ties)."""
        indegree = {n.name: len(n.deps) for n in self.nodes}
        remaining = {n.name: n for n in self.nodes}
        ordered: list[Node] = []
        while remaining:
            ready = [n for n in self.nodes if n.name in remaining and indegree[n.name] == 0]
            if not ready:
                raise ConfigError(
                    "dependency cycle among nodes: " + ", ".join(sorted(remaining))
                )
            for node in ready:
                ordered.append(node)
                remaining.pop(node.name)
                for other in self.nodes:
                    if node.name in other.deps and other.name in remaining:
                        indegree[other.name] -= 1
        return ordered

    def names(self) -> list[str]:
        return [n.name for n in self.order()]

    def run(
        self,
        ctx: dict[str, Any],
        completed: Iterable[str] = (),
        on_start: Callable[[Node], None] | None = None,
        on_done: Callable[[Node, dict[str, Any]], None] | None = None,
        on_skip: Callable[[Node], None] | None = None,
    ) -> dict[str, Any]:
        """Execute nodes in order, skipping those already complete.

        A node returns a dict of context updates, which are merged before the next node.
        A raise propagates immediately, leaving `completed` as the durable checkpoint.
        """
        done = set(completed)
        for node in self.order():
            if node.name in done:
                if on_skip is not None:
                    on_skip(node)
                continue
            if on_start is not None:
                on_start(node)
            updates = node(ctx)
            if not isinstance(updates, dict):
                # Strict on purpose: a node that forgets to return would otherwise look
                # like a successful no-op and silently truncate the pipeline.
                raise ConfigError(
                    f"node {node.name} must return a dict of context updates, "
                    f"got {type(updates).__name__}"
                )
            ctx.update(updates)
            done.add(node.name)
            if on_done is not None:
                on_done(node, ctx)
        return ctx


def build_dag(nodes: Sequence[Node]) -> DAG:
    return DAG(list(nodes))
