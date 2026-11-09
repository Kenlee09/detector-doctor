"""DAG: ordering, validation, checkpoint resume."""

from __future__ import annotations

import pytest

from conftest import REPO_ROOT  # noqa: F401  (ensures src on sys.path)

from dd.errors import ConfigError
from dd.orchestrator.dag import DAG, Node


def test_topological_order_respects_deps():
    calls: list[str] = []

    def make(name: str):
        def fn(ctx):
            calls.append(name)
            return {name: True}

        return fn

    dag = DAG(
        [
            Node("c", make("c"), ("a", "b")),
            Node("b", make("b"), ("a",)),
            Node("a", make("a")),
        ]
    )
    assert dag.names() == ["a", "b", "c"]
    dag.run({})
    assert calls == ["a", "b", "c"]


def test_unknown_dependency_is_rejected():
    with pytest.raises(ConfigError):
        DAG([Node("a", lambda ctx: {}, ("ghost",))])


def test_cycle_is_rejected():
    with pytest.raises(ConfigError):
        DAG([Node("a", lambda ctx: {}, ("b",)), Node("b", lambda ctx: {}, ("a",))])


def test_resume_skips_completed_nodes():
    ran: list[str] = []

    def make(name: str):
        def fn(ctx):
            ran.append(name)
            return {}

        return fn

    dag = DAG([Node("a", make("a")), Node("b", make("b"), ("a",)), Node("c", make("c"), ("b",))])
    skipped: list[str] = []
    dag.run({}, completed=["a", "b"], on_skip=lambda n: skipped.append(n.name))
    assert ran == ["c"]
    assert skipped == ["a", "b"]


def test_node_must_return_dict():
    dag = DAG([Node("a", lambda ctx: None)])
    with pytest.raises(ConfigError):
        dag.run({})


def test_context_updates_are_merged():
    dag = DAG(
        [
            Node("a", lambda ctx: {"x": 1}),
            Node("b", lambda ctx: {"y": ctx["x"] + 1}, ("a",)),
        ]
    )
    ctx = dag.run({})
    assert ctx == {"x": 1, "y": 2}
