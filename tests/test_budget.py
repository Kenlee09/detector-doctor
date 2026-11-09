"""Budget: hard pre-flight gates."""

from __future__ import annotations

import pytest

from conftest import REPO_ROOT  # noqa: F401

from dd.budget import Budget
from dd.errors import BudgetExceeded


def test_reservation_commits_atomically():
    b = Budget(gpu_minutes_max=10.0, experiments_max=2)
    b.reserve_experiment(4.0)
    assert b.gpu_minutes_used == 4.0
    assert b.experiments_used == 1
    assert b.gpu_minutes_remaining() == 6.0


def test_gpu_cap_rejects_without_spending():
    b = Budget(gpu_minutes_max=5.0)
    with pytest.raises(BudgetExceeded):
        b.reserve_experiment(6.0)
    # A rejected reservation must leave state untouched.
    assert b.gpu_minutes_used == 0.0
    assert b.experiments_used == 0
    assert len(b.rejections) == 1


def test_experiment_count_cap():
    b = Budget(experiments_max=1)
    b.reserve_experiment(0.1)
    with pytest.raises(BudgetExceeded):
        b.reserve_experiment(0.1)


def test_token_cap():
    b = Budget(tokens_max=100)
    b.reserve_tokens(90)
    with pytest.raises(BudgetExceeded):
        b.reserve_tokens(50)
    assert b.tokens_used == 90


def test_roundtrip():
    b = Budget(gpu_minutes_max=20.0, experiments_max=3, tokens_max=1000)
    b.reserve_experiment(5.0)
    b.reserve_tokens(100)
    restored = Budget.from_dict(b.to_dict())
    assert restored.gpu_minutes_used == 5.0
    assert restored.tokens_used == 100
    assert restored.remaining() == b.remaining()
