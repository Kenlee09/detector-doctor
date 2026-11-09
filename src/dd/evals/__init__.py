"""Evaluation suites (gold-standard replay, reliability counters)."""

from .gold_replay import GoldResult, load_gold, run_eval

__all__ = ["GoldResult", "load_gold", "run_eval"]
