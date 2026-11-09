"""Narrow reasoning layer.

The LLM lives here and nowhere else. `diagnostician` owns the single step where a model
may exercise judgement (rank playbook options); `reporter` renders verified state and
holds no model at all.
"""

from .diagnostician import Diagnostician, PlanItem, Proposal
from .reporter import ReportInputs, render, write_report

__all__ = ["Diagnostician", "PlanItem", "Proposal", "ReportInputs", "render", "write_report"]
