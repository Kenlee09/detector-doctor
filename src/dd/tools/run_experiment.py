"""Tool 4/5 — idempotent, budget-gated experiment execution.

Two invariants, both enforced here rather than asked for in a prompt:

1. **Idempotency.** The artifact path is derived from `ExperimentConfig.config_hash()`.
   If the artifact already exists, the call is a cache hit: no GPU-minutes are reserved
   and no duplicate run is enqueued. This is what makes the orchestrator safe to retry
   after a crash, and it is the single most important difference between this and a
   free-running ReAct loop that re-launches training on every retry.

2. **Pre-flight budget.** `Budget.reserve_experiment()` runs *before* any work is
   queued, so a rejected plan leaves the run state untouched.

In this repository the runner is a simulator: it derives the post-fix AP50 from the
playbook's recorded `expected_gain`, so the whole loop is reproducible offline. Point
`runner=` at a real Ultralytics/RT-DETR job and nothing else changes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from ..budget import Budget
from ..data.store import MetricsStore
from ..types import ExperimentConfig, RemediationOption
from .base import ToolSpec

# A simulated run returns (map50, detail). Swappable for a real trainer.
Runner = Callable[[ExperimentConfig], tuple[float, dict[str, Any]]]


def default_simulated_runner(factory: "ExperimentTool") -> Runner:
    def run(cfg: ExperimentConfig) -> tuple[float, dict[str, Any]]:
        return factory.simulate(cfg)

    return run


class ExperimentTool:
    spec = ToolSpec(
        name="run_experiment",
        description=(
            "Execute (or fetch from cache) one evaluation/training experiment for a "
            "detector configuration. Idempotent per config hash; subject to the run budget."
        ),
        params={"detector": "string", "overrides": "object", "gpu_minutes": "number"},
        returns="experiment result with status cached|ran|dry_run",
        side_effect=True,
        cost="gpu",
    )

    def __init__(
        self,
        store: MetricsStore,
        out_dir: str | Path,
        budget: Budget,
        trace: Any | None = None,
        runner: Runner | None = None,
        default_gpu_minutes: float = 4.0,
    ) -> None:
        self.store = store
        self.out_dir = Path(out_dir)
        self.exp_dir = self.out_dir / "experiments"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.budget = budget
        self.trace = trace
        self.runner: Runner = runner or default_simulated_runner(self)
        self.default_gpu_minutes = default_gpu_minutes

    # -- simulation ----------------------------------------------------------------
    def simulate(self, cfg: ExperimentConfig) -> tuple[float, dict[str, Any]]:
        """Deterministic stand-in for a real trainer.

        Baseline = the detector's worst-slice AP50. Each override contributes the
        `expected_gain` recorded in the playbook, minus a small penalty for stacking
        options in the same layer (they overlap in effect). Fully deterministic, so a
        test can assert on the exact number.
        """
        table = self.store.table(cfg.detector)
        baseline = min(table.values()) if table else 0.0
        gain = 0.0
        applied: list[str] = []
        per_layer: dict[str, int] = {}
        for key, value in cfg.overrides:
            gain += self._gain_for(key, value) / (1 + per_layer.get(key, 0))
            per_layer[key] = per_layer.get(key, 0) + 1
            applied.append(f"{key}={value}")
        result = round(baseline + gain, 2)
        detail = {
            "simulated": True,
            "baseline_map50": round(baseline, 2),
            "gain": round(gain, 2),
            "applied_overrides": applied,
        }
        return result, detail

    @staticmethod
    def _gain_for(key: str, value: str) -> float:
        """Gains are looked up from the playbook, keyed by override string."""
        try:
            from .playbook import Playbook

            pb = Playbook.load()
        except Exception:  # pragma: no cover - playbook is shipped with the repo
            return 0.0
        token = f"{key}={value}"
        for opt in pb.options:
            if token in opt.source or token in opt.action:
                return opt.expected_gain
        return 0.0

    # -- artifacts -----------------------------------------------------------------
    def artifact_path(self, cfg: ExperimentConfig) -> Path:
        return self.exp_dir / f"{cfg.config_hash()}.json"

    def _write_artifact(self, cfg: ExperimentConfig, map50: float, detail: dict[str, Any]) -> Path:
        path = self.artifact_path(cfg)
        payload = {"config": cfg.to_dict(), "map50": map50, "detail": detail}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def artifact_sha(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    # -- public API ----------------------------------------------------------------
    def __call__(
        self,
        detector: str,
        overrides: dict[str, str] | None = None,
        gpu_minutes: float | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        cfg = ExperimentConfig.build(detector, self.store.dataset, overrides)
        path = self.artifact_path(cfg)

        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if self.trace is not None:
                self.trace.emit("experiment_cache_hit", config_hash=cfg.config_hash(), detector=detector)
            return {
                "status": "cached",
                "config_hash": cfg.config_hash(),
                "detector": detector,
                "overrides": dict(cfg.overrides),
                "map50": cached["map50"],
                "artifact": str(path.relative_to(self.out_dir)),
                "artifact_sha256": self.artifact_sha(path),
                "gpu_minutes_reserved": 0.0,
            }

        cost = self.default_gpu_minutes if gpu_minutes is None else gpu_minutes

        if dry_run:
            return {
                "status": "dry_run",
                "config_hash": cfg.config_hash(),
                "detector": detector,
                "overrides": dict(cfg.overrides),
                "projected_gpu_minutes": cost,
                "budget_remaining": self.budget.remaining(),
                "would_fit": cost <= self.budget.gpu_minutes_remaining(),
            }

        # Pre-flight gate: raises BudgetExceeded before any work is queued.
        self.budget.reserve_experiment(cost, label=f"{detector}:{cfg.config_hash()}")

        map50, detail = self.runner(cfg)
        path = self._write_artifact(cfg, map50, detail)
        if self.trace is not None:
            self.trace.emit(
                "experiment_ran",
                config_hash=cfg.config_hash(),
                detector=detector,
                map50=map50,
                gpu_minutes=cost,
            )
        return {
            "status": "ran",
            "config_hash": cfg.config_hash(),
            "detector": detector,
            "overrides": dict(cfg.overrides),
            "map50": map50,
            "artifact": str(path.relative_to(self.out_dir)),
            "artifact_sha256": self.artifact_sha(path),
            "gpu_minutes_reserved": cost,
            "detail": detail,
        }
