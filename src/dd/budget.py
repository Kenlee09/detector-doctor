"""Hard cost gates.

Why this exists (and why it is boring on purpose): the expensive thing in this domain is
not tokens, it is GPU-minutes. An agent that can freely enqueue training runs will
silently burn a lab's quota and, worse, produce duplicate runs that make results
irreproducible. So the budget is a *pre-flight* check owned by the orchestrator, not a
suggestion inside a prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import BudgetExceeded


@dataclass
class Budget:
    """Reservation ledger for one run.

    All three counters are hard ceilings. Reservations are checked *and* committed in
    the same call, so a partially budgeted action can never start.
    """

    gpu_minutes_max: float = 90.0
    experiments_max: int = 16
    tokens_max: int = 250_000
    gpu_minutes_used: float = 0.0
    experiments_used: int = 0
    tokens_used: int = 0
    rejections: list[str] = field(default_factory=list)

    # -- experiments ---------------------------------------------------------------
    def reserve_experiment(self, gpu_minutes: float, label: str = "") -> None:
        label = label or "experiment"
        if self.experiments_used + 1 > self.experiments_max:
            self.rejections.append(f"{label}: experiment cap {self.experiments_max} reached")
            raise BudgetExceeded(
                f"experiment cap reached ({self.experiments_max}); requested by {label}"
            )
        if self.gpu_minutes_used + gpu_minutes > self.gpu_minutes_max:
            self.rejections.append(
                f"{label}: needs {gpu_minutes:.1f} GPU-min, "
                f"only {self.gpu_minutes_remaining():.1f} left"
            )
            raise BudgetExceeded(
                f"GPU budget exceeded: needs {gpu_minutes:.1f} min, "
                f"{self.gpu_minutes_remaining():.1f} of {self.gpu_minutes_max:.1f} remaining"
            )
        self.experiments_used += 1
        self.gpu_minutes_used += gpu_minutes

    # -- tokens --------------------------------------------------------------------
    def reserve_tokens(self, n: int, label: str = "") -> None:
        label = label or "llm"
        if self.tokens_used + n > self.tokens_max:
            self.rejections.append(f"{label}: token cap {self.tokens_max} reached")
            raise BudgetExceeded(f"token budget exceeded ({self.tokens_max}) for {label}")
        self.tokens_used += n

    # -- introspection -------------------------------------------------------------
    def gpu_minutes_remaining(self) -> float:
        return max(0.0, self.gpu_minutes_max - self.gpu_minutes_used)

    def remaining(self) -> dict[str, Any]:
        return {
            "gpu_minutes": round(self.gpu_minutes_remaining(), 2),
            "experiments": self.experiments_max - self.experiments_used,
            "tokens": self.tokens_max - self.tokens_used,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_minutes_max": self.gpu_minutes_max,
            "experiments_max": self.experiments_max,
            "tokens_max": self.tokens_max,
            "gpu_minutes_used": round(self.gpu_minutes_used, 2),
            "experiments_used": self.experiments_used,
            "tokens_used": self.tokens_used,
            "rejections": list(self.rejections),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Budget":
        return cls(
            gpu_minutes_max=float(d.get("gpu_minutes_max", 90.0)),
            experiments_max=int(d.get("experiments_max", 16)),
            tokens_max=int(d.get("tokens_max", 250_000)),
            gpu_minutes_used=float(d.get("gpu_minutes_used", 0.0)),
            experiments_used=int(d.get("experiments_used", 0)),
            tokens_used=int(d.get("tokens_used", 0)),
            rejections=list(d.get("rejections") or []),
        )
