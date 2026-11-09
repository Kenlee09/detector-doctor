"""Thin wrapper so the eval can run without installing the package.

    python eval/run_gold_replay.py

The real implementation lives in `src/dd/evals/gold_replay.py` so that `dd eval` works
from an installed console script too.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dd.evals.gold_replay import run_eval  # noqa: E402


def main() -> int:
    out = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "artifacts" / "eval")
    result = run_eval(out_dir=out)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
