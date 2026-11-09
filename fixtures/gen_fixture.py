"""Generate `fixtures/metrics.json`.

The fixture is **synthetic**. It is shaped to reproduce the qualitative findings of the
AngleDrone benchmark (viewpoint-robust UAV detection) so that the agent can be developed,
demoed and tested without a GPU or a licensed dataset:

* a non-monotonic pitch response — accuracy peaks in a mid-elevation band, then falls
* a best-to-worst domain gap of 25-32 AP50 points, larger than the spread between models
* consistent detector ordering across all pitch bins (no rank inversion)
* class dissociation — the top-1 class changes between the best and worst slice

Numbers are invented. Only the *shape* is borrowed. Regenerate with::

    python fixtures/gen_fixture.py
"""

from __future__ import annotations

import json
from pathlib import Path

SLICES = ["0-15", "15-30", "30-45", "45-60", "60-75", "75-90"]

# Deviation-from-base shape: peak at 45-60, trough at 75-90, range 26.1 points.
SHAPE = [0.0, 8.5, 16.2, 20.4, 7.1, -5.7]

# (detector, base level, scale) — scale sets the gap, base level keeps the ordering.
DETECTORS = [
    ("yolov8n", 52.0, 0.958),        # gap 25.0
    ("fasterrcnn_r50", 47.5, 1.100),  # gap 28.7
    ("yolov11m", 56.0, 1.000),       # gap 26.1
    ("rtdetr_l", 58.5, 1.050),       # gap 27.4
    ("dino_detr_vitl", 61.0, 1.190),  # gap 31.1
    ("yolo26x", 64.0, 1.226),        # gap 32.0
]

# Per-class factors as a function of slice index. `car` dominates near-nadir and decays
# with elevation; `truck` does the opposite. They cross between bin 3 and bin 5, which is
# exactly the dissociation the benchmark reports.
def car_factor(i: int) -> float:
    return 1.120 - 0.030 * i


def truck_factor(i: int) -> float:
    return 0.840 + 0.045 * i


CONSTANT_CLASSES = {
    "van": 0.880,
    "bus": 0.800,
    "person": 0.620,
    "bicycle": 0.550,
    "boat": 0.480,
    "other": 0.400,
}


def build() -> dict:
    values: dict[str, dict[str, dict]] = {}
    for det, base, scale in DETECTORS:
        per_det: dict[str, dict] = {}
        for i, sl in enumerate(SLICES):
            map50 = base + scale * SHAPE[i]
            per_class = {"car": round(map50 * car_factor(i), 2), "truck": round(map50 * truck_factor(i), 2)}
            for name, factor in CONSTANT_CLASSES.items():
                per_class[name] = round(map50 * factor, 2)
            per_det[sl] = {
                "map50": round(map50, 2),
                "precision": round(map50 * 0.96 + 2.0, 2),
                "recall": round(map50 * 0.90, 2),
                "n_gt": 3000 + 140 * i,
                "n_pred": 4200 + 260 * i,
                "per_class": dict(sorted(per_class.items(), key=lambda kv: -kv[1])),
            }
        values[det] = per_det

    return {
        "meta": {
            "dataset": "anglebench",
            "attr": "pitch",
            "scale": "AP50 on 0-100",
            "slices": "pitch bins in degrees",
            "synthetic": True,
            "note": (
                "Synthetic fixture that mirrors the shape of the AngleDrone findings. "
                "Numbers are invented; do not cite them as measurements."
            ),
            "source": "fixtures/gen_fixture.py",
        },
        "detectors": [d for d, _, _ in DETECTORS],
        "slices": SLICES,
        "values": values,
    }


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Write the synthetic metrics fixture.")
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "metrics.json"),
        help="output path (default: fixtures/metrics.json)",
    )
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = build()
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(payload['detectors'])} detectors x {len(payload['slices'])} slices)")


if __name__ == "__main__":
    main()
