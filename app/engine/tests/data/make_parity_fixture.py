"""Regenerate preprocess-parity.json: Python reference windows for a deterministic synthetic signal.

Run from app/training with its environment:  python ../engine/tests/data/make_parity_fixture.py
The Rust test rebuilds the same signal from the formula below, so no audio is committed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from whalemix_training.preprocess import PreprocessingSpec, analyze_channel

STRIDE = 101
CASES = [
    {"name": "8k-log", "source_rate": 8000, "seconds": 13.0, "compression": "log"},
    {"name": "8k-pcen", "source_rate": 8000, "seconds": 13.0, "compression": "pcen"},
    {"name": "4k-log-short", "source_rate": 4000, "seconds": 3.5, "compression": "log"},
    {"name": "6k-log", "source_rate": 6000, "seconds": 11.0, "compression": "log"},
]


def signal(rate: int, seconds: float) -> np.ndarray:
    n = int(round(rate * seconds))
    t = np.arange(n, dtype=np.float64) / rate
    chirp = np.sin(2 * math.pi * (60.0 * t + 15.0 * t * t))
    return 0.3 * np.sin(2 * math.pi * 17.0 * t) + 0.2 * np.sin(2 * math.pi * 400.0 * t) + 0.1 * chirp


def main() -> None:
    out = []
    for case in CASES:
        spec = PreprocessingSpec.from_config({"compression": case["compression"]})
        windows = analyze_channel(signal(case["source_rate"], case["seconds"]), case["source_rate"], spec)
        out.append(
            {
                **case,
                "preprocessing": spec.to_json(),
                "windows": [
                    {
                        "start_s": w["start_s"],
                        "end_s": w["end_s"],
                        "tensor_sum": float(np.sum(w["tensor"], dtype=np.float64)),
                        "samples": [float(v) for v in w["tensor"].ravel()[::STRIDE]],
                    }
                    for w in windows
                ],
            }
        )
    path = Path(__file__).with_name("preprocess-parity.json")
    path.write_text(json.dumps({"stride": STRIDE, "cases": out}, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
