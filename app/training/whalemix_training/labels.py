"""Versioned label map, profiles and target/mask construction.

Target states are `present`, `absent` (reviewed absent within supported bandwidth) and `unassessed`.
Only reviewed present/absent states become supervised targets; everything else is masked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any

import numpy as np

from whalemix_training.hashing import canonical_hash

PRESENT = "present"
ABSENT = "absent"
UNASSESSED = "unassessed"
STATES = (PRESENT, ABSENT, UNASSESSED)


@lru_cache(maxsize=1)
def load_profiles() -> dict[str, Any]:
    text = resources.files("whalemix_training").joinpath("profiles.json").read_text(encoding="utf-8")
    return json.loads(text)


@dataclass(frozen=True)
class LabelMap:
    version: str
    whale_targets: tuple[str, ...]
    context_targets: tuple[str, ...]

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.whale_targets + self.context_targets

    def index(self, name: str) -> int:
        return self.outputs.index(name)

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "whale_targets": list(self.whale_targets),
            "context_targets": list(self.context_targets),
            "outputs": list(self.outputs),
        }

    @property
    def hash(self) -> str:
        return canonical_hash(self.to_json())

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> LabelMap:
        return cls(obj["version"], tuple(obj["whale_targets"]), tuple(obj["context_targets"]))


def default_label_map() -> LabelMap:
    prof = load_profiles()
    return LabelMap(prof["profiles_version"], tuple(prof["whale_targets"]), tuple(prof["context_targets"]))


def required_max_hz(label: str) -> float | None:
    prof = load_profiles()
    for profile in prof["profiles"]:
        if label in profile.get("outputs", []):
            value = profile.get("required_max_hz")
            return None if value is None else float(value)
    value = prof["context_required_max_hz"].get(label)
    return None if value is None else float(value)


def is_assessable(label: str, sample_rate: float) -> bool:
    """A reviewed absence only counts when the recording's Nyquist covers the label's required band."""
    need = required_max_hz(label)
    return need is None or sample_rate / 2.0 >= need


def targets_and_mask(states: dict[str, str], label_map: LabelMap) -> tuple[np.ndarray, np.ndarray]:
    """Convert per-label states into float32 targets plus a mask (1 = supervised, 0 = masked)."""
    n = len(label_map.outputs)
    y = np.zeros(n, dtype=np.float32)
    mask = np.zeros(n, dtype=np.float32)
    for name, state in states.items():
        if name not in label_map.outputs:
            raise KeyError(f"unknown label {name!r} for label map {label_map.version}")
        if state not in STATES:
            raise ValueError(f"unknown state {state!r} for label {name!r}")
        idx = label_map.index(name)
        if state == PRESENT:
            y[idx], mask[idx] = 1.0, 1.0
        elif state == ABSENT:
            y[idx], mask[idx] = 0.0, 1.0
    return y, mask
