"""Bounded, stratified replay memory and a resumable deterministic sampler.

Replay stores pointers (example IDs and strata), never audio. Only reviewed training-split examples are
admitted; validation/test windows and unreviewed model guesses are rejected. Each stratum keeps a capped
reservoir so one abundant condition cannot displace old years or rare classes.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

from whalemix_training.hashing import canonical_hash
from whalemix_training.labels import PRESENT


def stratum_of(example: dict[str, Any]) -> str:
    present = sorted(k for k, v in example["states"].items() if v == PRESENT) or ["background"]
    return "|".join(
        ["+".join(present), str(example.get("year", "unknown")), str(example.get("deployment", ""))]
    )


class ReplayAdmissionError(ValueError):
    pass


@dataclass
class StratifiedReplay:
    capacity: int
    per_stratum_cap: int
    seed: int
    members: dict[str, list[str]] = field(default_factory=dict)
    seen: dict[str, int] = field(default_factory=dict)
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def __len__(self) -> int:
        return sum(len(v) for v in self.members.values())

    def offer(self, example: dict[str, Any]) -> bool:
        if example.get("split") != "train":
            raise ReplayAdmissionError(
                f"{example.get('example_id')}: only training-split examples may enter replay"
            )
        if not example.get("reviewed"):
            raise ReplayAdmissionError(
                f"{example.get('example_id')}: unreviewed examples may not enter replay"
            )
        key = stratum_of(example)
        bucket = self.members.setdefault(key, [])
        self.seen[key] = self.seen.get(key, 0) + 1
        admitted = False
        if len(bucket) < self.per_stratum_cap:
            bucket.append(example["example_id"])
            admitted = True
        else:
            slot = self._rng.randrange(self.seen[key])
            if slot < self.per_stratum_cap:
                bucket[slot] = example["example_id"]
                admitted = True
        while len(self) > self.capacity:
            largest = max(sorted(self.members), key=lambda k: len(self.members[k]))
            self.members[largest].pop(self._rng.randrange(len(self.members[largest])))
        return admitted

    def sample(self, n: int) -> list[str]:
        pool = [eid for key in sorted(self.members) for eid in self.members[key]]
        if not pool or n <= 0:
            return []
        return [pool[self._rng.randrange(len(pool))] for _ in range(n)]

    def state_dict(self) -> dict[str, Any]:
        version, internal, gauss = self._rng.getstate()
        return {
            "capacity": self.capacity,
            "per_stratum_cap": self.per_stratum_cap,
            "seed": self.seed,
            "members": {k: list(v) for k, v in sorted(self.members.items())},
            "seen": dict(sorted(self.seen.items())),
            "rng_state": [version, list(internal), gauss],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> StratifiedReplay:
        replay = cls(state["capacity"], state["per_stratum_cap"], state["seed"])
        replay.members = {k: list(v) for k, v in state["members"].items()}
        replay.seen = dict(state["seen"])
        version, internal, gauss = state["rng_state"]
        replay._rng.setstate((version, tuple(internal), gauss))
        return replay

    def manifest_hash(self) -> str:
        return canonical_hash({k: v for k, v in self.state_dict().items() if k != "rng_state"})


def blend(
    new_ids: list[str], replay: StratifiedReplay, batch_size: int, new_fraction: float
) -> dict[str, Any]:
    """Compose one effective batch; when replay is empty or new data is short the actual blend is recorded."""
    want_new = round(batch_size * new_fraction)
    take_new = new_ids[: min(want_new, len(new_ids))] if len(replay) else new_ids[:batch_size]
    replayed = replay.sample(batch_size - len(take_new))
    return {"new": take_new, "replay": replayed, "actual_new_fraction": len(take_new) / max(1, batch_size)}


@dataclass
class DeterministicSampler:
    """Seeded per-epoch permutation with a consumption cursor. Prefetching does not move the cursor."""

    example_ids: list[str]
    seed: int
    epoch: int = 0
    cursor: int = 0

    @property
    def ids_hash(self) -> str:
        return hashlib.sha256("\n".join(self.example_ids).encode()).hexdigest()

    def order(self) -> list[str]:
        rng = random.Random(f"{self.seed}:{self.epoch}")
        ids = list(self.example_ids)
        rng.shuffle(ids)
        return ids

    def peek(self, n: int) -> list[str]:
        return self.order()[self.cursor : self.cursor + n]

    def mark_consumed(self, n: int) -> None:
        self.cursor += n
        if self.cursor >= len(self.example_ids):
            self.epoch += 1
            self.cursor = 0

    def state_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "epoch": self.epoch, "cursor": self.cursor, "ids_hash": self.ids_hash}

    def load_state(self, state: dict[str, Any]) -> None:
        if state["ids_hash"] != self.ids_hash:
            raise ValueError("sampler state belongs to a different example manifest")
        self.seed, self.epoch, self.cursor = state["seed"], state["epoch"], state["cursor"]
