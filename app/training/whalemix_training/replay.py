"""Bounded, stratified replay memory and a resumable deterministic sampler.

Replay stores pointers (example IDs and strata), never audio. Only reviewed training-split examples are
admitted; validation/test windows and unreviewed model guesses are rejected. Each stratum keeps a capped
reservoir so one abundant condition cannot displace old years or rare classes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

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
    # Seeded reservoir sampling for reproducible resume; not used for anything security-related.
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def _below(self, n: int) -> int:
        return int(self._rng.integers(0, n))

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
            slot = self._below(self.seen[key])
            if slot < self.per_stratum_cap:
                bucket[slot] = example["example_id"]
                admitted = True
        while len(self) > self.capacity:
            largest = max(sorted(self.members), key=lambda k: len(self.members[k]))
            self.members[largest].pop(self._below(len(self.members[largest])))
        return admitted

    def sample(self, n: int) -> list[str]:
        pool = [eid for key in sorted(self.members) for eid in self.members[key]]
        if not pool or n <= 0:
            return []
        return [pool[int(i)] for i in self._rng.integers(0, len(pool), size=n)]

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "per_stratum_cap": self.per_stratum_cap,
            "seed": self.seed,
            "members": {k: list(v) for k, v in sorted(self.members.items())},
            "seen": dict(sorted(self.seen.items())),
            "rng_state": self._rng.bit_generator.state,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> StratifiedReplay:
        replay = cls(state["capacity"], state["per_stratum_cap"], state["seed"])
        replay.members = {k: list(v) for k, v in state["members"].items()}
        replay.seen = dict(state["seen"])
        replay._rng.bit_generator.state = state["rng_state"]
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


class ReplayMixer:
    """Batch source for fine-tuning: new-data IDs and replay IDs are mixed so the running share of new
    examples tracks `new_fraction` (0.7 by default), even when a micro-batch is too small to hold the exact
    ratio. If replay is empty the batch is all new data. State (sampler cursor, replay RNG and membership,
    counts) is checkpointed for exact resume."""

    def __init__(self, sampler: DeterministicSampler, replay: StratifiedReplay, new_fraction: float) -> None:
        if not 0.0 < new_fraction <= 1.0:
            raise ValueError("replay.new_fraction must be in (0, 1]")
        self.sampler = sampler
        self.replay = replay
        self.new_fraction = new_fraction
        self.new_examples = 0
        self.replay_examples = 0
        self._pending: tuple[int, int] = (0, 0)

    @staticmethod
    def validate_members(replay: StratifiedReplay, split_of: dict[str, str], cap: int) -> None:
        """Every replay member must be a current reviewed training window, and no stratum may exceed cap."""
        for key, members in sorted(replay.members.items()):
            if len(members) > cap:
                raise ReplayAdmissionError(
                    f"replay stratum {key!r} holds {len(members)} > replay.per_stratum_cap {cap}"
                )
            for eid in members:
                split = split_of.get(eid)
                if split is None:
                    raise ReplayAdmissionError(f"{eid}: unknown replay id (not a reviewed window here)")
                if split != "train":
                    raise ReplayAdmissionError(f"{eid}: held-out ({split}) id in replay memory")

    def draw(self, n: int) -> list[str]:
        if len(self.replay) == 0:
            want_new = n
        else:
            total = self.new_examples + self.replay_examples + n
            want_new = min(n, max(0, round(total * self.new_fraction) - self.new_examples))
        new = self.sampler.draw(min(want_new, n))
        replayed = self.replay.sample(n - len(new))
        self._pending = (len(new), len(replayed))
        return new + replayed

    def consume(self) -> None:
        self.sampler.consume()
        self.new_examples += self._pending[0]
        self.replay_examples += self._pending[1]
        self._pending = (0, 0)

    @property
    def epoch(self) -> int:
        return self.sampler.epoch

    def stats(self) -> dict[str, Any]:
        total = self.new_examples + self.replay_examples
        return {
            "new_examples": self.new_examples,
            "replay_examples": self.replay_examples,
            "actual_new_fraction": self.new_examples / total if total else None,
            "target_new_fraction": self.new_fraction,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "sampler": self.sampler.state_dict(),
            "replay": self.replay.state_dict(),
            "new_fraction": self.new_fraction,
            "new_examples": self.new_examples,
            "replay_examples": self.replay_examples,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if abs(float(state["new_fraction"]) - self.new_fraction) > 1e-12:
            raise ValueError("mixer state has a different replay.new_fraction")
        self.sampler.load_state(state["sampler"])
        self.replay = StratifiedReplay.from_state(state["replay"])
        self.new_examples = int(state["new_examples"])
        self.replay_examples = int(state["replay_examples"])


@dataclass
class DeterministicSampler:
    """Seeded per-epoch permutation with a consumption cursor. Prefetching does not move the cursor."""

    example_ids: list[str]
    seed: int
    epoch: int = 0
    cursor: int = 0
    _last: list[str] = field(default_factory=list, init=False, repr=False, compare=False)

    @property
    def ids_hash(self) -> str:
        return hashlib.sha256("\n".join(self.example_ids).encode()).hexdigest()

    def order(self) -> list[str]:
        # Deterministic per-epoch order (reproducible resume), derived from the seed and epoch.
        seed = int.from_bytes(hashlib.sha256(f"{self.seed}:{self.epoch}".encode()).digest()[:8], "little")
        perm = np.random.default_rng(seed).permutation(len(self.example_ids))
        return [self.example_ids[int(i)] for i in perm]

    def peek(self, n: int) -> list[str]:
        return self.order()[self.cursor : self.cursor + n]

    def mark_consumed(self, n: int) -> None:
        self.cursor += n
        if self.cursor >= len(self.example_ids):
            self.epoch += 1
            self.cursor = 0

    def draw(self, n: int) -> list[str]:
        """Batch-source interface: peek without moving the cursor until `consume`."""
        self._last = self.peek(n)
        return self._last

    def consume(self) -> None:
        self.mark_consumed(len(self._last))
        self._last = []

    def stats(self) -> dict[str, Any]:
        return {}

    def state_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "epoch": self.epoch, "cursor": self.cursor, "ids_hash": self.ids_hash}

    def load_state(self, state: dict[str, Any]) -> None:
        if state["ids_hash"] != self.ids_hash:
            raise ValueError("sampler state belongs to a different example manifest")
        self.seed, self.epoch, self.cursor = state["seed"], state["epoch"], state["cursor"]
