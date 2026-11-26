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


def _derived_seed(seed: int, stream: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{stream}".encode()).digest()[:8], "little") % (2**63)


@dataclass
class GroupedPositives:
    """Positive stream stratified by group (e.g. recorder): each positive slot first picks a group, then
    takes that group's next window from its own seeded `DeterministicSampler`.

    Group g gets weight n_g ** temperature (n_g = its positive windows), normalised: temperature 1 is the
    plain proportional draw, 0 is equal weight per group, 0.5 is square-root tempering. The group of the
    d-th positive draw is a pure function of (seed, d), so the saved draw counter plus each group's
    cursor fully determine the rest of the sequence and a resume is exact."""

    groups: dict[str, DeterministicSampler]
    temperature: float
    seed: int
    drawn: int = 0
    drawn_by_group: dict[str, int] = field(default_factory=dict)
    _orders: dict[tuple[str, int], list[str]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    @classmethod
    def build(cls, group_of: dict[str, str], seed: int, temperature: float) -> GroupedPositives:
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("positive group temperature must be between 0 and 1")
        members: dict[str, list[str]] = {}
        for eid, g in group_of.items():
            members.setdefault(str(g), []).append(eid)
        if not members:
            raise ValueError("grouped positives need at least one positive window")
        groups = {
            g: DeterministicSampler(sorted(ids), _derived_seed(seed, f"positive:{g}"))
            for g, ids in sorted(members.items())
        }
        return cls(groups, float(temperature), int(seed), drawn_by_group={g: 0 for g in groups})

    @property
    def example_ids(self) -> list[str]:
        return [i for s in self.groups.values() for i in s.example_ids]

    @property
    def epoch(self) -> int:
        """Completed passes over the largest group (smaller groups repeat more often by design)."""
        return max(self.groups.values(), key=lambda s: len(s.example_ids)).epoch

    def weights(self) -> dict[str, float]:
        raw = {g: len(s.example_ids) ** self.temperature for g, s in self.groups.items()}
        total = sum(raw.values())
        return {g: w / total for g, w in raw.items()}

    def group_of_draw(self, d: int) -> str:
        digest = hashlib.sha256(f"{self.seed}:positive-group:{d}".encode()).digest()[:8]
        u = int.from_bytes(digest, "little") / 2.0**64
        acc = 0.0
        names = list(self.groups)
        for g, w in self.weights().items():
            acc += w
            if u < acc:
                return g
        return names[-1]

    def _order(self, g: str, epoch: int) -> list[str]:
        key = (g, epoch)
        if key not in self._orders:
            if len(self._orders) > 4 * len(self.groups) + 8:
                self._orders.clear()
            s = self.groups[g]
            self._orders[key] = DeterministicSampler(s.example_ids, s.seed, epoch).order()
        return self._orders[key]

    def peek(self, n: int) -> list[str]:
        pos = {g: (s.epoch, s.cursor) for g, s in self.groups.items()}
        out = []
        for j in range(n):
            g = self.group_of_draw(self.drawn + j)
            epoch, cursor = pos[g]
            order = self._order(g, epoch)
            out.append(order[cursor])
            cursor += 1
            pos[g] = (epoch + 1, 0) if cursor >= len(order) else (epoch, cursor)
        return out

    def consume(self, n: int) -> None:
        for j in range(n):
            g = self.group_of_draw(self.drawn + j)
            self.groups[g].mark_consumed(1)
            self.drawn_by_group[g] += 1
        self.drawn += n

    def stats(self) -> dict[str, Any]:
        w = self.weights()
        return {
            g: {
                "available": len(s.example_ids),
                "weight": round(w[g], 6),
                "drawn": self.drawn_by_group[g],
                "epochs": s.epoch,
            }
            for g, s in self.groups.items()
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": "grouped",
            "temperature": self.temperature,
            "seed": self.seed,
            "drawn": self.drawn,
            "drawn_by_group": dict(self.drawn_by_group),
            "groups": {g: s.state_dict() for g, s in self.groups.items()},
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if state.get("kind") != "grouped":
            raise ValueError("sampler state has ungrouped positives")
        if float(state["temperature"]) != self.temperature or int(state["seed"]) != self.seed:
            raise ValueError("sampler state has a different positive grouping")
        if sorted(state["groups"]) != sorted(self.groups):
            raise ValueError("sampler state has different positive groups")
        for g, s in self.groups.items():
            s.load_state(state["groups"][g])
        self.drawn = int(state["drawn"])
        self.drawn_by_group = {g: int(v) for g, v in state["drawn_by_group"].items()}
        self._orders.clear()


@dataclass
class BalancedSampler:
    """Seeded positive oversampling: every micro-batch of n holds exactly round(n * positive_fraction)
    positive windows (at least one, at most n - 1) and the rest negatives.

    Optional confusers (hard negatives, e.g. windows inside publisher-labelled calls of another type) get
    their own fixed share of each batch, round(n * confuser_fraction) (at least one when enabled, always
    leaving one plain-negative slot), and are removed from the plain-negative pool.

    Every pool is an independent `DeterministicSampler` stream that wraps into its next epoch inside a
    batch, so small pools are reused in a fresh seeded order each pass. The full state (cursors, epochs and
    draw counters) goes into the checkpoint, so a resume continues the exact sequence. Without confusers the
    sequence and the saved state are exactly those of the positive-only sampler.

    Optional positive grouping (`GroupedPositives`) stratifies the positive slots by a group such as the
    recorder, with tempered weights, so a recorder with few calls is not drowned out by one with many. Off
    by default; without it the sequence and saved state are unchanged."""

    positives: DeterministicSampler | GroupedPositives
    negatives: DeterministicSampler
    positive_fraction: float
    drawn: dict[str, int] = field(default_factory=lambda: {"positive": 0, "negative": 0})
    confusers: DeterministicSampler | None = None
    confuser_fraction: float = 0.0
    _last: list[str] = field(default_factory=list, init=False, repr=False, compare=False)
    _plan: tuple[int, ...] = field(default=(0, 0, 0), init=False, repr=False, compare=False)
    _orders: dict[tuple[str, int], list[str]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    KIND = "balanced"

    @classmethod
    def build(
        cls,
        positive_ids: list[str],
        negative_ids: list[str],
        seed: int,
        positive_fraction: float,
        confuser_ids: list[str] | None = None,
        confuser_fraction: float = 0.0,
        positive_groups: dict[str, str] | None = None,
        group_temperature: float = 1.0,
    ) -> BalancedSampler:
        if not 0.0 < positive_fraction < 1.0:
            raise ValueError("positive_fraction must be between 0 and 1 (exclusive)")
        if not 0.0 <= confuser_fraction < 1.0 or positive_fraction + confuser_fraction >= 1.0:
            raise ValueError("confuser_fraction must be >= 0 and leave room for plain negatives")
        confusers = None
        if confuser_fraction > 0.0:
            pool = set(confuser_ids or [])
            if not pool:
                raise ValueError("confuser_fraction > 0 needs at least one confuser window")
            if pool & set(positive_ids):
                raise ValueError("confuser windows must not be positives")
            negative_ids = [i for i in negative_ids if i not in pool]
            confusers = DeterministicSampler(sorted(pool), _derived_seed(seed, "confuser"))
        if not positive_ids or not negative_ids:
            raise ValueError("balanced sampling needs at least one positive and one negative window")
        positives: DeterministicSampler | GroupedPositives
        if positive_groups is not None:
            missing = [i for i in positive_ids if i not in positive_groups]
            if missing:
                raise ValueError(f"{len(missing)} positive windows have no group")
            positives = GroupedPositives.build(
                {i: positive_groups[i] for i in positive_ids}, seed, group_temperature
            )
        else:
            positives = DeterministicSampler(sorted(positive_ids), _derived_seed(seed, "positive"))
        sampler = cls(
            positives,
            DeterministicSampler(sorted(negative_ids), _derived_seed(seed, "negative")),
            float(positive_fraction),
            confusers=confusers,
            confuser_fraction=float(confuser_fraction),
        )
        if confusers is not None:
            sampler.drawn["confuser"] = 0
        return sampler

    @property
    def epoch(self) -> int:
        """Passes over the plain negatives (the large stream); smaller pools repeat more often by design."""
        return self.negatives.epoch

    def positives_per_batch(self, n: int) -> int:
        if n < 2:
            raise ValueError("balanced sampling needs micro_batch >= 2")
        return min(n - 1, max(1, round(n * self.positive_fraction)))

    def confusers_per_batch(self, n: int) -> int:
        if self.confusers is None:
            return 0
        room = n - self.positives_per_batch(n) - 1
        if room < 1:
            raise ValueError("micro_batch too small for positives, confusers and a plain negative")
        return min(room, max(1, round(n * self.confuser_fraction)))

    def _order(self, name: str, s: DeterministicSampler, epoch: int) -> list[str]:
        key = (name, epoch)
        if key not in self._orders:
            if len(self._orders) > 8:
                self._orders.clear()
            self._orders[key] = DeterministicSampler(s.example_ids, s.seed, epoch).order()
        return self._orders[key]

    def _peek(self, name: str, s: DeterministicSampler | GroupedPositives, n: int) -> list[str]:
        if isinstance(s, GroupedPositives):
            return s.peek(n)
        out: list[str] = []
        epoch, cursor = s.epoch, s.cursor
        while len(out) < n:
            order = self._order(name, s, epoch)
            take = order[cursor : cursor + (n - len(out))]
            out.extend(take)
            cursor += len(take)
            if cursor >= len(order):
                epoch, cursor = epoch + 1, 0
        return out

    def _streams(self) -> list[tuple[str, DeterministicSampler | GroupedPositives]]:
        streams: list[tuple[str, DeterministicSampler | GroupedPositives]] = [("positive", self.positives)]
        if self.confusers is not None:
            streams.append(("confuser", self.confusers))
        return [*streams, ("negative", self.negatives)]

    def draw(self, n: int) -> list[str]:
        k = self.positives_per_batch(n)
        c = self.confusers_per_batch(n)
        counts = {"positive": k, "confuser": c, "negative": n - k - c}
        self._plan = tuple(counts[name] for name, _ in self._streams())
        self._last = [i for name, s in self._streams() for i in self._peek(name, s, counts[name])]
        return self._last

    def consume(self) -> None:
        for (name, s), count in zip(self._streams(), self._plan, strict=True):
            self.drawn[name] += count
            if isinstance(s, GroupedPositives):
                s.consume(count)
                continue
            while count > 0:
                step = min(count, len(s.example_ids) - s.cursor)
                s.mark_consumed(step)
                count -= step
        self._last, self._plan = [], tuple(0 for _ in self._plan)

    def stats(self) -> dict[str, Any]:
        total = sum(self.drawn.values())
        out = {
            "kind": self.KIND,
            "positive_fraction": self.positive_fraction,
            "positives_available": len(self.positives.example_ids),
            "negatives_available": len(self.negatives.example_ids),
            "drawn": dict(self.drawn),
            "positive_share_drawn": None if total == 0 else self.drawn["positive"] / total,
            "positive_epochs": self.positives.epoch,
        }
        if isinstance(self.positives, GroupedPositives):
            out["positive_grouping"] = {
                "temperature": self.positives.temperature,
                "groups": self.positives.stats(),
            }
        if self.confusers is not None:
            out.update(
                {
                    "confuser_fraction": self.confuser_fraction,
                    "confusers_available": len(self.confusers.example_ids),
                    "confuser_share_drawn": None if total == 0 else self.drawn["confuser"] / total,
                    "confuser_epochs": self.confusers.epoch,
                }
            )
        return out

    def state_dict(self) -> dict[str, Any]:
        state = {
            "kind": self.KIND,
            "positive_fraction": self.positive_fraction,
            "positives": self.positives.state_dict(),
            "negatives": self.negatives.state_dict(),
            "drawn": dict(self.drawn),
        }
        if self.confusers is not None:
            state["confuser_fraction"] = self.confuser_fraction
            state["confusers"] = self.confusers.state_dict()
        return state

    def load_state(self, state: dict[str, Any]) -> None:
        if state.get("kind") != self.KIND:
            raise ValueError("sampler state is not from a balanced sampler")
        if float(state["positive_fraction"]) != self.positive_fraction:
            raise ValueError("sampler state has a different positive_fraction")
        if float(state.get("confuser_fraction", 0.0)) != self.confuser_fraction:
            raise ValueError("sampler state has a different confuser_fraction")
        grouped_state = state["positives"].get("kind") == "grouped"
        if grouped_state != isinstance(self.positives, GroupedPositives):
            raise ValueError("sampler state has a different positive grouping")
        self.positives.load_state(state["positives"])
        self.negatives.load_state(state["negatives"])
        if self.confusers is not None:
            self.confusers.load_state(state["confusers"])
        self.drawn = {k: int(v) for k, v in state["drawn"].items()}
        self._orders.clear()
