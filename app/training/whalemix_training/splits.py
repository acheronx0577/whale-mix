"""Deterministic grouped train/validation/test splits with leakage checks.

Files are grouped by deployment, then session, then recording date, then the file itself. Exact
duplicates (same SHA-256) are merged into one group so a copy can never cross splits. Groups are ordered
by a seeded hash and assigned greedily by duration toward the configured fractions.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from whalemix_training.hashing import canonical_hash

SPLITS = ("train", "validation", "test")


def group_key(entry: dict[str, Any]) -> str:
    for field in ("deployment", "session", "recording_date"):
        value = entry.get(field)
        if value:
            return f"{field}:{value}"
    return f"file:{entry['file_id']}"


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def build_groups(catalog: list[dict[str, Any]]) -> dict[str, str]:
    """Map file_id -> merged group id (duplicates merged with their original's group)."""
    uf = _UnionFind()
    first_by_hash: dict[str, str] = {}
    for entry in catalog:
        key = group_key(entry)
        uf.find(key)
        sha = entry["sha256"]
        if sha in first_by_hash:
            uf.union(key, first_by_hash[sha])
        else:
            first_by_hash[sha] = key
    return {entry["file_id"]: uf.find(group_key(entry)) for entry in catalog}


def _order(group: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()


def assign_splits(
    catalog: list[dict[str, Any]],
    seed: int,
    fractions: dict[str, float],
    protected_test_groups: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if abs(sum(fractions[s] for s in SPLITS) - 1.0) > 1e-6:
        raise ValueError("split fractions must sum to 1")
    readable = [e for e in catalog if e.get("status") == "ok"]
    groups = build_groups(readable)
    duration: dict[str, float] = defaultdict(float)
    for entry in readable:
        duration[groups[entry["file_id"]]] += entry["duration_s"]
    total = sum(duration.values()) or 1.0
    filled = dict.fromkeys(SPLITS, 0.0)
    group_split: dict[str, str] = {}
    for group in sorted(duration, key=lambda g: _order(g, seed)):
        if group in protected_test_groups:
            choice = "test"
        else:
            choice = min(SPLITS, key=lambda s: (filled[s] / total - fractions[s], SPLITS.index(s)))
        group_split[group] = choice
        filled[choice] += duration[group]
    assignments = {
        entry["file_id"]: {"split": group_split[groups[entry["file_id"]]], "group": groups[entry["file_id"]]}
        for entry in readable
    }
    manifest = {
        "seed": seed,
        "fractions": fractions,
        "assignments": dict(sorted(assignments.items())),
        "split_hours": {s: round(filled[s] / 3600.0, 6) for s in SPLITS},
    }
    manifest["split_hash"] = canonical_hash({"seed": seed, "assignments": manifest["assignments"]})
    return manifest


def leakage_check(catalog: list[dict[str, Any]], manifest: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    assignments = manifest["assignments"]
    split_by_hash: dict[str, set[str]] = defaultdict(set)
    split_by_group: dict[str, set[str]] = defaultdict(set)
    for entry in catalog:
        a = assignments.get(entry["file_id"])
        if a is None:
            continue
        split_by_hash[entry["sha256"]].add(a["split"])
        split_by_group[a["group"]].add(a["split"])
    problems += [
        f"sha256 {h[:12]} appears in splits {sorted(s)}" for h, s in split_by_hash.items() if len(s) > 1
    ]
    problems += [f"group {g} appears in splits {sorted(s)}" for g, s in split_by_group.items() if len(s) > 1]
    return problems


def heldout_gaps(
    annotations: list[dict[str, str]], manifest: dict[str, Any], labels: tuple[str, ...]
) -> list[str]:
    """Labels with no reviewed positive event in validation or test cannot support a held-out comparison."""
    present: dict[str, set[str]] = defaultdict(set)
    for row in annotations:
        a = manifest["assignments"].get(row["file_id"])
        if a and row["review_status"] == "reviewed" and row["state"] == "present":
            present[row["label"]].add(a["split"])
    gaps = []
    for label in labels:
        missing = [s for s in ("validation", "test") if s not in present[label]]
        if missing:
            gaps.append(f"{label}: no reviewed positives in {', '.join(missing)}")
    return gaps
