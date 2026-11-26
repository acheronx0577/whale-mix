"""Deterministic grouped train/validation/test splits with leakage checks and a frozen held-out lock.

Files are grouped by deployment, then session, then recording date, then the file itself. Exact
duplicates (same SHA-256), the same decoded audio in another container (same PCM hash) and recordings
whose time ranges overlap on one recorder are merged into one group so a copy can never cross splits.
Groups are ordered by a seeded hash and assigned greedily by duration toward the configured fractions.

Once a split manifest exists, its validation and test membership is frozen: a later run keeps every
previous assignment, sends new groups to train, and refuses to continue when a held-out file is
missing or changed, or when an appended file would join a held-out group.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from whalemix_training.hashing import canonical_hash

SPLITS = ("train", "validation", "test")
HELDOUT = ("validation", "test")


class SplitLockError(ValueError):
    """The frozen held-out sets cannot be kept for this catalog; nothing was written."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = problems


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


def _interval(entry: dict[str, Any]) -> tuple[str, datetime, datetime] | None:
    recorder = entry.get("recorder_id") or entry.get("deployment")
    start = entry.get("start_utc")
    if not recorder or not start or entry.get("duration_s") is None:
        return None
    try:
        t0 = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
    except ValueError:
        return None
    return str(recorder), t0, t0 + timedelta(seconds=float(entry["duration_s"]))


def time_overlaps(catalog: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """(recorder, file_id_a, file_id_b) for recordings whose time ranges overlap on the same recorder."""
    by_recorder: dict[str, list[tuple[datetime, datetime, str]]] = defaultdict(list)
    for entry in catalog:
        iv = _interval(entry)
        if iv is not None:
            by_recorder[iv[0]].append((iv[1], iv[2], entry["file_id"]))
    pairs = []
    for recorder, items in sorted(by_recorder.items()):
        items.sort()
        for i, (_s0, e0, f0) in enumerate(items):
            for s1, _e1, f1 in items[i + 1 :]:
                if s1 >= e0:
                    break
                pairs.append((recorder, f0, f1))
    return pairs


def build_groups(catalog: list[dict[str, Any]]) -> dict[str, str]:
    """Map file_id -> merged group id (byte duplicates, PCM duplicates and time overlaps merged)."""
    uf = _UnionFind()
    first_by_content: dict[str, str] = {}
    keys = {entry["file_id"]: group_key(entry) for entry in catalog}
    for entry in catalog:
        key = keys[entry["file_id"]]
        uf.find(key)
        for content in (f"sha:{entry['sha256']}", f"pcm:{entry.get('pcm_sha256') or ''}"):
            if content == "pcm:":
                continue
            if content in first_by_content:
                uf.union(key, first_by_content[content])
            else:
                first_by_content[content] = key
    for _recorder, a, b in time_overlaps(catalog):
        uf.union(keys[a], keys[b])
    return {entry["file_id"]: uf.find(keys[entry["file_id"]]) for entry in catalog}


def _order(group: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()


def heldout_hash(assignments: dict[str, dict[str, str]]) -> str:
    return canonical_hash(sorted((f, a["split"]) for f, a in assignments.items() if a["split"] in HELDOUT))


def _frozen_choices(
    readable: list[dict[str, Any]], groups: dict[str, str], previous: dict[str, Any]
) -> dict[str, str]:
    """Group -> split from a previous manifest; raises SplitLockError if the held-out sets cannot hold."""
    old = previous["assignments"]
    present = {e["file_id"] for e in readable}
    problems = [
        f"held-out file {fid} ({a['split']}) is missing or its content changed"
        for fid, a in sorted(old.items())
        if a["split"] in HELDOUT and fid not in present
    ]
    by_group: dict[str, set[str]] = defaultdict(set)
    members: dict[str, list[str]] = defaultdict(list)
    for entry in readable:
        g = groups[entry["file_id"]]
        members[g].append(entry["file_id"])
        if entry["file_id"] in old:
            by_group[g].add(old[entry["file_id"]]["split"])
    choices: dict[str, str] = {}
    for g, fids in sorted(members.items()):
        prior = by_group.get(g, set())
        if len(prior) > 1:
            problems.append(f"group {g} now links files from splits {sorted(prior)} (leakage)")
            continue
        split = next(iter(prior)) if prior else "train"
        new = [f for f in fids if f not in old]
        if split in HELDOUT and new:
            problems.append(
                f"appended file {new[0]} duplicates or overlaps held-out ({split}) data in group {g}"
            )
        choices[g] = split
    if problems:
        raise SplitLockError(problems)
    return choices


def assign_splits(
    catalog: list[dict[str, Any]],
    seed: int,
    fractions: dict[str, float],
    protected_test_groups: frozenset[str] = frozenset(),
    previous: dict[str, Any] | None = None,
    protected_validation_groups: frozenset[str] = frozenset(),
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
    frozen = _frozen_choices(readable, groups, previous) if previous is not None else None
    for group in sorted(duration, key=lambda g: _order(g, seed)):
        if frozen is not None:
            choice = frozen[group]
        elif group in protected_test_groups:
            choice = "test"
        elif group in protected_validation_groups:
            choice = "validation"
        else:
            choice = min(SPLITS, key=lambda s: (filled[s] / total - fractions[s], SPLITS.index(s)))
        group_split[group] = choice
        filled[choice] += duration[group]
    assignments = {
        entry["file_id"]: {"split": group_split[groups[entry["file_id"]]], "group": groups[entry["file_id"]]}
        for entry in readable
    }
    manifest: dict[str, Any] = {
        "seed": seed,
        "fractions": fractions,
        "assignments": dict(sorted(assignments.items())),
        "split_hours": {s: round(filled[s] / 3600.0, 6) for s in SPLITS},
        "heldout_files": {
            e["file_id"]: e["sha256"] for e in readable if assignments[e["file_id"]]["split"] in HELDOUT
        },
    }
    manifest["lock"] = {
        "heldout_hash": heldout_hash(manifest["assignments"]),
        "frozen_from": None if previous is None else previous.get("split_hash"),
        "appended_files": 0 if previous is None else len(set(assignments) - set(previous["assignments"])),
    }
    if previous is not None and previous.get("lock", {}).get("heldout_hash") not in (
        None,
        manifest["lock"]["heldout_hash"],
    ):
        raise SplitLockError(["held-out membership changed while frozen"])
    manifest["split_hash"] = canonical_hash({"seed": seed, "assignments": manifest["assignments"]})
    return manifest


def leakage_check(catalog: list[dict[str, Any]], manifest: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    assignments = manifest["assignments"]
    split_by_hash: dict[str, set[str]] = defaultdict(set)
    split_by_pcm: dict[str, set[str]] = defaultdict(set)
    split_by_group: dict[str, set[str]] = defaultdict(set)
    for entry in catalog:
        a = assignments.get(entry["file_id"])
        if a is None:
            continue
        split_by_hash[entry["sha256"]].add(a["split"])
        if entry.get("pcm_sha256"):
            split_by_pcm[entry["pcm_sha256"]].add(a["split"])
        split_by_group[a["group"]].add(a["split"])
    problems += [
        f"sha256 {h[:12]} appears in splits {sorted(s)}" for h, s in split_by_hash.items() if len(s) > 1
    ]
    problems += [
        f"audio content (pcm {h[:12]}) appears in splits {sorted(s)}"
        for h, s in split_by_pcm.items()
        if len(s) > 1
    ]
    problems += [f"group {g} appears in splits {sorted(s)}" for g, s in split_by_group.items() if len(s) > 1]
    for recorder, a, b in time_overlaps([e for e in catalog if e["file_id"] in assignments]):
        sa, sb = assignments[a]["split"], assignments[b]["split"]
        if sa != sb:
            problems.append(f"time overlap on recorder {recorder}: {a} ({sa}) and {b} ({sb})")
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
