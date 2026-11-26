"""Event-level detection metrics with a fixed, documented matching rule.

Matching rule (v1, set before any comparison):
- Predictions of the same recording/channel/label whose intervals overlap or sit within `merge_gap_s`
  are merged first (max score kept); with the default gap of 0 s, distinct nearby pulses stay separate.
- A prediction can match a truth event of the same recording, channel and label when their intervals
  overlap by at least `min_overlap_fraction` of the shorter one, or their centres are within
  `center_tolerance_s`.
- Matching is one-to-one and greedy in descending score order (ties by start time), so overlapping
  windows cannot count one real call twice. Unmatched predictions are false alarms; unmatched truth
  events are misses.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

MATCH_RULE_VERSION = "event-match-v1"


@dataclass(frozen=True)
class MatchRule:
    min_overlap_fraction: float = 0.1
    center_tolerance_s: float = 2.0
    merge_gap_s: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {"version": MATCH_RULE_VERSION, **self.__dict__}


def _key(e: dict[str, Any]) -> tuple[str, int, str]:
    return (str(e["recording_id"]), int(e.get("channel", 0)), str(e["label"]))


def merge_predictions(preds: list[dict[str, Any]], gap_s: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in sorted({_key(p) for p in preds}):
        group = sorted((p for p in preds if _key(p) == key), key=lambda p: (p["start_s"], p["end_s"]))
        current: dict[str, Any] | None = None
        for p in group:
            overlaps = current is not None and (
                p["start_s"] < current["end_s"] or (gap_s > 0 and p["start_s"] <= current["end_s"] + gap_s)
            )
            if current is not None and overlaps:
                current["end_s"] = max(current["end_s"], p["end_s"])
                current["score"] = max(current["score"], p["score"])
                current["merged"] = current.get("merged", 1) + 1
            else:
                if current is not None:
                    out.append(current)
                current = dict(p)
        if current is not None:
            out.append(current)
    return out


def _compatible(p: dict[str, Any], t: dict[str, Any], rule: MatchRule) -> float | None:
    """Return a match quality (larger is better) or None."""
    overlap = min(p["end_s"], t["end_s"]) - max(p["start_s"], t["start_s"])
    shorter = max(1e-9, min(p["end_s"] - p["start_s"], t["end_s"] - t["start_s"]))
    centre_gap = abs((p["start_s"] + p["end_s"]) / 2 - (t["start_s"] + t["end_s"]) / 2)
    if overlap > 0 and overlap / shorter >= rule.min_overlap_fraction:
        return 1.0 + overlap / shorter
    if centre_gap <= rule.center_tolerance_s:
        return 1.0 - centre_gap / (rule.center_tolerance_s + 1e-9)
    return None


def match(truth: list[dict[str, Any]], preds: list[dict[str, Any]], rule: MatchRule) -> dict[str, Any]:
    merged = merge_predictions(preds, rule.merge_gap_s)
    ordered = sorted(merged, key=lambda p: (-p["score"], p["start_s"], str(p["recording_id"])))
    truth_by_key: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for i, t in enumerate(truth):
        truth_by_key[_key(t)].append(i)
    used: set[int] = set()
    outcomes = []
    for p in ordered:
        best, best_q = None, None
        for i in truth_by_key.get(_key(p), []):
            if i in used:
                continue
            q = _compatible(p, truth[i], rule)
            if q is not None and (best_q is None or q > best_q):
                best, best_q = i, q
        if best is not None:
            used.add(best)
        outcomes.append({"prediction": p, "truth_index": best})
    misses = [truth[i] for i in range(len(truth)) if i not in used]
    return {"outcomes": outcomes, "misses": misses}


def average_precision(outcomes: list[dict[str, Any]], n_truth: int) -> float | None:
    if n_truth == 0:
        return None
    tp = 0
    total = 0.0
    for rank, o in enumerate(outcomes, start=1):
        if o["truth_index"] is not None:
            tp += 1
            total += tp / rank
    return total / n_truth


def _in_ranges(p: dict[str, Any], ranges: list[dict[str, Any]]) -> bool:
    centre = (p["start_s"] + p["end_s"]) / 2
    return any(
        str(r["recording_id"]) == str(p["recording_id"]) and r["start_s"] <= centre < r["end_s"]
        for r in ranges
    )


def _union(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for start, end in sorted(spans):
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        elif end > start:
            out.append((start, end))
    return out


def _by_track(events: list[dict[str, Any]], label: str) -> dict[tuple[str, int], list[tuple[float, float]]]:
    tracks: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    for e in events:
        if e["label"] == label:
            tracks[(str(e["recording_id"]), int(e.get("channel", 0)))].append((e["start_s"], e["end_s"]))
    return {k: _union(v) for k, v in tracks.items()}


def _overlap_s(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    return sum(max(0.0, min(e1, e2) - max(s1, s2)) for s1, e1 in a for s2, e2 in b)


def reviewed_exposure(
    truth: list[dict[str, Any]], reviewed: list[dict[str, Any]], label: str
) -> tuple[dict[tuple[str, int], list[tuple[float, float]]], float, float]:
    """Reviewed assessable spans per (recording, channel), their hours, and the background hours (reviewed
    time outside this label's reviewed calls)."""
    spans = _by_track(reviewed, label)
    calls = _by_track(truth, label)
    total = sum(e - s for track in spans.values() for s, e in track)
    inside_calls = sum(_overlap_s(track, calls.get(key, [])) for key, track in spans.items())
    return spans, total / 3600.0, (total - inside_calls) / 3600.0


def _centre_in(p: dict[str, Any], spans: dict[tuple[str, int], list[tuple[float, float]]]) -> bool:
    centre = (p["start_s"] + p["end_s"]) / 2
    track = spans.get((str(p["recording_id"]), int(p.get("channel", 0))), [])
    return any(s <= centre < e for s, e in track)


def label_metrics(
    truth: list[dict[str, Any]],
    preds: list[dict[str, Any]],
    label: str,
    analyzed_hours: float,
    rule: MatchRule,
    threshold: float,
    slices: dict[str, list[dict[str, Any]]] | None = None,
    min_events: int = 20,
    reviewed: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """With `reviewed` (the eval set's reviewed assessable intervals) false alarms are counted only inside
    reviewed time and divided by reviewed hours; detections elsewhere are reported as unreviewed, since
    nobody checked whether a call was there. Without it the legacy analysed-hours basis is used and the
    report carries no reviewed background exposure."""
    t = [e for e in truth if e["label"] == label]
    p_all = [e for e in preds if e["label"] == label]
    ranked = match(t, p_all, rule)
    ap = average_precision(ranked["outcomes"], len(t))
    result = match(t, [e for e in p_all if e["score"] >= threshold], rule)
    tp = sum(1 for o in result["outcomes"] if o["truth_index"] is not None)
    fp_events = [o["prediction"] for o in result["outcomes"] if o["truth_index"] is None]
    unreviewed = 0
    exposure: dict[str, Any] = {"exposure_basis": "analyzed"}
    if reviewed is not None:
        spans, analyzed_hours, background = reviewed_exposure(truth, reviewed, label)
        unreviewed = sum(1 for p in fp_events if not _centre_in(p, spans))
        fp_events = [p for p in fp_events if _centre_in(p, spans)]
        exposure = {
            "exposure_basis": "reviewed",
            "reviewed_hours": analyzed_hours,
            "reviewed_background_hours": background,
            "unreviewed_detections": unreviewed,
        }
    fn = len(result["misses"])
    review = sum(1 for o in result["outcomes"] if o["prediction"].get("review"))
    out: dict[str, Any] = {
        "label": label,
        "threshold": threshold,
        "truth_events": len(t),
        "tp": tp,
        "fp": len(fp_events),
        "fn": fn,
        "precision": tp / (tp + len(fp_events)) if tp + len(fp_events) else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "average_precision": ap,
        "analyzed_hours": analyzed_hours,
        "false_alarms_per_hour": len(fp_events) / analyzed_hours if analyzed_hours > 0 else None,
        "review_events_per_hour": review / analyzed_hours if analyzed_hours > 0 else None,
        "evidence": "sufficient" if len(t) >= min_events else "insufficient_evidence",
        "recall_ci95": grouped_recall_ci(result, t) if len(t) >= min_events else None,
        **exposure,
    }
    if slices:
        out["false_alarms_by_slice"] = {}
        for name, ranges in sorted(slices.items()):
            hours = sum(r["end_s"] - r["start_s"] for r in ranges) / 3600.0
            n = sum(1 for p in fp_events if _in_ranges(p, ranges))
            out["false_alarms_by_slice"][name] = {
                "hours": hours,
                "false_alarms": n,
                "per_hour": n / hours if hours > 0 else None,
                "evidence": "sufficient" if hours > 0 else "insufficient_evidence",
            }
    return out


def grouped_recall_ci(result: dict[str, Any], truth: list[dict[str, Any]], n_boot: int = 1000, seed: int = 0):
    """95% recording-grouped bootstrap interval for recall."""
    found: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    matched_ids = {id(truth[o["truth_index"]]) for o in result["outcomes"] if o["truth_index"] is not None}
    for e in truth:
        found[str(e["recording_id"])][0] += int(id(e) in matched_ids)
        found[str(e["recording_id"])][1] += 1
    groups = sorted(found)
    # Seeded, reproducible resampling for a confidence interval; not used for anything security-related.
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = [found[groups[int(i)]] for i in rng.integers(0, len(groups), size=len(groups))]
        total = sum(g[1] for g in pick)
        stats.append(sum(g[0] for g in pick) / total if total else 0.0)
    stats.sort()
    return [stats[int(0.025 * (n_boot - 1))], stats[int(0.975 * (n_boot - 1))]]
