"""Training-window manifest: window timing from the preprocessing contract plus per-label states.

A label is `present` in a window when a reviewed present annotation overlaps it, `absent` only when a
reviewed absent annotation covers the whole window, and `unassessed` otherwise. Both supervised states
also need the effective bandwidth (source Nyquist capped by the preprocessing band) to cover the label;
out-of-band positives and negatives stay masked. Unreviewed or ambiguous annotations never become
supervised targets.

Window generation is bounded: `max_windows` caps how many window records are materialised, and
`reviewed_only` keeps just the windows that carry at least one supervised target (training needs only
those). `count_windows` sizes a corpus without building records.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from whalemix_training.labels import ABSENT, PRESENT, UNASSESSED, LabelMap, is_assessable
from whalemix_training.preprocess import PreprocessingSpec, resample_ratio, resampled_length, window_starts


def frame_count(duration_frames: int, src_sr: int, spec: PreprocessingSpec) -> int:
    up, down = resample_ratio(src_sr, spec.sample_rate)
    n = max(
        resampled_length(duration_frames, up, down), spec.n_fft + (spec.window_frames - 1) * spec.hop_length
    )
    return 1 + (n - spec.n_fft) // spec.hop_length


DEFAULT_MAX_WINDOWS = 2_000_000


class WindowLimitError(ValueError):
    """Raised instead of materialising more window records than the configured bound."""


def window_states(
    start_s: float,
    end_s: float,
    rows: list[dict[str, str]],
    sample_rate: int,
    label_map: LabelMap,
    spec: PreprocessingSpec | None = None,
) -> dict[str, str]:
    states = dict.fromkeys(label_map.outputs, UNASSESSED)
    for row in rows:
        if row["review_status"] != "reviewed":
            continue
        a0, a1, label = float(row["start_s"]), float(row["end_s"]), row["label"]
        if not is_assessable(label, sample_rate, spec):
            continue
        if row["state"] == PRESENT and a0 < end_s and a1 > start_s:
            states[label] = PRESENT
        elif row["state"] == ABSENT and a0 <= start_s and a1 >= end_s and states[label] != PRESENT:
            states[label] = ABSENT
    return states


def count_windows(
    catalog: list[dict[str, Any]], split_manifest: dict[str, Any], spec: PreprocessingSpec
) -> dict[str, int]:
    """Windows per split (all channels) without building any record."""
    counts: dict[str, int] = {}
    for entry in catalog:
        assignment = split_manifest["assignments"].get(entry["file_id"])
        if entry.get("status") != "ok" or assignment is None:
            continue
        n = len(window_starts(frame_count(entry["frames"], entry["sample_rate"], spec), spec))
        counts[assignment["split"]] = counts.get(assignment["split"], 0) + n * int(entry["channels"])
    return counts


def build_windows(
    catalog: list[dict[str, Any]],
    annotations: list[dict[str, str]],
    split_manifest: dict[str, Any],
    spec: PreprocessingSpec,
    label_map: LabelMap,
    *,
    reviewed_only: bool = False,
    max_windows: int | None = None,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for window in iter_windows(catalog, annotations, split_manifest, spec, label_map):
        if reviewed_only and not window["reviewed"]:
            continue
        if max_windows is not None and len(windows) >= max_windows:
            raise WindowLimitError(
                f"more than {max_windows} {'reviewed ' if reviewed_only else ''}windows; raise "
                "training.max_windows deliberately or narrow the corpus"
            )
        windows.append(window)
    return windows


def iter_windows(
    catalog: list[dict[str, Any]],
    annotations: list[dict[str, str]],
    split_manifest: dict[str, Any],
    spec: PreprocessingSpec,
    label_map: LabelMap,
) -> Iterator[dict[str, Any]]:
    by_source: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in annotations:
        by_source.setdefault((row["file_id"], int(row["channel"])), []).append(row)
    for entry in catalog:
        assignment = split_manifest["assignments"].get(entry["file_id"])
        if entry.get("status") != "ok" or assignment is None:
            continue
        n_frames = frame_count(entry["frames"], entry["sample_rate"], spec)
        span = ((spec.window_frames - 1) * spec.hop_length + spec.n_fft) / spec.sample_rate
        for channel in range(entry["channels"]):
            rows = by_source.get((entry["file_id"], channel), [])
            for start in window_starts(n_frames, spec):
                start_s = start * spec.hop_length / spec.sample_rate
                states = window_states(start_s, start_s + span, rows, entry["sample_rate"], label_map, spec)
                yield (
                    {
                        "example_id": f"{entry['file_id']}:{channel}:{start}",
                        "file_id": entry["file_id"],
                        "channel": channel,
                        "start_frame": start,
                        "start_s": start_s,
                        "end_s": start_s + span,
                        "split": assignment["split"],
                        "group": assignment["group"],
                        "year": (entry.get("recording_date") or "")[:4] or "unknown",
                        "deployment": entry.get("deployment") or assignment["group"],
                        "states": states,
                        "reviewed": any(v != UNASSESSED for v in states.values()),
                    }
                )
