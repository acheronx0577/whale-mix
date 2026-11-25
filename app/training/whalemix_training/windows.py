"""Training-window manifest: window timing from the preprocessing contract plus per-label states.

A label is `present` in a window when a reviewed present annotation overlaps it, `absent` only when a
reviewed absent annotation covers the whole window and the recording bandwidth supports the label, and
`unassessed` otherwise. Unreviewed or ambiguous annotations never become supervised targets.
"""

from __future__ import annotations

from typing import Any

from whalemix_training.labels import ABSENT, PRESENT, UNASSESSED, LabelMap, is_assessable
from whalemix_training.preprocess import PreprocessingSpec, resample_ratio, resampled_length, window_starts


def frame_count(duration_frames: int, src_sr: int, spec: PreprocessingSpec) -> int:
    up, down = resample_ratio(src_sr, spec.sample_rate)
    n = max(
        resampled_length(duration_frames, up, down), spec.n_fft + (spec.window_frames - 1) * spec.hop_length
    )
    return 1 + (n - spec.n_fft) // spec.hop_length


def window_states(
    start_s: float, end_s: float, rows: list[dict[str, str]], sample_rate: int, label_map: LabelMap
) -> dict[str, str]:
    states = dict.fromkeys(label_map.outputs, UNASSESSED)
    for row in rows:
        if row["review_status"] != "reviewed":
            continue
        a0, a1, label = float(row["start_s"]), float(row["end_s"]), row["label"]
        if row["state"] == PRESENT and a0 < end_s and a1 > start_s:
            states[label] = PRESENT
        elif (
            row["state"] == ABSENT
            and a0 <= start_s
            and a1 >= end_s
            and states[label] != PRESENT
            and is_assessable(label, sample_rate)
        ):
            states[label] = ABSENT
    return states


def build_windows(
    catalog: list[dict[str, Any]],
    annotations: list[dict[str, str]],
    split_manifest: dict[str, Any],
    spec: PreprocessingSpec,
    label_map: LabelMap,
) -> list[dict[str, Any]]:
    by_source: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in annotations:
        by_source.setdefault((row["file_id"], int(row["channel"])), []).append(row)
    windows = []
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
                states = window_states(start_s, start_s + span, rows, entry["sample_rate"], label_map)
                windows.append(
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
    return windows
