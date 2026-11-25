"""Annotation manifest validation and the coverage matrix.

Annotations are CSV rows with explicit review status and provenance. Overlaps are allowed. Only
`reviewed` rows become supervised targets; ambiguous/unreviewed rows stay masked. Model call-index output
is an annotation aid, never ground truth, unless a reviewer marked it reviewed.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from whalemix_training.labels import ABSENT, PRESENT, STATES, LabelMap, is_assessable

REQUIRED_COLUMNS = (
    "annotation_id",
    "file_id",
    "channel",
    "start_s",
    "end_s",
    "label",
    "state",
    "review_status",
    "provenance",
)
REVIEW_STATUSES = ("reviewed", "unreviewed", "ambiguous")
DURATION_TOLERANCE_S = 0.05


def read_annotations(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: annotation CSV is missing columns: {', '.join(missing)}")
        return list(reader)


def _check_row(
    row: dict[str, str], source: dict[str, Any] | None, label_map: LabelMap
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    label, state, review = row["label"], row["state"], row["review_status"]
    if label not in label_map.outputs:
        errors.append(f"unknown label {label!r}")
    if state not in STATES:
        errors.append(f"unknown state {state!r}")
    if review not in REVIEW_STATUSES:
        errors.append(f"unknown review_status {review!r}")
    if not row["provenance"].strip():
        errors.append("missing provenance")
    try:
        start, end, channel = float(row["start_s"]), float(row["end_s"]), int(row["channel"])
    except ValueError:
        errors.append("start_s/end_s/channel must be numeric")
        return errors, warnings
    if not (0.0 <= start < end):
        errors.append("start_s must be >= 0 and < end_s")
    if source is None:
        errors.append(f"file_id {row['file_id']!r} is not in the source catalog")
        return errors, warnings
    if source.get("status") != "ok":
        errors.append("source file is unreadable")
        return errors, warnings
    if end > source["duration_s"] + DURATION_TOLERANCE_S:
        errors.append("end_s is beyond the recording duration")
    if not 0 <= channel < source["channels"]:
        errors.append("channel is out of range for the recording")
    if state == ABSENT and label in label_map.outputs and not is_assessable(label, source["sample_rate"]):
        errors.append("reviewed absence is outside the recording's bandwidth (mark it unassessed)")
    if "call-index" in row["provenance"].lower() and review == "reviewed":
        warnings.append("call-index derived row marked reviewed; confirm a human review happened")
    return errors, warnings


def validate(
    rows: list[dict[str, str]], catalog: list[dict[str, Any]], label_map: LabelMap
) -> dict[str, Any]:
    sources = {entry["file_id"]: entry for entry in catalog}
    issues: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for number, row in enumerate(rows, start=2):
        errors, warnings = _check_row(row, sources.get(row["file_id"]), label_map)
        if row["annotation_id"] in seen_ids:
            errors.append("duplicate annotation_id")
        seen_ids.add(row["annotation_id"])
        for message in errors:
            issues.append(
                {"line": number, "annotation_id": row["annotation_id"], "level": "error", "message": message}
            )
        for message in warnings:
            issues.append(
                {
                    "line": number,
                    "annotation_id": row["annotation_id"],
                    "level": "warning",
                    "message": message,
                }
            )
    error_count = sum(1 for i in issues if i["level"] == "error")
    return {
        "rows": len(rows),
        "errors": error_count,
        "warnings": len(issues) - error_count,
        "issues": issues,
        "coverage": coverage_matrix(rows, sources, label_map),
    }


def coverage_matrix(
    rows: list[dict[str, str]], sources: dict[str, dict[str, Any]], label_map: LabelMap
) -> dict[str, Any]:
    """Reviewed coverage per label: present events, reviewed-absent hours, deployments and years."""
    events: dict[str, int] = defaultdict(int)
    absent_s: dict[str, float] = defaultdict(float)
    deployments: dict[str, set[str]] = defaultdict(set)
    years: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("review_status") != "reviewed" or row.get("label") not in label_map.outputs:
            continue
        source = sources.get(row["file_id"])
        if source is None or source.get("status") != "ok":
            continue
        label = row["label"]
        if row["state"] == PRESENT:
            events[label] += 1
        elif row["state"] == ABSENT:
            absent_s[label] += max(0.0, float(row["end_s"]) - float(row["start_s"]))
        deployments[label].add(source.get("deployment") or f"file:{source['file_id']}")
        years[label].add((source.get("recording_date") or "")[:4] or "unknown")
    matrix = {}
    for label in label_map.outputs:
        n = events.get(label, 0)
        matrix[label] = {
            "reviewed_present_events": n,
            "reviewed_absent_hours": round(absent_s.get(label, 0.0) / 3600.0, 6),
            "independent_groups": len(deployments.get(label, set())),
            "years": sorted(years.get(label, set())),
            "status": "covered" if n > 0 else "absent_or_unverified",
        }
    return matrix
