"""Read-only source inventory.

Walks a configured recording root, hashes and probes each audio file, and writes an incremental on-disk
catalog. Unchanged files (same relative path, size and mtime) reuse their previous entry so a large
corpus is not re-hashed on every run. The recording root is opened read-only and never written.
"""

from __future__ import annotations

import csv
import hashlib
import os
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import soundfile as sf

from whalemix_training.hashing import canonical_hash, sha256_file
from whalemix_training.io_utils import assert_outside_root, atomic_write_json, atomic_write_jsonl, read_jsonl

METADATA_FIELDS = (
    "recording_date",
    "session",
    "deployment",
    "calibration_status",
    "provenance",
    "license",
    "annotation_status",
)


def file_id_for(relative_path: str, digest: str) -> str:
    """Unique per catalog entry: exact duplicates at different paths keep distinct IDs but share sha256."""
    return hashlib.sha256(f"{relative_path}\0{digest}".encode()).hexdigest()[:16]


def iter_audio_files(root: Path, extensions: list[str], max_files: int | None = None) -> Iterator[Path]:
    """Deterministic lazy walk (sorted per directory), following no symlinks to avoid loops."""
    allowed = {ext.lower() for ext in extensions}
    count = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for name in sorted(filenames):
            if Path(name).suffix.lower() in allowed:
                yield Path(dirpath) / name
                count += 1
                if max_files is not None and count >= max_files:
                    return


def load_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "relative_path" not in reader.fieldnames:
            raise ValueError(f"{path}: metadata CSV needs a relative_path column")
        return {row["relative_path"].replace("\\", "/"): row for row in reader}


def probe(path: Path) -> dict[str, Any]:
    try:
        info = sf.info(str(path))
    except (RuntimeError, sf.LibsndfileError) as exc:
        return {"status": "unreadable", "error": str(exc).splitlines()[0][:200]}
    return {
        "status": "ok",
        "duration_s": float(info.frames) / float(info.samplerate) if info.samplerate else 0.0,
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "encoding": f"{info.format}/{info.subtype}",
        "frames": int(info.frames),
    }


def build_inventory(
    root: Path,
    out_dir: Path,
    extensions: list[str],
    metadata_csv: Path | None = None,
    max_files: int | None = None,
) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(f"recording root does not exist or is not a directory: {root}")
    assert_outside_root(out_dir, root)
    catalog_path = out_dir / "source-catalog.jsonl"
    previous: dict[str, dict[str, Any]] = {}
    if catalog_path.exists():
        previous = {row["relative_path"]: row for row in read_jsonl(catalog_path)}
    metadata = load_metadata(metadata_csv)

    entries: list[dict[str, Any]] = []
    reused = 0
    for path in iter_audio_files(root, extensions, max_files):
        rel = path.relative_to(root).as_posix()
        stat = path.stat()
        old = previous.get(rel)
        if old and old.get("size_bytes") == stat.st_size and old.get("mtime_ns") == stat.st_mtime_ns:
            entry = dict(old)
            reused += 1
        else:
            digest = sha256_file(path)
            entry = {
                "file_id": file_id_for(rel, digest),
                "relative_path": rel,
                "sha256": digest,
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                **probe(path),
            }
        meta = metadata.get(rel, {})
        for field in METADATA_FIELDS:
            entry[field] = (meta.get(field) or "").strip() or None
        entry["annotation_status"] = entry["annotation_status"] or "unannotated"
        entry["calibration_status"] = entry["calibration_status"] or "unknown"
        entries.append(entry)

    report = summarize(entries)
    report["reused_entries"] = reused
    report["catalog_hash"] = canonical_hash([e["sha256"] + ":" + e["relative_path"] for e in entries])
    atomic_write_jsonl(catalog_path, entries)
    atomic_write_json(out_dir / "inventory-report.json", report)
    return report


def summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    hours_by_year: dict[str, float] = defaultdict(float)
    deployments: set[str] = set()
    missing_provenance: list[str] = []
    unreadable: list[str] = []
    total_s = 0.0
    for entry in entries:
        by_hash[entry["sha256"]].append(entry["relative_path"])
        if entry.get("status") != "ok":
            unreadable.append(entry["relative_path"])
            continue
        total_s += entry["duration_s"]
        year = (entry.get("recording_date") or "")[:4] or "unknown"
        hours_by_year[year] += entry["duration_s"] / 3600.0
        if entry.get("deployment"):
            deployments.add(entry["deployment"])
        if not entry.get("provenance") or not entry.get("license"):
            missing_provenance.append(entry["relative_path"])
    duplicates = {h: paths for h, paths in by_hash.items() if len(paths) > 1}
    return {
        "files": len(entries),
        "readable_files": len(entries) - len(unreadable),
        "unreadable_files": unreadable,
        "total_hours": round(total_s / 3600.0, 6),
        "hours_by_year": {k: round(v, 6) for k, v in sorted(hours_by_year.items())},
        "independent_deployments": len(deployments),
        "exact_duplicate_groups": duplicates,
        "missing_provenance_or_license": missing_provenance,
    }


def load_catalog(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))
