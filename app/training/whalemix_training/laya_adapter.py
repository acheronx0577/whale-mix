"""The single training-side Laya adapter (Laya 0.4.6 CLI, typed `choice` questions).

Deterministic code ranks evidence groups by measured impact first. Laya then classifies the top
unresolved groups into one versioned set of next actions. Its answer is advisory: it is stored next to the
measurements and never changes a gate, label or threshold. Any failure (not configured, missing
checkpoint, timeout, invalid JSON, unsupported answer) falls back to the deterministic ranking.
Results are cached by (package version, model revision, question-set hash, evidence hash).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from importlib import resources
from pathlib import Path
from typing import Any

from whalemix_training.hashing import canonical_hash, canonical_json

Runner = Callable[..., subprocess.CompletedProcess]


def question_set() -> dict[str, Any]:
    text = resources.files("whalemix_training").joinpath("laya_questions.json").read_text(encoding="utf-8")
    return json.loads(text)


def rank_groups(groups: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Distinct actionable groups, highest measured impact first (ties by id)."""
    distinct = {g["group_id"]: g for g in groups if g.get("actionable", True)}
    return sorted(distinct.values(), key=lambda g: (-float(g["impact"]), g["group_id"]))[:limit]


def _request_line(group: dict[str, Any]) -> str:
    return canonical_json(
        {"stage": group["stage"], "group": group["group_id"], "evidence": group["evidence"]}
    )


def _peak_child_rss_mb() -> float | None:
    if sys.platform.startswith("win"):
        return None
    import resource

    return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0


def advise(
    groups: list[dict[str, Any]],
    laya_cfg: dict[str, Any],
    cache_dir: Path,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    qs = question_set()
    ranked = rank_groups(groups, int(laya_cfg.get("batch_size", 8)))
    fallback = {
        "source": "deterministic_ranking",
        "ranking": [g["group_id"] for g in ranked],
        "recommendations": [],
    }
    if not ranked:
        return {**fallback, "reason": "no actionable evidence groups"}
    if not laya_cfg.get("enabled"):
        return {**fallback, "reason": "laya disabled in config"}
    if not laya_cfg.get("model") or not laya_cfg.get("package_version") or not laya_cfg.get("model_revision"):
        return {
            **fallback,
            "reason": "SETUP_REQUIRED: pin laya.model, laya.model_revision and laya.package_version",
        }
    key_parts = {
        "package_version": laya_cfg["package_version"],
        "model_revision": laya_cfg["model_revision"],
        "question_set_hash": canonical_hash(qs),
        "evidence_hash": canonical_hash([_request_line(g) for g in ranked]),
    }
    cache_path = cache_dir / f"{canonical_hash(key_parts)}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("key") == key_parts:
            return {**cached["result"], "cache": "hit"}
    with tempfile.TemporaryDirectory(prefix="whalemix-laya-") as tmp:
        qfile, bfile = Path(tmp) / "questions.json", Path(tmp) / "batch.txt"
        qfile.write_text(json.dumps({"questions": qs["questions"]}), encoding="utf-8")
        bfile.write_text("\n".join(_request_line(g) for g in ranked) + "\n", encoding="utf-8")
        args = [
            str(laya_cfg.get("executable", "laya")),
            "--questions",
            str(qfile),
            "--batch",
            str(bfile),
            "--batch-size",
            str(len(ranked)),
            "--json",
            "--device",
            "cpu",
            "--model",
            str(laya_cfg["model"]),
        ]
        started = time.monotonic()
        try:
            proc = runner(
                args,
                capture_output=True,
                text=True,
                timeout=float(laya_cfg.get("timeout_s", 120)),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {**fallback, "reason": f"laya unavailable: {type(exc).__name__}"}
        latency = time.monotonic() - started
    if proc.returncode != 0:
        return {**fallback, "reason": f"laya exited {proc.returncode}"}
    labels = set(qs["questions"]["next_action"]["criteria"])
    recs = []
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if len(lines) != len(ranked):
        return {**fallback, "reason": "laya returned an unexpected number of answers"}
    for group, line in zip(ranked, lines, strict=True):
        try:
            answer = json.loads(line)["answers"]["next_action"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return {**fallback, "reason": "laya returned invalid JSON"}
        if answer.get("type") != "choice" or answer.get("choice") not in labels:
            return {**fallback, "reason": "laya returned an unsupported answer"}
        recs.append(
            {
                "group_id": group["group_id"],
                "choice": answer["choice"],
                "confidence": answer.get("confidence"),
                "evidence_link": group.get("evidence_link"),
                "advisory": True,
            }
        )
    result = {
        "source": "laya",
        "ranking": [g["group_id"] for g in ranked],
        "recommendations": recs,
        "latency_s": round(latency, 3),
        "peak_child_rss_mb": _peak_child_rss_mb(),
        "key": key_parts,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"key": key_parts, "result": result}, sort_keys=True), encoding="utf-8")
    return {**result, "cache": "miss"}
