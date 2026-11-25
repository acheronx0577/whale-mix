"""Deterministic candidate gates and atomic selection. Laya advice can never change a gate outcome."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from whalemix_training.io_utils import atomic_write_json

PASS_ELIGIBLE = "PASS_ELIGIBLE"
FAIL = "FAIL"
INCONCLUSIVE = "INCONCLUSIVE"
SETUP_REQUIRED = "SETUP_REQUIRED"
COMPARABLE_KEYS = ("eval_set_hash", "match_rule", "operating_budget", "preprocessing_hash")


def _recall(report: dict[str, Any], label: str) -> float | None:
    m = report["labels"].get(label)
    return None if m is None else m.get("recall")


def gate(candidate: dict[str, Any], parent: dict[str, Any] | None, policy: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    required = ("eval_set_hash", "labels", "resources", "match_rule", "operating_budget")
    missing = [k for k in required if k not in candidate]
    if missing:
        return {"status": SETUP_REQUIRED, "reasons": [f"candidate report missing {', '.join(missing)}"]}
    if parent is not None:
        differing = [k for k in COMPARABLE_KEYS if candidate.get(k) != parent.get(k)]
        if differing:
            return {
                "status": SETUP_REQUIRED,
                "reasons": [f"not comparable with parent: {', '.join(differing)}"],
            }
    if candidate["resources"].get("within_budget") is False:
        return {"status": FAIL, "reasons": ["resource budget exceeded"]}
    labels = sorted(candidate["labels"])
    insufficient = [lbl for lbl in labels if candidate["labels"][lbl].get("evidence") != "sufficient"]
    for label in labels:
        fa = candidate["labels"][label].get("false_alarms_per_hour")
        if fa is not None and fa > policy["false_alarms_per_hour_budget"]:
            reasons.append(f"{label}: false alarms/hour {fa:.3f} above budget")
    if parent is not None:
        for label in labels:
            new, old = _recall(candidate, label), _recall(parent, label)
            if new is not None and old is not None and old - new > policy["max_retention_drop"]:
                reasons.append(f"{label}: recall dropped {old - new:.3f} (retention)")
        for slice_name, old_value in parent.get("retention", {}).items():
            new_value = candidate.get("retention", {}).get(slice_name)
            if new_value is None:
                reasons.append(f"retention slice {slice_name} missing")
            elif old_value - new_value > policy["max_retention_drop"]:
                reasons.append(f"retention slice {slice_name} dropped {old_value - new_value:.3f}")
    if reasons:
        return {"status": FAIL, "reasons": reasons}
    if insufficient:
        return {"status": INCONCLUSIVE, "reasons": [f"insufficient evidence for {', '.join(insufficient)}"]}
    if parent is not None:
        gains = [
            (_recall(candidate, lbl) or 0.0) - (_recall(parent, lbl) or 0.0)
            for lbl in labels
            if _recall(parent, lbl) is not None
        ]
        primary = sum(gains) / len(gains) if gains else 0.0
        lower = [candidate["labels"][lbl].get("recall_ci95") for lbl in labels]
        if primary < policy["min_primary_gain"]:
            return {"status": INCONCLUSIVE, "reasons": [f"primary gain {primary:.4f} below minimum"]}
        if any(ci is None for ci in lower):
            return {"status": INCONCLUSIVE, "reasons": ["missing uncertainty intervals"]}
    return {"status": PASS_ELIGIBLE, "reasons": []}


def select(selection_dir: Path, candidate: dict[str, Any], outcome: dict[str, Any]) -> dict[str, Any]:
    """Append to the improvement ledger; atomically move the selection pointer only on PASS_ELIGIBLE."""
    selection_dir.mkdir(parents=True, exist_ok=True)
    pointer = selection_dir / "selected.json"
    previous = json.loads(pointer.read_text(encoding="utf-8")) if pointer.exists() else None
    entry = {
        "candidate_weights_sha256": candidate.get("weights_sha256"),
        "parent_weights_sha256": None if previous is None else previous.get("weights_sha256"),
        "status": outcome["status"],
        "reasons": outcome["reasons"],
        "eval_set_hash": candidate.get("eval_set_hash"),
    }
    with (selection_dir / "improvement-ledger.jsonl").open("a", encoding="utf-8") as ledger:
        ledger.write(json.dumps(entry, sort_keys=True) + "\n")
    if outcome["status"] == PASS_ELIGIBLE:
        atomic_write_json(pointer, {"weights_sha256": candidate.get("weights_sha256"), "report": candidate})
        return {"selected": candidate.get("weights_sha256"), "changed": True}
    return {"selected": None if previous is None else previous.get("weights_sha256"), "changed": False}
