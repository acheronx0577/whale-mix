"""quality_report.json plus a short readable report for each evaluated stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from whalemix_training.io_utils import atomic_write_json, atomic_write_text


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_reports(out_dir: Path, report: dict[str, Any]) -> None:
    atomic_write_json(out_dir / "quality_report.json", report)
    lines = [
        f"# Quality report: {report.get('stage', 'stage')}",
        "",
        f"- Gate: **{report['gate']['status']}**",
        f"- Model weights SHA-256: `{report.get('weights_sha256')}`",
        f"- Eval set hash: `{report.get('eval_set_hash')}`",
        f"- Match rule: `{report['match_rule']['version']}`",
        f"- Operating budget: {report['operating_budget']} false alarms/hour",
        f"- Fixture data: {report.get('fixture', False)}",
        "",
        "| Label | Truth | TP | FP | FN | Precision | Recall | FA/hour | AP | Evidence |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, m in sorted(report["labels"].items()):
        lines.append(
            f"| {label} | {m['truth_events']} | {m['tp']} | {m['fp']} | {m['fn']} | {_fmt(m['precision'])} | "
            f"{_fmt(m['recall'])} | {_fmt(m['false_alarms_per_hour'])} | {_fmt(m['average_precision'])} | "
            f"{m['evidence']} |"
        )
    lines += ["", "## Gate reasons", ""] + [f"- {r}" for r in report["gate"]["reasons"] or ["none"]]
    advice = report.get("advice", {})
    lines += ["", f"## Advice ({advice.get('source', 'none')}, advisory only)", ""]
    for rec in advice.get("recommendations", []):
        lines.append(f"- {rec['group_id']}: {rec['choice']} (confidence {_fmt(rec.get('confidence'))})")
    if advice.get("reason"):
        lines.append(f"- {advice['reason']}")
    atomic_write_text(out_dir / "Quality-Report.md", "\n".join(lines) + "\n")
