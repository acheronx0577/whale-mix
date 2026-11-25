"""evaluate, export, quantize and benchmark commands."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from whalemix_training import checkpoint as ckpt
from whalemix_training import export as exp
from whalemix_training import laya_adapter, selection
from whalemix_training.cli import _EXTRA_ARGS, _out_dir, _print, command
from whalemix_training.hashing import canonical_hash
from whalemix_training.io_utils import atomic_write_json
from whalemix_training.labels import default_label_map
from whalemix_training.metrics import MatchRule, label_metrics
from whalemix_training.model import build_model
from whalemix_training.preprocess import PreprocessingSpec, analyze_channel
from whalemix_training.report import write_reports


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evidence_groups(report: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    groups = []
    for label, m in report["labels"].items():
        if m["fn"]:
            groups.append(
                {
                    "group_id": f"{label}:misses",
                    "stage": stage,
                    "impact": m["fn"],
                    "evidence": {"label": label, "misses": m["fn"], "recall": m["recall"]},
                    "evidence_link": f"labels.{label}",
                }
            )
        for name, s in (m.get("false_alarms_by_slice") or {}).items():
            if s["false_alarms"]:
                groups.append(
                    {
                        "group_id": f"{label}:fa:{name}",
                        "stage": stage,
                        "impact": s["false_alarms"],
                        "evidence": {"label": label, "slice": name, "false_alarms": s["false_alarms"]},
                        "evidence_link": f"labels.{label}.false_alarms_by_slice.{name}",
                    }
                )
        if m["evidence"] != "sufficient":
            groups.append(
                {
                    "group_id": f"{label}:coverage",
                    "stage": stage,
                    "impact": 0.5,
                    "evidence": {"label": label, "truth_events": m["truth_events"]},
                    "evidence_link": f"labels.{label}.evidence",
                }
            )
    return groups


def evaluate(eval_set: dict[str, Any], predictions: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    policy = cfg["selection"]
    rule = MatchRule(**eval_set.get("match_rule", {}))
    hours = sum(r["duration_s"] for r in eval_set["recordings"]) / 3600.0
    labels = sorted({e["label"] for e in eval_set["truth"]} | set(eval_set.get("labels", [])))
    thresholds = predictions.get("thresholds", {})
    report: dict[str, Any] = {
        "stage": eval_set.get("stage", "evaluation"),
        "fixture": bool(eval_set.get("fixture", False)),
        "eval_set_hash": canonical_hash(
            {k: eval_set[k] for k in ("recordings", "truth", "slices") if k in eval_set}
        ),
        "weights_sha256": predictions.get("weights_sha256"),
        "preprocessing_hash": predictions.get("preprocessing_hash"),
        "match_rule": rule.to_json(),
        "operating_budget": policy["false_alarms_per_hour_budget"],
        "resources": predictions.get("resources", {}),
        "labels": {},
        "retention": {},
    }
    for label in labels:
        report["labels"][label] = label_metrics(
            eval_set["truth"],
            predictions["events"],
            label,
            hours,
            rule,
            float(thresholds.get(label, 0.5)),
            eval_set.get("slices"),
            int(policy["min_events_per_slice"]),
        )
    for panel, recording_ids in sorted(eval_set.get("retention_panels", {}).items()):
        ids = set(map(str, recording_ids))
        panel_hours = (
            sum(r["duration_s"] for r in eval_set["recordings"] if str(r["recording_id"]) in ids) / 3600
        )
        truth = [e for e in eval_set["truth"] if str(e["recording_id"]) in ids]
        preds = [e for e in predictions["events"] if str(e["recording_id"]) in ids]
        for label in labels:
            m = label_metrics(truth, preds, label, panel_hours, rule, float(thresholds.get(label, 0.5)))
            if m["recall"] is not None:
                report["retention"][f"{label}@{panel}"] = m["recall"]
    return report


@command("evaluate", "event-level evaluation, deterministic gate, optional selection and Laya advice")
def cmd_evaluate(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    out = _out_dir(cfg) / "reports" / args.stage
    report = evaluate(_load_json(args.eval_set), _load_json(args.predictions), cfg)
    report["stage"] = args.stage
    parent = _load_json(args.parent_report) if args.parent_report else None
    report["gate"] = selection.gate(report, parent, cfg["selection"])
    report["advice"] = laya_adapter.advise(
        evidence_groups(report, args.stage), cfg["laya"], out / "laya-cache"
    )
    if args.select:
        report["selection"] = selection.select(_out_dir(cfg) / "selection", report, report["gate"])
    write_reports(out, report)
    _print({"gate": report["gate"]["status"], "report": str(out / "quality_report.json")})
    return 0 if report["gate"]["status"] == selection.PASS_ELIGIBLE else 1


@command("export", "export a selected checkpoint (or an explicit untrained fixture) to an ONNX bundle")
def cmd_export(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    spec = PreprocessingSpec.from_config(cfg["preprocessing"])
    label_map = default_label_map()
    torch.manual_seed(int(cfg["training"]["seed"]))
    model = build_model(label_map, int(cfg["model"]["in_channels"]))
    if args.fixture:
        bundle_dir = _out_dir(cfg) / "fixture-bundles" / args.name
        lineage = {
            "initialization": "random",
            "trained": False,
            "optimizer_updates": 0,
            "note": "FIXTURE: untrained random weights for plumbing tests only; never ship",
        }
        kind = "fixture-untrained"
    else:
        if args.checkpoint is None:
            _print(
                {"status": "SETUP_REQUIRED", "reason": "export needs --checkpoint (selected) or --fixture"}
            )
            return 1
        payload = ckpt.load_checkpoint(
            args.checkpoint, {"preprocessing_hash": spec.hash, "label_map_hash": label_map.hash}
        )
        meta = payload["metadata"]
        if meta["artifact_kind"] != "selected":
            _print(
                {
                    "status": "SETUP_REQUIRED",
                    "reason": "only a selected checkpoint can be exported for release",
                }
            )
            return 1
        model.load_state_dict(payload["model"])
        bundle_dir = _out_dir(cfg) / "bundles" / meta["weights_sha256"][:16]
        lineage = {
            k: meta[k]
            for k in (
                "initialization",
                "parent_checkpoint",
                "code_revision",
                "dataset_hash",
                "split_hash",
                "weights_sha256",
                "stage",
            )
        }
        kind = "selected"
    manifest = exp.write_bundle(model, spec, label_map, bundle_dir, kind, lineage)
    if args.parity_wav:
        data, sr = sf.read(str(args.parity_wav), dtype="float64", always_2d=True)
        session = exp.ort_session(bundle_dir / exp.MODEL_FILE)
        ref = exp.parity_reference([data[:, c] for c in range(data.shape[1])], sr, spec, session)
        atomic_write_json(bundle_dir.parent / f"{args.name}-parity-reference.json", ref)
    _print(
        {
            "bundle": str(bundle_dir),
            "kind": kind,
            "sha256": manifest["model"]["sha256"],
            "size_bytes": manifest["model"]["size_bytes"],
            "validation_passed": manifest["validation"]["passed"],
            "max_abs_diff_vs_torch": manifest["validation"]["max_abs_diff_vs_torch"],
        }
    )
    return 0


@command("quantize", "static INT8 quantisation (deferred until real calibration windows exist)")
def cmd_quantize(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    _print(
        {
            "status": "SETUP_REQUIRED",
            "reason": "INT8 needs a selected trained checkpoint and representative calibration windows; "
            "FP32 stays the reference until a measured INT8 comparison passes",
        }
    )
    return 1


@command(
    "benchmark", "time ONNX session load, warm batch inference and Python preprocessing (fixture-labelled)"
)
def cmd_benchmark(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    manifest = exp.load_bundle(args.bundle)
    spec = PreprocessingSpec.from_config(
        {k: v for k, v in manifest["preprocessing"].items() if k in PreprocessingSpec.__dataclass_fields__}
    )
    t0 = time.perf_counter()
    session = exp.ort_session(args.bundle / exp.MODEL_FILE, int(cfg["training"]["compute_threads"]))
    load_s = time.perf_counter() - t0
    batch = (
        np.random.default_rng(0).standard_normal((8, 1, spec.n_bins, spec.window_frames)).astype(np.float32)
    )
    session.run([exp.OUTPUT_NAME], {exp.INPUT_NAME: batch})
    t0 = time.perf_counter()
    reps = 5
    for _ in range(reps):
        session.run([exp.OUTPUT_NAME], {exp.INPUT_NAME: batch})
    per_batch = (time.perf_counter() - t0) / reps
    result: dict[str, Any] = {
        "artifact_kind": manifest["artifact_kind"],
        "measurement_kind": "fixture",
        "session_load_s": load_s,
        "warm_batch8_s": per_batch,
        "windows_per_s": 8 / per_batch,
        "audio_seconds_per_s_at_window_hop": 8 * spec.window_hop_seconds / per_batch,
    }
    if args.wav:
        data, sr = sf.read(str(args.wav), dtype="float64", always_2d=True)
        t0 = time.perf_counter()
        n = len(analyze_channel(data[:, 0], sr, spec))
        result["python_preprocessing_x_realtime"] = (data.shape[0] / sr) / (time.perf_counter() - t0)
        result["python_preprocessing_windows"] = n
    atomic_write_json(_out_dir(cfg) / "benchmark.json", result)
    _print(result)
    return 0


def _evaluate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--eval-set", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--parent-report", type=Path, default=None)
    p.add_argument("--stage", default="evaluation")
    p.add_argument("--select", action="store_true", help="update the selection pointer on PASS_ELIGIBLE")


def _export_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument(
        "--fixture", action="store_true", help="export an untrained random fixture (plumbing only)"
    )
    p.add_argument("--name", default="fixture-b0")
    p.add_argument("--parity-wav", type=Path, default=None)


def _benchmark_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--wav", type=Path, default=None)


_EXTRA_ARGS.update({"evaluate": _evaluate_args, "export": _export_args, "benchmark": _benchmark_args})
