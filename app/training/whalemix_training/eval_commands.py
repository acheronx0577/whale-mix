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
from whalemix_training.datalock import SetupRequired
from whalemix_training.device import DeviceError, resolve_device
from whalemix_training.hashing import canonical_hash, sha256_file
from whalemix_training.io_utils import atomic_write_json
from whalemix_training.labels import default_label_map
from whalemix_training.metrics import MatchRule, label_metrics
from whalemix_training.model import build_model
from whalemix_training.preprocess import PreprocessingSpec, analyze_channel
from whalemix_training.report import write_reports
from whalemix_training.scoring import PREDICTIONS_SCHEMA, eval_set_hash, score_split


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


def identity_problems(
    eval_set: dict[str, Any], predictions: dict[str, Any], cfg: dict[str, Any]
) -> list[str]:
    """Predictions must belong to this eval set, preprocessing contract and label map."""
    if predictions.get("schema") != PREDICTIONS_SCHEMA:
        return ["predictions were not written by `score` (no identity block)"]
    expected = {
        "eval_set_hash": eval_set_hash(eval_set),
        "preprocessing_hash": PreprocessingSpec.from_config(cfg["preprocessing"]).hash,
        "label_map_hash": default_label_map().hash,
        "split_hash": eval_set.get("split_hash"),
        "split": eval_set.get("split"),
    }
    problems = [
        f"{key} differs between predictions and eval set"
        for key, value in expected.items()
        if predictions.get(key) != value
    ]
    candidate = predictions.get("candidate") or {}
    if candidate.get("weights_sha256") != predictions.get("weights_sha256"):
        problems.append("candidate weights differ from the predictions' weights")
    return problems


PURPOSES = ("release", "dev-pilot")
DEV_PILOT_CLAIM = "dev pilot: development results only, not release qualification"


def evaluate(eval_set: dict[str, Any], predictions: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    policy = cfg["selection"]
    purpose = cfg["evaluation"].get("purpose", "release")
    if purpose not in PURPOSES:
        raise SetupRequired(f"evaluation.purpose must be one of {', '.join(PURPOSES)}")
    rule = MatchRule(**eval_set.get("match_rule", {}))
    hours = sum(r["duration_s"] for r in eval_set["recordings"]) / 3600.0
    labels = sorted({e["label"] for e in eval_set["truth"]} | set(eval_set.get("labels", [])))
    thresholds = predictions.get("thresholds", {})
    report: dict[str, Any] = {
        "stage": eval_set.get("stage", "evaluation"),
        "split": eval_set.get("split"),
        "purpose": purpose,
        "fixture": bool(eval_set.get("fixture", False)),
        "eval_set_hash": eval_set_hash(eval_set),
        "weights_sha256": predictions.get("weights_sha256"),
        "candidate": predictions.get("candidate"),
        "identity_problems": identity_problems(eval_set, predictions, cfg),
        "preprocessing_hash": predictions.get("preprocessing_hash"),
        "match_rule": rule.to_json(),
        "operating_budget": policy["false_alarms_per_hour_budget"],
        "resources": predictions.get("resources", {}),
        "labels": {},
        "retention": {},
    }
    if purpose == "dev-pilot":
        report["claim"] = DEV_PILOT_CLAIM
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
            eval_set.get("reviewed"),
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
    try:
        report = evaluate(_load_json(args.eval_set), _load_json(args.predictions), cfg)
    except SetupRequired as exc:
        _print({"status": "SETUP_REQUIRED", "reason": str(exc)})
        return 1
    if args.select and report.get("split") == "test":
        _print(
            {"status": "SETUP_REQUIRED", "reason": "selection cannot use the test split (sealed for qualify)"}
        )
        return 1
    report["stage"] = args.stage
    parent = _load_json(args.parent_report) if args.parent_report else None
    report["gate"] = selection.gate(report, parent, cfg["selection"])
    report["advice"] = laya_adapter.advise(
        evidence_groups(report, args.stage), cfg["laya"], out / "laya-cache"
    )
    if args.select:
        report["selection"] = selection.select(_out_dir(cfg) / "selection", report, report["gate"])
    write_reports(out, report)
    summary = {"gate": report["gate"]["status"], "report": str(out / "quality_report.json")}
    if args.select:
        summary["selection"] = {k: v for k, v in report["selection"].items() if k != "report"}
    _print(summary)
    passed = report["gate"]["status"] == selection.PASS_ELIGIBLE
    return 0 if passed and (not args.select or report["selection"]["changed"]) else 1


def _seal_path(cfg: dict[str, Any]) -> Path:
    return _out_dir(cfg) / "selection" / "test-seal.json"


def _check_test_seal(cfg: dict[str, Any], checkpoint: Path, pointer: dict[str, Any]) -> None:
    """Test scoring is for the selected checkpoint only, and once scored the test split is sealed to
    those weights: tuning after looking at test results cannot be scored on the same test split."""
    if not checkpoint.is_file() or sha256_file(checkpoint) != pointer["selected_checkpoint_sha256"]:
        raise SetupRequired("the test split is sealed to selection/selected.pt; score that checkpoint only")
    seal = _seal_path(cfg)
    if seal.exists() and _load_json(seal)["weights_sha256"] != pointer["weights_sha256"]:
        raise SetupRequired(
            "the test split was already sealed to other weights; it cannot score a re-tuned model"
        )


def _write_test_seal(cfg: dict[str, Any], predictions: dict[str, Any], eval_set: dict[str, Any]) -> None:
    seal = _seal_path(cfg)
    if not seal.exists():
        atomic_write_json(
            seal,
            {
                "weights_sha256": predictions["weights_sha256"],
                "checkpoint_sha256": predictions["candidate"]["checkpoint_sha256"],
                "eval_set_hash": eval_set_hash(eval_set),
                "sealed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )


@command("score", "score a trained checkpoint on a locked split; writes predictions.json and eval-set.json")
def cmd_score(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    try:
        device = resolve_device(cfg["training"])
        thresholds = dict(cfg["evaluation"].get("thresholds") or {})
        pointer = _out_dir(cfg) / "selection" / "selected.json"
        if args.use_selected_thresholds or args.split == "test":
            if not pointer.exists():
                raise SetupRequired(
                    "the test split stays sealed until a selection exists (score + evaluate --select on "
                    "validation); selected thresholds need selection/selected.json"
                )
            thresholds = _load_json(pointer)["thresholds"]
        if args.split == "test":
            _check_test_seal(cfg, args.checkpoint, _load_json(pointer))
        predictions, eval_set = score_split(cfg, args.checkpoint, args.split, device, thresholds)
    except (SetupRequired, DeviceError) as exc:
        _print({"status": "SETUP_REQUIRED", "reason": str(exc)})
        return 1
    if args.split == "test":
        _write_test_seal(cfg, predictions, eval_set)
    out = _out_dir(cfg) / "scores" / (args.name or args.split)
    atomic_write_json(out / "eval-set.json", eval_set)
    atomic_write_json(out / "predictions.json", predictions)
    _print(
        {
            "predictions": str(out / "predictions.json"),
            "eval_set": str(out / "eval-set.json"),
            "events": len(predictions["events"]),
            "truth_events": len(eval_set["truth"]),
            "weights_sha256": predictions["weights_sha256"],
        }
    )
    return 0


def _qualify(cfg: dict[str, Any], test_report_path: Path) -> dict[str, Any]:
    """Check the selection, the selected checkpoint and an independent test-split report; return the
    lineage that a qualified bundle records. Raises SetupRequired on any mismatch."""
    out = _out_dir(cfg)
    pointer_path = out / "selection" / "selected.json"
    if not pointer_path.exists():
        raise SetupRequired("no selection; run score + evaluate --select on the validation split first")
    pointer = _load_json(pointer_path)
    selected_path = out / "selection" / pointer["selected_checkpoint"]
    if not selected_path.is_file() or sha256_file(selected_path) != pointer["selected_checkpoint_sha256"]:
        raise SetupRequired("selected checkpoint is missing or changed since selection")
    spec = PreprocessingSpec.from_config(cfg["preprocessing"])
    label_map = default_label_map()
    payload = ckpt.load_checkpoint(
        selected_path, {"preprocessing_hash": spec.hash, "label_map_hash": label_map.hash}
    )
    meta = payload["metadata"]
    if meta["artifact_kind"] != "selected" or meta["weights_sha256"] != pointer["weights_sha256"]:
        raise SetupRequired("selected checkpoint does not match the selection pointer")
    report = _load_json(test_report_path)
    candidate = report.get("candidate") or {}
    problems = []
    if report.get("split") != "test":
        problems.append("qualification needs a report on the test split")
    if report.get("purpose") != "release":
        problems.append(f"a {report.get('purpose', 'unlabelled')} report cannot qualify a release")
    if candidate.get("weights_sha256") != pointer["weights_sha256"]:
        problems.append("test report is for different weights than the selection")
    if report.get("eval_set_hash") == pointer["eval_set_hash"]:
        problems.append("test report reuses the selection eval set")
    thresholds = {label: m.get("threshold") for label, m in sorted(report.get("labels", {}).items())}
    if thresholds != pointer["thresholds"]:
        problems.append("test report thresholds differ from the selected thresholds")
    regate = selection.gate(
        {k: v for k, v in report.items() if k not in selection.NON_REPORT_KEYS}, None, cfg["selection"]
    )
    recorded = report.get("gate", {}).get("status")
    if regate["status"] != selection.PASS_ELIGIBLE or recorded != regate["status"]:
        reasons = "; ".join(regate["reasons"])
        status = regate["status"]
        problems.append(f"re-run gate on the test report is {status} (recorded {recorded}): {reasons}")
    if problems:
        raise SetupRequired("; ".join(problems))
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
            "global_update",
            "preprocessing_hash",
            "label_map_hash",
        )
    }
    lineage.update(
        {
            "selected_checkpoint_sha256": pointer["selected_checkpoint_sha256"],
            "source_checkpoint_sha256": pointer["source_checkpoint_sha256"],
            "validation_report_hash": pointer["report_hash"],
            "validation_eval_set_hash": pointer["eval_set_hash"],
            "test_report_hash": selection.report_hash(report),
            "test_eval_set_hash": report["eval_set_hash"],
        }
    )
    return {"payload": payload, "lineage": lineage, "thresholds": pointer["thresholds"], "spec": spec}


@command("qualify", "qualify the selected checkpoint on the test split and publish a qualified bundle")
def cmd_qualify(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    try:
        q = _qualify(cfg, args.test_report)
    except (SetupRequired, ckpt.CheckpointError, KeyError) as exc:
        _print({"status": "NOT_QUALIFIED", "reason": str(exc)})
        return 1
    label_map = default_label_map()
    torch.manual_seed(int(cfg["training"]["seed"]))
    model = build_model(label_map, int(cfg["model"]["in_channels"]))
    model.load_state_dict(q["payload"]["model"])
    lineage = q["lineage"]
    qualification = {
        "status": "qualified",
        "method": "whalemix_training qualify",
        "gate": selection.PASS_ELIGIBLE,
        "lineage": lineage,
        "lineage_hash": canonical_hash(lineage),
    }
    bundle_dir = _out_dir(cfg) / "bundles" / lineage["weights_sha256"][:16]
    manifest = exp.write_bundle(
        model, q["spec"], label_map, bundle_dir, "selected", lineage, q["thresholds"], qualification
    )
    _print(
        {
            "status": "QUALIFIED",
            "bundle": str(bundle_dir),
            "sha256": manifest["model"]["sha256"],
            "lineage_hash": qualification["lineage_hash"],
        }
    )
    return 0


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
    p.add_argument(
        "--select", action="store_true", help="promote the scored checkpoint on PASS_ELIGIBLE (validation)"
    )


def _export_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument(
        "--fixture", action="store_true", help="export an untrained random fixture (plumbing only)"
    )
    p.add_argument("--name", default="fixture-b0")
    p.add_argument("--parity-wav", type=Path, default=None)


def _score_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=("validation", "test"), required=True)
    p.add_argument("--name", default=None, help="output folder under scores/ (default: the split name)")
    p.add_argument(
        "--use-selected-thresholds",
        action="store_true",
        help="score with the thresholds recorded at selection (for the test split)",
    )


def _qualify_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--test-report", type=Path, required=True, help="quality_report.json on the test split")


def _benchmark_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--wav", type=Path, default=None)


_EXTRA_ARGS.update(
    {
        "evaluate": _evaluate_args,
        "export": _export_args,
        "benchmark": _benchmark_args,
        "score": _score_args,
        "qualify": _qualify_args,
    }
)
