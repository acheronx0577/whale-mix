"""Slice 3 checks on fixed predictions and an explicitly labelled untrained fixture model.

Hand-counted event fixtures validate the evaluator, not model accuracy.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.conftest import write_tone
from whalemix_training import checkpoint as ckpt
from whalemix_training import export as exp
from whalemix_training import laya_adapter, selection
from whalemix_training.cli import main
from whalemix_training.config import DEFAULTS
from whalemix_training.hashing import sha256_file
from whalemix_training.labels import default_label_map
from whalemix_training.metrics import MatchRule, label_metrics, match, merge_predictions
from whalemix_training.model import build_model
from whalemix_training.preprocess import PreprocessingSpec

SMALL = PreprocessingSpec(
    sample_rate=2000, n_fft=256, hop_length=100, fmax_hz=500.0, window_seconds=2.0, window_hop_seconds=1.0
)


def ev(rec, label, a, b, score=1.0, **kw):
    return {"recording_id": rec, "label": label, "start_s": a, "end_s": b, "score": score, **kw}


TRUTH = [
    ev("r1", "fin_20hz_pulse", 10.0, 11.0),
    ev("r1", "fin_20hz_pulse", 13.0, 14.0),  # neighbouring pulse must stay distinct
    ev("r1", "blue_a", 20.0, 35.0),
    ev("r2", "blue_a", 5.0, 20.0),  # missed
    ev("r2", "fin_20hz_pulse", 40.0, 41.0),  # missed
]
PREDS = [
    ev("r1", "fin_20hz_pulse", 10.1, 11.1, 0.9),
    ev("r1", "fin_20hz_pulse", 13.2, 14.0, 0.8),
    ev("r1", "blue_a", 18.0, 28.0, 0.95),
    ev("r1", "blue_a", 27.0, 37.0, 0.7),  # overlapping duplicate window of the same call
    ev("r1", "blue_b", 20.0, 35.0, 0.9),  # wrong call type: false alarm for blue_b
    ev("r2", "fin_20hz_pulse", 100.0, 101.0, 0.6, review=True),  # false alarm in vessel slice
    ev("r2", "blue_a", 60.0, 70.0, 0.3),  # below threshold
]
SLICES = {"vessel_only": [{"recording_id": "r2", "start_s": 90.0, "end_s": 190.0}]}


def test_merge_keeps_distinct_pulses_and_merges_overlaps():
    merged = merge_predictions(PREDS, 0.0)
    assert len([p for p in merged if p["label"] == "fin_20hz_pulse" and p["recording_id"] == "r1"]) == 2
    blue = [p for p in merged if p["label"] == "blue_a" and p["recording_id"] == "r1"]
    assert len(blue) == 1 and blue[0]["end_s"] == 37.0 and blue[0]["score"] == 0.95


def test_hand_counted_metrics():
    rule = MatchRule()
    hours = 200.0 / 3600.0
    fin = label_metrics(TRUTH, PREDS, "fin_20hz_pulse", hours, rule, 0.5, SLICES, min_events=1)
    assert (fin["tp"], fin["fp"], fin["fn"]) == (2, 1, 1)
    assert fin["precision"] == pytest.approx(2 / 3) and fin["recall"] == pytest.approx(2 / 3)
    assert fin["false_alarms_per_hour"] == pytest.approx(1 / hours)
    assert fin["review_events_per_hour"] == pytest.approx(1 / hours)
    assert fin["false_alarms_by_slice"]["vessel_only"]["false_alarms"] == 1
    blue = label_metrics(TRUTH, PREDS, "blue_a", hours, rule, 0.5, min_events=1)
    assert (blue["tp"], blue["fp"], blue["fn"]) == (1, 0, 1)
    assert blue["average_precision"] == pytest.approx(0.5)
    wrong = label_metrics(TRUTH, PREDS, "blue_b", hours, rule, 0.5, min_events=1)
    assert (wrong["tp"], wrong["fp"], wrong["fn"]) == (0, 1, 0) and wrong["recall"] is None
    assert label_metrics(TRUTH, PREDS, "blue_a", hours, rule, 0.5)["evidence"] == "insufficient_evidence"


def test_one_to_one_matching():
    truth = [ev("r", "blue_a", 0, 10)]
    preds = [ev("r", "blue_a", 0, 10, 0.9), ev("r", "blue_a", 30, 40, 0.8)]
    result = match(truth, preds, MatchRule(center_tolerance_s=100.0))
    assert sum(o["truth_index"] is not None for o in result["outcomes"]) == 1


def _report(recall: float, fa: float = 0.5, evidence: str = "sufficient", **extra) -> dict:
    return {
        "eval_set_hash": "e",
        "match_rule": {"version": "v"},
        "operating_budget": 2.0,
        "preprocessing_hash": "p",
        "resources": {"within_budget": True},
        "candidate": {"artifact_kind": "training-state", "weights_sha256": extra.get("sha", "cand")},
        "labels": {
            "blue_a": {
                "truth_events": 20,
                "threshold": 0.5,
                "recall": recall,
                "false_alarms_per_hour": fa,
                "evidence": evidence,
                "recall_ci95": [recall - 0.05, recall + 0.05],
                "reviewed_background_hours": 10.0,
            }
        },
        "retention": extra.get("retention", {}),
        "weights_sha256": extra.get("sha", "cand"),
    }


POLICY = {**DEFAULTS["selection"], "min_recall": {"blue_a": 0.0}, "min_background_hours": {"blue_a": 0.0}}


def test_gate_outcomes():
    assert selection.gate(_report(0.8), None, POLICY)["status"] == selection.PASS_ELIGIBLE
    assert (
        selection.gate(_report(0.8, evidence="insufficient_evidence"), None, POLICY)["status"]
        == "INCONCLUSIVE"
    )
    assert selection.gate({"labels": {}}, None, POLICY)["status"] == selection.SETUP_REQUIRED
    assert selection.gate(_report(0.9), _report(0.8), POLICY)["status"] == selection.PASS_ELIGIBLE
    assert selection.gate(_report(0.805), _report(0.8), POLICY)["status"] == selection.INCONCLUSIVE
    assert selection.gate(_report(0.7), _report(0.8), POLICY)["status"] == selection.FAIL
    assert selection.gate(_report(0.9, fa=5.0), None, POLICY)["status"] == selection.FAIL
    old = _report(0.8, retention={"blue_a@2013": 0.9})
    assert (
        selection.gate(_report(0.9, retention={"blue_a@2013": 0.8}), old, POLICY)["status"] == selection.FAIL
    )
    other = {**_report(0.9), "eval_set_hash": "different"}
    assert selection.gate(other, _report(0.8), POLICY)["status"] == selection.SETUP_REQUIRED
    starved = {**_report(0.9), "resources": {"within_budget": False}}
    assert selection.gate(starved, None, POLICY)["status"] == selection.FAIL


def _scored_checkpoint(path: Path, seed: int) -> dict:
    """A real training-state checkpoint and the candidate identity `score` would record for it."""
    torch.manual_seed(seed)
    model = torch.nn.Linear(2, 1)
    meta = {k: "x" for k in ckpt.REQUIRED_METADATA} | {"artifact_kind": "training-state", "global_update": 1}
    weights = ckpt.save_checkpoint(path, {"model": model.state_dict(), "metadata": meta})
    identity = {
        "artifact_kind": "training-state",
        "checkpoint_path": str(path),
        "checkpoint_sha256": sha256_file(path),
        "weights_sha256": weights,
    }
    return {**_report(0.8, sha=weights), "candidate": identity}


def test_selection_is_atomic_and_keeps_previous_on_failure(tmp_path):
    a = _scored_checkpoint(tmp_path / "a.pt", 1)
    first = selection.select(tmp_path / "sel", a, {"status": "PASS_ELIGIBLE", "reasons": []})
    assert first == {"selected": a["weights_sha256"], "changed": True, "status": "PASS_ELIGIBLE"}
    b = _scored_checkpoint(tmp_path / "b.pt", 2)
    for status in ("FAIL", "INCONCLUSIVE", "SETUP_REQUIRED"):
        kept = selection.select(tmp_path / "sel", b, {"status": status, "reasons": ["x"]})
        assert kept["selected"] == a["weights_sha256"] and kept["changed"] is False
    (tmp_path / "b.pt").write_bytes(b"tampered after scoring")
    refused = selection.select(tmp_path / "sel", b, {"status": "PASS_ELIGIBLE", "reasons": []})
    assert refused["changed"] is False and refused["status"] == "SETUP_REQUIRED"
    ledger = (tmp_path / "sel" / "improvement-ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(ledger) == 5
    pointer = json.loads((tmp_path / "sel" / "selected.json").read_text(encoding="utf-8"))
    assert pointer["weights_sha256"] == a["weights_sha256"]
    assert pointer["source_checkpoint_sha256"] == a["candidate"]["checkpoint_sha256"]
    promoted = ckpt.load_checkpoint(tmp_path / "sel" / "selected.pt")["metadata"]
    assert promoted["artifact_kind"] == "selected" and promoted["weights_sha256"] == a["weights_sha256"]


GROUPS = [
    {"group_id": "fin:fa:vessel", "stage": "s", "impact": 9, "evidence": {"fa": 9}},
    {"group_id": "blue:misses", "stage": "s", "impact": 3, "evidence": {"fn": 3}},
    {"group_id": "noise", "stage": "s", "impact": 99, "evidence": {}, "actionable": False},
]
LAYA_CFG = {
    "enabled": True,
    "executable": "laya",
    "model": "pinned",
    "model_revision": "rev1",
    "package_version": "0.4.6",
    "timeout_s": 5,
    "batch_size": 8,
}


def test_laya_fallbacks_and_ranking(tmp_path):
    off = laya_adapter.advise(GROUPS, {**LAYA_CFG, "enabled": False}, tmp_path)
    assert off["source"] == "deterministic_ranking" and off["ranking"] == ["fin:fa:vessel", "blue:misses"]
    unpinned = laya_adapter.advise(GROUPS, {**LAYA_CFG, "model": None}, tmp_path)
    assert "SETUP_REQUIRED" in unpinned["reason"]

    def timeout(*a, **k):
        raise subprocess.TimeoutExpired("laya", 5)

    assert "TimeoutExpired" in laya_adapter.advise(GROUPS, LAYA_CFG, tmp_path, timeout)["reason"]
    bad = lambda args, **k: subprocess.CompletedProcess(args, 0, "not json\nx\n", "")  # noqa: E731
    assert laya_adapter.advise(GROUPS, LAYA_CFG, tmp_path, bad)["reason"] == "laya returned invalid JSON"
    odd = json.dumps({"answers": {"next_action": {"type": "choice", "choice": "set_threshold"}}})
    weird = lambda args, **k: subprocess.CompletedProcess(args, 0, odd + "\n" + odd + "\n", "")  # noqa: E731
    assert (
        laya_adapter.advise(GROUPS, LAYA_CFG, tmp_path, weird)["reason"]
        == "laya returned an unsupported answer"
    )


def test_laya_success_is_cached_and_advisory(tmp_path):
    calls = []

    def fake(args, **kwargs):
        calls.append(args)
        assert args[args.index("--device") + 1] == "cpu" and "--json" in args
        assert args[args.index("--model") + 1] == "pinned"
        q = json.loads(Path(args[args.index("--questions") + 1]).read_text(encoding="utf-8"))
        assert q["questions"]["next_action"]["type"] == "choice"
        line = json.dumps(
            {
                "answers": {
                    "next_action": {"type": "choice", "choice": "mine_hard_negatives", "confidence": 0.7}
                }
            }
        )
        return subprocess.CompletedProcess(args, 0, line + "\n" + line + "\n", "")

    first = laya_adapter.advise(GROUPS, LAYA_CFG, tmp_path, fake)
    second = laya_adapter.advise(GROUPS, LAYA_CFG, tmp_path, fake)
    assert first["source"] == "laya" and first["cache"] == "miss" and second["cache"] == "hit"
    assert len(calls) == 1 and all(r["advisory"] for r in first["recommendations"])
    failing = _report(0.7)
    gate = selection.gate(failing, _report(0.8), POLICY)
    assert gate["status"] == selection.FAIL  # a positive recommendation cannot change the gate


def test_fixture_export_parity_and_bundle(tmp_path):
    torch.manual_seed(0)
    lm = default_label_map()
    model = build_model(lm)
    manifest = exp.write_bundle(
        model,
        SMALL,
        lm,
        tmp_path / "bundle",
        "fixture-untrained",
        {"initialization": "random", "trained": False},
    )
    assert manifest["validation"]["passed"] and manifest["validation"]["max_abs_diff_vs_torch"] < 1e-4
    assert manifest["model"]["input"]["shape"] == ["batch", 1, SMALL.n_bins, SMALL.window_frames]
    assert manifest["label_map"]["outputs"][0] == "blue_a" and manifest["threshold_status"] == "uncalibrated"
    assert exp.load_bundle(tmp_path / "bundle")["artifact_kind"] == "fixture-untrained"
    with pytest.raises(ValueError):
        exp.write_bundle(model, SMALL, lm, tmp_path / "b2", "selected", {"initialization": "imagenet"})
    (tmp_path / "bundle" / "model.onnx").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        exp.load_bundle(tmp_path / "bundle")


def test_parity_reference_matches_preprocessing(tmp_path):
    torch.manual_seed(0)
    model = build_model(default_label_map())
    exp.write_bundle(
        model, SMALL, default_label_map(), tmp_path / "b", "fixture-untrained", {"initialization": "random"}
    )
    wav = tmp_path / "tone.wav"
    write_tone(wav, 5.0, 100.0, sr=8000)
    import soundfile as sf

    data, sr = sf.read(str(wav), dtype="float64")
    ref = exp.parity_reference([data], sr, SMALL, exp.ort_session(tmp_path / "b" / "model.onnx"))
    assert len(ref["windows"]) == 4
    w = ref["windows"][0]
    with torch.no_grad():
        torch_probs = torch.sigmoid(model(torch.tensor(w["tensor"]).reshape(1, 1, SMALL.n_bins, -1)))[
            0
        ].numpy()
    np.testing.assert_allclose(w["probabilities"], torch_probs, atol=1e-4)


def test_cli_evaluate_export_quantize_benchmark(tmp_path, capsys):
    out = tmp_path / "out"
    cfg = tmp_path / "c.json"
    cfg.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "paths": {"output_dir": str(out)},
                "preprocessing": {
                    "sample_rate": 2000,
                    "n_fft": 256,
                    "hop_length": 100,
                    "fmax_hz": 500.0,
                    "window_seconds": 2.0,
                    "window_hop_seconds": 1.0,
                },
                "selection": {"min_events_per_slice": 1},
            }
        ),
        encoding="utf-8",
    )
    eval_set = tmp_path / "eval.json"
    eval_set.write_text(
        json.dumps(
            {
                "fixture": True,
                "recordings": [
                    {"recording_id": "r1", "duration_s": 100},
                    {"recording_id": "r2", "duration_s": 200},
                ],
                "truth": TRUTH,
                "slices": SLICES,
                "retention_panels": {"2013": ["r1"]},
            }
        ),
        encoding="utf-8",
    )
    preds = tmp_path / "preds.json"
    preds.write_text(
        json.dumps({"weights_sha256": "fixture", "events": PREDS, "resources": {"within_budget": True}}),
        encoding="utf-8",
    )
    code = main(
        [
            "evaluate",
            "--config",
            str(cfg),
            "--eval-set",
            str(eval_set),
            "--predictions",
            str(preds),
            "--stage",
            "fixture",
            "--select",
        ]
    )
    report = json.loads((out / "reports" / "fixture" / "quality_report.json").read_text(encoding="utf-8"))
    # a fixture evaluation is never eligible, and hand-written predictions carry no candidate identity
    assert code == 1 and report["gate"]["status"] == "SETUP_REQUIRED"
    assert report["fixture"] is True and report["identity_problems"]
    assert report["retention"]["blue_a@2013"] == 1.0
    assert report["advice"]["source"] == "deterministic_ranking"
    assert (out / "reports" / "fixture" / "Quality-Report.md").exists()
    assert not (out / "selection" / "selected.json").exists()
    wav = tmp_path / "t.wav"
    write_tone(wav, 6.0, 120.0, sr=4000)
    assert main(["export", "--config", str(cfg), "--fixture", "--name", "t", "--parity-wav", str(wav)]) == 0
    assert (out / "fixture-bundles" / "t" / "bundle.json").exists()
    assert (out / "fixture-bundles" / "t-parity-reference.json").exists()
    assert main(["export", "--config", str(cfg)]) == 1
    assert main(["quantize", "--config", str(cfg)]) == 1
    assert (
        main(
            [
                "benchmark",
                "--config",
                str(cfg),
                "--bundle",
                str(out / "fixture-bundles" / "t"),
                "--wav",
                str(wav),
            ]
        )
        == 0
    )
    bench = json.loads((out / "benchmark.json").read_text(encoding="utf-8"))
    assert bench["measurement_kind"] == "fixture" and bench["windows_per_s"] > 0


def test_reproduce_script_plans_stages_and_stops_at_first_failure(corpus: dict[str, Path]) -> None:
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "reproduce-acoustic-v1.py"
    spec = importlib.util.spec_from_file_location("reproduce", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = str(corpus["config"])
    assert mod.main(["--config", cfg, "--dry-run"]) == 0
    # No annotations file exists yet, so validate-labels must fail and nothing after it may run.
    code = mod.main(["--config", cfg])
    assert code != 0
    entries = [json.loads(x) for x in (corpus["out"] / "reproduce-ledger.jsonl").read_text().splitlines()]
    assert [e["stage"] for e in entries] == ["inventory", "validate-labels"]
    assert entries[0]["exit_code"] == 0 and entries[1]["exit_code"] == code


def test_export_has_dynamic_batch_axis_and_validates_batches_1_3_16(tmp_path):
    import onnx

    torch.manual_seed(0)
    lm = default_label_map()
    model = build_model(lm)
    path = tmp_path / "model.onnx"
    exp.export_onnx(model, SMALL, path)
    graph = onnx.load(str(path)).graph
    for value in (graph.input[0], graph.output[0]):
        dim0 = value.type.tensor_type.shape.dim[0]
        assert dim0.dim_param == "batch" and not dim0.HasField("dim_value")
    report = exp.validate_onnx(model, SMALL, path)
    assert report["passed"] and report["dynamic_batch"]
    assert report["batch_sizes"] == [1, 3, 16]
    assert set(report["max_abs_diff_by_batch"]) == {"1", "3", "16"}
    assert all(d < 1e-4 for d in report["max_abs_diff_by_batch"].values())
    assert report["max_abs_diff_alone_vs_batched"] < 1e-4
    session = exp.ort_session(path)
    for size in (1, 3, 16):
        x = np.zeros((size, 1, SMALL.n_bins, SMALL.window_frames), dtype=np.float32)
        assert session.run([exp.OUTPUT_NAME], {exp.INPUT_NAME: x})[0].shape == (size, len(lm.outputs))


def test_static_batch_export_is_refused(tmp_path):
    import onnx

    torch.manual_seed(0)
    model = build_model(default_label_map())
    path = tmp_path / "static.onnx"
    dummy = torch.zeros(1, 1, SMALL.n_bins, SMALL.window_frames)
    from whalemix_training.model import ProbabilityHead

    torch.onnx.export(
        ProbabilityHead(model).eval(),
        (dummy,),
        str(path),
        input_names=[exp.INPUT_NAME],
        output_names=[exp.OUTPUT_NAME],
        opset_version=exp.OPSET,
        dynamo=False,
    )
    assert onnx.load(str(path)).graph.input[0].type.tensor_type.shape.dim[0].dim_value == 1
    with pytest.raises(RuntimeError):
        exp.ensure_dynamic_batch(path)
    assert not exp.validate_onnx(model, SMALL, path)["passed"]
