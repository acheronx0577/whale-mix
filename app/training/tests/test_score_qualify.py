"""Fix 4 (training readiness): score/predict with identity checks, a gate that cannot pass empty,
zero-recall or unsupported candidates, selection bound to the exact checkpoint, and a supported qualify
step. Fixture data and a 2-update fixture model; thresholds and budgets are opened up on purpose so the
plumbing reaches every branch. None of this says anything about whale-detection accuracy."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from whalemix_training import checkpoint as ckpt
from whalemix_training import export as exp
from whalemix_training import inventory, selection
from whalemix_training.annotations import REQUIRED_COLUMNS
from whalemix_training.cli import main
from whalemix_training.config import DEFAULTS

POLICY = {**DEFAULTS["selection"], "min_recall": {"blue_a": 0.0}, "min_background_hours": {"blue_a": 0.0}}


def _full(labels: dict, **extra) -> dict:
    base = {
        "eval_set_hash": "e",
        "match_rule": {"version": "v"},
        "operating_budget": 2.0,
        "preprocessing_hash": "p",
        "resources": {"within_budget": True},
        "labels": labels,
        "retention": {},
        "weights_sha256": "w",
        "candidate": {"artifact_kind": "training-state", "weights_sha256": "w", "checkpoint_sha256": "c"},
    }
    return {**base, **extra}


def _label(recall, truth=25, fa=0.1):
    return {
        "recall": recall,
        "truth_events": truth,
        "false_alarms_per_hour": fa,
        "evidence": "sufficient" if truth >= 20 else "insufficient_evidence",
        "recall_ci95": None if recall is None else [recall, recall],
        "reviewed_background_hours": 10.0,
    }


def test_gate_rejects_empty_zero_recall_and_unsupported_candidates():
    # Audit repro: on da0dee7 every one of these returned PASS_ELIGIBLE.
    assert selection.gate(_full({}), None, POLICY)["status"] == selection.SETUP_REQUIRED
    zero = selection.gate(_full({"blue_a": _label(0.0)}), None, POLICY)
    assert zero["status"] == selection.FAIL and "zero recall" in zero["reasons"][0]
    fixture = selection.gate(_full({"blue_a": _label(0.9)}, fixture=True), None, POLICY)
    assert fixture["status"] == selection.SETUP_REQUIRED
    anonymous = _full({"blue_a": _label(0.9)})
    del anonymous["candidate"]
    assert selection.gate(anonymous, None, POLICY)["status"] == selection.SETUP_REQUIRED
    untrained = _full({"blue_a": _label(0.9)}, candidate={"artifact_kind": "fixture-untrained"})
    assert selection.gate(untrained, None, POLICY)["status"] == selection.SETUP_REQUIRED
    mismatched = _full({"blue_a": _label(0.9)}, identity_problems=["eval_set_hash differs"])
    assert selection.gate(mismatched, None, POLICY)["status"] == selection.SETUP_REQUIRED
    assert selection.gate(_full({"blue_a": _label(0.9)}), None, POLICY)["status"] == selection.PASS_ELIGIBLE


SMALL_PRE = {
    "sample_rate": 2000,
    "n_fft": 256,
    "hop_length": 100,
    "fmax_hz": 500.0,
    "window_seconds": 2.0,
    "window_hop_seconds": 1.0,
}


@pytest.fixture()
def trained(corpus):
    """Fixture pipeline: every file gets a reviewed blue_a call, then a 2-update fixture model."""
    cfg = json.loads(Path(corpus["config"]).read_text(encoding="utf-8"))
    cfg["preprocessing"] = SMALL_PRE
    cfg["training"] = {"max_updates": 2, "micro_batch": 2, "compute_threads": 1}
    cfg["selection"] = {
        "min_events_per_slice": 1,
        "false_alarms_per_hour_budget": 1e9,
        "min_recall": {"blue_a": 0.0},
        "min_background_hours": {"blue_a": 0.0},
    }
    cfg["evaluation"] = {"labels": ["blue_a"], "thresholds": {"blue_a": 0.0}}
    Path(corpus["config"]).write_text(json.dumps(cfg), encoding="utf-8")
    assert main(["inventory", "--config", str(corpus["config"])]) == 0
    catalog = inventory.load_catalog(corpus["out"] / "source-catalog.jsonl")
    with (corpus["tmp"] / "annotations.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(REQUIRED_COLUMNS)
        for n, e in enumerate(x for x in catalog if x.get("status") == "ok"):
            w.writerow([f"p{n}", e["file_id"], "0", "1.0", "2.0", "blue_a", "present", "reviewed", "fixture"])
            absent = ["3.0", "5.5", "fin_20hz_pulse", "absent", "reviewed", "fixture"]
            w.writerow([f"a{n}", e["file_id"], "0", *absent])
    for step in ("validate-labels", "split", "train"):
        assert main([step, "--config", str(corpus["config"])]) == 0
    return corpus


def _cli(corpus, *argv: str) -> int:
    return main([argv[0], "--config", str(corpus["config"]), *argv[1:]])


def test_score_writes_structured_predictions_with_identity(trained, capsys):
    latest = trained["out"] / "checkpoints" / "latest.pt"
    assert _cli(trained, "score", "--checkpoint", str(latest), "--split", "validation") == 0
    pred = json.loads((trained["out"] / "scores" / "validation" / "predictions.json").read_text("utf-8"))
    eval_set = json.loads((trained["out"] / "scores" / "validation" / "eval-set.json").read_text("utf-8"))
    payload = ckpt.load_checkpoint(latest)
    assert pred["schema"] == "whalemix-predictions/1"
    assert pred["candidate"]["weights_sha256"] == payload["metadata"]["weights_sha256"]
    assert pred["candidate"]["checkpoint_sha256"] == ckpt_sha(latest)
    assert pred["candidate"]["artifact_kind"] == "training-state"
    assert pred["split_hash"] == payload["metadata"]["split_hash"]
    assert pred["events"] and {"recording_id", "channel", "label", "start_s", "end_s", "score"} <= set(
        pred["events"][0]
    )
    assert {e["recording_id"] for e in pred["events"]} == {r["recording_id"] for r in eval_set["recordings"]}
    assert eval_set["split"] == "validation" and eval_set["truth"]
    # identity: evaluating these predictions against a different eval set is refused
    eval_set["truth"] = eval_set["truth"][1:]
    other = trained["tmp"] / "other-eval.json"
    other.write_text(json.dumps(eval_set), encoding="utf-8")
    pred_path = trained["out"] / "scores" / "validation" / "predictions.json"
    assert _cli(trained, "evaluate", "--eval-set", str(other), "--predictions", str(pred_path)) == 1
    report_path = trained["out"] / "reports" / "evaluation" / "quality_report.json"
    report = json.loads(report_path.read_text("utf-8"))
    assert report["gate"]["status"] == "SETUP_REQUIRED" and report["identity_problems"]


def ckpt_sha(path: Path) -> str:
    from whalemix_training.hashing import sha256_file

    return sha256_file(path)


def _score_and_select(corpus) -> Path:
    latest = corpus["out"] / "checkpoints" / "latest.pt"
    assert _cli(corpus, "score", "--checkpoint", str(latest), "--split", "validation") == 0
    d = corpus["out"] / "scores" / "validation"
    code = _cli(
        corpus,
        "evaluate",
        "--eval-set",
        str(d / "eval-set.json"),
        "--predictions",
        str(d / "predictions.json"),
        "--stage",
        "validation",
        "--select",
    )
    assert code == 0
    return latest


def test_selection_is_bound_to_the_exact_checkpoint(trained):
    latest = trained["out"] / "checkpoints" / "latest.pt"
    assert _cli(trained, "score", "--checkpoint", str(latest), "--split", "validation") == 0
    d = trained["out"] / "scores" / "validation"
    shutil.copy(latest, trained["tmp"] / "kept.pt")
    latest.write_bytes(latest.read_bytes() + b"\0")  # the checkpoint changed after it was scored
    argv = ["--eval-set", str(d / "eval-set.json"), "--predictions", str(d / "predictions.json"), "--select"]
    assert _cli(trained, "evaluate", *argv) == 1
    assert not (trained["out"] / "selection" / "selected.json").exists()
    shutil.copy(trained["tmp"] / "kept.pt", latest)
    assert _cli(trained, "evaluate", *argv) == 0
    pointer = json.loads((trained["out"] / "selection" / "selected.json").read_text("utf-8"))
    promoted = ckpt.load_checkpoint(trained["out"] / "selection" / "selected.pt")
    assert promoted["metadata"]["artifact_kind"] == "selected"
    assert promoted["metadata"]["weights_sha256"] == pointer["weights_sha256"]
    assert pointer["source_checkpoint_sha256"] == ckpt_sha(latest)
    assert pointer["thresholds"] == {"blue_a": 0.0}


def test_qualify_publishes_a_qualified_bundle_with_lineage(trained, capsys):
    _score_and_select(trained)
    selected = trained["out"] / "selection" / "selected.pt"
    flags = ["--checkpoint", str(selected), "--split", "test", "--use-selected-thresholds"]
    assert _cli(trained, "score", *flags) == 0
    d = trained["out"] / "scores" / "test"
    argv = ["--eval-set", str(d / "eval-set.json"), "--predictions", str(d / "predictions.json")]
    assert _cli(trained, "evaluate", *argv, "--stage", "test") == 0
    test_report = trained["out"] / "reports" / "test" / "quality_report.json"

    tampered = json.loads(test_report.read_text("utf-8"))
    tampered["labels"]["blue_a"]["false_alarms_per_hour"] = 1e12  # hand edit; the stored gate says PASS
    forged = trained["tmp"] / "forged.json"
    forged.write_text(json.dumps(tampered), encoding="utf-8")
    capsys.readouterr()
    assert _cli(trained, "qualify", "--test-report", str(forged)) == 1
    assert "re-run gate" in capsys.readouterr().out

    validation_report = trained["out"] / "reports" / "validation" / "quality_report.json"
    assert _cli(trained, "qualify", "--test-report", str(validation_report)) == 1

    assert _cli(trained, "qualify", "--test-report", str(test_report)) == 0
    bundles = list((trained["out"] / "bundles").iterdir())
    assert len(bundles) == 1
    manifest = exp.load_bundle(bundles[0])
    q = manifest["qualification"]
    assert manifest["artifact_kind"] == "selected" and q["status"] == "qualified"
    lineage = q["lineage"]
    assert q["lineage_hash"] and lineage["test_eval_set_hash"] != lineage["validation_eval_set_hash"]
    assert manifest["thresholds"] == {"blue_a": 0.0}
    assert manifest["threshold_status"] == "validation-calibrated"
    assert manifest["lineage"]["weights_sha256"] == q["lineage"]["weights_sha256"]


def test_export_checkpoint_stays_unqualified_and_fixture_cannot_be_scored(trained, tmp_path):
    _score_and_select(trained)
    selected = trained["out"] / "selection" / "selected.pt"
    assert _cli(trained, "export", "--checkpoint", str(selected)) == 0
    bundle = next((trained["out"] / "bundles").iterdir())
    assert exp.load_bundle(bundle)["qualification"]["status"] == "unqualified"
    payload = ckpt.load_checkpoint(selected)
    meta = {**payload["metadata"], "artifact_kind": "fixture-untrained"}
    meta.pop("weights_sha256")
    fixture = tmp_path / "fixture.pt"
    ckpt.save_checkpoint(fixture, {"model": payload["model"], "metadata": meta})
    assert _cli(trained, "score", "--checkpoint", str(fixture), "--split", "validation") == 1


def test_test_split_is_sealed_to_the_selected_checkpoint(trained, capsys):
    latest = trained["out"] / "checkpoints" / "latest.pt"
    capsys.readouterr()
    assert _cli(trained, "score", "--checkpoint", str(latest), "--split", "test") == 1
    assert "seal" in capsys.readouterr().out
    _score_and_select(trained)
    assert _cli(trained, "score", "--checkpoint", str(latest), "--split", "test") == 1
    selected = trained["out"] / "selection" / "selected.pt"
    assert _cli(trained, "score", "--checkpoint", str(selected), "--split", "test") == 0
    seal_path = trained["out"] / "selection" / "test-seal.json"
    seal = json.loads(seal_path.read_text("utf-8"))
    pointer = json.loads((trained["out"] / "selection" / "selected.json").read_text("utf-8"))
    assert seal["weights_sha256"] == pointer["weights_sha256"]
    assert _cli(trained, "score", "--checkpoint", str(selected), "--split", "test") == 0  # same weights
    seal_path.write_text(json.dumps({**seal, "weights_sha256": "tuned-after-test"}), encoding="utf-8")
    capsys.readouterr()
    assert _cli(trained, "score", "--checkpoint", str(selected), "--split", "test") == 1
    assert "sealed" in capsys.readouterr().out
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    d = trained["out"] / "scores" / "test"
    argv = ["--eval-set", str(d / "eval-set.json"), "--predictions", str(d / "predictions.json")]
    capsys.readouterr()
    assert _cli(trained, "evaluate", *argv, "--stage", "test-select", "--select") == 1
    assert "test split" in capsys.readouterr().out


def test_scored_eval_set_lists_reviewed_time_and_attributable_recordings(trained):
    latest = trained["out"] / "checkpoints" / "latest.pt"
    assert _cli(trained, "score", "--checkpoint", str(latest), "--split", "validation") == 0
    eval_set = json.loads((trained["out"] / "scores" / "validation" / "eval-set.json").read_text("utf-8"))
    reviewed_ids = {r["recording_id"] for r in eval_set["reviewed"]}
    assert reviewed_ids and {r["recording_id"] for r in eval_set["recordings"]} <= reviewed_ids
    assert "unattributable_recordings" in eval_set


def test_dev_pilot_reports_say_so_and_cannot_qualify(trained, capsys):
    cfg = json.loads(Path(trained["config"]).read_text(encoding="utf-8"))
    cfg["evaluation"]["purpose"] = "dev-pilot"
    Path(trained["config"]).write_text(json.dumps(cfg), encoding="utf-8")
    _score_and_select(trained)
    report = json.loads(
        (trained["out"] / "reports" / "validation" / "quality_report.json").read_text("utf-8")
    )
    assert report["purpose"] == "dev-pilot" and "not release" in report["claim"]
    selected = trained["out"] / "selection" / "selected.pt"
    assert _cli(trained, "score", "--checkpoint", str(selected), "--split", "test") == 0
    d = trained["out"] / "scores" / "test"
    argv = ["--eval-set", str(d / "eval-set.json"), "--predictions", str(d / "predictions.json")]
    _cli(trained, "evaluate", *argv, "--stage", "test")
    capsys.readouterr()
    test_report = trained["out"] / "reports" / "test" / "quality_report.json"
    assert _cli(trained, "qualify", "--test-report", str(test_report)) == 1
    assert "dev-pilot" in capsys.readouterr().out
