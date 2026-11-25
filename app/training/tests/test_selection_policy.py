"""Blocker 3 (local training handoff): per-profile minimum recall and minimum reviewed background
exposure, false alarms counted only on reviewed assessable time, a sealed test split and dev-pilot reports
that cannot qualify. Fixture numbers only; nothing here measures whale-detection accuracy."""

from __future__ import annotations

from whalemix_training import selection
from whalemix_training.config import DEFAULTS
from whalemix_training.metrics import MatchRule, label_metrics
from whalemix_training.scoring import reviewed_intervals

POLICY = {**DEFAULTS["selection"], "min_recall": {"blue_a": 0.5}, "min_background_hours": {"blue_a": 1.0}}
CANDIDATE = {"artifact_kind": "training-state", "weights_sha256": "w", "checkpoint_sha256": "c"}


def _calls(n: int) -> list[dict]:
    return [
        {"recording_id": "r1", "channel": 0, "label": "blue_a", "start_s": 60.0 * i, "end_s": 60.0 * i + 10}
        for i in range(n)
    ]


def _report(metrics: dict) -> dict:
    return {
        "eval_set_hash": "e",
        "match_rule": {"version": "v"},
        "operating_budget": 2.0,
        "preprocessing_hash": "p",
        "resources": {"within_budget": True},
        "labels": {"blue_a": metrics},
        "retention": {},
        "weights_sha256": "w",
        "candidate": CANDIDATE,
    }


def _one_of_twenty() -> dict:
    truth = _calls(20)
    preds = [{**truth[0], "score": 0.9}]  # finds exactly one of the 20 reviewed calls
    reviewed = [{"recording_id": "r1", "channel": 0, "label": "blue_a", "start_s": 0.0, "end_s": 7200.0}]
    return label_metrics(truth, preds, "blue_a", 2.0, MatchRule(), 0.5, min_events=20, reviewed=reviewed)


def test_one_of_twenty_calls_fails_when_the_minimum_recall_is_higher():
    m = _one_of_twenty()
    assert m["recall"] == 1 / 20 and m["evidence"] == "sufficient"
    outcome = selection.gate(_report(m), None, POLICY)
    assert outcome["status"] == selection.FAIL
    assert any("below minimum recall" in r for r in outcome["reasons"])
    lenient = {**POLICY, "min_recall": {"blue_a": 0.04}}
    assert selection.gate(_report(m), None, lenient)["status"] == selection.PASS_ELIGIBLE


def test_undeclared_minimums_are_setup_required():
    m = _one_of_twenty()
    for key in ("min_recall", "min_background_hours"):
        policy = {**POLICY, key: {}}
        outcome = selection.gate(_report(m), None, policy)
        assert outcome["status"] == selection.SETUP_REQUIRED and key in outcome["reasons"][0]


def test_too_little_reviewed_background_is_inconclusive():
    m = {**_one_of_twenty(), "recall": 0.9}
    thin = {**POLICY, "min_recall": {"blue_a": 0.5}, "min_background_hours": {"blue_a": 5.0}}
    outcome = selection.gate(_report(m), None, thin)
    assert outcome["status"] == selection.INCONCLUSIVE and "background" in outcome["reasons"][0]
    unreviewed = {k: v for k, v in m.items() if k != "reviewed_background_hours"}
    assert selection.gate(_report(unreviewed), None, POLICY)["status"] == selection.INCONCLUSIVE


def test_false_alarms_count_only_reviewed_assessable_time():
    truth = [{"recording_id": "r1", "channel": 0, "label": "blue_a", "start_s": 100.0, "end_s": 110.0}]
    reviewed = [{"recording_id": "r1", "channel": 0, "label": "blue_a", "start_s": 0.0, "end_s": 1800.0}]
    preds = [
        {**truth[0], "score": 0.9},
        {
            "recording_id": "r1",
            "channel": 0,
            "label": "blue_a",
            "start_s": 500.0,
            "end_s": 502.0,
            "score": 0.8,
        },
        # outside reviewed time on the same recording, and on a recording nobody reviewed
        {
            "recording_id": "r1",
            "channel": 0,
            "label": "blue_a",
            "start_s": 3000.0,
            "end_s": 3002.0,
            "score": 0.8,
        },
        {"recording_id": "r2", "channel": 0, "label": "blue_a", "start_s": 10.0, "end_s": 12.0, "score": 0.8},
    ]
    m = label_metrics(truth, preds, "blue_a", 10.0, MatchRule(), 0.5, min_events=1, reviewed=reviewed)
    assert m["exposure_basis"] == "reviewed"
    assert m["fp"] == 1 and m["unreviewed_detections"] == 2
    assert m["reviewed_hours"] == 0.5 and m["false_alarms_per_hour"] == 2.0
    assert abs(m["reviewed_background_hours"] - (1800.0 - 10.0) / 3600.0) < 1e-9


def test_reviewed_intervals_are_merged_per_recording_channel_and_label():
    files = {"f1": {"sample_rate": 2000}, "f2": {"sample_rate": 2000}}
    rows = [
        {
            "file_id": "f1",
            "channel": "0",
            "label": "blue_a",
            "state": "present",
            "review_status": "reviewed",
            "start_s": "0",
            "end_s": "10",
        },
        {
            "file_id": "f1",
            "channel": "0",
            "label": "blue_a",
            "state": "absent",
            "review_status": "reviewed",
            "start_s": "5",
            "end_s": "30",
        },
        {
            "file_id": "f1",
            "channel": "0",
            "label": "blue_a",
            "state": "present",
            "review_status": "unreviewed",
            "start_s": "40",
            "end_s": "50",
        },
        {
            "file_id": "f2",
            "channel": "0",
            "label": "blue_a",
            "state": "present",
            "review_status": "ambiguous",
            "start_s": "0",
            "end_s": "10",
        },
    ]
    got = reviewed_intervals(rows, files, ["blue_a"], None)
    assert got == [{"recording_id": "f1", "channel": 0, "label": "blue_a", "start_s": 0.0, "end_s": 30.0}]
