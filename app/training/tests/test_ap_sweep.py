"""Average precision must follow the operating-point merge and match rule at every threshold.

Regression for the collapse where unthresholded overlapping windows (10.156 s long, 5 s hop) chain-merged
into one event per recording, so AP fell to 1/N however well the calls were ranked.
"""

from __future__ import annotations

import random

import pytest

from whalemix_training.metrics import (
    AP_METHOD,
    MatchRule,
    _match_track,
    average_precision,
    label_metrics,
    match,
    precision_recall_sweep,
    sweep_average_precision,
)

WINDOW_S = 10.156
HOP_S = 5.0


def ev(rec, a, b, score=1.0, label="blue_a"):
    return {"recording_id": rec, "label": label, "start_s": a, "end_s": b, "score": score}


def long_recording(n_calls=30, hours=1.5, call_score=0.95, background_score=0.05):
    """One continuous recording scored with dense overlapping windows; calls well apart."""
    duration = hours * 3600.0
    truth = [ev("long", 60.0 + 150.0 * i, 60.0 + 150.0 * i + 15.0) for i in range(n_calls)]
    assert truth[-1]["end_s"] < duration
    preds, start = [], 0.0
    while start + WINDOW_S <= duration:
        end = start + WINDOW_S
        on_call = any(start < t["end_s"] and end > t["start_s"] for t in truth)
        preds.append(ev("long", start, end, call_score if on_call else background_score))
        start += HOP_S
    return truth, preds


def test_ap_does_not_collapse_to_one_over_n_on_a_long_recording():
    truth, preds = long_recording()
    n = len(truth)
    rule = MatchRule()
    # The legacy ranking (all windows, one merge) collapses the recording into a single event.
    legacy = match(truth, preds, rule)
    assert len(legacy["outcomes"]) == 1
    assert average_precision(legacy["outcomes"], n) == pytest.approx(1 / n)
    m = label_metrics(truth, preds, "blue_a", 1.5, rule, 0.5, min_events=1)
    assert (m["tp"], m["fp"], m["fn"]) == (n, 0, 0)
    assert m["average_precision"] == pytest.approx(1.0)
    assert m["average_precision"] > 10 / n
    assert m["average_precision_method"] == AP_METHOD


def test_ap_reflects_ranking_on_a_long_recording():
    truth, preds = long_recording(n_calls=20)
    # Weaken half the calls below some background windows: AP must fall strictly between 1/N and 1.
    weak = {i for i in range(0, 20, 2)}
    for p in preds:
        hit = [i for i, t in enumerate(truth) if p["start_s"] < t["end_s"] and p["end_s"] > t["start_s"]]
        if hit and hit[0] in weak:
            p["score"] = 0.6
    for p in preds[5::40]:
        if p["score"] == 0.05:
            p["score"] = 0.8
    ap = label_metrics(truth, preds, "blue_a", 1.5, MatchRule(), 0.5, min_events=1)["average_precision"]
    assert 1 / 20 < ap < 1.0
    assert ap > 0.5


def random_case(seed):
    rng = random.Random(seed)
    truth, preds = [], []
    for rec in ("a", "b", "c"):
        for _ in range(rng.randint(0, 8)):
            s = rng.uniform(0, 600)
            truth.append(ev(rec, s, s + rng.uniform(1, 20)))
        t = 0.0
        while t < 620:
            preds.append(ev(rec, t, t + WINDOW_S, round(rng.random(), 2)))
            t += HOP_S * rng.choice([1, 1, 2, 3])
    reviewed = [{"recording_id": r, "label": "blue_a", "start_s": 0.0, "end_s": 400.0} for r in ("a", "b")]
    return truth, preds, reviewed


@pytest.mark.parametrize("seed", range(6))
def test_sweep_points_equal_the_operating_point_at_every_threshold(seed):
    truth, preds, reviewed = random_case(seed)
    rule = MatchRule()
    points = precision_recall_sweep(truth, preds, rule)
    assert [p["threshold"] for p in points] == sorted({p["score"] for p in preds}, reverse=True)
    for point in points[::7]:
        m = label_metrics(truth, preds, "blue_a", 1.0, rule, point["threshold"], min_events=1)
        assert (point["tp"], point["fp"]) == (m["tp"], m["fp"])
        mr = label_metrics(
            truth, preds, "blue_a", 1.0, rule, point["threshold"], min_events=1, reviewed=reviewed
        )
        assert mr["tp"] == m["tp"]
        assert 0.0 <= mr["average_precision"] <= 1.0


@pytest.mark.parametrize("seed", range(6))
def test_track_matcher_is_identical_to_match(seed):
    truth, preds, _ = random_case(seed)
    rule = MatchRule()
    for threshold in (0.0, 0.4, 0.8):
        active = [p for p in preds if p["score"] >= threshold]
        full = match(truth, active, rule)
        expected = sorted(
            (o["prediction"]["recording_id"], o["prediction"]["start_s"], o["truth_index"] is not None)
            for o in full["outcomes"]
        )
        got = []
        for rec in ("a", "b", "c"):
            for p, hit in _match_track(
                [t for t in truth if t["recording_id"] == rec],
                [p for p in active if p["recording_id"] == rec],
                rule,
            ):
                got.append((rec, p["start_s"], hit))
        assert sorted(got) == expected


def test_threshold_cap_keeps_extremes_and_bounds():
    truth, preds, _ = random_case(3)
    for i, p in enumerate(preds):
        p["score"] = i / len(preds)
    points = precision_recall_sweep(truth, preds, MatchRule(), max_thresholds=16)
    assert len(points) == 16
    assert points[0]["threshold"] == max(p["score"] for p in preds)
    assert points[-1]["threshold"] == min(p["score"] for p in preds)
    ap = sweep_average_precision(points, len(truth))
    assert ap is None or 0.0 <= ap <= 1.0


def test_no_truth_gives_no_ap():
    assert sweep_average_precision([{"threshold": 0.5, "tp": 0, "fp": 3}], 0) is None
    assert sweep_average_precision([], 4) == 0.0
