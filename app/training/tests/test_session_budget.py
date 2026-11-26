"""Blocker 1: each session is capped by min(remaining global updates, per-session limit), so a 3,000-update
run with a 1,000-update session cap finishes in three resumed sessions instead of stopping at 1,000."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from tests.test_model_and_trainer import _prepare_cli, _small_config
from tests.test_resume_replay import _set_training, _train, _weights
from whalemix_training.trainer import SessionBudget


def _budget(max_updates: int, session: int) -> SessionBudget:
    return SessionBudget(max_updates, session, 1e9, 10**9, 1e9)


def test_session_cap_applies_to_the_remaining_global_budget():
    b = _budget(3000, 1000)
    assert [b.updates_allowed(done) for done in (0, 1000, 2000)] == [1000, 1000, 1000]
    assert b.updates_allowed(2500) == 500  # boundary: only the remainder
    assert b.updates_allowed(2999) == 1
    assert b.updates_allowed(3000) == 0 and b.updates_allowed(3100) == 0
    assert _budget(500, 1000).updates_allowed(0) == 500
    assert _budget(0, 1000).updates_allowed(0) == 0


def _sample_log(corpus) -> list[dict]:
    path = corpus["out"] / "checkpoints" / "train-sample-log.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_three_capped_sessions_match_one_straight_run(corpus):
    _small_config(corpus)
    _prepare_cli(corpus)
    _set_training(corpus, max_updates=6, session_max_updates=6)
    assert _train(corpus) == 0
    straight, _ = _weights(corpus)
    straight_log = _sample_log(corpus)
    assert [e["global_update"] for e in straight_log] == [1, 2, 3, 4, 5, 6]
    shutil.rmtree(corpus["out"] / "checkpoints")

    _set_training(corpus, max_updates=6, session_max_updates=2)
    latest = corpus["out"] / "checkpoints" / "latest.pt"
    assert _train(corpus) == 0
    sessions = []
    for _ in range(2):
        assert _train(corpus, "--resume", str(latest)) == 0
        plan = json.loads((corpus["out"] / "train-plan.json").read_text(encoding="utf-8"))
        sessions.append(plan["session"]["optimizer_updates"])
    assert sessions == [2, 2]  # was [0, 0]: the cap was applied to the global count
    resumed, payload = _weights(corpus)
    assert payload["metadata"]["global_update"] == 6
    assert resumed == straight
    assert _sample_log(corpus) == straight_log  # same sample IDs in the same order across sessions
    # A fourth resume has nothing left to do and changes nothing.
    assert _train(corpus, "--resume", str(latest)) == 0
    plan = json.loads(Path(corpus["out"] / "train-plan.json").read_text(encoding="utf-8"))
    assert plan["session"]["optimizer_updates"] == 0 and _weights(corpus)[0] == straight
