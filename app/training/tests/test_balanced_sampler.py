"""Seeded positive oversampling (`training.imbalance: balanced`): fixed positives per batch, deterministic,
resume-exact, off by default."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tests.test_model_and_trainer import _prepare_cli, _small_config
from tests.test_resume_replay import _set_training, _train, _weights
from whalemix_training.config import DEFAULTS
from whalemix_training.replay import BalancedSampler, DeterministicSampler

POS = [f"p{i}" for i in range(5)]
NEG = [f"n{i}" for i in range(40)]


def _run(s: BalancedSampler, batches: int, n: int = 8) -> list[list[str]]:
    out = []
    for _ in range(batches):
        out.append(list(s.draw(n)))
        s.consume()
    return out


def test_each_batch_has_the_fixed_positive_count_and_positives_wrap_epochs():
    s = BalancedSampler.build(POS, NEG, seed=7, positive_fraction=0.25)
    batches = _run(s, 12)
    for b in batches:
        assert sum(x.startswith("p") for x in b) == 2 and len(b) == 8
    # 24 positives drawn from 5: every pass is a full permutation before any repeat
    flat = [x for b in batches for x in b if x.startswith("p")]
    for k in range(0, 20, 5):
        assert sorted(flat[k : k + 5]) == POS
    assert s.stats()["drawn"] == {"positive": 24, "negative": 72}
    assert s.stats()["positive_epochs"] == 4 and s.epoch == 1


def test_same_seed_same_sequence_and_other_seed_differs():
    a = _run(BalancedSampler.build(POS, NEG, 3, 0.5), 10)
    b = _run(BalancedSampler.build(list(reversed(POS)), list(reversed(NEG)), 3, 0.5), 10)
    c = _run(BalancedSampler.build(POS, NEG, 4, 0.5), 10)
    assert a == b and a != c


def test_resume_from_state_continues_the_exact_sequence():
    straight = _run(BalancedSampler.build(POS, NEG, 11, 0.3), 15)
    first = BalancedSampler.build(POS, NEG, 11, 0.3)
    head = _run(first, 6)
    state = json.loads(json.dumps(first.state_dict()))  # survives a JSON/checkpoint round trip
    resumed = BalancedSampler.build(POS, NEG, 11, 0.3)
    resumed.load_state(state)
    assert head + _run(resumed, 9) == straight


def test_draw_without_consume_does_not_advance():
    s = BalancedSampler.build(POS, NEG, 1, 0.25)
    assert s.draw(8) == s.draw(8)


def test_mismatched_states_are_refused():
    s = BalancedSampler.build(POS, NEG, 1, 0.25)
    with pytest.raises(ValueError):
        BalancedSampler.build(POS, NEG, 1, 0.5).load_state(s.state_dict())
    with pytest.raises(ValueError):
        s.load_state(DeterministicSampler(POS + NEG, 1).state_dict())
    with pytest.raises(ValueError):
        BalancedSampler.build(POS, NEG[:-1], 1, 0.25).load_state(s.state_dict())
    with pytest.raises(ValueError):
        BalancedSampler.build([], NEG, 1, 0.25)
    with pytest.raises(ValueError):
        BalancedSampler.build(POS, NEG, 1, 1.0)


def test_off_by_default():
    assert DEFAULTS["training"]["imbalance"] == "none"


def _plan(corpus) -> dict:
    plan = json.loads((corpus["out"] / "train-plan.json").read_text(encoding="utf-8"))
    assert plan["sampler"]["kind"] == "balanced"
    return plan


def test_cli_balanced_training_resumes_exactly_and_batches_hold_positives(corpus):
    _small_config(corpus)
    _prepare_cli(corpus)
    balance = {"positive_fraction": 0.5, "labels": None}
    _set_training(corpus, max_updates=4, imbalance="balanced", balance=balance)
    assert _train(corpus) == 0
    straight, _ = _weights(corpus)
    plan = _plan(corpus)
    assert plan["sampler"]["positives_available"] >= 1 and plan["sampler"]["negatives_available"] >= 1
    assert plan["session"]["mix"]["drawn"] == {"positive": 4, "negative": 4}
    log = (corpus["out"] / "checkpoints" / "train-sample-log.jsonl").read_text(encoding="utf-8").splitlines()
    assert [len(json.loads(x)["ids"]) for x in log] == [2, 2, 2, 2]
    shutil.rmtree(corpus["out"] / "checkpoints")

    _set_training(corpus, max_updates=2, imbalance="balanced", balance=balance)
    assert _train(corpus) == 0
    _set_training(corpus, max_updates=4, imbalance="balanced", balance=balance)
    assert _train(corpus, "--resume", str(corpus["out"] / "checkpoints" / "latest.pt")) == 0
    resumed, payload = _weights(corpus)
    assert payload["metadata"]["global_update"] == 4 and resumed == straight
    assert payload["sampler"]["kind"] == "balanced" and payload["sampler"]["drawn"] == {
        "positive": 4,
        "negative": 4,
    }


def test_cli_resume_refuses_a_changed_sampler(corpus, capsys):
    _small_config(corpus)
    _prepare_cli(corpus)
    _set_training(corpus, max_updates=2)
    assert _train(corpus) == 0
    latest = Path(corpus["out"]) / "checkpoints" / "latest.pt"
    _set_training(corpus, max_updates=4, imbalance="balanced")
    assert _train(corpus, "--resume", str(latest)) == 1
    assert "SETUP_REQUIRED" in capsys.readouterr().out


def test_cli_rejects_an_unknown_imbalance_mode(corpus, capsys):
    _small_config(corpus)
    _prepare_cli(corpus)
    _set_training(corpus, max_updates=0, imbalance="weighted")
    assert _train(corpus) == 1
    assert "imbalance" in capsys.readouterr().out
