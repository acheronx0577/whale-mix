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


CONF = [f"c{i}" for i in range(3)]


def test_confusers_get_fixed_slots_and_leave_the_plain_negative_pool():
    s = BalancedSampler.build(
        POS, NEG + CONF, seed=5, positive_fraction=0.25, confuser_ids=CONF, confuser_fraction=0.25
    )
    assert s.stats()["negatives_available"] == len(NEG) and s.stats()["confusers_available"] == 3
    for b in _run(s, 9):
        assert [sum(x.startswith(p) for x in b) for p in "pcn"] == [2, 2, 4]
    flat = [
        x
        for b in _run(BalancedSampler.build(POS, NEG + CONF, 5, 0.25, CONF, 0.25), 9)
        for x in b
        if x[0] == "c"
    ]
    for k in range(0, 18, 3):
        assert sorted(flat[k : k + 3]) == CONF  # each pass is a full seeded permutation
    assert s.stats()["drawn"] == {"positive": 18, "negative": 36, "confuser": 18}


def test_confuser_resume_is_exact_and_settings_must_match():
    build = lambda: BalancedSampler.build(POS, NEG + CONF, 8, 0.25, CONF, 0.25)  # noqa: E731
    straight = _run(build(), 14)
    first = build()
    head = _run(first, 5)
    state = json.loads(json.dumps(first.state_dict()))
    resumed = build()
    resumed.load_state(state)
    assert head + _run(resumed, 9) == straight
    with pytest.raises(ValueError):
        BalancedSampler.build(POS, NEG + CONF, 8, 0.25).load_state(state)
    with pytest.raises(ValueError):
        BalancedSampler.build(POS, NEG + CONF, 8, 0.25, CONF, 0.125).load_state(state)


def test_without_confusers_the_sequence_and_state_are_unchanged():
    plain = BalancedSampler.build(POS, NEG, 3, 0.25)
    explicit = BalancedSampler.build(POS, NEG, 3, 0.25, confuser_ids=CONF, confuser_fraction=0.0)
    assert _run(plain, 6) == _run(explicit, 6)
    assert set(plain.state_dict()) == {"kind", "positive_fraction", "positives", "negatives", "drawn"}
    with pytest.raises(ValueError):
        BalancedSampler.build(POS, NEG, 3, 0.25, confuser_ids=[], confuser_fraction=0.25)
    with pytest.raises(ValueError):
        BalancedSampler.build(POS, NEG, 3, 0.25, confuser_ids=POS[:1], confuser_fraction=0.25)
    with pytest.raises(ValueError):
        BalancedSampler.build(POS, NEG, 3, 0.5, confuser_ids=CONF, confuser_fraction=0.5)


def _add_confuser_rows(corpus) -> None:
    """Mark 4-7 s of every annotated file as a reviewed blue_a-absent confuser interval (fixture only)."""
    import csv

    path = corpus["tmp"] / "annotations.csv"
    rows = list(csv.reader(path.open(encoding="utf-8")))
    files = sorted({r[1] for r in rows[1:]})
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        for n, fid in enumerate(files):
            w.writerow(
                [f"conf-{n}", fid, "0", "4.0", "7.0", "blue_a", "absent", "reviewed", "fixture confuser"]
            )
    from whalemix_training.cli import main

    assert main(["validate-labels", "--config", str(corpus["config"])]) == 0
    assert main(["split", "--config", str(corpus["config"])]) == 0


def test_cli_confuser_slots_are_drawn_and_resume_exactly(corpus):
    _small_config(corpus)
    _prepare_cli(corpus)
    _add_confuser_rows(corpus)
    balance = {
        "positive_fraction": 0.25,
        "labels": ["blue_a"],
        "confuser_fraction": 0.25,
        "confuser_id_prefixes": ["conf-"],
    }
    _set_training(corpus, max_updates=4, micro_batch=4, imbalance="balanced", balance=balance)
    assert _train(corpus) == 0
    straight, _ = _weights(corpus)
    plan = _plan(corpus)
    assert plan["sampler"]["confusers_available"] >= 1
    assert plan["session"]["mix"]["drawn"] == {"positive": 4, "negative": 8, "confuser": 4}
    sampled = set()
    manifest = json.loads((corpus["out"] / "split-manifest.json").read_text(encoding="utf-8"))
    for line in (
        (corpus["out"] / "checkpoints" / "train-sample-log.jsonl").read_text(encoding="utf-8").splitlines()
    ):
        for eid in json.loads(line)["ids"]:
            assert manifest["assignments"][eid.split(":")[0]]["split"] == "train"
            sampled.add(eid)
    shutil.rmtree(corpus["out"] / "checkpoints")
    _set_training(corpus, max_updates=2, micro_batch=4, imbalance="balanced", balance=balance)
    assert _train(corpus) == 0
    _set_training(corpus, max_updates=4, micro_batch=4, imbalance="balanced", balance=balance)
    assert _train(corpus, "--resume", str(corpus["out"] / "checkpoints" / "latest.pt")) == 0
    resumed, payload = _weights(corpus)
    assert resumed == straight and payload["sampler"]["drawn"] == {
        "positive": 4,
        "negative": 8,
        "confuser": 4,
    }


def test_cli_confusers_need_prefixes(corpus, capsys):
    _small_config(corpus)
    _prepare_cli(corpus)
    _set_training(corpus, max_updates=0, imbalance="balanced", balance={"confuser_fraction": 0.25})
    assert _train(corpus) == 1
    assert "confuser_id_prefixes" in capsys.readouterr().out


# --- recorder-balanced (grouped) positives -----------------------------------------------------------

GROUPS = {**{f"p{i}": "big" for i in range(36)}, **{f"q{i}": "mid" for i in range(9)}, "r0": "small"}
GPOS = sorted(GROUPS)


def _grouped(seed: int = 7, temperature: float = 0.5, confusers: bool = False) -> BalancedSampler:
    extra = {"confuser_ids": [f"c{i}" for i in range(6)], "confuser_fraction": 0.25} if confusers else {}
    return BalancedSampler.build(
        GPOS,
        NEG,
        seed=seed,
        positive_fraction=0.25,
        positive_groups=GROUPS,
        group_temperature=temperature,
        **extra,
    )


def test_grouped_weights_follow_the_temperature():
    for temperature, expected in ((1.0, (36, 9, 1)), (0.5, (6, 3, 1)), (0.0, (1, 1, 1))):
        w = _grouped(temperature=temperature).positives.weights()
        total = sum(expected)
        assert [round(w[g], 9) for g in ("big", "mid", "small")] == [round(x / total, 9) for x in expected]


def test_grouped_draws_match_weights_keep_slot_counts_and_only_use_positives():
    s = _grouped(temperature=0.5)
    batches = _run(s, 1500, n=8)
    pos_ids = set(GPOS)
    counts = {"big": 0, "mid": 0, "small": 0}
    for b in batches:
        assert sum(i in pos_ids for i in b) == 2 and len(b) == 8
        for i in b[:2]:
            counts[GROUPS[i]] += 1
    total = sum(counts.values())
    assert abs(counts["big"] / total - 0.6) < 0.03
    assert abs(counts["mid"] / total - 0.3) < 0.03
    assert abs(counts["small"] / total - 0.1) < 0.02
    assert s.stats()["positive_grouping"]["groups"]["small"]["drawn"] == counts["small"]
    # each group cycles through all of its own windows before repeating
    big_seq = [i for b in batches for i in b[:2] if GROUPS[i] == "big"]
    assert sorted(big_seq[:36]) == sorted(i for i in GPOS if GROUPS[i] == "big")


def test_grouped_resume_from_state_continues_the_exact_sequence():
    for confusers in (False, True):
        straight = _run(_grouped(confusers=confusers), 40)
        first = _grouped(confusers=confusers)
        head = _run(first, 17)
        state = json.loads(json.dumps(first.state_dict()))
        second = _grouped(confusers=confusers)
        second.load_state(state)
        assert head + _run(second, 23) == straight
        assert second.drawn == first.drawn | second.drawn


def test_grouped_seed_matters_and_mismatched_states_are_refused():
    assert _run(_grouped(seed=1), 10) != _run(_grouped(seed=2), 10)
    state = _grouped(temperature=0.5).state_dict()
    with pytest.raises(ValueError, match="grouping"):
        _grouped(temperature=0.0).load_state(state)
    plain = BalancedSampler.build(GPOS, NEG, seed=7, positive_fraction=0.25)
    with pytest.raises(ValueError, match="grouping"):
        plain.load_state(state)
    with pytest.raises(ValueError, match="grouping"):
        _grouped().load_state(plain.state_dict())
    with pytest.raises(ValueError, match="no group"):
        BalancedSampler.build(GPOS, NEG, seed=7, positive_fraction=0.25, positive_groups={"p0": "big"})


def test_temperature_one_with_one_group_is_a_plain_shuffle_of_that_group():
    one = {i: "only" for i in GPOS}
    s = BalancedSampler.build(GPOS, NEG, seed=7, positive_fraction=0.25, positive_groups=one)
    seq = [i for b in _run(s, 23) for i in b[:2]]
    assert sorted(seq[:46]) == sorted(GPOS)


def test_cli_grouped_positives_resume_exactly_and_report_groups(corpus):
    _small_config(corpus)
    _prepare_cli(corpus)
    balance = {
        "positive_fraction": 0.5,
        "labels": None,
        "positive_group_by": "deployment",
        "positive_group_temperature": 0.5,
    }
    _set_training(corpus, max_updates=4, imbalance="balanced", balance=balance)
    assert _train(corpus) == 0
    straight, _ = _weights(corpus)
    plan = _plan(corpus)
    grouping = plan["sampler"]["positive_grouping"]
    assert grouping["temperature"] == 0.5 and len(grouping["groups"]) >= 1
    assert abs(sum(g["weight"] for g in grouping["groups"].values()) - 1.0) < 1e-5
    assert sum(g["drawn"] for g in plan["session"]["mix"]["positive_grouping"]["groups"].values()) == 4
    shutil.rmtree(corpus["out"] / "checkpoints")
    _set_training(corpus, max_updates=2, imbalance="balanced", balance=balance)
    assert _train(corpus) == 0
    _set_training(corpus, max_updates=4, imbalance="balanced", balance=balance)
    assert _train(corpus, "--resume", str(corpus["out"] / "checkpoints" / "latest.pt")) == 0
    resumed, payload = _weights(corpus)
    assert resumed == straight and payload["sampler"]["positives"]["kind"] == "grouped"
    assert payload["sampler"]["positives"]["drawn"] == 4


def test_cli_grouping_field_must_exist_for_every_positive(corpus, capsys):
    _small_config(corpus)
    _prepare_cli(corpus)
    balance = {"positive_fraction": 0.5, "labels": None, "positive_group_by": "recorder_id"}
    _set_training(corpus, max_updates=0, imbalance="balanced", balance=balance)
    assert _train(corpus) == 1
    assert "positive_group_by" in capsys.readouterr().out
