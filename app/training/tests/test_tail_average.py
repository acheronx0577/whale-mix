"""Opt-in second-half learning-rate decay and tail weight averaging: off by default, exact across
`train --resume`, BatchNorm statistics refreshed on training windows only. Fixture data only."""

from __future__ import annotations

import itertools
import json
import math
import shutil
from pathlib import Path

import pytest
import torch
from torch import nn

from tests.test_model_and_trainer import _prepare_cli, _small_config
from whalemix_training import checkpoint as ckpt
from whalemix_training import tail_average as tail_mod
from whalemix_training.cli import main
from whalemix_training.replay import DeterministicSampler
from whalemix_training.trainer import SessionBudget, run_session

SCHEDULE = {"kind": "cosine", "start_update": 2, "end_update": 6, "final_factor": 0.1}
TAIL = {"start_update": 3, "every_updates": 1, "bn_refresh_batches": 2}


def _set_training(corpus, **training) -> None:
    cfg = json.loads(Path(corpus["config"]).read_text(encoding="utf-8"))
    cfg["training"] = {
        "max_updates": 0,
        "micro_batch": 2,
        "checkpoint_every_updates": 1,
        "compute_threads": 1,
        **training,
    }
    Path(corpus["config"]).write_text(json.dumps(cfg), encoding="utf-8")


def _train(corpus, *extra: str) -> int:
    return main(["train", "--config", str(corpus["config"]), *extra])


def _ckpts(corpus) -> Path:
    return corpus["out"] / "checkpoints"


def _plan(corpus) -> dict:
    return json.loads((corpus["out"] / "train-plan.json").read_text(encoding="utf-8"))


def test_lr_factor_is_a_pure_cosine_of_the_global_update():
    s = tail_mod.validate_lr_schedule(
        {"kind": "cosine", "start_update": 3000, "end_update": 6000, "final_factor": 0.1}
    )
    assert tail_mod.lr_factor(0, s) == 1.0 and tail_mod.lr_factor(2999, s) == 1.0
    assert tail_mod.lr_factor(3000, s) == pytest.approx(1.0)
    assert tail_mod.lr_factor(4500, s) == pytest.approx(0.55)
    assert tail_mod.lr_factor(5999, s) == pytest.approx(0.1, abs=1e-6)
    assert tail_mod.lr_factor(6000, s) == 0.1 and tail_mod.lr_factor(9000, s) == 0.1
    values = [tail_mod.lr_factor(u, s) for u in range(3000, 6000)]
    assert all(a >= b for a, b in itertools.pairwise(values))
    assert tail_mod.lr_factor(123, None) == 1.0
    for bad in (
        {"kind": "step", "start_update": 1, "end_update": 2, "final_factor": 0.1},
        {"kind": "cosine", "start_update": 5, "end_update": 5, "final_factor": 0.1},
        {"kind": "cosine", "start_update": 1, "end_update": 5, "final_factor": 0.0},
    ):
        with pytest.raises(ValueError):
            tail_mod.validate_lr_schedule(bad)


def test_run_session_applies_the_rate_by_global_index():
    model = nn.Linear(4, 11)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    seen = []

    def load(ids):
        return torch.ones(len(ids), 4), torch.ones(len(ids), 11), torch.ones(len(ids), 11)

    run_session(
        model,
        opt,
        DeterministicSampler([f"e{i}" for i in range(8)], 1),
        load,
        SessionBudget(6, 3, 60, 100, 600),
        2,
        lambda r, n: None,
        start_update=2,
        on_update=lambda u, ids: seen.append((u, opt.param_groups[0]["lr"])),
        lr_at=lambda i: 0.5**i,
    )
    assert seen == [(3, 0.25), (4, 0.125), (5, 0.0625)]


def test_tail_average_is_a_uniform_mean_of_scheduled_snapshots():
    model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    tail = tail_mod.TailAverage(start_update=2, every_updates=2, bn_refresh_batches=0)
    snaps = []
    for update in range(1, 7):
        with torch.no_grad():
            model[0].weight.fill_(float(update))
            model[1].num_batches_tracked.fill_(update)
        if tail.observe(update, model):
            snaps.append(update)
    assert snaps == [2, 4, 6] and tail.count == 3 == tail.expected_count(6)
    avg = tail.averaged_state(model.state_dict())
    assert torch.allclose(avg["0.weight"], torch.full((2, 2), 4.0))
    assert avg["0.weight"].dtype == torch.float32 and int(avg["1.num_batches_tracked"]) == 6
    with pytest.raises(ValueError):
        tail_mod.TailAverage(0, 1, 0)


def test_tail_state_round_trips_and_rejects_mismatches():
    model = nn.Linear(2, 2)
    tail = tail_mod.TailAverage(1, 1, 0)
    tail.observe(1, model)
    saved = tail.state_dict()
    other = tail_mod.TailAverage(1, 1, 0)
    other.load_state(saved, 1)
    assert other.count == 1 and other.last_update == 1
    with pytest.raises(ValueError):
        tail_mod.TailAverage(1, 2, 0).load_state(saved, 1)  # settings differ
    with pytest.raises(ValueError):
        tail_mod.TailAverage(1, 1, 0).load_state(None, 3)  # window started but no state saved
    with pytest.raises(ValueError):
        tail_mod.TailAverage(1, 1, 0).load_state(saved, 0)  # state ahead of the checkpoint
    tail_mod.TailAverage(5, 1, 0).load_state(None, 3)  # window not started yet: nothing to restore


def test_bn_refresh_recomputes_exact_statistics_with_other_layers_in_eval():
    model = nn.Sequential(nn.Dropout(0.9), nn.BatchNorm1d(3))
    model[1].running_mean.fill_(99.0)
    data = {f"w{i}": torch.tensor([[float(i), 2.0 * i, -1.0]]) for i in range(6)}
    batches = [["w0", "w1"], ["w2", "w3"], ["w4", "w5"]]
    used = tail_mod.refresh_batchnorm(model, batches, lambda ids: torch.cat([data[i] for i in ids]))
    assert used == 3 and not model.training and model[1].momentum == 0.1
    batch_means = torch.stack([torch.cat([data[i] for i in ids]).mean(0) for ids in batches])
    assert torch.allclose(model[1].running_mean, batch_means.mean(0))  # dropout was off (eval)
    assert tail_mod.refresh_batchnorm(nn.Linear(2, 2), batches, lambda ids: None) == 0


def test_bn_refresh_ids_are_seeded_and_drawn_from_the_given_pool():
    pool = [f"t{i:03d}" for i in range(100)]
    a = tail_mod.bn_refresh_ids(pool, 5, 4, 7)
    assert a == tail_mod.bn_refresh_ids(list(reversed(pool)), 5, 4, 7)
    assert a != tail_mod.bn_refresh_ids(pool, 5, 4, 8)
    flat = [x for b in a for x in b]
    assert len(flat) == 20 == len(set(flat)) and set(flat) <= set(pool)
    small = tail_mod.bn_refresh_ids(pool[:3], 5, 4, 7)  # fewer windows than requested: use them all once
    assert len(small) == 1 and sorted(small[0]) == pool[:3]
    assert tail_mod.bn_refresh_ids([], 5, 4, 7) == []


def test_defaults_leave_training_unchanged(corpus):
    _small_config(corpus)
    _prepare_cli(corpus)
    _set_training(corpus, max_updates=2)
    assert _train(corpus) == 0
    plan = _plan(corpus)
    assert plan["lr_schedule"] is None and plan["tail_average"] is None
    assert "tail_average_result" not in plan
    payload = ckpt.load_checkpoint(_ckpts(corpus) / "latest.pt")
    assert "tail_average" not in payload
    assert payload["optimizer"]["param_groups"][0]["lr"] == pytest.approx(0.001)
    assert not list(_ckpts(corpus).glob("tail-average-*.pt"))


def test_schedule_and_tail_average_are_exact_across_resume(corpus):
    _small_config(corpus)
    _prepare_cli(corpus)
    opts = {"lr_schedule": SCHEDULE, "tail_average": TAIL}
    _set_training(corpus, max_updates=6, **opts)
    assert _train(corpus) == 0
    straight = ckpt.load_checkpoint(_ckpts(corpus) / "latest.pt")
    straight_tail = ckpt.load_checkpoint(_ckpts(corpus) / "tail-average-u6.pt")
    plan = _plan(corpus)
    result = plan["tail_average_result"]
    assert result["snapshots"] == 4 and result["first_update"] == 3 and result["last_update"] == 6
    assert result["bn_refresh_split"] == "train" and result["bn_refresh_batches_used"] >= 1
    assert result["weights_sha256"] == straight_tail["metadata"]["weights_sha256"]
    assert straight_tail["metadata"]["weights_sha256"] != straight["metadata"]["weights_sha256"]
    assert straight_tail["metadata"]["artifact_kind"] == "training-state"
    last_lr = straight["optimizer"]["param_groups"][0]["lr"]
    assert last_lr == pytest.approx(0.001 * tail_mod.lr_factor(5, plan["lr_schedule"]))
    assert last_lr < 0.001
    shutil.rmtree(_ckpts(corpus))

    _set_training(corpus, max_updates=6, session_max_updates=3, **opts)
    assert _train(corpus) == 0
    half = ckpt.load_checkpoint(_ckpts(corpus) / "latest.pt")
    assert half["metadata"]["global_update"] == 3 and half["tail_average"]["count"] == 1
    assert not list(_ckpts(corpus).glob("tail-average-*.pt"))
    shutil.copy(_ckpts(corpus) / "latest.pt", _ckpts(corpus) / "session1.pt")
    assert _train(corpus, "--resume", str(_ckpts(corpus) / "session1.pt")) == 0
    resumed = ckpt.load_checkpoint(_ckpts(corpus) / "latest.pt")
    resumed_tail = ckpt.load_checkpoint(_ckpts(corpus) / "tail-average-u6.pt")
    assert resumed["metadata"]["weights_sha256"] == straight["metadata"]["weights_sha256"]
    assert resumed_tail["metadata"]["weights_sha256"] == straight_tail["metadata"]["weights_sha256"]
    assert _plan(corpus)["exact_resume"] is True


def test_resume_rejects_changed_late_training_settings(corpus, capsys):
    _small_config(corpus)
    _prepare_cli(corpus)
    _set_training(corpus, max_updates=2, lr_schedule=SCHEDULE, tail_average=TAIL)
    assert _train(corpus) == 0
    changed = {**SCHEDULE, "final_factor": 0.5}
    _set_training(corpus, max_updates=6, lr_schedule=changed, tail_average=TAIL)
    capsys.readouterr()
    assert _train(corpus, "--resume", str(_ckpts(corpus) / "latest.pt")) == 1
    assert "SETUP_REQUIRED" in capsys.readouterr().out
    _set_training(corpus, max_updates=2, lr_schedule={**SCHEDULE, "kind": "step"})
    assert _train(corpus) == 1


def test_cosine_matches_the_closed_form_inside_the_window():
    assert math.isclose(tail_mod.lr_factor(3, SCHEDULE), 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi / 4)))
