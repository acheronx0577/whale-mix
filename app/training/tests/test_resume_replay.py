"""Fix 3 (training readiness): exact `train --resume`, last-good checkpoints, and fine-tune batches that
really mix replay IDs. Fixture data only; a few optimizer updates on a tiny spectrogram spec."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

from tests.test_model_and_trainer import _prepare_cli, _small_config
from whalemix_training import checkpoint as ckpt
from whalemix_training import model_commands
from whalemix_training import replay as replay_mod
from whalemix_training.cli import main
from whalemix_training.replay import DeterministicSampler, ReplayAdmissionError, StratifiedReplay
from whalemix_training.trainer import SessionBudget, run_session


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


def _weights(corpus) -> str:
    payload = ckpt.load_checkpoint(corpus["out"] / "checkpoints" / "latest.pt")
    return payload["metadata"]["weights_sha256"], payload


def _train(corpus, *extra: str) -> int:
    return main(["train", "--config", str(corpus["config"]), *extra])


def test_resume_is_exact_and_continues_the_update_count(corpus):
    _small_config(corpus)
    _prepare_cli(corpus)
    _set_training(corpus, max_updates=4)
    assert _train(corpus) == 0
    straight, _ = _weights(corpus)
    shutil.rmtree(corpus["out"] / "checkpoints")

    _set_training(corpus, max_updates=2)
    assert _train(corpus) == 0
    half, payload = _weights(corpus)
    assert payload["metadata"]["global_update"] == 2 and half != straight
    _set_training(corpus, max_updates=4)
    assert _train(corpus, "--resume", str(corpus["out"] / "checkpoints" / "latest.pt")) == 0
    resumed, payload = _weights(corpus)
    assert payload["metadata"]["global_update"] == 4
    assert resumed == straight
    plan = json.loads((corpus["out"] / "train-plan.json").read_text(encoding="utf-8"))
    assert plan["resumed_from"]["global_update"] == 2 and plan["session"]["optimizer_updates"] == 2


def test_resume_rejects_mismatched_runs(corpus, capsys):
    _small_config(corpus)
    _prepare_cli(corpus)
    _set_training(corpus, max_updates=1)
    assert _train(corpus) == 0
    latest = corpus["out"] / "checkpoints" / "latest.pt"
    _set_training(corpus, max_updates=3, learning_rate=0.5)  # optimizer settings changed
    capsys.readouterr()
    assert _train(corpus, "--resume", str(latest)) == 1
    out = capsys.readouterr().out
    assert "SETUP_REQUIRED" in out and "resume" in out
    _set_training(corpus, max_updates=3)
    assert _train(corpus, "--resume", str(corpus["out"] / "missing.pt")) == 1


def test_non_finite_loss_stops_without_overwriting_last_good(tmp_path):
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 11))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    calls = {"n": 0}
    saved = []

    def load(ids):
        calls["n"] += 1
        x = torch.ones(len(ids), 1, 2, 2)
        if calls["n"] == 3:
            x = x * float("nan")
        return x, torch.ones(len(ids), 11), torch.ones(len(ids), 11)

    result = run_session(
        model,
        opt,
        DeterministicSampler([f"e{i}" for i in range(8)], 1),
        load,
        SessionBudget(10, 10, 60, 1, 600),
        2,
        lambda reason, n: saved.append((reason, n)),
    )
    assert result["stopped"] == "non_finite_loss" and result["optimizer_updates"] == 2
    assert saved == [("periodic", 1), ("periodic", 2)]
    assert all(torch.isfinite(p).all() for p in model.parameters())


def _replay_state(example_ids: list[str]) -> dict:
    replay = StratifiedReplay(capacity=1000, per_stratum_cap=500, seed=3)
    for eid in example_ids:
        replay.offer({"example_id": eid, "split": "train", "reviewed": True, "states": {}, "year": "2013"})
    return replay.state_dict()


def test_mixer_keeps_the_new_fraction_and_resumes_exactly():
    sampler = DeterministicSampler([f"n{i}" for i in range(20)], 5)
    replay = StratifiedReplay.from_state(_replay_state([f"r{i}" for i in range(6)]))
    mixer = replay_mod.ReplayMixer(sampler, replay, 0.7)
    first = mixer.draw(10)
    assert sum(i.startswith("n") for i in first) == 7 and sum(i.startswith("r") for i in first) == 3
    mixer.consume()
    state = mixer.state_dict()
    nxt = mixer.draw(10)
    fresh = DeterministicSampler([f"n{i}" for i in range(20)], 5)
    clone = replay_mod.ReplayMixer(fresh, StratifiedReplay(1, 1, 0), 0.7)
    clone.load_state(json.loads(json.dumps(state)))
    assert clone.draw(10) == nxt


def _finetune_setup(corpus, tmp_path):
    _small_config(corpus)
    _prepare_cli(corpus)
    _set_training(corpus, max_updates=1)
    assert _train(corpus) == 0
    parent = tmp_path / "parent.pt"
    shutil.copy(corpus["out"] / "checkpoints" / "latest.pt", parent)
    shutil.rmtree(corpus["out"] / "checkpoints")
    plan = json.loads((corpus["out"] / "train-plan.json").read_text(encoding="utf-8"))
    return parent, plan


def test_finetune_feeds_replay_ids_and_rejects_heldout_ids(corpus, tmp_path, capsys, monkeypatch):
    parent, _ = _finetune_setup(corpus, tmp_path)
    from whalemix_training.config import load_config

    prep = model_commands._prepare(load_config(corpus["config"]))
    train_ids = sorted(w["example_id"] for w in prep["windows"] if w["split"] == "train")
    heldout_ids = sorted(w["example_id"] for w in prep["windows"] if w["split"] != "train")
    assert len(train_ids) >= 4 and heldout_ids
    replay_ids = train_ids[:2]
    manifest = tmp_path / "replay.jsonl"
    cfg = json.loads(Path(corpus["config"]).read_text(encoding="utf-8"))
    cfg["paths"]["replay_manifest"] = str(manifest)
    cfg["training"]["max_updates"] = 3
    Path(corpus["config"]).write_text(json.dumps(cfg), encoding="utf-8")

    seen: list[str] = []
    real_loader = model_commands.WindowLoader

    class SpyLoader(real_loader):  # type: ignore[misc, valid-type]
        def __call__(self, ids):
            seen.extend(ids)
            return super().__call__(ids)

    monkeypatch.setattr(model_commands, "WindowLoader", SpyLoader)

    manifest.write_text(json.dumps(_replay_state([*replay_ids, heldout_ids[0]])) + "\n", encoding="utf-8")
    capsys.readouterr()
    assert main(["finetune", "--config", str(corpus["config"]), "--parent", str(parent)]) == 1
    assert "held-out" in capsys.readouterr().out

    manifest.write_text(json.dumps(_replay_state(replay_ids)) + "\n", encoding="utf-8")
    cfg["replay"] = {"per_stratum_cap": 1}  # both members share one stratum
    Path(corpus["config"]).write_text(json.dumps(cfg), encoding="utf-8")
    capsys.readouterr()
    assert main(["finetune", "--config", str(corpus["config"]), "--parent", str(parent)]) == 1
    assert "per_stratum_cap" in capsys.readouterr().out
    del cfg["replay"]
    Path(corpus["config"]).write_text(json.dumps(cfg), encoding="utf-8")

    manifest.write_text(json.dumps(_replay_state(replay_ids)) + "\n", encoding="utf-8")
    seen.clear()
    assert main(["finetune", "--config", str(corpus["config"]), "--parent", str(parent)]) == 0
    plan = json.loads((corpus["out"] / "finetune-plan.json").read_text(encoding="utf-8"))
    used_replay = [i for i in seen if i in replay_ids]
    assert used_replay, "fine-tune never fed a replay example"
    assert not set(seen) & set(heldout_ids)
    mix = plan["session"]["mix"]
    assert mix["replay_examples"] > 0 and mix["new_examples"] > mix["replay_examples"]
    assert plan["replay"]["new_fraction"] == pytest.approx(0.7)
    assert plan["replay"]["members"] == 2
    assert (corpus["out"] / "finetune-replay-next.jsonl").exists()


def test_replay_mixer_rejects_unknown_or_heldout_members():
    replay = StratifiedReplay.from_state(_replay_state(["a:0:0", "b:0:0"]))
    with pytest.raises(ReplayAdmissionError, match="held-out"):
        replay_mod.ReplayMixer.validate_members(replay, {"a:0:0": "train", "b:0:0": "test"}, cap=500)
    with pytest.raises(ReplayAdmissionError, match="unknown"):
        replay_mod.ReplayMixer.validate_members(replay, {"a:0:0": "train"}, cap=500)
