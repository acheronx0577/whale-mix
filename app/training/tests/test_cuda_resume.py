"""Blocker 2: CUDA RNG and the fp16 gradient-scaler state are checkpointed and restored with the model,
optimizer, sampler/replay and global update; a resume on different hardware is refused unless it is
explicitly recorded as a hardware transition (not bit-exact).

The CPU tests run everywhere. `test_cuda_interrupt_and_resume_matches_straight_run` needs a CUDA GPU and
is skipped elsewhere (it is exempt from WHALEMIX_REQUIRE_FIXTURE because CI has no GPU)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
import torch

from tests.test_model_and_trainer import _prepare_cli, _small_config
from tests.test_resume_replay import _set_training, _train, _weights
from whalemix_training import checkpoint as ckpt
from whalemix_training.device import resolve_device
from whalemix_training.replay import DeterministicSampler
from whalemix_training.trainer import SessionBudget, run_session

CUDA_WEIGHT_TOLERANCE = 1e-5  # declared max abs difference, straight vs interrupted+resumed (fp32)


def test_rng_states_include_the_selected_cuda_device(monkeypatch):
    stored: dict[int, torch.Tensor] = {1: torch.arange(4, dtype=torch.uint8)}
    restored: dict[int, torch.Tensor] = {}
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device=0: stored[int(device)].clone())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda s, device=0: restored.__setitem__(int(device), s))
    states = ckpt.rng_states(torch.device("cuda:1"))
    assert states["cuda"]["index"] == 1 and torch.equal(states["cuda"]["state"], stored[1])
    ckpt.restore_rng_states(states, torch.device("cuda:1"))
    assert torch.equal(restored[1], stored[1])
    assert "cuda" not in ckpt.rng_states(torch.device("cpu"))
    restored.clear()
    ckpt.restore_rng_states(states, torch.device("cpu"))  # CPU target: CUDA state not applied
    assert restored == {}


class _FakeScaler:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def scale(self, loss):
        self.calls.append("scale")
        return loss

    def step(self, optimizer):
        self.calls.append("step")
        optimizer.step()

    def update(self):
        self.calls.append("update")


def test_run_session_uses_the_scaler_it_is_given():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 11))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    ids = [f"w{i}" for i in range(4)]

    def load(batch):
        n = len(batch)
        return torch.ones(n, 1, 2, 2), torch.zeros(n, 11), torch.ones(n, 11)

    scaler = _FakeScaler()
    out = run_session(
        model,
        opt,
        DeterministicSampler(ids, 1),
        load,
        SessionBudget(2, 10, 1e9, 10**9, 1e9),
        2,
        lambda *a: None,
        device=resolve_device({}),
        scaler=scaler,
    )
    assert out["optimizer_updates"] == 2 and scaler.calls == ["scale", "step", "update"] * 2


def _rewrite_device(path: Path, device: dict) -> None:
    payload = torch.load(path, weights_only=True)
    payload["metadata"] = {k: v for k, v in payload["metadata"].items() if k != "weights_sha256"}
    payload["metadata"]["device"] = device
    ckpt.save_checkpoint(path, payload)


def test_resume_on_different_hardware_is_refused_unless_recorded(corpus, capsys):
    _small_config(corpus)
    _prepare_cli(corpus)
    _set_training(corpus, max_updates=1)
    assert _train(corpus) == 0
    latest = corpus["out"] / "checkpoints" / "latest.pt"
    gpu = {
        "requested": "cuda",
        "actual": "cuda",
        "precision": "fp32",
        "name": "NVIDIA GeForce RTX 3070 Laptop GPU",
        "capability": "sm_86",
        "torch": "2.15.0+cu132",
        "torch_cuda_build": "13.2",
    }
    _rewrite_device(latest, gpu)
    _set_training(corpus, max_updates=2)
    capsys.readouterr()
    assert _train(corpus, "--resume", str(latest)) == 1
    out = capsys.readouterr().out
    assert "hardware transition" in out and "--allow-hardware-transition" in out
    assert _train(corpus, "--resume", str(latest), "--allow-hardware-transition") == 0
    plan = json.loads((corpus["out"] / "train-plan.json").read_text(encoding="utf-8"))
    assert plan["exact_resume"] is False
    assert plan["hardware_transition"]["from"]["name"] == gpu["name"]
    assert plan["hardware_transition"]["to"]["actual"] == "cpu"
    meta = ckpt.load_checkpoint(latest)["metadata"]
    assert meta["global_update"] == 2 and meta["hardware_transitions"][0]["at_update"] == 1


def test_same_device_resume_is_marked_exact(corpus):
    _small_config(corpus)
    _prepare_cli(corpus)
    _set_training(corpus, max_updates=1)
    assert _train(corpus) == 0
    _set_training(corpus, max_updates=2)
    assert _train(corpus, "--resume", str(corpus["out"] / "checkpoints" / "latest.pt")) == 0
    plan = json.loads((corpus["out"] / "train-plan.json").read_text(encoding="utf-8"))
    assert plan["exact_resume"] is True and plan["hardware_transition"] is None


def test_fp16_checkpoint_without_scaler_state_cannot_resume():
    from whalemix_training.model_commands import SetupRequired, resume_scaler_state

    with pytest.raises(SetupRequired, match="scaler"):
        resume_scaler_state({"metadata": {}}, "fp16", transition=False)
    assert resume_scaler_state({"metadata": {}}, "fp32", transition=False) is None
    assert resume_scaler_state({"scaler": {"scale": 1.0}}, "fp16", transition=False) == {"scale": 1.0}
    assert resume_scaler_state({"metadata": {}}, "fp16", transition=True) is None  # fresh scaler, recorded


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU (run on the training host)")
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_cuda_interrupt_and_resume_matches_straight_run(corpus, precision):
    _small_config(corpus)
    _prepare_cli(corpus)
    cuda = {"device": "cuda", "precision": precision}
    _set_training(corpus, max_updates=4, session_max_updates=4, **cuda)
    assert _train(corpus) == 0
    _, straight = _weights(corpus)
    log = (corpus["out"] / "checkpoints" / "train-sample-log.jsonl").read_text(encoding="utf-8")
    assert straight["metadata"]["device"]["actual"].startswith("cuda")
    shutil.rmtree(corpus["out"] / "checkpoints")

    _set_training(corpus, max_updates=4, session_max_updates=2, **cuda)
    latest = corpus["out"] / "checkpoints" / "latest.pt"
    assert _train(corpus) == 0  # "interrupted" after 2 updates
    first = torch.load(latest, weights_only=True)
    assert "cuda" in first["rng"] and (precision != "fp16" or "scaler" in first)
    assert _train(corpus, "--resume", str(latest)) == 0
    _, resumed = _weights(corpus)
    plan = json.loads((corpus["out"] / "train-plan.json").read_text(encoding="utf-8"))
    assert plan["exact_resume"] is True and plan["device"]["actual"].startswith("cuda")
    assert (corpus["out"] / "checkpoints" / "train-sample-log.jsonl").read_text(encoding="utf-8") == log
    diff = max(
        float((straight["model"][k].float() - resumed["model"][k].float()).abs().max())
        for k in straight["model"]
        if straight["model"][k].is_floating_point()
    )
    report = os.environ.get("WHALEMIX_CUDA_REPORT")
    if report:
        with open(report, "a", encoding="utf-8") as fh:
            device = {k: plan["device"].get(k) for k in ("actual", "name", "capability", "precision")}
            fh.write(json.dumps({"device": device, "max_abs_weight_diff": diff}) + "\n")
    tolerance = CUDA_WEIGHT_TOLERANCE if precision == "fp32" else 1e-3
    assert diff <= tolerance
