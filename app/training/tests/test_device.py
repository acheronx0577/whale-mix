"""Fix 2 (training readiness): explicit device selection, one tensor-placement path, clear CUDA errors and
the actual device/precision recorded in the run ledger. CUDA is mocked; no GPU is needed."""

from __future__ import annotations

import json

import pytest
import torch

from tests.test_model_and_trainer import _prepare_cli, _small_config
from whalemix_training import device as dev
from whalemix_training import trainer
from whalemix_training.cli import main
from whalemix_training.replay import DeterministicSampler
from whalemix_training.trainer import SessionBudget, run_session


def test_cpu_is_the_explicit_default():
    info = dev.resolve_device({})
    assert info.device == torch.device("cpu") and info.precision == "fp32"
    assert info.to_json()["actual"] == "cpu" and info.to_json()["requested"] == "cpu"


@pytest.mark.parametrize(
    ("cfg", "needle"),
    [
        ({"device": "tpu"}, "training.device"),
        ({"device": "cpu", "precision": "fp16"}, "fp32"),
        ({"device": "cpu", "precision": "int4"}, "training.precision"),
    ],
)
def test_invalid_device_settings_are_rejected(cfg, needle):
    with pytest.raises(dev.DeviceError, match=needle):
        dev.resolve_device(cfg)


def test_cuda_requested_but_unavailable_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(dev.DeviceError, match="CUDA is unavailable"):
        dev.resolve_device({"device": "cuda"})


def test_cuda_build_without_kernels_for_the_card_is_rejected(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "NVIDIA GeForce RTX 5090")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (12, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_80", "sm_86", "sm_90"])
    with pytest.raises(dev.DeviceError, match="sm_120"):
        dev.resolve_device({"device": "cuda"})
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_86", "sm_90", "sm_120"])
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    info = dev.resolve_device({"device": "cuda", "precision": "bf16"})
    ledger = info.to_json()
    assert info.device == torch.device("cuda:0")
    assert ledger["actual"] == "cuda:0" and ledger["name"] == "NVIDIA GeForce RTX 5090"
    assert ledger["capability"] == "sm_120" and ledger["precision"] == "bf16"


def test_session_moves_every_batch_through_one_path(monkeypatch):
    seen = []
    real = dev.move_batch

    def spy(batch, device):
        seen.append(device)
        return real(batch, device)

    monkeypatch.setattr(trainer, "move_batch", spy)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 11))
    info = dev.resolve_device({"device": "cpu"})
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    batch = (torch.ones(2, 1, 2, 2), torch.ones(2, 11), torch.ones(2, 11))
    result = run_session(
        model,
        opt,
        DeterministicSampler(["a", "b", "c", "d"], 1),
        lambda ids: batch,
        SessionBudget(2, 10, 60, 100, 600),
        2,
        lambda *a: None,
        device=info,
    )
    assert result["optimizer_updates"] == 2 and seen == [info.device, info.device]
    assert all(p.device == info.device for p in model.parameters())


def test_cli_records_device_and_refuses_missing_cuda(corpus, capsys, monkeypatch):
    _small_config(corpus)
    _prepare_cli(corpus)
    assert main(["train", "--config", str(corpus["config"])]) == 0
    plan = json.loads((corpus["out"] / "train-plan.json").read_text(encoding="utf-8"))
    assert plan["device"]["actual"] == "cpu" and plan["device"]["precision"] == "fp32"
    cfg = json.loads(corpus["config"].read_text(encoding="utf-8"))
    cfg["training"]["device"] = "cuda"
    corpus["config"].write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    capsys.readouterr()
    assert main(["train", "--config", str(corpus["config"])]) == 1
    out = capsys.readouterr().out
    assert "DEVICE_UNAVAILABLE" in out and "CUDA is unavailable" in out
