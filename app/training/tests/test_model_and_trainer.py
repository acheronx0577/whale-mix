"""Slice 2 checks: shapes, masking, preprocessing, sampling and state serialisation.

No test runs an optimizer step; every model here is a randomly initialised fixture.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from scipy import signal as sps

from tests.conftest import write_tone
from whalemix_training import checkpoint as ckpt
from whalemix_training import inventory
from whalemix_training.annotations import REQUIRED_COLUMNS
from whalemix_training.cli import main
from whalemix_training.dataset import read_window
from whalemix_training.labels import default_label_map
from whalemix_training.losses import masked_bce, training_pos_weight
from whalemix_training.model import build_model, parameter_count, profile_scores
from whalemix_training.preprocess import (
    PreprocessingSpec,
    analyze_channel,
    fit_global_standardize,
    power_frames,
    resample,
    window_starts,
)
from whalemix_training.replay import DeterministicSampler, ReplayAdmissionError, StratifiedReplay, blend
from whalemix_training.trainer import SessionBudget, run_session
from whalemix_training.windows import window_states

SMALL = PreprocessingSpec(
    sample_rate=2000, n_fft=256, hop_length=100, fmax_hz=500.0, window_seconds=2.0, window_hop_seconds=1.0
)


def test_spec_dimensions_and_hash():
    spec = PreprocessingSpec()
    assert spec.window_frames == 100 and spec.window_hop_frames == 50
    assert spec.n_bins == 257
    assert spec.hash == PreprocessingSpec().hash
    assert spec.hash != PreprocessingSpec(compression="pcen").hash
    with pytest.raises(ValueError):
        PreprocessingSpec.from_config({"compression": "mel"})


def test_resampler_is_antialiased_and_matches_scipy_shape():
    sr = 8000
    t = np.arange(sr * 2) / sr
    low = np.sin(2 * np.pi * 100 * t)
    high = np.sin(2 * np.pi * 3000 * t)  # above the 2 kHz Nyquist of a 4 kHz output
    y_low, y_high = (
        resample(low, sr, 4000, PreprocessingSpec()),
        resample(high, sr, 4000, PreprocessingSpec()),
    )
    ref = sps.resample_poly(low, 1, 2)
    assert y_low.size == ref.size == 8000
    mid = slice(500, -500)
    assert np.max(np.abs(y_low[mid] - np.sin(2 * np.pi * 100 * np.arange(8000) / 4000)[mid])) < 1e-3
    assert np.sqrt(np.mean(y_high[mid] ** 2)) < 1e-3


def test_window_tiling_covers_tail():
    spec = SMALL
    assert window_starts(20, spec) == [0]
    assert window_starts(5, spec) == [0]
    assert window_starts(45, spec) == [0, 5]
    assert window_starts(100, spec) == [0, 20, 40, 60]
    assert window_starts(105, spec) == [0, 20, 40, 60, 65]


def test_power_peak_follows_tone_and_short_signals_pad():
    spec = SMALL
    t = np.arange(spec.sample_rate // 2) / spec.sample_rate
    power = power_frames(np.sin(2 * np.pi * 250 * t), spec)
    assert power.shape == (spec.window_frames, spec.n_bins)
    assert abs(np.argmax(power[0]) * spec.sample_rate / spec.n_fft - 250) < spec.sample_rate / spec.n_fft


def test_windowed_read_equals_full_reference(tmp_path):
    path = tmp_path / "tone.wav"
    write_tone(path, 7.3, 140.0, sr=3000, channels=2, seed=3)
    for spec in (SMALL, PreprocessingSpec(**{**SMALL.__dict__, "compression": "pcen"})):
        data, sr = sf.read(str(path), dtype="float64", always_2d=True)
        ref = analyze_channel(data[:, 1], sr, spec)
        for w in ref:
            got = read_window(path, 1, w["start_frame"], spec)
            assert got.shape == (1, spec.n_bins, spec.window_frames)
            np.testing.assert_allclose(got, w["tensor"], rtol=1e-5, atol=1e-5)


def test_normalisation_fits_on_train_only():
    tensors = [np.ones((1, 4, 4), np.float32), np.zeros((1, 4, 4), np.float32)]
    norm = fit_global_standardize(tensors, ["train", "train"])
    assert norm["mean"] == pytest.approx(0.5) and norm["std"] == pytest.approx(0.5)
    with pytest.raises(ValueError):
        fit_global_standardize(tensors, ["train", "validation"])


def test_model_shapes_random_init_and_profiles():
    torch.manual_seed(0)
    lm = default_label_map()
    model = build_model(lm)
    assert all(p.requires_grad for p in model.parameters())
    assert 3_500_000 < parameter_count(model) < 5_000_000
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 1, SMALL.n_bins, SMALL.window_frames))
    assert out.shape == (2, len(lm.outputs))
    scores = profile_scores(torch.sigmoid(out), lm, "blue_whale")
    assert set(scores) == {"blue_a", "blue_b"}
    with pytest.raises(ValueError):
        profile_scores(torch.sigmoid(out), lm, "sei_whale")
    with pytest.raises(ValueError):
        build_model(lm, initialization="imagenet")


def test_masked_loss_ignores_masked_targets():
    logits = torch.tensor([[2.0, -1.0, 5.0]])
    y = torch.tensor([[1.0, 0.0, 0.0]])
    full = masked_bce(logits, y, torch.ones_like(y))
    masked = masked_bce(logits, y, torch.tensor([[1.0, 1.0, 0.0]]))
    flipped = masked_bce(logits, torch.tensor([[1.0, 0.0, 1.0]]), torch.tensor([[1.0, 1.0, 0.0]]))
    assert masked == pytest.approx(flipped.item())
    assert masked < full
    assert masked_bce(logits, y, torch.zeros_like(y)).item() == 0.0
    w = training_pos_weight(torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]), torch.ones(3, 2))
    assert w.tolist() == [2.0, 2.0]


def _example(i: int, year: str, label: str = "blue_a", split: str = "train", reviewed: bool = True) -> dict:
    return {
        "example_id": f"e{i}",
        "split": split,
        "reviewed": reviewed,
        "year": year,
        "deployment": "d",
        "states": {label: "present"},
    }


def test_replay_is_bounded_stratified_and_serialisable():
    replay = StratifiedReplay(capacity=30, per_stratum_cap=10, seed=5)
    for i in range(500):
        replay.offer(_example(i, "2014", "vessel"))
    for i in range(500, 505):
        replay.offer(_example(i, "2013"))
    assert len(replay) <= 30
    old = [k for k in replay.members if k.startswith("blue_a|2013")]
    assert old and len(replay.members[old[0]]) == 5
    with pytest.raises(ReplayAdmissionError):
        replay.offer(_example(999, "2013", split="validation"))
    with pytest.raises(ReplayAdmissionError):
        replay.offer(_example(998, "2013", reviewed=False))
    restored = StratifiedReplay.from_state(json.loads(json.dumps(replay.state_dict())))
    assert restored.sample(7) == replay.sample(7)
    assert restored.manifest_hash() == replay.manifest_hash()
    mix = blend([f"n{i}" for i in range(10)], replay, 10, 0.7)
    assert len(mix["new"]) == 7 and len(mix["replay"]) == 3
    empty = blend(["n1", "n2"], StratifiedReplay(5, 5, 1), 10, 0.7)
    assert empty["actual_new_fraction"] == pytest.approx(0.2)


def test_sampler_cursor_only_moves_on_consumption():
    s = DeterministicSampler([f"x{i}" for i in range(10)], seed=3)
    first = s.peek(4)
    assert s.peek(4) == first and s.cursor == 0
    s.mark_consumed(4)
    nxt = s.peek(4)
    clone = DeterministicSampler([f"x{i}" for i in range(10)], seed=0)
    clone.load_state(s.state_dict())
    assert clone.peek(4) == nxt
    with pytest.raises(ValueError):
        DeterministicSampler(["other"], 3).load_state(s.state_dict())


def _meta(kind: str = "fixture-untrained") -> dict:
    lm = default_label_map()
    return {
        "initialization": "random",
        "parent_checkpoint": None,
        "architecture": "efficientnet_b0",
        "label_map": lm.to_json(),
        "label_map_hash": lm.hash,
        "config_hash": "c",
        "code_revision": {},
        "dependency_versions": ckpt.dependency_versions(),
        "dataset_hash": "d",
        "split_hash": "s",
        "preprocessing_hash": SMALL.hash,
        "seed": 1,
        "epoch": 0,
        "global_update": 0,
        "stage": "test",
        "artifact_kind": kind,
    }


def test_checkpoint_roundtrip_lkg_and_compat(tmp_path):
    torch.manual_seed(1)
    model = build_model(default_label_map())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sampler = DeterministicSampler(["a", "b", "c"], 9)
    sampler.mark_consumed(2)
    path = tmp_path / "ckpt" / "latest.pt"
    payload = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "sampler": sampler.state_dict(),
        "rng": ckpt.rng_states(),
        "metadata": _meta(),
    }
    sha1 = ckpt.save_checkpoint(path, payload)
    expected_next = torch.rand(3)
    loaded = ckpt.load_checkpoint(path, {"preprocessing_hash": SMALL.hash})
    ckpt.restore_rng_states(loaded["rng"])
    assert torch.equal(torch.rand(3), expected_next)
    clone = build_model(default_label_map())
    clone.load_state_dict(loaded["model"])
    assert ckpt.state_dict_sha256(clone.state_dict()) == sha1
    assert loaded["sampler"]["cursor"] == 2
    ckpt.save_checkpoint(path, payload)
    assert (tmp_path / "ckpt" / "latest.lkg.pt").exists()
    path.write_bytes(b"corrupt")
    assert ckpt.load_checkpoint(path)["loaded_from"].endswith("latest.lkg.pt")
    with pytest.raises(ckpt.CheckpointError):
        ckpt.load_checkpoint(path, {"preprocessing_hash": "other"})
    with pytest.raises(ckpt.CheckpointError):
        ckpt.save_checkpoint(tmp_path / "x.pt", {"model": model.state_dict(), "metadata": {}})


def test_zero_update_budget_runs_no_optimizer_step():
    model = build_model(default_label_map())
    before = ckpt.state_dict_sha256(model.state_dict())
    opt = torch.optim.AdamW(model.parameters())
    calls = []
    budget = SessionBudget(0, 1000, 2700, 250, 300)
    result = run_session(
        model,
        opt,
        DeterministicSampler(["a"], 1),
        lambda ids: calls.append(ids),
        budget,
        4,
        lambda *a: calls.append(a),
    )
    assert result["optimizer_updates"] == 0 and calls == []
    assert ckpt.state_dict_sha256(model.state_dict()) == before
    assert opt.state_dict()["state"] == {}


def test_window_states_and_manifest(corpus):
    lm = default_label_map()
    rows = [
        {"review_status": "reviewed", "start_s": "1", "end_s": "2", "label": "blue_a", "state": "present"},
        {
            "review_status": "reviewed",
            "start_s": "0",
            "end_s": "60",
            "label": "fin_20hz_pulse",
            "state": "absent",
        },
        {
            "review_status": "reviewed",
            "start_s": "0",
            "end_s": "60",
            "label": "invertebrate",
            "state": "absent",
        },
        {"review_status": "ambiguous", "start_s": "0", "end_s": "60", "label": "vessel", "state": "present"},
    ]
    states = window_states(0.0, 10.0, rows, 4000, lm)
    assert states["blue_a"] == "present" and states["fin_20hz_pulse"] == "absent"
    assert states["invertebrate"] == "unassessed" and states["vessel"] == "unassessed"
    assert window_states(5.0, 15.0, rows, 4000, lm)["blue_a"] == "unassessed"


def _prepare_cli(corpus) -> None:
    assert main(["inventory", "--config", str(corpus["config"])]) == 0
    catalog = inventory.load_catalog(corpus["out"] / "source-catalog.jsonl")
    ids = {e["relative_path"]: e["file_id"] for e in catalog}
    with (corpus["tmp"] / "annotations.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(REQUIRED_COLUMNS)
        for n, rel in enumerate(
            ["dep-a/a1.wav", "dep-b/b1.wav", "dep-c/c1.wav", "dep-d/d1.wav", "dep-e/e1.wav"]
        ):
            w.writerow([f"p{n}", ids[rel], "0", "1.0", "2.0", "blue_a", "present", "reviewed", "fixture"])
            w.writerow(
                [f"a{n}", ids[rel], "0", "0.0", "7.0", "fin_20hz_pulse", "absent", "reviewed", "fixture"]
            )
    assert main(["validate-labels", "--config", str(corpus["config"])]) == 0
    assert main(["split", "--config", str(corpus["config"])]) == 0


def _small_config(corpus) -> None:
    cfg = json.loads(Path(corpus["config"]).read_text(encoding="utf-8"))
    cfg["preprocessing"] = {
        "sample_rate": 2000,
        "n_fft": 256,
        "hop_length": 100,
        "fmax_hz": 500.0,
        "window_seconds": 2.0,
        "window_hop_seconds": 1.0,
    }
    cfg["training"] = {"max_updates": 0, "micro_batch": 2}
    Path(corpus["config"]).write_text(json.dumps(cfg), encoding="utf-8")


def test_cli_train_is_setup_only_with_zero_updates(corpus, capsys):
    _small_config(corpus)
    assert main(["train", "--config", str(corpus["config"])]) == 1  # SETUP_REQUIRED before data exists
    _prepare_cli(corpus)
    assert main(["train", "--config", str(corpus["config"])]) == 0
    plan = json.loads((corpus["out"] / "train-plan.json").read_text(encoding="utf-8"))
    assert plan["session"]["optimizer_updates"] == 0 and plan["weights_written"] is False
    assert plan["setup_check"]["output_shape"] == [2, 11] and plan["setup_check"]["finite"]
    assert plan["model"]["initialization"] == "random"
    assert not (corpus["out"] / "checkpoints").exists()
    assert main(["finetune", "--config", str(corpus["config"])]) == 1
    assert "SETUP_REQUIRED" in capsys.readouterr().out


def test_finetune_rejects_fixture_parent(corpus, tmp_path, capsys):
    _small_config(corpus)
    _prepare_cli(corpus)
    parent = tmp_path / "fixture-parent.pt"
    ckpt.save_checkpoint(
        parent, {"model": build_model(default_label_map()).state_dict(), "metadata": _meta()}
    )
    assert main(["finetune", "--config", str(corpus["config"]), "--parent", str(parent)]) == 1
    assert "fixture checkpoint cannot parent" in capsys.readouterr().out
