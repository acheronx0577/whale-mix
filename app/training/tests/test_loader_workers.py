"""Faster real-audio input, part 2: bounded loader threads that return windows in the requested order,
catalog audio info instead of re-opening each file for its header, and worker count treated like device
placement (it may change between sessions without breaking an exact resume)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from tests.test_feature_cache import _loaders, prepared  # noqa: F401  (fixture)
from tests.test_resume_replay import _set_training, _train, _weights
from tests.test_session_budget import _sample_log
from whalemix_training.dataset import WindowLoader, read_window
from whalemix_training.labels import default_label_map


def test_workers_return_identical_windows_in_the_requested_order(prepared):  # noqa: F811
    serial, ids, prep = _loaders(prepared, None)
    root = serial.root
    threaded = WindowLoader(
        root, prep["catalog"], prep["windows"], prep["spec"], default_label_map(), workers=3
    )
    try:
        for order in (ids, list(reversed(ids)), ids[::2] + ids[1::2]):
            for a, b in zip(serial(order), threaded(order), strict=True):
                assert np.array_equal(a.numpy(), b.numpy())
    finally:
        threaded.close()


def test_catalog_audio_info_gives_the_same_window_without_reopening_the_header(prepared):  # noqa: F811
    loader, ids, prep = _loaders(prepared, None)
    w = loader.windows[ids[0]]
    entry = loader.entries[w["file_id"]]
    path = loader.paths[w["file_id"]]
    direct = read_window(path, w["channel"], w["start_frame"], prep["spec"])
    hinted = read_window(
        path, w["channel"], w["start_frame"], prep["spec"], audio=(entry["sample_rate"], entry["frames"])
    )
    assert np.array_equal(direct, hinted)


def test_worker_count_can_change_between_sessions_and_resume_stays_exact(prepared, tmp_path):  # noqa: F811
    _set_training(prepared, max_updates=4)
    assert _train(prepared) == 0
    straight, _ = _weights(prepared)
    straight_log = _sample_log(prepared)
    shutil.rmtree(prepared["out"] / "checkpoints")

    cfg = json.loads(Path(prepared["config"]).read_text(encoding="utf-8"))
    cfg["paths"]["feature_cache_dir"] = str(tmp_path / "feature-cache")
    Path(prepared["config"]).write_text(json.dumps(cfg), encoding="utf-8")
    latest = prepared["out"] / "checkpoints" / "latest.pt"
    _set_training(prepared, max_updates=4, session_max_updates=2, loader_workers=0)
    assert _train(prepared) == 0
    _set_training(prepared, max_updates=4, session_max_updates=2, loader_workers=3)
    assert _train(prepared, "--resume", str(latest)) == 0
    plan = json.loads((prepared["out"] / "train-plan.json").read_text(encoding="utf-8"))
    assert plan["exact_resume"] is True and plan["input"]["loader_workers"] == 3
    assert _weights(prepared)[0] == straight
    assert _sample_log(prepared) == straight_log
