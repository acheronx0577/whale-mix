"""Faster real-audio input: a verified, deterministic feature cache. A cached window must be bit-identical
to live preprocessing (declared tolerance: exact), every entry is keyed by the source SHA-256 and the
preprocessing hash, damaged or mismatched entries are rejected and recomputed, a source that changed since
inventory is refused, and training with the cache consumes the same samples and reaches the same weights."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from tests.test_model_and_trainer import _prepare_cli, _small_config
from tests.test_resume_replay import _set_training, _train, _weights
from tests.test_session_budget import _sample_log
from whalemix_training.config import load_config
from whalemix_training.dataset import WindowLoader
from whalemix_training.feature_cache import CacheError, FeatureCache, SourceChanged
from whalemix_training.labels import default_label_map
from whalemix_training.model_commands import _prepare


def _loaders(corpus, cache_dir: Path | None):
    cfg = load_config(Path(corpus["config"]))
    prep = _prepare(cfg)
    root = Path(cfg["paths"]["recordings_root"])
    cache = None if cache_dir is None else FeatureCache(cache_dir, prep["spec"])
    loader = WindowLoader(
        root, prep["catalog"], prep["windows"], prep["spec"], default_label_map(), cache=cache
    )
    ids = sorted(w["example_id"] for w in prep["windows"])[:6]
    return loader, ids, prep


@pytest.fixture()
def prepared(corpus):
    _small_config(corpus)
    _prepare_cli(corpus)
    return corpus


def test_cached_windows_are_bit_identical_to_live_preprocessing(prepared, tmp_path):
    live, ids, _ = _loaders(prepared, None)
    cached, _, _ = _loaders(prepared, tmp_path / "cache")
    reference = live(ids)
    first = cached(ids)
    assert cached.cache.stats == {"hits": 0, "misses": 6, "rejected": 0, "written": 6}
    second = cached(ids)
    assert cached.cache.stats["hits"] == 6
    for a, b, c in zip(reference, first, second, strict=True):
        assert a.dtype == b.dtype == c.dtype and np.array_equal(a.numpy(), b.numpy())
        assert np.array_equal(a.numpy(), c.numpy())
    info = json.loads((cached.cache.dir / "cache-info.json").read_text("utf-8"))
    assert info["preprocessing_hash"] == cached.spec.hash and info["tolerance"] == "exact"


def test_damaged_or_mismatched_entries_are_rejected_and_recomputed(prepared, tmp_path):
    loader, ids, _ = _loaders(prepared, tmp_path / "cache")
    reference = loader(ids)[0].numpy()
    entries = sorted(loader.cache.dir.rglob("*.npz"))
    entries[0].write_bytes(b"not a cache entry")
    # a well-formed entry stored under the wrong key (another window's features)
    entries[2].write_bytes(entries[1].read_bytes())
    again, _, _ = _loaders(prepared, tmp_path / "cache")
    assert np.array_equal(again(ids)[0].numpy(), reference)
    assert again.cache.stats["rejected"] == 2 and again.cache.stats["hits"] == 4
    assert again.cache.stats["written"] == 2


def test_cache_directory_is_bound_to_the_preprocessing_hash(prepared, tmp_path):
    loader, ids, prep = _loaders(prepared, tmp_path / "cache")
    loader(ids)
    info_path = loader.cache.dir / "cache-info.json"
    info = json.loads(info_path.read_text("utf-8"))
    info_path.write_text(json.dumps({**info, "preprocessing_hash": "other"}), encoding="utf-8")
    with pytest.raises(CacheError, match="preprocessing"):
        FeatureCache(tmp_path / "cache", prep["spec"])


def test_a_source_changed_since_inventory_is_refused(prepared, tmp_path):
    loader, ids, prep = _loaders(prepared, tmp_path / "cache")
    loader(ids)
    first = next(w for w in prep["windows"] if w["example_id"] == ids[0])
    entry = next(e for e in prep["catalog"] if e["file_id"] == first["file_id"])
    source = Path(load_config(Path(prepared["config"]))["paths"]["recordings_root"]) / entry["relative_path"]
    os.utime(source, ns=(entry["mtime_ns"] + 10**9, entry["mtime_ns"] + 10**9))
    fresh, _, _ = _loaders(prepared, tmp_path / "cache")
    with pytest.raises(SourceChanged, match="inventory"):
        fresh(ids)


def test_training_with_the_cache_matches_a_plain_run_and_resumes_exactly(prepared, tmp_path):
    _set_training(prepared, max_updates=4)
    assert _train(prepared) == 0
    straight, _ = _weights(prepared)
    straight_log = _sample_log(prepared)
    shutil.rmtree(prepared["out"] / "checkpoints")

    cfg = json.loads(Path(prepared["config"]).read_text(encoding="utf-8"))
    cfg["paths"]["feature_cache_dir"] = str(tmp_path / "feature-cache")
    Path(prepared["config"]).write_text(json.dumps(cfg), encoding="utf-8")
    _set_training(prepared, max_updates=4, session_max_updates=2)
    latest = prepared["out"] / "checkpoints" / "latest.pt"
    assert _train(prepared) == 0
    assert _train(prepared, "--resume", str(latest)) == 0
    plan = json.loads((prepared["out"] / "train-plan.json").read_text(encoding="utf-8"))
    assert plan["exact_resume"] is True
    assert plan["input"]["feature_cache"]["stats"]["hits"] > 0
    assert _weights(prepared)[0] == straight
    assert _sample_log(prepared) == straight_log
