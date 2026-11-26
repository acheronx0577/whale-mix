"""Verified, deterministic on-disk cache of preprocessed training windows.

A cached window is the exact float32 tensor that `dataset.read_window` returns (declared tolerance: exact;
the bytes are stored unchanged), so live and cached training see identical inputs and Python/Rust parity
is untouched. Entries are keyed by the source file's SHA-256 (from the inventory catalog), channel, window
start and the preprocessing hash. Every read re-checks the stored key and a digest of the tensor bytes;
anything that does not match is rejected, recomputed live and rewritten. The cache directory is private
run storage (never the recording root, never Git) and is bound to one preprocessing hash.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from whalemix_training.datalock import SetupRequired
from whalemix_training.preprocess import PreprocessingSpec

CACHE_FORMAT = 1


class CacheError(SetupRequired):
    """The cache directory belongs to other preprocessing or is not a WhaleMix feature cache."""


class SourceChanged(SetupRequired):
    """A recording no longer matches the inventory catalog (size or modification time)."""


class FeatureCache:
    def __init__(self, root: Path, spec: PreprocessingSpec) -> None:
        self.spec = spec
        self.dir = Path(root) / spec.hash[:16]
        self.dir.mkdir(parents=True, exist_ok=True)
        info_path = self.dir / "cache-info.json"
        expected = {
            "format": CACHE_FORMAT,
            "preprocessing_hash": spec.hash,
            "tolerance": "exact",
            "content": "float32 [1, bins, frames] windows exactly as returned by dataset.read_window",
        }
        if info_path.exists():
            found = json.loads(info_path.read_text(encoding="utf-8"))
            if found.get("format") != CACHE_FORMAT or found.get("preprocessing_hash") != spec.hash:
                raise CacheError(f"{self.dir} holds features for other preprocessing; use a new cache dir")
        else:
            tmp = info_path.with_name(f".cache-info.{os.getpid()}.tmp")
            tmp.write_text(json.dumps(expected, indent=1, sort_keys=True), encoding="utf-8")
            os.replace(tmp, info_path)
        self._lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0, "rejected": 0, "written": 0}

    def _count(self, name: str) -> None:
        with self._lock:
            self.stats[name] += 1

    def _key(self, source_sha256: str, channel: int, start_frame: int) -> dict[str, Any]:
        return {
            "format": CACHE_FORMAT,
            "source_sha256": source_sha256,
            "channel": int(channel),
            "start_frame": int(start_frame),
            "preprocessing_hash": self.spec.hash,
        }

    def path(self, source_sha256: str, channel: int, start_frame: int) -> Path:
        return self.dir / source_sha256[:2] / f"{source_sha256}-c{channel}-f{start_frame}.npz"

    def get(self, source_sha256: str, channel: int, start_frame: int) -> np.ndarray | None:
        path = self.path(source_sha256, channel, start_frame)
        if not path.exists():
            self._count("misses")
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                stored = json.loads(bytes(data["key"]).decode("utf-8"))
                x = np.array(data["x"])
            digest = stored.pop("x_sha256")
            ok = (
                stored == self._key(source_sha256, channel, start_frame)
                and x.dtype == np.float32
                and x.shape == (1, self.spec.n_bins, self.spec.window_frames)
                and hashlib.sha256(x.tobytes()).hexdigest() == digest
            )
        except (OSError, ValueError, KeyError, zipfile.BadZipFile, json.JSONDecodeError, UnicodeDecodeError):
            ok = False
        if not ok:
            self._count("rejected")
            path.unlink(missing_ok=True)
            return None
        self._count("hits")
        return x

    def put(self, source_sha256: str, channel: int, start_frame: int, x: np.ndarray) -> None:
        path = self.path(source_sha256, channel, start_frame)
        path.parent.mkdir(parents=True, exist_ok=True)
        key = {
            **self._key(source_sha256, channel, start_frame),
            "x_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        }
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        with tmp.open("wb") as fh:
            np.savez(
                fh, x=x, key=np.frombuffer(json.dumps(key, sort_keys=True).encode("utf-8"), dtype=np.uint8)
            )
        os.replace(tmp, path)
        self._count("written")

    def to_json(self) -> dict[str, Any]:
        return {"preprocessing_hash": self.spec.hash, "tolerance": "exact", "stats": dict(self.stats)}
