"""Read one training window from its read-only source file, exactly matching the full-file reference.

Only the source range that feeds the window (plus the resampling filter support) is read, so long
recordings are never loaded whole. Results equal `preprocess.analyze_channel` for the same window.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from whalemix_training.feature_cache import FeatureCache, SourceChanged
from whalemix_training.labels import LabelMap, targets_and_mask
from whalemix_training.preprocess import (
    PreprocessingSpec,
    hann_periodic,
    resample_filter,
    resample_ratio,
    resample_segment,
    source_range_for,
    window_tensor,
)

MAX_LOADER_WORKERS = 32


@lru_cache(maxsize=16)
def _taps(up: int, down: int, half_taps_per_phase: int, beta: float) -> np.ndarray:
    taps = resample_filter(up, down, half_taps_per_phase, beta)
    taps.setflags(write=False)
    return taps


def read_window(
    path: Path,
    channel: int,
    start_frame: int,
    spec: PreprocessingSpec,
    audio: tuple[int, int] | None = None,
) -> np.ndarray:
    """`audio` = (sample_rate, frames) from the inventory catalog skips re-reading the file header; the
    arithmetic is unchanged, so the window is identical."""
    samplerate, frames = audio if audio is not None else _header(path)
    up, down = resample_ratio(samplerate, spec.sample_rate)
    out_start = start_frame * spec.hop_length
    out_len = spec.n_fft + (spec.window_frames - 1) * spec.hop_length
    if up == down:
        lo, hi = out_start, out_start + out_len
        data = sf.read(str(path), start=lo, stop=min(hi, frames), dtype="float64", always_2d=True)[0]
        signal = np.zeros(out_len)
        signal[: data.shape[0]] = data[:, channel]
    else:
        taps = _taps(up, down, spec.resample_half_taps_per_phase, spec.resample_kaiser_beta)
        lo, hi = source_range_for(out_start, out_start + out_len, up, down, taps.size)
        hi = min(hi, frames)
        data = sf.read(str(path), start=lo, stop=hi, dtype="float64", always_2d=True)[0]
        signal = resample_segment(data[:, channel], lo, frames, out_start, out_len, up, down, taps)
    idx = np.arange(spec.n_fft)[None, :] + spec.hop_length * np.arange(spec.window_frames)[:, None]
    spectrum = np.fft.rfft(signal[idx] * hann_periodic(spec.n_fft)[None, :], axis=1)
    b0, b1 = spec.bin_range
    power = (spectrum.real**2 + spectrum.imag**2)[:, b0:b1]
    return window_tensor(power, 0, spec)


def _header(path: Path) -> tuple[int, int]:
    info = sf.info(str(path))
    return int(info.samplerate), int(info.frames)


class WindowLoader:
    """Callable used by the trainer: example IDs -> (x, y, mask) tensors, in the order given.

    With a `FeatureCache`, each window is served from the cache when a verified entry exists and is
    otherwise computed live and stored; the tensors are identical either way. `workers` > 0 reads the
    windows of one micro-batch on that many threads (bounded; results keep the requested order, so the
    sampler sequence is unchanged). 0, the default, reads serially."""

    def __init__(
        self,
        root: Path,
        catalog: list[dict[str, Any]],
        windows: list[dict[str, Any]],
        spec: PreprocessingSpec,
        label_map: LabelMap,
        cache: FeatureCache | None = None,
        workers: int = 0,
    ) -> None:
        if not 0 <= int(workers) <= MAX_LOADER_WORKERS:
            raise ValueError(f"loader_workers must be between 0 and {MAX_LOADER_WORKERS}")
        self.root = root
        self.paths = {e["file_id"]: root / e["relative_path"] for e in catalog}
        self.entries = {e["file_id"]: e for e in catalog}
        self.windows = {w["example_id"]: w for w in windows}
        self.spec = spec
        self.label_map = label_map
        self.cache = cache
        self._checked: set[str] = set()
        self.workers = int(workers)
        self._pool: ThreadPoolExecutor | None = None

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def _check_source(self, file_id: str) -> None:
        """The cache key is the catalog SHA-256, so the file must still be the inventoried one."""
        if file_id in self._checked:
            return
        entry, stat = self.entries[file_id], self.paths[file_id].stat()
        if stat.st_size != entry.get("size_bytes") or stat.st_mtime_ns != entry.get("mtime_ns"):
            raise SourceChanged(f"{entry['relative_path']} changed since inventory; re-run inventory")
        self._checked.add(file_id)

    def window(self, w: dict[str, Any]) -> np.ndarray:
        path, entry = self.paths[w["file_id"]], self.entries[w["file_id"]]
        self._check_source(w["file_id"])  # the catalog audio info below must describe this file
        audio = (int(entry["sample_rate"]), int(entry["frames"]))
        if self.cache is None:
            return read_window(path, w["channel"], w["start_frame"], self.spec, audio)
        sha = entry["sha256"]
        x = self.cache.get(sha, w["channel"], w["start_frame"])
        if x is None:
            x = read_window(path, w["channel"], w["start_frame"], self.spec, audio)
            self.cache.put(sha, w["channel"], w["start_frame"], x)
        return x

    def __call__(self, ids: list[str]) -> tuple[Any, Any, Any]:
        import torch

        windows = [self.windows[eid] for eid in ids]
        if self.workers > 0 and len(windows) > 1:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(
                    max_workers=self.workers, thread_name_prefix="whalemix-loader"
                )
            xs = list(self._pool.map(self.window, windows))  # map keeps the input order
        else:
            xs = [self.window(w) for w in windows]
        ys, masks = [], []
        for w in windows:
            y, m = targets_and_mask(w["states"], self.label_map)
            ys.append(y)
            masks.append(m)
        return (
            torch.from_numpy(np.stack(xs)),
            torch.from_numpy(np.stack(ys)),
            torch.from_numpy(np.stack(masks)),
        )
