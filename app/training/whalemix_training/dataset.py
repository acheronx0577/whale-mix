"""Read one training window from its read-only source file, exactly matching the full-file reference.

Only the source range that feeds the window (plus the resampling filter support) is read, so long
recordings are never loaded whole. Results equal `preprocess.analyze_channel` for the same window.
"""

from __future__ import annotations

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


def read_window(path: Path, channel: int, start_frame: int, spec: PreprocessingSpec) -> np.ndarray:
    info = sf.info(str(path))
    up, down = resample_ratio(info.samplerate, spec.sample_rate)
    out_start = start_frame * spec.hop_length
    out_len = spec.n_fft + (spec.window_frames - 1) * spec.hop_length
    if up == down:
        lo, hi = out_start, out_start + out_len
        data = sf.read(str(path), start=lo, stop=min(hi, info.frames), dtype="float64", always_2d=True)[0]
        signal = np.zeros(out_len)
        signal[: data.shape[0]] = data[:, channel]
    else:
        taps = resample_filter(up, down, spec.resample_half_taps_per_phase, spec.resample_kaiser_beta)
        lo, hi = source_range_for(out_start, out_start + out_len, up, down, taps.size)
        hi = min(hi, info.frames)
        data = sf.read(str(path), start=lo, stop=hi, dtype="float64", always_2d=True)[0]
        signal = resample_segment(data[:, channel], lo, info.frames, out_start, out_len, up, down, taps)
    idx = np.arange(spec.n_fft)[None, :] + spec.hop_length * np.arange(spec.window_frames)[:, None]
    spectrum = np.fft.rfft(signal[idx] * hann_periodic(spec.n_fft)[None, :], axis=1)
    b0, b1 = spec.bin_range
    power = (spectrum.real**2 + spectrum.imag**2)[:, b0:b1]
    return window_tensor(power, 0, spec)


class WindowLoader:
    """Callable used by the trainer: example IDs -> (x, y, mask) tensors, in the order given.

    With a `FeatureCache`, each window is served from the cache when a verified entry exists and is
    otherwise computed live and stored; the tensors are identical either way."""

    def __init__(
        self,
        root: Path,
        catalog: list[dict[str, Any]],
        windows: list[dict[str, Any]],
        spec: PreprocessingSpec,
        label_map: LabelMap,
        cache: FeatureCache | None = None,
    ) -> None:
        self.root = root
        self.paths = {e["file_id"]: root / e["relative_path"] for e in catalog}
        self.entries = {e["file_id"]: e for e in catalog}
        self.windows = {w["example_id"]: w for w in windows}
        self.spec = spec
        self.label_map = label_map
        self.cache = cache
        self._checked: set[str] = set()

    def _check_source(self, file_id: str) -> None:
        """The cache key is the catalog SHA-256, so the file must still be the inventoried one."""
        if file_id in self._checked:
            return
        entry, stat = self.entries[file_id], self.paths[file_id].stat()
        if stat.st_size != entry.get("size_bytes") or stat.st_mtime_ns != entry.get("mtime_ns"):
            raise SourceChanged(f"{entry['relative_path']} changed since inventory; re-run inventory")
        self._checked.add(file_id)

    def window(self, w: dict[str, Any]) -> np.ndarray:
        path = self.paths[w["file_id"]]
        if self.cache is None:
            return read_window(path, w["channel"], w["start_frame"], self.spec)
        self._check_source(w["file_id"])
        sha = self.entries[w["file_id"]]["sha256"]
        x = self.cache.get(sha, w["channel"], w["start_frame"])
        if x is None:
            x = read_window(path, w["channel"], w["start_frame"], self.spec)
            self.cache.put(sha, w["channel"], w["start_frame"], x)
        return x

    def __call__(self, ids: list[str]) -> tuple[Any, Any, Any]:
        import torch

        xs, ys, masks = [], [], []
        for eid in ids:
            w = self.windows[eid]
            xs.append(self.window(w))
            y, m = targets_and_mask(w["states"], self.label_map)
            ys.append(y)
            masks.append(m)
        return (
            torch.from_numpy(np.stack(xs)),
            torch.from_numpy(np.stack(ys)),
            torch.from_numpy(np.stack(masks)),
        )
