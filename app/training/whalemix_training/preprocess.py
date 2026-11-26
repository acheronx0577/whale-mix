"""Deterministic acoustic preprocessing contract (Python reference; the Rust engine mirrors it).

Pipeline per source channel (channels are never averaged):
1. Rational polyphase resampling to `sample_rate` with a Kaiser-windowed sinc anti-alias filter whose taps
   are computed by formula, so the Rust engine reproduces them exactly.
2. Short-time power spectra on a global frame grid (no centering): frame f covers resampled samples
   [f * hop, f * hop + n_fft) with a periodic Hann window. A signal shorter than one model window is
   zero-padded to exactly one window.
3. Bins between fmin_hz and fmax_hz are kept.
4. Model windows are `window_frames` frames long, start every `window_hop_frames` frames, and a final
   end-aligned window covers any remaining tail. Compression (`log` or per-window `pcen`) and optional
   global standardisation are applied per window, so overlapping windows reuse the same power frames.
Output tensor layout: float32 [channels=1, bins, frames] per window.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from whalemix_training.hashing import canonical_hash

CONTRACT_VERSION = 1


@dataclass(frozen=True)
class PreprocessingSpec:
    sample_rate: int = 4000
    n_fft: int = 1024
    hop_length: int = 400
    window: str = "hann_periodic"
    fmin_hz: float = 0.0
    fmax_hz: float = 1000.0
    compression: str = "log"
    log_floor: float = 1e-6
    pcen: dict[str, float] = field(
        default_factory=lambda: {"time_constant_s": 2.0, "gain": 0.98, "bias": 2.0, "power": 0.5, "eps": 1e-6}
    )
    window_seconds: float = 10.0
    window_hop_seconds: float = 5.0
    resample_half_taps_per_phase: int = 16
    resample_kaiser_beta: float = 8.6
    normalization: dict[str, float | str] = field(default_factory=lambda: {"kind": "none"})
    channel_policy: str = "independent"
    dtype: str = "float32"
    layout: str = "NCFT"

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> PreprocessingSpec:
        keys = {f for f in cls.__dataclass_fields__}
        values = {k: v for k, v in cfg.items() if k in keys}
        if values.get("window") == "hann":
            values["window"] = "hann_periodic"
        spec = cls(**values)
        spec.validate()
        return spec

    def validate(self) -> None:
        if self.compression not in ("log", "pcen"):
            raise ValueError("compression must be 'log' or 'pcen'")
        if self.window != "hann_periodic":
            raise ValueError("only hann_periodic windows are part of contract v1")
        if not 0 <= self.fmin_hz < self.fmax_hz <= self.sample_rate / 2:
            raise ValueError("need 0 <= fmin_hz < fmax_hz <= sample_rate / 2")
        if self.window_frames < 1 or self.window_hop_frames < 1:
            raise ValueError("window_seconds and window_hop_seconds must cover at least one hop")
        if self.normalization.get("kind") not in ("none", "global_standardize"):
            raise ValueError("normalization.kind must be 'none' or 'global_standardize'")

    @property
    def window_frames(self) -> int:
        return round(self.window_seconds * self.sample_rate / self.hop_length)

    @property
    def window_hop_frames(self) -> int:
        return round(self.window_hop_seconds * self.sample_rate / self.hop_length)

    @property
    def bin_range(self) -> tuple[int, int]:
        res = self.sample_rate / self.n_fft
        lo = math.ceil(self.fmin_hz / res - 1e-9)
        hi = math.floor(self.fmax_hz / res + 1e-9)
        return lo, hi + 1

    @property
    def n_bins(self) -> int:
        lo, hi = self.bin_range
        return hi - lo

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["contract_version"] = CONTRACT_VERSION
        out["window_frames"] = self.window_frames
        out["window_hop_frames"] = self.window_hop_frames
        out["bin_start"], out["bin_end"] = self.bin_range
        return out

    @property
    def hash(self) -> str:
        return canonical_hash(self.to_json())


# --- resampling -------------------------------------------------------------------------------------


def _bessel_i0(x: float) -> float:
    """Power series for I0; identical summation order is used by the Rust engine."""
    total, term, k = 1.0, 1.0, 1
    half = x / 2.0
    while True:
        term *= (half / k) ** 2
        total += term
        if term < 1e-17 * total:
            return total
        k += 1


def resample_ratio(src_sr: int, dst_sr: int) -> tuple[int, int]:
    g = math.gcd(int(src_sr), int(dst_sr))
    return int(dst_sr) // g, int(src_sr) // g


def resample_filter(up: int, down: int, half_taps_per_phase: int, beta: float) -> np.ndarray:
    factor = max(up, down)
    half = half_taps_per_phase * factor
    n = 2 * half + 1
    cutoff = 1.0 / factor
    i0_beta = _bessel_i0(beta)
    taps = np.empty(n, dtype=np.float64)
    for k in range(n):
        t = k - half
        x = cutoff * t
        sinc = 1.0 if t == 0 else math.sin(math.pi * x) / (math.pi * x)
        r = 2.0 * k / (n - 1) - 1.0
        win = _bessel_i0(beta * math.sqrt(max(0.0, 1.0 - r * r))) / i0_beta
        taps[k] = up * cutoff * sinc * win
    return taps


def resampled_length(n_in: int, up: int, down: int) -> int:
    return -(-n_in * up // down)


def resample_segment(
    x_sub: np.ndarray,
    sub_offset: int,
    n_in_total: int,
    out_start: int,
    out_len: int,
    up: int,
    down: int,
    taps: np.ndarray,
) -> np.ndarray:
    """Resampled samples [out_start, out_start + out_len) of a signal of `n_in_total` samples, given only
    the source slice `x_sub` that starts at `sub_offset`. Outputs past the resampled length are zero, which
    matches the zero padding of the full-signal path."""
    half = (taps.size - 1) // 2
    n_out_total = resampled_length(n_in_total, up, down)
    n = np.arange(out_start, out_start + out_len, dtype=np.int64)
    base = n * down + half
    i_max = base // up
    k0 = base - i_max * up
    y = np.zeros(out_len, dtype=np.float64)
    for q in range(-(-taps.size // up)):
        k = k0 + q * up
        local = i_max - q - sub_offset
        valid = (
            (k < taps.size)
            & (i_max - q >= 0)
            & (i_max - q < n_in_total)
            & (local >= 0)
            & (local < x_sub.size)
        )
        y[valid] += taps[k[valid]] * x_sub[local[valid]]
    y[n >= n_out_total] = 0.0
    return y


def resample(x: np.ndarray, src_sr: int, dst_sr: int, spec: PreprocessingSpec) -> np.ndarray:
    """Resample a full 1-D signal; samples outside the signal are zero."""
    x = np.asarray(x, dtype=np.float64)
    up, down = resample_ratio(src_sr, dst_sr)
    if up == down:
        return x.copy()
    taps = resample_filter(up, down, spec.resample_half_taps_per_phase, spec.resample_kaiser_beta)
    return resample_segment(x, 0, x.size, 0, resampled_length(x.size, up, down), up, down, taps)


def source_range_for(out_start: int, out_end: int, up: int, down: int, n_taps: int) -> tuple[int, int]:
    """Source sample range [lo, hi) that feeds resampled outputs [out_start, out_end)."""
    half = (n_taps - 1) // 2
    lo = (out_start * down + half - (n_taps - 1)) // up - 1
    hi = (max(out_end - 1, out_start) * down + half) // up + 2
    return max(lo, 0), hi


# --- spectra and windows ----------------------------------------------------------------------------


def hann_periodic(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def power_frames(signal: np.ndarray, spec: PreprocessingSpec) -> np.ndarray:
    """Return power spectra [frames, bins] (float64) on the global frame grid, padding short signals."""
    need = spec.n_fft + (spec.window_frames - 1) * spec.hop_length
    if signal.size < need:
        signal = np.concatenate([signal, np.zeros(need - signal.size)])
    n_frames = 1 + (signal.size - spec.n_fft) // spec.hop_length
    idx = np.arange(spec.n_fft)[None, :] + spec.hop_length * np.arange(n_frames)[:, None]
    frames = signal[idx] * hann_periodic(spec.n_fft)[None, :]
    spectrum = np.fft.rfft(frames, axis=1)
    lo, hi = spec.bin_range
    return (spectrum.real**2 + spectrum.imag**2)[:, lo:hi]


def window_starts(n_frames: int, spec: PreprocessingSpec) -> list[int]:
    width, step = spec.window_frames, spec.window_hop_frames
    starts = list(range(0, n_frames - width + 1, step))
    if not starts:
        return [0]
    if starts[-1] + width < n_frames:
        starts.append(n_frames - width)
    return starts


def pcen(power: np.ndarray, spec: PreprocessingSpec) -> np.ndarray:
    """Per-window PCEN over [frames, bins]; smoother state starts at the window's first frame."""
    p = spec.pcen
    t_frames = p["time_constant_s"] * spec.sample_rate / spec.hop_length
    s = (math.sqrt(1.0 + 4.0 * t_frames**2) - 1.0) / (2.0 * t_frames**2)
    smooth = np.empty_like(power)
    smooth[0] = power[0]
    for t in range(1, power.shape[0]):
        smooth[t] = (1.0 - s) * smooth[t - 1] + s * power[t]
    gain, bias, r, eps = p["gain"], p["bias"], p["power"], p["eps"]
    return (power / (eps + smooth) ** gain + bias) ** r - bias**r


def compress(power: np.ndarray, spec: PreprocessingSpec) -> np.ndarray:
    out = np.log(power + spec.log_floor) if spec.compression == "log" else pcen(power, spec)
    if spec.normalization.get("kind") == "global_standardize":
        out = (out - float(spec.normalization["mean"])) / float(spec.normalization["std"])
    return out


def window_tensor(power: np.ndarray, start: int, spec: PreprocessingSpec) -> np.ndarray:
    """[1, bins, frames] float32 tensor for one window of the global power-frame grid."""
    block = compress(power[start : start + spec.window_frames], spec)
    return block.T[None, :, :].astype(np.float32)


def analyze_channel(samples: np.ndarray, src_sr: int, spec: PreprocessingSpec) -> list[dict[str, Any]]:
    """Reference path: every model window of one channel with its timing."""
    power = power_frames(resample(samples, src_sr, spec.sample_rate, spec), spec)
    out = []
    for start in window_starts(power.shape[0], spec):
        start_s = start * spec.hop_length / spec.sample_rate
        end_s = start_s + ((spec.window_frames - 1) * spec.hop_length + spec.n_fft) / spec.sample_rate
        out.append(
            {
                "start_frame": start,
                "start_s": start_s,
                "end_s": end_s,
                "tensor": window_tensor(power, start, spec),
            }
        )
    return out


def fit_global_standardize(tensors: list[np.ndarray], split_of_each: list[str]) -> dict[str, float | str]:
    """Fit normalisation on training windows only; any non-train window is a leakage error."""
    if any(s != "train" for s in split_of_each):
        raise ValueError("normalisation must be fitted on training windows only")
    if not tensors:
        raise ValueError("no training windows to fit normalisation")
    stacked = np.concatenate([t.ravel().astype(np.float64) for t in tensors])
    std = float(stacked.std())
    return {"kind": "global_standardize", "mean": float(stacked.mean()), "std": std if std > 0 else 1.0}
