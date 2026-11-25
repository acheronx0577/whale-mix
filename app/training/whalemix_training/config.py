"""Run configuration. Paths and runtime targets come only from the config file or environment."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

# Environment overrides. Values are paths, never credentials; nothing here is printed by default.
ENV_OVERRIDES: dict[str, tuple[str, str]] = {
    "WHALEMIX_RECORDINGS_ROOT": ("paths", "recordings_root"),
    "WHALEMIX_METADATA_CSV": ("paths", "metadata_csv"),
    "WHALEMIX_ANNOTATIONS_CSV": ("paths", "annotations_csv"),
    "WHALEMIX_OUTPUT_DIR": ("paths", "output_dir"),
}

DEFAULTS: dict[str, Any] = {
    "schema_version": 1,
    "paths": {
        "recordings_root": None,
        "metadata_csv": None,
        "annotations_csv": None,
        "output_dir": None,
    },
    "inventory": {"extensions": [".wav", ".flac"], "max_files": None},
    "split": {"seed": 20261125, "fractions": {"train": 0.70, "validation": 0.15, "test": 0.15}},
    "preprocessing": {
        "sample_rate": 4000,
        "n_fft": 1024,
        "hop_length": 400,
        "window": "hann",
        "fmin_hz": 0.0,
        "fmax_hz": 1000.0,
        "compression": "log",
        "log_floor": 1e-6,
        "pcen": {"time_constant_s": 2.0, "gain": 0.98, "bias": 2.0, "power": 0.5, "eps": 1e-6},
        "window_seconds": 10.0,
        "window_hop_seconds": 5.0,
        "resample_half_taps_per_phase": 16,
        "resample_kaiser_beta": 8.6,
    },
    "model": {"architecture": "efficientnet_b0", "in_channels": 1, "initialization": "random"},
    "training": {
        "optimizer": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "micro_batch": 4,
        "accumulation_steps": 1,
        "max_updates": 0,
        "checkpoint_every_updates": 250,
        "checkpoint_every_seconds": 300,
        "session_max_seconds": 2700,
        "session_max_updates": 1000,
        "seed": 1234,
        "compute_threads": 2,
        "loader_workers": 0,
        "imbalance": "none",
    },
    "replay": {"capacity": 10000, "per_stratum_cap": 500, "new_fraction": 0.7, "seed": 77},
    "selection": {
        "false_alarms_per_hour_budget": 2.0,
        "min_primary_gain": 0.01,
        "max_retention_drop": 0.02,
        "min_events_per_slice": 20,
    },
    "laya": {"enabled": False, "executable": "laya", "model": None, "timeout_s": 120, "batch_size": 8},
}


class ConfigError(ValueError):
    """Raised for missing or invalid configuration. Messages name keys, never values of secrets."""


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path | None, env: dict[str, str] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        try:
            user = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigError(f"config file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"config file is not valid JSON: {path}: {exc.msg}") from exc
        if not isinstance(user, dict):
            raise ConfigError("config root must be a JSON object")
        cfg = _merge(cfg, user)
    environ = os.environ if env is None else env
    for var, (section, key) in ENV_OVERRIDES.items():
        if environ.get(var):
            cfg[section][key] = environ[var]
    if cfg.get("schema_version") != 1:
        raise ConfigError("unsupported config schema_version")
    return cfg


def require_path(cfg: dict[str, Any], key: str) -> Path:
    value = cfg["paths"].get(key)
    if not value:
        raise ConfigError(
            f"paths.{key} is not configured (set it in --config or the matching WHALEMIX_* variable)"
        )
    return Path(value)


def optional_path(cfg: dict[str, Any], key: str) -> Path | None:
    value = cfg["paths"].get(key)
    return Path(value) if value else None
