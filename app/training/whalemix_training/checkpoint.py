"""Atomic, resumable checkpoints with lineage metadata and a last-known-good copy.

A checkpoint stores model/optimizer/scheduler state, the sampler cursor (consumed examples only), replay
membership and RNG states, plus metadata describing initialisation, parent, data, split and
preprocessing hashes. Loading checks compatibility instead of silently accepting a mismatched file.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import os
import pickle
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

REQUIRED_METADATA = (
    "initialization",
    "parent_checkpoint",
    "architecture",
    "label_map",
    "label_map_hash",
    "config_hash",
    "code_revision",
    "dependency_versions",
    "dataset_hash",
    "split_hash",
    "preprocessing_hash",
    "seed",
    "epoch",
    "global_update",
    "stage",
    "artifact_kind",
)
ARTIFACT_KINDS = ("fixture-untrained", "training-state", "selected")


class CheckpointError(RuntimeError):
    pass


def dependency_versions() -> dict[str, str]:
    out = {}
    for name in ("torch", "torchvision", "numpy", "onnx", "onnxruntime"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "absent"
    return out


def code_revision(repo_dir: Path) -> dict[str, str]:
    """Git HEAD plus a hash of the uncommitted diff (never the diff text itself)."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "HEAD"], cwd=repo_dir, capture_output=True, check=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"head": "unknown", "diff_sha256": "unknown"}
    return {"head": head, "diff_sha256": hashlib.sha256(diff).hexdigest()}


def rng_states() -> dict[str, Any]:
    """RNG states in plain containers and tensors, so checkpoints load with `weights_only=True`."""
    kind, keys, pos, has_gauss, cached = np.random.get_state()
    version, internal, gauss = random.getstate()
    return {
        "python": [version, list(internal), gauss],
        "numpy": [
            kind,
            torch.from_numpy(np.asarray(keys, dtype=np.int64)),
            int(pos),
            int(has_gauss),
            float(cached),
        ],
        "torch": torch.get_rng_state(),
    }


def restore_rng_states(states: dict[str, Any]) -> None:
    version, internal, gauss = states["python"]
    random.setstate((version, tuple(internal), gauss))
    kind, keys, pos, has_gauss, cached = states["numpy"]
    np.random.set_state((kind, keys.numpy().astype(np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(states["torch"])


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(path: Path, payload: dict[str, Any]) -> str:
    """Write atomically; the previous file becomes `<name>.lkg.pt`. Returns the weights SHA-256."""
    meta = payload.get("metadata", {})
    missing = [k for k in REQUIRED_METADATA if k not in meta]
    if missing:
        raise CheckpointError(f"checkpoint metadata missing: {', '.join(missing)}")
    if meta["artifact_kind"] not in ARTIFACT_KINDS:
        raise CheckpointError("unknown artifact_kind")
    weights_sha = state_dict_sha256(payload["model"])
    payload = {**payload, "metadata": {**meta, "weights_sha256": weights_sha}}
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(buffer.getvalue())
        handle.flush()
        os.fsync(handle.fileno())
    lkg = path.with_name(path.stem + ".lkg" + path.suffix)
    if path.exists():
        os.replace(path, lkg)
    os.replace(tmp, path)
    return weights_sha


def load_checkpoint(path: Path, expect: dict[str, str] | None = None) -> dict[str, Any]:
    """Load (falling back to the last-known-good copy) and verify compatibility keys in `expect`."""
    candidates = [path, path.with_name(path.stem + ".lkg" + path.suffix)]
    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = torch.load(candidate, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError) as exc:
            last_error = exc
            continue
        meta = payload.get("metadata", {})
        if state_dict_sha256(payload["model"]) != meta.get("weights_sha256"):
            last_error = CheckpointError(f"{candidate.name}: weights hash mismatch")
            continue
        for key, value in (expect or {}).items():
            if meta.get(key) != value:
                raise CheckpointError(f"incompatible checkpoint: {key} differs")
        payload["loaded_from"] = str(candidate)
        return payload
    raise CheckpointError(f"no loadable checkpoint at {path}: {last_error}")
