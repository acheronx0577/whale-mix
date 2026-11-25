"""Score a trained checkpoint on one locked split and write structured predictions plus the eval set.

Predictions carry the candidate identity (artifact kind, checkpoint file SHA-256, weights SHA-256) and the
hashes of the eval set, preprocessing, label map, split and dataset, so `evaluate` and selection can refuse
anything that does not belong together. Windows are read one by one from the read-only sources (the same
reader training uses), so long recordings are never loaded whole. Labels the effective bandwidth cannot
assess for a recording get no predictions and no truth events there.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from whalemix_training import checkpoint as ckpt
from whalemix_training.config import require_path
from whalemix_training.datalock import SetupRequired, load_locked_data
from whalemix_training.dataset import read_window
from whalemix_training.device import DeviceInfo, move_batch
from whalemix_training.hashing import canonical_hash, sha256_file
from whalemix_training.labels import is_assessable
from whalemix_training.model import build_model
from whalemix_training.preprocess import window_starts
from whalemix_training.windows import frame_count

PREDICTIONS_SCHEMA = "whalemix-predictions/1"
SCORABLE = ("training-state", "selected")


def eval_set_hash(eval_set: dict[str, Any]) -> str:
    return canonical_hash({k: eval_set[k] for k in ("recordings", "truth", "slices") if k in eval_set})


def score_split(
    cfg: dict[str, Any],
    checkpoint_path: Path,
    split_name: str,
    device: DeviceInfo,
    thresholds: dict[str, float],
    batch_size: int = 16,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data = load_locked_data(cfg)
    spec, label_map, split = data["spec"], data["label_map"], data["split"]
    try:
        payload = ckpt.load_checkpoint(
            checkpoint_path,
            {
                "label_map_hash": label_map.hash,
                "preprocessing_hash": spec.hash,
                "split_hash": split["split_hash"],
                "dataset_hash": data["dataset_hash"],
            },
        )
    except ckpt.CheckpointError as exc:
        raise SetupRequired(f"cannot score {checkpoint_path.name}: {exc}") from exc
    if Path(payload["loaded_from"]) != checkpoint_path:
        raise SetupRequired(f"{checkpoint_path.name} did not load; refusing a last-known-good fallback")
    meta = payload["metadata"]
    if meta["artifact_kind"] not in SCORABLE:
        raise SetupRequired(f"only trained checkpoints can be scored, not {meta['artifact_kind']}")
    labels = list(cfg["evaluation"].get("labels") or label_map.whale_targets)
    unknown = [lbl for lbl in labels if lbl not in label_map.outputs]
    if unknown:
        raise SetupRequired(f"evaluation.labels has unknown labels: {', '.join(unknown)}")
    files = [
        e
        for e in data["catalog"]
        if e.get("status") == "ok" and split["assignments"].get(e["file_id"], {}).get("split") == split_name
    ]
    if not files:
        raise SetupRequired(f"no readable recordings in the {split_name} split")

    model = build_model(label_map, int(cfg["model"]["in_channels"]))
    model.load_state_dict(payload["model"])
    model.to(device.device).eval()
    root = require_path(cfg, "recordings_root")
    events: list[dict[str, Any]] = []
    recordings = []
    windows = 0
    t0 = time.perf_counter()
    span = ((spec.window_frames - 1) * spec.hop_length + spec.n_fft) / spec.sample_rate
    for entry in files:
        recordings.append({"recording_id": entry["file_id"], "duration_s": entry["duration_s"]})
        assessable = [lbl for lbl in labels if is_assessable(lbl, entry["sample_rate"], spec)]
        starts = window_starts(frame_count(entry["frames"], entry["sample_rate"], spec), spec)
        for channel in range(int(entry["channels"])):
            for i in range(0, len(starts), batch_size):
                chunk = starts[i : i + batch_size]
                x = np.stack([read_window(root / entry["relative_path"], channel, s, spec) for s in chunk])
                (xt,) = move_batch((torch.from_numpy(x),), device.device)
                with torch.no_grad():
                    probs = torch.sigmoid(model(xt)).float().cpu().numpy()
                windows += len(chunk)
                for row, start in zip(probs, chunk, strict=True):
                    start_s = start * spec.hop_length / spec.sample_rate
                    for lbl in assessable:
                        events.append(
                            {
                                "recording_id": entry["file_id"],
                                "channel": channel,
                                "label": lbl,
                                "start_s": start_s,
                                "end_s": start_s + span,
                                "score": float(row[label_map.index(lbl)]),
                            }
                        )
    seconds = time.perf_counter() - t0
    by_id = {e["file_id"]: e for e in files}
    truth = [
        {
            "recording_id": r["file_id"],
            "channel": int(r["channel"]),
            "label": r["label"],
            "start_s": float(r["start_s"]),
            "end_s": float(r["end_s"]),
        }
        for r in data["rows"]
        if r["file_id"] in by_id
        and r["review_status"] == "reviewed"
        and r["state"] == "present"
        and r["label"] in labels
        and is_assessable(r["label"], by_id[r["file_id"]]["sample_rate"], spec)
    ]
    eval_set: dict[str, Any] = {
        "stage": split_name,
        "split": split_name,
        "fixture": False,
        "labels": labels,
        "recordings": recordings,
        "truth": truth,
        "slices": {},
        "split_hash": split["split_hash"],
        "dataset_hash": data["dataset_hash"],
    }
    predictions = {
        "schema": PREDICTIONS_SCHEMA,
        "candidate": {
            "artifact_kind": meta["artifact_kind"],
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "weights_sha256": meta["weights_sha256"],
            "stage": meta["stage"],
            "global_update": meta["global_update"],
        },
        "weights_sha256": meta["weights_sha256"],
        "split": split_name,
        "eval_set_hash": eval_set_hash(eval_set),
        "preprocessing_hash": spec.hash,
        "label_map_hash": label_map.hash,
        "split_hash": split["split_hash"],
        "dataset_hash": data["dataset_hash"],
        "thresholds": {lbl: float(thresholds.get(lbl, 0.5)) for lbl in labels},
        "resources": {
            "within_budget": True,
            "windows": windows,
            "seconds": seconds,
            "device": device.to_json(),
        },
        "events": events,
    }
    return predictions, eval_set
