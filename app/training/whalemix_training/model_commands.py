"""Commands that need torch: train and finetune (setup-only when the update budget is zero)."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from whalemix_training import annotations as ann
from whalemix_training import checkpoint as ckpt
from whalemix_training import inventory
from whalemix_training.cli import _EXTRA_ARGS, _out_dir, _print, command, data_binding
from whalemix_training.config import ConfigError, optional_path, require_path
from whalemix_training.dataset import WindowLoader
from whalemix_training.device import DeviceError, DeviceInfo, move_batch, resolve_device
from whalemix_training.hashing import canonical_hash
from whalemix_training.io_utils import atomic_write_json, atomic_write_jsonl, read_jsonl
from whalemix_training.labels import LabelMap, default_label_map
from whalemix_training.losses import masked_bce
from whalemix_training.model import build_model, describe
from whalemix_training.preprocess import PreprocessingSpec
from whalemix_training.replay import (
    DeterministicSampler,
    ReplayAdmissionError,
    ReplayMixer,
    StratifiedReplay,
)
from whalemix_training.splits import SPLITS
from whalemix_training.trainer import SessionBudget, build_optimizer, run_session
from whalemix_training.windows import WindowLimitError, build_windows, count_windows

REPO_DIR = Path(__file__).resolve().parents[3]


class SetupRequired(RuntimeError):
    """A prerequisite (labels, split, parent checkpoint, replay manifest) is missing; nothing was trained."""


def _prepare(cfg: dict[str, Any]) -> dict[str, Any]:
    out = _out_dir(cfg)
    catalog_path, split_path = out / "source-catalog.jsonl", out / "split-manifest.json"
    validation_path = out / "label-validation.json"
    for needed in (catalog_path, split_path, validation_path):
        if not needed.exists():
            raise SetupRequired(f"missing {needed.name}; run inventory, validate-labels and split first")
    import json

    split = json.loads(split_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if split.get("leakage_problems"):
        raise SetupRequired("split manifest has leakage problems")
    if validation.get("errors"):
        raise SetupRequired("label validation has errors")
    try:
        binding = data_binding(out, cfg)
    except ConfigError as exc:
        raise SetupRequired(str(exc)) from exc
    stale = [k for k in ("catalog_hash", "annotations_sha256") if split.get(k) != binding[k]]
    if stale or split.get("label_validation_sha256") != binding["label_validation_sha256"]:
        raise SetupRequired(
            f"split manifest is stale ({', '.join(stale) or 'label validation'} changed); rerun split"
        )
    catalog = inventory.load_catalog(catalog_path)
    rows = ann.read_annotations(require_path(cfg, "annotations_csv"))
    spec = PreprocessingSpec.from_config(cfg["preprocessing"])
    label_map = default_label_map()
    try:
        windows = build_windows(
            catalog,
            rows,
            split,
            spec,
            label_map,
            reviewed_only=True,
            max_windows=int(cfg["training"]["max_windows"]),
        )
    except WindowLimitError as exc:
        raise SetupRequired(str(exc)) from exc
    return {
        "out": out,
        "catalog": catalog,
        "split": split,
        "spec": spec,
        "label_map": label_map,
        "windows": windows,
        "windows_by_split": count_windows(catalog, split, spec),
        "dataset_hash": canonical_hash([e["sha256"] for e in catalog]),
    }


def preflight(prep: dict[str, Any]) -> dict[str, Any]:
    """Per-label supervised counts per split, plus warnings and blocking problems for a real run."""
    label_map: LabelMap = prep["label_map"]
    totals = prep["windows_by_split"]
    per_label = {
        label: {s: {"present": 0, "absent": 0, "unassessed": totals.get(s, 0)} for s in SPLITS}
        for label in label_map.outputs
    }
    for w in prep["windows"]:
        for label, state in w["states"].items():
            if state != "unassessed":
                cell = per_label[label][w["split"]]
                cell[state] += 1
                cell["unassessed"] -= 1
    warnings, blocking = [], []
    for label in label_map.whale_targets:
        train = per_label[label]["train"]
        if train["present"] == 0:
            warnings.append(f"{label}: no reviewed positive windows in train")
        elif train["absent"] == 0:
            warnings.append(f"{label}: positives but no reviewed negatives in train")
        for s in ("validation", "test"):
            if per_label[label][s]["present"] == 0:
                warnings.append(f"{label}: no reviewed positive windows in {s}")
    supervised = sum(c["train"]["present"] + c["train"]["absent"] for c in per_label.values())
    if supervised == 0:
        blocking.append("no supervised targets in train")
    return {
        "windows_by_split": {s: totals.get(s, 0) for s in SPLITS},
        "reviewed_windows_by_split": dict(Counter(w["split"] for w in prep["windows"])),
        "per_label": per_label,
        "warnings": warnings,
        "blocking": blocking,
    }


def _metadata(
    cfg: dict[str, Any],
    prep: dict[str, Any],
    label_map: LabelMap,
    stage: str,
    parent: str | None,
    epoch: int,
    update: int,
    kind: str,
) -> dict[str, Any]:
    return {
        "initialization": "random" if parent is None else "resume",
        "parent_checkpoint": parent,
        "architecture": cfg["model"]["architecture"],
        "label_map": label_map.to_json(),
        "label_map_hash": label_map.hash,
        "config_hash": canonical_hash(cfg),
        "code_revision": ckpt.code_revision(REPO_DIR),
        "dependency_versions": ckpt.dependency_versions(),
        "dataset_hash": prep["dataset_hash"],
        "split_hash": prep["split"]["split_hash"],
        "preprocessing_hash": prep["spec"].hash,
        "seed": int(cfg["training"]["seed"]),
        "epoch": epoch,
        "global_update": update,
        "stage": stage,
        "artifact_kind": kind,
    }


RESUME_EXCLUDED_TRAINING_KEYS = (
    "max_updates",
    "session_max_seconds",
    "session_max_updates",
    "checkpoint_every_updates",
    "checkpoint_every_seconds",
    "device",
    "cuda_index",
    "max_windows",
)


def resume_identity(cfg: dict[str, Any], stage: str) -> str:
    """Everything that must match for an exact resume. Budgets, device placement and paths may change."""
    training = {k: v for k, v in cfg["training"].items() if k not in RESUME_EXCLUDED_TRAINING_KEYS}
    return canonical_hash(
        {
            "stage": stage,
            "model": cfg["model"],
            "preprocessing": cfg["preprocessing"],
            "training": training,
            "replay": cfg["replay"] if stage == "finetune" else None,
        }
    )


def _load_resume(
    path: Path | None, cfg: dict[str, Any], prep: dict[str, Any], stage: str
) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise SetupRequired(f"--resume checkpoint not found: {path}")
    try:
        payload = ckpt.load_checkpoint(
            path,
            {
                "label_map_hash": prep["label_map"].hash,
                "preprocessing_hash": prep["spec"].hash,
                "split_hash": prep["split"]["split_hash"],
                "dataset_hash": prep["dataset_hash"],
                "stage": stage,
                "artifact_kind": "training-state",
                "resume_identity": resume_identity(cfg, stage),
            },
        )
    except ckpt.CheckpointError as exc:
        raise SetupRequired(f"cannot resume from {path.name}: {exc}") from exc
    for key in ("optimizer", "sampler", "rng"):
        if key not in payload:
            raise SetupRequired(f"cannot resume from {path.name}: no {key} state")
    if stage == "finetune" and "mixer" not in payload:
        raise SetupRequired(f"cannot resume from {path.name}: no replay mixer state")
    return payload


def _replay_source(
    cfg: dict[str, Any], prep: dict[str, Any], train_ids: list[str]
) -> tuple[ReplayMixer, list[str]]:
    replay_path = optional_path(cfg, "replay_manifest")
    if replay_path is None or not replay_path.exists():
        raise SetupRequired("finetune needs paths.replay_manifest")
    replay = StratifiedReplay.from_state(next(iter(read_jsonl(replay_path))))
    split_of = {w["example_id"]: w["split"] for w in prep["windows"]}
    held_out_files = {f for f, a in prep["split"]["assignments"].items() if a["split"] != "train"}
    for members in replay.members.values():
        for eid in members:
            if eid not in split_of and eid.split(":")[0] in held_out_files:
                split_of[eid] = prep["split"]["assignments"][eid.split(":")[0]]["split"]
    try:
        ReplayMixer.validate_members(replay, split_of, int(cfg["replay"]["per_stratum_cap"]))
    except ReplayAdmissionError as exc:
        raise SetupRequired(f"replay manifest rejected: {exc}") from exc
    in_replay = {eid for members in replay.members.values() for eid in members}
    new_ids = [eid for eid in train_ids if eid not in in_replay]
    sampler = DeterministicSampler(new_ids, int(cfg["training"]["seed"]))
    return ReplayMixer(sampler, replay, float(cfg["replay"]["new_fraction"])), new_ids


def _session(args: argparse.Namespace, cfg: dict[str, Any], stage: str, parent: dict[str, Any] | None) -> int:
    training = cfg["training"]
    info: DeviceInfo = resolve_device(training)
    prep = _prepare(cfg)
    resume = _load_resume(getattr(args, "resume", None), cfg, prep, stage)
    torch.manual_seed(int(training["seed"]))
    torch.set_num_threads(int(training["compute_threads"]))
    label_map = prep["label_map"]
    model = build_model(label_map, int(cfg["model"]["in_channels"]), "random")
    if parent is not None:
        model.load_state_dict(parent["model"])
    if resume is not None:
        model.load_state_dict(resume["model"])
    model.to(info.device)
    optimizer = build_optimizer(model, training)
    train_ids = sorted(w["example_id"] for w in prep["windows"] if w["split"] == "train" and w["reviewed"])
    source: DeterministicSampler | ReplayMixer
    if stage == "finetune":
        source, new_ids = _replay_source(cfg, prep, train_ids)
    else:
        source, new_ids = DeterministicSampler(train_ids, int(training["seed"])), train_ids
    start_update = 0
    if resume is not None:
        try:
            optimizer.load_state_dict(resume["optimizer"])
            source.load_state(resume["mixer"] if stage == "finetune" else resume["sampler"])
        except (ValueError, KeyError, RuntimeError) as exc:
            raise SetupRequired(f"cannot resume: {exc}") from exc
        start_update = int(resume["metadata"]["global_update"])
    budget = SessionBudget.from_config(training)
    loader = WindowLoader(
        require_path(cfg, "recordings_root"), prep["catalog"], prep["windows"], prep["spec"], label_map
    )
    lineage_parent = (
        resume["metadata"]["parent_checkpoint"]
        if resume is not None
        else None
        if parent is None
        else parent["metadata"]["weights_sha256"]
    )
    plan: dict[str, Any] = {
        "stage": stage,
        "model": describe(model),
        "preprocessing": prep["spec"].to_json(),
        "preprocessing_hash": prep["spec"].hash,
        "split_hash": prep["split"]["split_hash"],
        "windows_by_split": prep["windows_by_split"],
        "reviewed_train_windows": len(train_ids),
        "preflight": preflight(prep),
        "budget": vars(budget),
        "imbalance_handling": training["imbalance"],
        "device": info.to_json(),
        "parent": lineage_parent,
        "resumed_from": None
        if resume is None
        else {
            "checkpoint": str(resume["loaded_from"]),
            "weights_sha256": resume["metadata"]["weights_sha256"],
            "global_update": start_update,
        },
    }
    if isinstance(source, ReplayMixer):
        plan["replay"] = {
            "members": len(source.replay),
            "new_examples_available": len(new_ids),
            "new_fraction": source.new_fraction,
            "per_stratum_cap": int(cfg["replay"]["per_stratum_cap"]),
            "manifest_hash": source.replay.manifest_hash(),
        }
    if train_ids:
        probe_ids = (new_ids or train_ids)[: min(int(training["micro_batch"]), len(new_ids or train_ids))]
        x, y, mask = move_batch(loader(probe_ids), info.device)
        model.eval()
        with torch.no_grad():
            logits = model(x)
        plan["setup_check"] = {
            "input_shape": list(x.shape),
            "output_shape": list(logits.shape),
            "finite": bool(torch.isfinite(logits).all()),
            "masked_loss_untrained": float(masked_bce(logits, y, mask)),
            "supervised_targets": int(mask.sum()),
        }
    elif budget.max_updates > 0:
        raise SetupRequired("no reviewed training windows; refusing to train on unlabeled data")
    if budget.max_updates > 0 and plan["preflight"]["blocking"]:
        raise SetupRequired(f"preflight blocking: {'; '.join(plan['preflight']['blocking'])}")
    if resume is not None:
        ckpt.restore_rng_states(resume["rng"])

    ckpt_dir = prep["out"] / "checkpoints"
    identity = resume_identity(cfg, stage)
    root_init = "random" if parent is None else parent["metadata"].get("initialization", "random")

    def on_checkpoint(reason: str, updates: int) -> None:
        meta = _metadata(cfg, prep, label_map, stage, plan["parent"], source.epoch, updates, "training-state")
        meta.update({"device": plan["device"], "resume_identity": identity, "initialization": root_init})
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "sampler": (source.sampler if isinstance(source, ReplayMixer) else source).state_dict(),
            "rng": ckpt.rng_states(),
            "reason": reason,
            "metadata": meta,
        }
        if isinstance(source, ReplayMixer):
            payload["mixer"] = source.state_dict()
        ckpt.save_checkpoint(ckpt_dir / "latest.pt", payload)

    result = run_session(
        model,
        optimizer,
        source,
        loader,
        budget,
        int(training["micro_batch"]),
        on_checkpoint,
        device=info,
        start_update=start_update,
    )
    plan["session"] = result
    plan["global_update"] = start_update + result["optimizer_updates"]
    plan["weights_written"] = result["optimizer_updates"] > 0
    if isinstance(source, ReplayMixer):
        _write_next_replay(prep, source, new_ids)
    atomic_write_json(prep["out"] / f"{stage}-plan.json", plan)
    _print(
        {
            "stage": stage,
            "optimizer_updates": result["optimizer_updates"],
            "global_update": plan["global_update"],
            "stopped": result["stopped"],
            "weights_written": plan["weights_written"],
        }
    )
    return 0


def _write_next_replay(prep: dict[str, Any], mixer: ReplayMixer, new_ids: list[str]) -> None:
    """Replay memory for the next fine-tune: the current members plus this run's new training windows,
    offered in a fixed order to a copy (the checkpointed mixer state is not touched)."""
    nxt = StratifiedReplay.from_state(mixer.replay.state_dict())
    by_id = {w["example_id"]: w for w in prep["windows"]}
    for eid in new_ids:
        nxt.offer(by_id[eid])
    atomic_write_jsonl(prep["out"] / "finetune-replay-next.jsonl", [nxt.state_dict()])


@command("train", "train from random initialisation (setup-only when training.max_updates is 0)")
def cmd_train(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    try:
        return _session(args, cfg, "train", None)
    except SetupRequired as exc:
        _print({"status": "SETUP_REQUIRED", "reason": str(exc)})
        return 1
    except DeviceError as exc:
        _print({"status": "DEVICE_UNAVAILABLE", "reason": str(exc)})
        return 1


@command("finetune", "continue a recorded project checkpoint with replay (setup-only when max_updates is 0)")
def cmd_finetune(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    try:
        if args.parent is None or not args.parent.exists():
            raise SetupRequired("finetune needs --parent pointing at a recorded project checkpoint")
        parent = ckpt.load_checkpoint(args.parent, {"label_map_hash": default_label_map().hash})
        if parent["metadata"]["artifact_kind"] == "fixture-untrained":
            raise SetupRequired("a fixture checkpoint cannot parent a fine-tune")
        return _session(args, cfg, "finetune", parent)
    except (SetupRequired, ckpt.CheckpointError) as exc:
        _print({"status": "SETUP_REQUIRED", "reason": str(exc)})
        return 1
    except DeviceError as exc:
        _print({"status": "DEVICE_UNAVAILABLE", "reason": str(exc)})
        return 1


def _resume_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--resume", type=Path, default=None, help="training-state checkpoint of this run to continue exactly"
    )


def _finetune_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--parent", type=Path, default=None, help="recorded project checkpoint to continue")
    _resume_arg(parser)


_EXTRA_ARGS["finetune"] = _finetune_args
_EXTRA_ARGS["train"] = _resume_arg
