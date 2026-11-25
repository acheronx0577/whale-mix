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
from whalemix_training.cli import _EXTRA_ARGS, _out_dir, _print, command
from whalemix_training.config import optional_path, require_path
from whalemix_training.dataset import WindowLoader
from whalemix_training.hashing import canonical_hash
from whalemix_training.io_utils import atomic_write_json, read_jsonl
from whalemix_training.labels import LabelMap, default_label_map
from whalemix_training.losses import masked_bce
from whalemix_training.model import build_model, describe
from whalemix_training.preprocess import PreprocessingSpec
from whalemix_training.replay import DeterministicSampler, StratifiedReplay
from whalemix_training.trainer import SessionBudget, build_optimizer, run_session
from whalemix_training.windows import build_windows

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
    catalog = inventory.load_catalog(catalog_path)
    rows = ann.read_annotations(require_path(cfg, "annotations_csv"))
    spec = PreprocessingSpec.from_config(cfg["preprocessing"])
    label_map = default_label_map()
    windows = build_windows(catalog, rows, split, spec, label_map)
    return {
        "out": out,
        "catalog": catalog,
        "split": split,
        "spec": spec,
        "label_map": label_map,
        "windows": windows,
        "dataset_hash": canonical_hash([e["sha256"] for e in catalog]),
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


def _session(args: argparse.Namespace, cfg: dict[str, Any], stage: str, parent: dict[str, Any] | None) -> int:
    prep = _prepare(cfg)
    training = cfg["training"]
    torch.manual_seed(int(training["seed"]))
    torch.set_num_threads(int(training["compute_threads"]))
    label_map = prep["label_map"]
    model = build_model(label_map, int(cfg["model"]["in_channels"]), "random")
    optimizer = build_optimizer(model, training)
    if parent is not None:
        model.load_state_dict(parent["model"])
    train_ids = sorted(w["example_id"] for w in prep["windows"] if w["split"] == "train" and w["reviewed"])
    sampler = DeterministicSampler(train_ids, int(training["seed"]))
    budget = SessionBudget.from_config(training)
    loader = WindowLoader(
        require_path(cfg, "recordings_root"), prep["catalog"], prep["windows"], prep["spec"], label_map
    )
    plan: dict[str, Any] = {
        "stage": stage,
        "model": describe(model),
        "preprocessing": prep["spec"].to_json(),
        "preprocessing_hash": prep["spec"].hash,
        "split_hash": prep["split"]["split_hash"],
        "windows_by_split": dict(Counter(w["split"] for w in prep["windows"])),
        "reviewed_train_windows": len(train_ids),
        "budget": vars(budget),
        "imbalance_handling": training["imbalance"],
        "parent": None if parent is None else parent["metadata"]["weights_sha256"],
    }
    if train_ids:
        x, y, mask = loader(sampler.peek(min(int(training["micro_batch"]), len(train_ids))))
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

    ckpt_dir = prep["out"] / "checkpoints"

    def on_checkpoint(reason: str, updates: int) -> None:
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "sampler": sampler.state_dict(),
            "rng": ckpt.rng_states(),
            "reason": reason,
            "metadata": _metadata(
                cfg, prep, label_map, stage, plan["parent"], sampler.epoch, updates, "training-state"
            ),
        }
        ckpt.save_checkpoint(ckpt_dir / "latest.pt", payload)

    result = run_session(
        model, optimizer, sampler, loader, budget, int(training["micro_batch"]), on_checkpoint
    )
    plan["session"] = result
    plan["weights_written"] = result["optimizer_updates"] > 0
    atomic_write_json(prep["out"] / f"{stage}-plan.json", plan)
    _print(
        {
            "stage": stage,
            "optimizer_updates": result["optimizer_updates"],
            "stopped": result["stopped"],
            "weights_written": plan["weights_written"],
        }
    )
    return 0


@command("train", "train from random initialisation (setup-only when training.max_updates is 0)")
def cmd_train(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    try:
        return _session(args, cfg, "train", None)
    except SetupRequired as exc:
        _print({"status": "SETUP_REQUIRED", "reason": str(exc)})
        return 1


@command("finetune", "continue a recorded project checkpoint with replay (setup-only when max_updates is 0)")
def cmd_finetune(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    try:
        if args.parent is None or not args.parent.exists():
            raise SetupRequired("finetune needs --parent pointing at a recorded project checkpoint")
        parent = ckpt.load_checkpoint(args.parent, {"label_map_hash": default_label_map().hash})
        if parent["metadata"]["artifact_kind"] == "fixture-untrained":
            raise SetupRequired("a fixture checkpoint cannot parent a fine-tune")
        replay_path = optional_path(cfg, "replay_manifest")
        if replay_path is None or not replay_path.exists():
            raise SetupRequired("finetune needs paths.replay_manifest")
        StratifiedReplay.from_state(next(iter(read_jsonl(replay_path))))
        return _session(args, cfg, "finetune", parent)
    except (SetupRequired, ckpt.CheckpointError) as exc:
        _print({"status": "SETUP_REQUIRED", "reason": str(exc)})
        return 1


def _finetune_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--parent", type=Path, default=None, help="recorded project checkpoint to continue")


_EXTRA_ARGS["finetune"] = _finetune_args
