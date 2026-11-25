"""Single training entrypoint: `python -m whalemix_training <command> --config FILE`.

Exit codes: 0 success, 1 check failed (errors found / gate not passed), 2 invalid configuration or input.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from whalemix_training import annotations as ann
from whalemix_training import inventory, splits
from whalemix_training.config import ConfigError, load_config, optional_path, require_path
from whalemix_training.hashing import sha256_file
from whalemix_training.io_utils import assert_outside_root, atomic_write_json
from whalemix_training.labels import default_label_map
from whalemix_training.preprocess import PreprocessingSpec

Handler = Callable[[argparse.Namespace, dict[str, Any]], int]
COMMANDS: dict[str, tuple[str, Handler]] = {}


def command(name: str, help_text: str) -> Callable[[Handler], Handler]:
    def register(func: Handler) -> Handler:
        COMMANDS[name] = (help_text, func)
        return func

    return register


def _out_dir(cfg: dict[str, Any]) -> Path:
    out = require_path(cfg, "output_dir")
    assert_outside_root(out, optional_path(cfg, "recordings_root"))
    out.mkdir(parents=True, exist_ok=True)
    return out


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


@command("inventory", "hash and probe configured recordings into a read-only source catalog")
def cmd_inventory(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    report = inventory.build_inventory(
        require_path(cfg, "recordings_root"),
        _out_dir(cfg),
        cfg["inventory"]["extensions"],
        optional_path(cfg, "metadata_csv"),
        args.max_files if args.max_files is not None else cfg["inventory"]["max_files"],
    )
    _print(
        {k: report[k] for k in ("files", "readable_files", "total_hours", "catalog_hash", "reused_entries")}
    )
    return 0


@command("validate-labels", "validate annotations against the catalog and write the coverage matrix")
def cmd_validate_labels(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    out = _out_dir(cfg)
    catalog = inventory.load_catalog(out / "source-catalog.jsonl")
    annotations_path = require_path(cfg, "annotations_csv")
    rows = ann.read_annotations(annotations_path)
    spec = PreprocessingSpec.from_config(cfg["preprocessing"])
    report = ann.validate(rows, catalog, default_label_map(), spec)
    report["catalog_hash"] = inventory.catalog_hash(catalog)
    report["annotations_sha256"] = sha256_file(annotations_path)
    atomic_write_json(out / "label-validation.json", report)
    _print({"rows": report["rows"], "errors": report["errors"], "warnings": report["warnings"]})
    return 1 if report["errors"] else 0


@command("split", "assign deterministic grouped splits, check leakage and lock the split hash")
def cmd_split(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    out = _out_dir(cfg)
    catalog = inventory.load_catalog(out / "source-catalog.jsonl")
    current_catalog = inventory.catalog_hash(catalog)
    annotations_path = optional_path(cfg, "annotations_csv")
    binding: dict[str, Any] = {"catalog_hash": current_catalog, "annotations_sha256": None}
    if annotations_path is not None:
        binding = data_binding(out, cfg)  # raises ConfigError when label validation is stale
    manifest_path = out / "split-manifest.json"
    previous = None
    if manifest_path.exists() and cfg["split"].get("freeze_heldout", True):
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    try:
        manifest = splits.assign_splits(
            catalog, int(cfg["split"]["seed"]), cfg["split"]["fractions"], previous=previous
        )
    except splits.SplitLockError as exc:
        _print({"status": "HELDOUT_LOCK_FAILED", "problems": exc.problems})
        return 1
    problems = splits.leakage_check(catalog, manifest)
    gaps = []
    if annotations_path is not None:
        gaps = splits.heldout_gaps(
            ann.read_annotations(annotations_path), manifest, default_label_map().outputs
        )
    manifest.update(binding)
    manifest["leakage_problems"] = problems
    manifest["heldout_gaps"] = gaps
    atomic_write_json(out / "split-manifest.json", manifest)
    _print(
        {
            "split_hash": manifest["split_hash"],
            "split_hours": manifest["split_hours"],
            "leakage_problems": len(problems),
            "heldout_gaps": len(gaps),
        }
    )
    return 1 if problems else 0


def data_binding(out: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Hashes that tie label validation to the exact catalog and annotation file in use.

    Raises ConfigError when label-validation.json is missing or was produced for different inputs."""
    validation_path = out / "label-validation.json"
    if not validation_path.exists():
        raise ConfigError("label-validation.json is missing; run validate-labels first")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    catalog = inventory.load_catalog(out / "source-catalog.jsonl")
    current = {
        "catalog_hash": inventory.catalog_hash(catalog),
        "annotations_sha256": sha256_file(require_path(cfg, "annotations_csv")),
    }
    stale = [k for k, v in current.items() if validation.get(k) != v]
    if stale:
        raise ConfigError(f"label validation is stale ({', '.join(stale)} changed); rerun validate-labels")
    return {**current, "label_validation_sha256": sha256_file(validation_path)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m whalemix_training", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (help_text, _) in COMMANDS.items():
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", type=Path, required=True, help="run configuration JSON")
        p.add_argument("--max-files", type=int, default=None, help="bound the inventory walk (fixtures)")
        _EXTRA_ARGS.get(name, lambda _p: None)(p)
    return parser


_OPTIONAL_MODULES = {"torch", "torchvision", "onnx", "onnxruntime", "onnxscript"}
_EXTRA_ARGS: dict[str, Callable[[argparse.ArgumentParser], None]] = {}


def main(argv: list[str] | None = None) -> int:
    _load_optional_commands()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        return COMMANDS[args.command][1](args, cfg)
    except (ConfigError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _load_optional_commands() -> None:
    """Model commands need torch; register them only when the import works so data commands stay light."""
    try:
        import importlib

        importlib.import_module("whalemix_training.model_commands")
        importlib.import_module("whalemix_training.eval_commands")
    except ImportError as exc:  # pragma: no cover - exercised only without the model extras
        missing = exc.name or ""
        if missing.split(".")[0] not in _OPTIONAL_MODULES:
            raise
