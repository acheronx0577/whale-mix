"""Load the locked data products (catalog, split, annotations) and refuse stale or leaky ones.

Training, scoring and qualification all start here, so none of them can run on a catalog, label file or
split manifest that no longer matches what was validated.
"""

from __future__ import annotations

import json
from typing import Any

from whalemix_training import annotations as ann
from whalemix_training import inventory
from whalemix_training.cli import _out_dir, data_binding
from whalemix_training.config import ConfigError, require_path
from whalemix_training.hashing import canonical_hash
from whalemix_training.labels import default_label_map
from whalemix_training.preprocess import PreprocessingSpec


class SetupRequired(RuntimeError):
    """A prerequisite (labels, split, parent checkpoint, replay manifest) is missing; nothing was trained."""


def load_locked_data(cfg: dict[str, Any]) -> dict[str, Any]:
    out = _out_dir(cfg)
    catalog_path, split_path = out / "source-catalog.jsonl", out / "split-manifest.json"
    validation_path = out / "label-validation.json"
    for needed in (catalog_path, split_path, validation_path):
        if not needed.exists():
            raise SetupRequired(f"missing {needed.name}; run inventory, validate-labels and split first")
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
    return {
        "out": out,
        "catalog": catalog,
        "split": split,
        "rows": ann.read_annotations(require_path(cfg, "annotations_csv")),
        "spec": PreprocessingSpec.from_config(cfg["preprocessing"]),
        "label_map": default_label_map(),
        "dataset_hash": canonical_hash([e["sha256"] for e in catalog]),
    }
