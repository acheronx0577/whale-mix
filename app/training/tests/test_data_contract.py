from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from whalemix_training import annotations as ann
from whalemix_training import inventory, splits
from whalemix_training.cli import main
from whalemix_training.config import ConfigError, load_config
from whalemix_training.io_utils import ReadOnlyRootError, assert_outside_root
from whalemix_training.labels import default_label_map, is_assessable, targets_and_mask


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def test_inventory_is_read_only_and_reports_duplicates(corpus):
    before = _snapshot(corpus["root"])
    report = inventory.build_inventory(corpus["root"], corpus["out"], [".wav"], corpus["meta"])
    assert _snapshot(corpus["root"]) == before
    assert report["files"] == 9
    assert report["unreadable_files"] == ["dep-c/broken.wav"]
    assert len(report["exact_duplicate_groups"]) == 1
    assert report["missing_provenance_or_license"] == []
    assert report["hours_by_year"]["2013"] > 0
    catalog = inventory.load_catalog(corpus["out"] / "source-catalog.jsonl")
    entry = next(e for e in catalog if e["relative_path"] == "dep-a/a1.wav")
    assert entry["sample_rate"] == 4000 and entry["channels"] == 1
    assert abs(entry["duration_s"] - 12.0) < 1e-6
    assert entry["annotation_status"] == "unannotated"


def test_inventory_is_incremental(corpus):
    inventory.build_inventory(corpus["root"], corpus["out"], [".wav"], corpus["meta"])
    again = inventory.build_inventory(corpus["root"], corpus["out"], [".wav"], corpus["meta"])
    assert again["reused_entries"] == 9


def test_inventory_bounded_walk(corpus):
    report = inventory.build_inventory(corpus["root"], corpus["out"], [".wav"], None, max_files=3)
    assert report["files"] == 3


def test_output_inside_raw_root_is_refused(corpus):
    with pytest.raises(ReadOnlyRootError):
        assert_outside_root(corpus["root"] / "derived", corpus["root"])
    with pytest.raises(ReadOnlyRootError):
        inventory.build_inventory(corpus["root"], corpus["root"] / "out", [".wav"])


def test_config_env_override_and_missing_key(tmp_path):
    cfg = load_config(None, env={"WHALEMIX_OUTPUT_DIR": str(tmp_path)})
    assert cfg["paths"]["output_dir"] == str(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad)


def test_targets_and_mask_keep_unassessed_masked():
    lm = default_label_map()
    y, mask = targets_and_mask({"blue_a": "present", "fin_20hz_pulse": "absent", "vessel": "unassessed"}, lm)
    assert y[lm.index("blue_a")] == 1 and mask[lm.index("blue_a")] == 1
    assert y[lm.index("fin_20hz_pulse")] == 0 and mask[lm.index("fin_20hz_pulse")] == 1
    assert mask[lm.index("vessel")] == 0 and mask[lm.index("humpback_song")] == 0
    assert len(lm.outputs) == 11 and lm.outputs[:4] == ("blue_a", "blue_b", "fin_20hz_pulse", "humpback_song")


def test_bandwidth_assessability():
    assert is_assessable("blue_a", 2000)
    assert not is_assessable("humpback_song", 1000)
    assert not is_assessable("invertebrate", 8000)


def _write_annotations(path: Path, catalog: list[dict], extra: list[list[str]] | None = None) -> None:
    ids = {e["relative_path"]: e["file_id"] for e in catalog}
    rows = [
        ["n1", ids["dep-a/a1.wav"], "0", "1.0", "3.0", "blue_a", "present", "reviewed", "fixture"],
        ["n2", ids["dep-a/a1.wav"], "0", "1.0", "3.0", "vessel", "present", "reviewed", "fixture"],
        ["n3", ids["dep-b/b1.wav"], "0", "0.0", "10.0", "blue_a", "absent", "reviewed", "fixture"],
        [
            "n4",
            ids["dep-d/d1.wav"],
            "0",
            "0.0",
            "2.0",
            "fin_20hz_pulse",
            "present",
            "ambiguous",
            "call-index",
        ],
        ["n5", ids["dep-e/e1.wav"], "0", "2.0", "4.0", "blue_a", "present", "reviewed", "fixture"],
        ["n6", ids["dep-c/c1.wav"], "0", "2.0", "4.0", "blue_a", "present", "reviewed", "fixture"],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(ann.REQUIRED_COLUMNS)
        w.writerows(rows + (extra or []))


def test_label_validation_and_coverage(corpus):
    inventory.build_inventory(corpus["root"], corpus["out"], [".wav"], corpus["meta"])
    catalog = inventory.load_catalog(corpus["out"] / "source-catalog.jsonl")
    ids = {e["relative_path"]: e["file_id"] for e in catalog}
    path = corpus["tmp"] / "annotations.csv"
    _write_annotations(
        path,
        catalog,
        [
            ["bad1", ids["dep-a/a2.wav"], "0", "5.0", "99.0", "blue_a", "present", "reviewed", "fixture"],
            ["bad2", ids["dep-a/a2.wav"], "3", "1.0", "2.0", "orca", "maybe", "reviewed", ""],
            ["bad3", ids["dep-a/a2.wav"], "0", "1.0", "2.0", "invertebrate", "absent", "reviewed", "fixture"],
        ],
    )
    report = ann.validate(ann.read_annotations(path), catalog, default_label_map())
    messages = {(i["annotation_id"], i["message"]) for i in report["issues"]}
    assert ("bad1", "end_s is beyond the recording duration") in messages
    assert ("bad2", "unknown label 'orca'") in messages
    assert ("bad2", "missing provenance") in messages
    assert ("bad2", "channel is out of range for the recording") in messages
    assert any(m[0] == "bad3" and "bandwidth" in m[1] for m in messages)
    cov = report["coverage"]
    assert cov["blue_a"]["reviewed_present_events"] >= 3
    assert cov["fin_20hz_pulse"]["status"] == "absent_or_unverified"  # ambiguous row is not coverage
    assert cov["humpback_song"]["status"] == "absent_or_unverified"


def test_grouped_split_is_deterministic_and_leak_free(corpus):
    inventory.build_inventory(corpus["root"], corpus["out"], [".wav"], corpus["meta"])
    catalog = inventory.load_catalog(corpus["out"] / "source-catalog.jsonl")
    fr = {"train": 0.7, "validation": 0.15, "test": 0.15}
    m1 = splits.assign_splits(catalog, 7, fr)
    m2 = splits.assign_splits(list(reversed(catalog)), 7, fr)
    assert m1["split_hash"] == m2["split_hash"]
    assert splits.leakage_check(catalog, m1) == []
    by_rel = {
        e["relative_path"]: m1["assignments"][e["file_id"]]
        for e in catalog
        if e["file_id"] in m1["assignments"]
    }
    assert by_rel["dep-a/a1.wav"]["split"] == by_rel["dep-c/copy-of-a1.wav"]["split"]
    assert by_rel["dep-a/a1.wav"]["split"] == by_rel["dep-a/a2.wav"]["split"]
    assert {a["split"] for a in m1["assignments"].values()} == {"train", "validation", "test"}
    forced = splits.assign_splits(catalog, 7, fr, frozenset({"deployment:dep-e"}))
    e1 = next(e["file_id"] for e in catalog if e["relative_path"] == "dep-e/e1.wav")
    assert forced["assignments"][e1]["split"] == "test"


def test_leakage_check_detects_cross_split_duplicate(corpus):
    inventory.build_inventory(corpus["root"], corpus["out"], [".wav"], corpus["meta"])
    catalog = inventory.load_catalog(corpus["out"] / "source-catalog.jsonl")
    manifest = splits.assign_splits(catalog, 7, {"train": 0.7, "validation": 0.15, "test": 0.15})
    dup = next(e for e in catalog if e["relative_path"] == "dep-c/copy-of-a1.wav")
    original = manifest["assignments"][
        next(e["file_id"] for e in catalog if e["relative_path"] == "dep-a/a1.wav")
    ]
    other = "test" if original["split"] != "test" else "train"
    manifest["assignments"][dup["file_id"]] = {"split": other, "group": "tampered"}
    assert any("sha256" in p for p in splits.leakage_check(catalog, manifest))


def test_cli_data_commands(corpus, capsys):
    assert main(["inventory", "--config", str(corpus["config"])]) == 0
    catalog = inventory.load_catalog(corpus["out"] / "source-catalog.jsonl")
    _write_annotations(corpus["tmp"] / "annotations.csv", catalog)
    assert main(["validate-labels", "--config", str(corpus["config"])]) == 0
    assert main(["split", "--config", str(corpus["config"])]) == 0
    manifest = json.loads((corpus["out"] / "split-manifest.json").read_text(encoding="utf-8"))
    assert manifest["leakage_problems"] == []
    assert any("fin_20hz_pulse" in g for g in manifest["heldout_gaps"])
    assert not any(p.name.startswith("source-catalog") for p in corpus["root"].rglob("*"))


def test_cli_missing_config_value_exits_2(tmp_path, capsys):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    env_backup = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith("WHALEMIX_")}
    try:
        assert main(["inventory", "--config", str(cfg)]) == 2
    finally:
        os.environ.update(env_backup)
    assert "paths.recordings_root" in capsys.readouterr().err
