"""Fix 1 (training readiness): hash-bound data lock, frozen held-out splits, leakage, bandwidth masks,
bounded windows and the zero-update preflight. Each test reproduces an audit failure on da0dee7."""

from __future__ import annotations

import json

import pytest
import soundfile as sf

from tests.conftest import write_tone
from tests.test_model_and_trainer import _prepare_cli, _small_config
from whalemix_training import inventory, splits
from whalemix_training import windows as win
from whalemix_training.cli import main
from whalemix_training.labels import default_label_map
from whalemix_training.preprocess import PreprocessingSpec
from whalemix_training.windows import build_windows, window_states


def _split_manifest(corpus) -> dict:
    return json.loads((corpus["out"] / "split-manifest.json").read_text(encoding="utf-8"))


def _heldout(manifest: dict) -> dict[str, str]:
    return {f: a["split"] for f, a in manifest["assignments"].items() if a["split"] != "train"}


def test_training_rejects_annotations_changed_after_validation(corpus, capsys):
    _small_config(corpus)
    _prepare_cli(corpus)
    with (corpus["tmp"] / "annotations.csv").open("a", newline="", encoding="utf-8") as handle:
        handle.write("late,unknown-file,0,0.0,1.0,blue_a,present,reviewed,fixture\n")
    capsys.readouterr()
    assert main(["train", "--config", str(corpus["config"])]) == 1
    assert "stale" in capsys.readouterr().out


def test_training_rejects_catalog_changed_after_split(corpus, capsys):
    _small_config(corpus)
    _prepare_cli(corpus)
    write_tone(corpus["root"] / "dep-f" / "f1.wav", 6.0, 30.0, seed=9)
    assert main(["inventory", "--config", str(corpus["config"])]) == 0
    capsys.readouterr()
    assert main(["train", "--config", str(corpus["config"])]) == 1
    assert "stale" in capsys.readouterr().out


def test_split_rejects_stale_label_validation(corpus, capsys):
    _small_config(corpus)
    _prepare_cli(corpus)
    with (corpus["tmp"] / "annotations.csv").open("a", newline="", encoding="utf-8") as handle:
        handle.write("late2,unknown-file,0,0.0,1.0,blue_a,present,reviewed,fixture\n")
    assert main(["split", "--config", str(corpus["config"])]) == 2
    assert "validate-labels" in capsys.readouterr().err


def test_heldout_sets_stay_frozen_across_appends(corpus):
    _small_config(corpus)
    _prepare_cli(corpus)
    before = _split_manifest(corpus)
    for n in range(6):  # a large append that would reshuffle a greedy re-split
        write_tone(corpus["root"] / f"new-{n}" / "x.wav", 30.0, 40.0 + n, seed=20 + n)
    assert main(["inventory", "--config", str(corpus["config"])]) == 0
    assert main(["validate-labels", "--config", str(corpus["config"])]) == 0
    assert main(["split", "--config", str(corpus["config"])]) == 0
    after = _split_manifest(corpus)
    assert _heldout(after) == _heldout(before)
    assert after["lock"]["heldout_hash"] == before["lock"]["heldout_hash"]
    new_ids = set(after["assignments"]) - set(before["assignments"])
    assert len(new_ids) == 6 and all(after["assignments"][f]["split"] == "train" for f in new_ids)


def test_removed_or_changed_heldout_file_is_rejected(corpus, capsys):
    _small_config(corpus)
    _prepare_cli(corpus)
    manifest = _split_manifest(corpus)
    catalog = {e["file_id"]: e for e in inventory.load_catalog(corpus["out"] / "source-catalog.jsonl")}
    victim = next(f for f, s in _heldout(manifest).items())
    write_tone(corpus["root"] / catalog[victim]["relative_path"], 5.0, 77.0, seed=99)  # content changed
    assert main(["inventory", "--config", str(corpus["config"])]) == 0
    assert main(["validate-labels", "--config", str(corpus["config"])]) in (0, 1)
    capsys.readouterr()
    assert main(["split", "--config", str(corpus["config"])]) == 1
    assert "held-out" in capsys.readouterr().out


def test_same_audio_in_another_container_is_grouped_and_leak_checked(corpus):
    data, sr = sf.read(str(corpus["root"] / "dep-b" / "b1.wav"), dtype="int16")
    sf.write(str(corpus["root"] / "dep-e" / "b1-reencoded.flac"), data, sr, subtype="PCM_16")
    inventory.build_inventory(corpus["root"], corpus["out"], [".wav", ".flac"], corpus["meta"])
    catalog = inventory.load_catalog(corpus["out"] / "source-catalog.jsonl")
    by_rel = {e["relative_path"]: e for e in catalog}
    wav, flac = by_rel["dep-b/b1.wav"], by_rel["dep-e/b1-reencoded.flac"]
    assert wav["sha256"] != flac["sha256"] and wav["pcm_sha256"] == flac["pcm_sha256"]
    manifest = splits.assign_splits(catalog, 7, {"train": 0.7, "validation": 0.15, "test": 0.15})
    a = manifest["assignments"]
    assert a[wav["file_id"]]["split"] == a[flac["file_id"]]["split"]
    other = "test" if a[wav["file_id"]]["split"] != "test" else "train"
    a[flac["file_id"]] = {"split": other, "group": "tampered"}
    assert any("audio content" in p for p in splits.leakage_check(catalog, manifest))


def test_time_overlap_across_splits_is_detected():
    catalog = [
        {
            "file_id": f"f{i}",
            "sha256": f"s{i}",
            "status": "ok",
            "duration_s": 600.0,
            "recorder_id": "hydrophone-1",
            "start_utc": start,
        }
        for i, start in enumerate(["2015-01-01T00:00:00Z", "2015-01-01T00:05:00Z", "2015-01-02T00:00:00Z"])
    ]
    manifest = {
        "assignments": {
            "f0": {"split": "train", "group": "g0"},
            "f1": {"split": "test", "group": "g1"},
            "f2": {"split": "test", "group": "g2"},
        }
    }
    problems = splits.leakage_check(catalog, manifest)
    assert any("overlap" in p and "f0" in p and "f1" in p for p in problems)
    assert not any("f2" in p for p in problems)
    grouped = splits.build_groups(catalog)
    assert grouped["f0"] == grouped["f1"] != grouped["f2"]


def test_bandwidth_masks_apply_to_positives_and_negatives():
    lm = default_label_map()
    spec = PreprocessingSpec()  # fmax 1000 Hz
    rows = [
        {"review_status": "reviewed", "start_s": "0", "end_s": "60", "label": lbl, "state": st}
        for lbl, st in (("humpback_song", "present"), ("invertebrate", "absent"), ("blue_a", "present"))
    ]
    narrow = window_states(0.0, 10.0, rows, 1000, lm, spec)  # Nyquist 500 Hz < humpback's 1000 Hz
    assert narrow["humpback_song"] == "unassessed" and narrow["blue_a"] == "present"
    wide = window_states(0.0, 10.0, rows, 96000, lm, spec)  # source is wide, but the model sees <= 1 kHz
    assert wide["invertebrate"] == "unassessed" and wide["humpback_song"] == "present"


def _windows_inputs(corpus):
    inventory.build_inventory(corpus["root"], corpus["out"], [".wav"], corpus["meta"])
    catalog = inventory.load_catalog(corpus["out"] / "source-catalog.jsonl")
    manifest = splits.assign_splits(catalog, 7, {"train": 0.7, "validation": 0.15, "test": 0.15})
    ids = {e["relative_path"]: e["file_id"] for e in catalog}
    rows = [
        {
            "file_id": ids["dep-a/a1.wav"],
            "channel": "0",
            "review_status": "reviewed",
            "start_s": "1",
            "end_s": "2",
            "label": "blue_a",
            "state": "present",
        }
    ]
    spec = PreprocessingSpec(
        sample_rate=2000, n_fft=256, hop_length=100, fmax_hz=500.0, window_seconds=2.0, window_hop_seconds=1.0
    )
    return catalog, rows, manifest, spec


def test_window_generation_is_bounded(corpus):
    catalog, rows, manifest, spec = _windows_inputs(corpus)
    lm = default_label_map()
    everything = build_windows(catalog, rows, manifest, spec, lm)
    assert len(everything) > 20
    with pytest.raises(win.WindowLimitError):
        build_windows(catalog, rows, manifest, spec, lm, max_windows=10)
    reviewed = build_windows(catalog, rows, manifest, spec, lm, reviewed_only=True, max_windows=10)
    assert reviewed and all(w["reviewed"] for w in reviewed)


def test_zero_update_preflight_reports_per_label_counts(corpus, capsys):
    _small_config(corpus)
    _prepare_cli(corpus)
    assert main(["train", "--config", str(corpus["config"])]) == 0
    plan = json.loads((corpus["out"] / "train-plan.json").read_text(encoding="utf-8"))
    pre = plan["preflight"]
    assert plan["session"]["optimizer_updates"] == 0
    blue = pre["per_label"]["blue_a"]
    assert sum(blue[s]["present"] for s in ("train", "validation", "test")) >= 5
    assert pre["per_label"]["fin_20hz_pulse"]["train"]["absent"] > 0
    humpback = pre["per_label"]["humpback_song"]["train"]
    assert humpback == {"present": 0, "absent": 0, "unassessed": pre["windows_by_split"]["train"]}
    assert "humpback_song" in " ".join(pre["warnings"])
    assert pre["blocking"] == []
