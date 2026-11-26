"""Publisher-held-out recordings can be sealed as the test split by group key (Dryad pilot)."""

from __future__ import annotations

import json

from tests.test_model_and_trainer import _prepare_cli, _small_config
from whalemix_training.cli import main


def _set_split(corpus, **split) -> None:
    cfg = json.loads(corpus["config"].read_text(encoding="utf-8"))
    cfg.setdefault("split", {}).update(split)
    corpus["config"].write_text(json.dumps(cfg), encoding="utf-8")


def _manifest(corpus) -> dict:
    return json.loads((corpus["out"] / "split-manifest.json").read_text(encoding="utf-8"))


def _test_dirs(manifest: dict, catalog: list[dict]) -> set[str]:
    by_id = {e["file_id"]: e["relative_path"].split("/")[0] for e in catalog}
    return {by_id[f] for f, a in manifest["assignments"].items() if a["split"] == "test"}


def _catalog(corpus) -> list[dict]:
    lines = (corpus["out"] / "source-catalog.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_sealed_groups_are_the_whole_test_split(corpus):
    _small_config(corpus)
    _prepare_cli(corpus)
    _set_split(
        corpus,
        fractions={"train": 0.8, "validation": 0.2, "test": 0.0},
        sealed_test_groups=["deployment:dep-d"],
    )
    (corpus["out"] / "split-manifest.json").unlink()
    assert main(["split", "--config", str(corpus["config"])]) == 0
    manifest = _manifest(corpus)
    assert _test_dirs(manifest, _catalog(corpus)) == {"dep-d"}
    assert manifest["sealed_test_groups"] == ["deployment:dep-d"]
    assert {a["split"] for a in manifest["assignments"].values()} >= {"train", "validation"}
    # a rerun keeps the frozen held-out sets and the same hash
    assert main(["split", "--config", str(corpus["config"])]) == 0
    assert _manifest(corpus)["split_hash"] == manifest["split_hash"]


def test_unknown_sealed_group_is_refused(corpus, capsys):
    _small_config(corpus)
    _prepare_cli(corpus)
    _set_split(corpus, sealed_test_groups=["deployment:no-such-deployment"])
    (corpus["out"] / "split-manifest.json").unlink()
    assert main(["split", "--config", str(corpus["config"])]) == 2
    assert "no-such-deployment" in capsys.readouterr().err
    assert not (corpus["out"] / "split-manifest.json").exists()
