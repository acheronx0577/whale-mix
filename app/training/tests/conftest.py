"""Purpose-built fixtures: short synthetic tones and noise written to a temp directory.

They exercise interfaces and input contracts only. They are not training data and say nothing about
whale-detection accuracy.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

FIXTURE_SR = 4000


def write_tone(
    path: Path, seconds: float, freq: float, sr: int = FIXTURE_SR, channels: int = 1, seed: int = 0
) -> None:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    sig = 0.3 * np.sin(2 * np.pi * freq * t) + 0.01 * rng.standard_normal(t.size)
    data = np.stack([sig] * channels, axis=1) if channels > 1 else sig
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data.astype(np.float32), sr, subtype="PCM_16")


@pytest.fixture()
def corpus(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "raw"
    files = {
        "dep-a/a1.wav": (12.0, 20.0, 0),
        "dep-a/a2.wav": (8.0, 90.0, 1),
        "dep-b/b1.wav": (10.0, 20.0, 2),
        "dep-b/b2.wav": (6.0, 300.0, 3),
        "dep-c/c1.wav": (9.0, 50.0, 4),
        "dep-d/d1.wav": (7.0, 400.0, 5),
        "dep-e/e1.wav": (11.0, 25.0, 6),
    }
    for rel, (secs, freq, seed) in files.items():
        write_tone(root / rel, secs, freq, seed=seed)
    # exact duplicate placed in a different deployment folder
    (root / "dep-c").mkdir(parents=True, exist_ok=True)
    (root / "dep-c" / "copy-of-a1.wav").write_bytes((root / "dep-a" / "a1.wav").read_bytes())
    (root / "dep-c" / "broken.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunk")
    meta = tmp_path / "metadata.csv"
    with meta.open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["relative_path", "recording_date", "deployment", "provenance", "license"])
        for rel in [*files, "dep-c/copy-of-a1.wav"]:
            dep = rel.split("/")[0]
            year = {"dep-a": "2013", "dep-b": "2013", "dep-c": "2014", "dep-d": "2014", "dep-e": "2015"}[dep]
            w.writerow([rel, f"{year}-06-01", dep, "fixture:synthetic-tone", "CC0-fixture"])
    out = tmp_path / "out"
    cfg = tmp_path / "config.json"
    cfg.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "paths": {
                    "recordings_root": str(root),
                    "metadata_csv": str(meta),
                    "output_dir": str(out),
                    "annotations_csv": str(tmp_path / "annotations.csv"),
                },
            }
        ),
        encoding="utf-8",
    )
    return {"root": root, "meta": meta, "out": out, "config": cfg, "tmp": tmp_path}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """With WHALEMIX_REQUIRE_FIXTURE=1 (CI integration job) a skipped test fails instead of passing."""
    outcome = yield
    report = outcome.get_result()
    if (
        report.skipped
        and not hasattr(report, "wasxfail")
        and os.environ.get("WHALEMIX_REQUIRE_FIXTURE") == "1"
    ):
        report.outcome = "failed"
        report.longrepr = f"required test was skipped: {report.longrepr}"
