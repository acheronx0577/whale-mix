# whale-mix-engine

Rust inference worker for WhaleMix Acoustic v1. The desktop app starts it with `QProcess` only when
audio is queued. It never trains and never downloads weights.

**Status (2026-11-26): light model v1 bundle.** The first trained bundle (Pilot 12, seed 14141, model SHA-256
`5d90c70b…728b`, 16.1 MB fp32 ONNX, dynamic batch) is distributed outside Git as a release asset. Install it
as `<app>/models/acoustic-v1/` or point `WHALEMIX_MODEL_BUNDLE` at it. At its fixed Blue A threshold 0.897 it
scored recall 0.825 at 0.76 false alarms/h on the sealed MB01 test (Oct 7–27 2020, 1,272 calls, 494 h
reviewed background); recorders it was not trained on are weaker. Its `qualification-evidence.json` lists the
full metrics and disclosures. The fixture timings and tests below still use an untrained fixture bundle; they
check plumbing, not detection quality.

Measured with the light model v1 bundle on a box CPU (8 vCPU, `--threads 2 --batch 8`): 10,000 real Dryad WAVs
(479.5 h of audio, 335,450 windows) in 888 s, 1,944× real time, 0 file errors, peak worker RSS 120 MB.

## Command

```text
whale-mix-engine --inputs <manifest.jsonl> --out <dir> --model-bundle <dir>
                 [--profile <id>]... [--threads N] [--batch N] [--run-id ID]
                 [--check-bundle] [--allow-fixture-bundle]
```

- `--inputs`: JSON Lines with one `{"source_id": "...", "path": "..."}` per recording. It is read lazily.
- `--profile`: a required profile id from the bundle, or `auto` (the default) for every required profile.
- `--threads`: ONNX Runtime intra-op threads, capped at 2. Execution is sequential and inter-op is fixed at 1.
- `--batch`: windows per inference call (default 8).
- `--allow-fixture-bundle`: accepts an untrained `fixture-untrained` bundle. This is for tests only. The app never passes it.

| Exit | Meaning |
|---|---|
| 0 | Completed. `results.csv` and `run.json` were validated |
| 1 | Run failed (model/runtime/I/O). Committed rows stay in `results.partial.jsonl` |
| 2 | Invalid arguments or configuration (unknown profile, bad numbers) |
| 3 | Model bundle missing, invalid, hash mismatch, unqualified, or a refused fixture |
| 4 | Canceled. Partial output is kept as `results.partial.csv` / `run.partial.json` |

## Bundle gate

`bundle.json` must have schema `whalemix-model-bundle/1`. It must also have a consistent label map
(`whale_targets` followed by `context_targets`), an input shape that matches the preprocessing contract,
and a model SHA-256 that matches the file. Only `artifact_kind = "selected"` with
`qualification.status = "qualified"` runs by default.

## Protocol (`whalemix-engine/1`, stdout JSON Lines)

Every event has `protocol`, `type`, `run_id` and a monotonic `seq`. The types are `started`,
`dataset_added`, `schema_updated`, `progress`, `results_batch`, `file_error`, `model_unavailable`,
`model_ready`, `paused`, `resumed`, `cancel_requested`, `canceled`, `completed` and `failed`. A
`results_batch` is written only after its rows are appended and fsynced to `results.partial.jsonl`.
Events are capped at 256 KiB. On stdin, the app can send `pause`, `resume` or `cancel`. These take
effect at chunk/batch boundaries.

## Memory bounds

- Audio is decoded in 65,536-frame chunks.
- Each channel keeps only the frames needed for the current window.
- The decode/preprocess producer hands batches to the inference thread over a channel of capacity 1.
- The ONNX session is created lazily on the first batch and dropped when the run ends.

## Results

`results.csv` has one row per channel window. Each row holds:

- stable `event_id`, source, channel, start/end seconds, granularity `coarse_window`
- one `score_<output>` column per label-map output; an empty cell means *unavailable for this
  recording's bandwidth*, not zero
- automatic label/reason, model SHA-256 and artifact kind

While thresholds are `uncalibrated` (fixture bundles), every row is `review` with reason "scores only".
`run.json` records:

- bundle hashes
- the inputs-manifest hash
- per-file status, in `files.partial.jsonl`
- totals
- resources: `model_load_ms`, `first_progress_ms`, `x_realtime`, `peak_rss_mb` (Linux VmHWM) and `measurement_kind`

## Tests

```bash
cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test --release
# Fixture inference, parity and cancel tests (bundle exported outside Git):
#   cd app/training && python -m whalemix_training export --config <cfg> --fixture \
#       --parity-wav ../../samples/audio/blue-whale-ne-pacific.wav
WHALEMIX_FIXTURE_BUNDLE=<out>/fixture-bundles/fixture-b0 \
WHALEMIX_PARITY_WAV=../../samples/audio/blue-whale-ne-pacific.wav cargo test --release
```

`tests/data/preprocess-parity.json` is a small committed Python reference for a synthetic signal.
Regenerate it with `tests/data/make_parity_fixture.py`. No audio or weights are committed.
