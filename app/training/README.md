# WhaleMix Acoustic v1 training package

Development-side code for the WhaleMix Acoustic v1 detector. The desktop app never imports it,
never trains, and never runs Laya. It only consumes an exported, selected model bundle.

**Status (2026-11-25): untrained.** No optimizer update has run and no trained weights exist.
The tests use short synthetic tones (purpose-built fixtures) to check interfaces and input contracts.
They say nothing about detection accuracy.

## Install (isolated environment)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 torchvision==0.29.2
pip install -e "app/training[model,dev]"
```

Exact versions verified on the build box are in `requirements-lock.txt`.

## Entry point

`python -m whalemix_training <command> --config <run-config.json>`

| Command | Slice | What it does |
|---|---|---|
| `inventory` | data | Hash and probe recordings into `source-catalog.jsonl` (read-only, incremental) |
| `validate-labels` | data | Check annotations against the catalog; write the coverage matrix |
| `split` | data | Deterministic grouped 70/15/15 split, leakage check, locked split hash |
| `train` | model | Random-init EfficientNet-B0 session; with `training.max_updates = 0` (this build) it only builds, checks shapes/loss and writes `train-plan.json` |
| `finetune` | model | Continue a recorded project checkpoint with replay (`--parent`, `paths.replay_manifest`); fixture checkpoints are refused |
| `evaluate` | eval | Event matching (`event-match-v1`), per-label/slice metrics, retention, gate, optional `--select`, Laya advice |
| `export` | export | `--checkpoint` (selected only) or `--fixture` (untrained plumbing model) to an ONNX bundle + `bundle.json`; `--parity-wav` writes a Rust parity reference |
| `quantize` | export | Returns `SETUP_REQUIRED` until real calibration windows and a selected checkpoint exist |
| `benchmark` | export | Session load, warm batch-8 latency and Python preprocessing speed, labelled `fixture` |

`scripts/reproduce-acoustic-v1.py --config <cfg>` runs these stages in order: inventory, validate-labels,
split, train, then optional finetune (`--parent`), evaluate (`--eval-set --predictions [--select]`) and
export (`--export-checkpoint`). It stops at the first non-zero exit and appends each stage to
`<output_dir>/reproduce-ledger.jsonl`. `--dry-run` prints the plan.

Gates are deterministic: `PASS_ELIGIBLE`, `FAIL`, `INCONCLUSIVE`, `SETUP_REQUIRED`. Only `PASS_ELIGIBLE`
moves `selection/selected.json` (atomic); every attempt is appended to `improvement-ledger.jsonl`.
Selection needs a per-label policy in the config: `selection.min_recall` and
`selection.min_background_hours` (for example `{"blue_a": 0.5}`). An undeclared label is
`SETUP_REQUIRED`, recall below the minimum is `FAIL`, and too little reviewed background is
`INCONCLUSIVE`. `score` keeps only recordings with reviewed rows (attributable) and writes their merged
reviewed intervals into `eval-set.json`. False alarms are counted only inside reviewed assessable time
and divided by those hours; other detections are reported as `unreviewed_detections`.
The test split is sealed: `score --split test` accepts only `selection/selected.pt`, uses the selected
thresholds and writes `selection/test-seal.json`; a re-tuned model cannot be scored on it afterwards,
and `evaluate --select` refuses test-split reports. `evaluation.purpose = "dev-pilot"` marks reports as
development results; `qualify` accepts only `release` reports.
The Laya 0.4.6 adapter (`laya --questions --batch --batch-size --json --device cpu --model`) is advisory
and falls back to the deterministic ranking on any failure. It never runs in the desktop app.

### Preprocessing contract (v1)

Per channel, never averaged: Kaiser-sinc polyphase resampling (taps computed by formula), periodic-Hann
power spectra on a global frame grid, a frequency crop, then fixed-length model windows with an
end-aligned tail window. `log` or per-window `pcen` compression; optional train-only global
standardisation. Tensor `float32 [N, 1, bins, frames]`. The full definition and its hash go into every
checkpoint and bundle. The Rust engine implements the same contract (parity tested).

Paths come only from the config file or `WHALEMIX_RECORDINGS_ROOT`, `WHALEMIX_METADATA_CSV`,
`WHALEMIX_ANNOTATIONS_CSV` and `WHALEMIX_OUTPUT_DIR`. Outputs inside the recording root are refused.

Exit codes: `0` success, `1` a check or gate failed, `2` invalid configuration or input.

## Checks

```powershell
cd app/training
ruff check . ; ruff format --check . ; mypy whalemix_training ; pytest -q
```

## Credits and citations

- Team: original implementation, data curation and annotations, evaluation, profiles, the analysis
  controller and the Rust/Qt integration.
- Architecture: EfficientNet-B0 (Tan and Le, 2019) via torchvision, instantiated with `weights=None`.
- Libraries: PyTorch, torchvision, NumPy, SciPy, soundfile/libsndfile, ONNX, ONNX Runtime.
- Comparison baseline: MBARI Blue A (attributed, kept separate; its weights are never relabelled).
