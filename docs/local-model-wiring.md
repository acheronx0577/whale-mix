# Local model wiring (WhaleMix light model v1)

How the desktop app turns dropped audio into Blue A detections on your own machine. Nothing is uploaded
and nothing is downloaded at run time.

## Pieces

| Piece | What it is | Where it lives |
|---|---|---|
| `whale-mix` | Qt 6 desktop app (C++20) | build output, next to the worker |
| `whale-mix-engine` | Rust inference worker, one process per analysis run | next to `whale-mix` (or `WHALEMIX_ENGINE`) |
| Model bundle | `model.onnx` + `bundle.json` (+ `qualification-evidence.json`) | `<app>/models/acoustic-v1/` (or `WHALEMIX_MODEL_BUNDLE`) |

The bundle is **not** in Git. It is published as GitHub Release assets; `.gitignore` blocks `*.onnx`,
`*.pt`, `bundles/` and `models/`.

## Light model v1 bundle

- Pilot 12 lean MB01, seed 14141, randomly initialised, 6,000 updates; selected checkpoint `b94b69c1…5a9e`.
- `model.onnx`: fp32, opset 18, 16,086,384 bytes, SHA-256 `5d90c70b5deff5faf6a4194d11c51c2b5f9efeff9f4112df9b354967af53728b`.
- Input `spectrogram` `[batch, 1, 52, 100]` (dynamic batch, checked at batch 1, 3 and 16 against PyTorch,
  max abs diff 6e-8); output `probabilities` `[batch, 11]` (sigmoid, label-map order, `blue_a` first).
- Blue A threshold `0.897`, fixed before training from training-window counts.
- `qualification.status = "qualified"`, written by `whalemix_training qualify`. Read the disclosures in
  `qualification-evidence.json`: the sealed test is the MB01 recorder (also in training, four days later), and
  recorders the model was not trained on score lower.

## Install

1. Download `model.onnx`, `bundle.json` and `qualification-evidence.json` from the release.
2. Put them in `<folder with whale-mix>/models/acoustic-v1/`, or set
   `WHALEMIX_MODEL_BUNDLE=<folder>`.
3. Check: `whale-mix-engine --check-bundle --model-bundle <folder>` prints a `model_ready` event. The worker
   re-checks the model SHA-256 and refuses unqualified or fixture bundles.

## Drop audio → results

1. Drop WAV files or a folder on the window (or use New source → Raw audio). Files are discovered off the UI
   thread and listed in `inputs.jsonl` under the run folder (`<AppLocalData>/runs/<run-id>/`).
2. The app starts `whale-mix-engine` with `QProcess` (no shell). The worker streams `progress` and
   `results_batch` events on stdout and appends every row to `out/results.partial.jsonl` before announcing it.
3. The table fills while the run goes: rows are kept as they arrive and the views refresh at most four times
   a second, appending only new rows, so a 10,000-file drop stays responsive.
4. Selecting a row shows its Blue A score, the engine's reason and a 0–250 Hz spectrogram of that window,
   drawn from the WAV.
5. When the run completes the worker writes `out/results.csv` and `out/run.json`. Export report writes
   `decisions.csv`, `decisions.json` and the report for every row, including hand-set labels.

## Measured on a box CPU (8 vCPU, `--threads 2 --batch 8`)

- 10,000 real Dryad WAVs (479.5 h of audio): 335,450 windows, 0 file errors, 888 s in the worker alone
  (1,944× real time, worker peak RSS 120 MB).
- Same drop through the desktop app: 880 s, longest UI event-loop gap 196 ms (2 gaps over 100 ms).
  Before the streaming fix the app stalled for up to 4.5 s at only 1,000 files.

## Tests

```bash
cd app/training && python -m pytest -q                       # includes dynamic-batch export tests
cd app/engine && cargo test --release
ctest --test-dir build/linux --output-on-failure             # GUI smoke + engine bridge tests
# Opt-in stress run with a real bundle (not run in CI):
WHALEMIX_STRESS_AUDIO_DIR=<folder> WHALEMIX_MODEL_BUNDLE=<bundle> WHALEMIX_ENGINE=<worker> \
  build/linux/app/whalemix_gui_tests --gtest_filter='*QualifiedBundleFolderDrop*'
```
