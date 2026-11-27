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
                 [--device auto|hybrid|gpu|cuda|cpu]
                 [--check-bundle] [--allow-fixture-bundle]
```

- `--inputs`: JSON Lines with one `{"source_id": "...", "path": "..."}` per recording. It is read lazily.
- `--profile`: a required profile id from the bundle, or `auto` (the default) for every required profile.
- `--threads`: ONNX Runtime intra-op threads, capped at 2. Execution is sequential and inter-op is fixed at 1.
- `--batch`: windows per inference call (default 8).
- `--device`: `auto` (default), `hybrid`, `gpu`, `cuda` or `cpu`. The default build has no CUDA provider, so every mode except an explicit `cpu` runs on CPU and the `started` event says so. A `--features cuda` build uses CUDA for `auto`, `hybrid`, `gpu` and `cuda` when a GPU initializes, and falls back to CPU with a reason when it does not.
- `--allow-fixture-bundle`: accepts an untrained `fixture-untrained` bundle. This is for tests only. The app never passes it.

| Exit | Meaning |
|---|---|
| 0 | Completed. `results.csv` and `run.json` were validated |
| 1 | Run failed (model/runtime/I/O). Committed rows stay in `results.partial.jsonl` |
| 2 | Invalid arguments or configuration (unknown profile, bad numbers) |
| 3 | Model bundle missing, invalid, hash mismatch, unqualified, or a refused fixture |
| 4 | Canceled. Partial output is kept as `results.partial.csv` / `run.partial.json` |

## CUDA

CI builds the default CPU feature set and does not install a CUDA toolkit. On a machine with an NVIDIA GPU:

```bash
cd app/engine
cargo build --release --features cuda
```

That enables `ort`'s `cuda` feature. The build downloads pyke's ONNX Runtime 1.22 `cu12` archive and statically links it (`libonnxruntime.a` on Linux, `onnxruntime.lib` on Windows). It does not compile against the CUDA toolkit. The CUDA provider shared library is copied next to the executable (`libonnxruntime_providers_cuda.so` and `libonnxruntime_providers_shared.so` on Linux; `onnxruntime_providers_cuda.dll` and `onnxruntime_providers_shared.dll` on Windows). At run time that provider still has to load the CUDA 12 and cuDNN 9 shared libraries from `LD_LIBRARY_PATH` (Linux) or `PATH` (Windows).

`ORT_DYLIB_PATH` is not read by this build. `ort` only consults it with the `load-dynamic` feature, and the downloaded `cu12` archive contains no `libonnxruntime.so` / `onnxruntime.dll` (the runtime is the static library). To link a different ONNX Runtime 1.22.x, such as one compiled with CUDA 12.8 and `sm_120` for an RTX 5090, point the build at it:

```bash
ORT_LIB_LOCATION=/path/to/onnxruntime/lib cargo build --release --features cuda
```

That directory must contain the 1.22 library and the CUDA provider library. Set `ORT_PREFER_DYNAMIC_LINK=1` when the directory holds a shared library instead of the static archive. A hand-built `load-dynamic` binary is the case that uses `ORT_DYLIB_PATH`: it must be an ONNX Runtime 1.22.x shared library, with the CUDA provider library beside it. This crate does not enable that feature.

| Machine | What the runtime needs |
|---|---|
| Linux, Vast.ai GTX 1080 (Pascal, sm_61) | Driver whose `nvidia-smi` CUDA version is 12 or newer, plus CUDA 12 (`libcudart.so.12`, cuBLAS, cuFFT, NVRTC) and cuDNN 9 (`libcudnn.so.9` and the `libcudnn_*.so.9` libraries) on `LD_LIBRARY_PATH`. Pascal is in the standard CUDA 12 builds. A CUDA 11 / cuDNN 8 image will not load this provider. |
| Linux, Vast.ai RTX 5090 (Blackwell, sm_120) | A driver new enough for CUDA 12.8 (Blackwell) and cuDNN 9 on `LD_LIBRARY_PATH`. The bundled ONNX Runtime 1.22 `cu12` archive may not contain sm_120 kernels. If the session fails with a missing kernel image, rebuild with `ORT_LIB_LOCATION` pointing at an ONNX Runtime 1.22.x library compiled with CUDA 12.8 or newer and architecture 120. `ORT_DYLIB_PATH` does not apply to this feature. |
| Windows, RTX 3070 (Ampere, sm_86) | Current NVIDIA driver, CUDA 12 toolkit (`cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll`, `cufft64_11.dll` on `PATH`, usually `CUDA_PATH\bin`) and cuDNN 9 (`cudnn64_9.dll` plus the `cudnn_*64_9.dll` libraries). cuDNN 8 does not load. |

`--device cpu` never selects CUDA. `--device auto` uses CUDA only when this build has the feature and a GPU is visible (`/dev/nvidia0`, `/dev/nvidiactl` or `/proc/driver/nvidia/gpus` on Linux, `C:\Windows\System32\nvcuda.dll` on Windows) and the provider registers. `CUDA_VISIBLE_DEVICES` empty, `none` or `-1` hides the GPU. The `started` event reports `device_active` of `"cuda"` or `"cpu"` and a `device_note`.

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
