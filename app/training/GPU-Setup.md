# GPU Setup (training only)

The desktop app and the Rust engine stay CPU-only. GPU support is for the training host and is
selected explicitly in the run config:

```json
{"training": {"device": "cuda", "precision": "bf16", "cuda_index": 0}}
```

- `device`: `cpu` (default) or `cuda`. There is no silent fallback. If `cuda` is requested and torch
  cannot use the GPU, `train`/`finetune` exit 1 with `DEVICE_UNAVAILABLE` and the cause.
- `precision`: `fp32` (default, and the only CPU option), `bf16`, or `fp16` (uses a gradient scaler).
- The actual device, card name, compute capability, precision and torch CUDA build go into
  `<stage>-plan.json` and every checkpoint's metadata (`device`).

## Which card

The RTX 5090 vs RTX 3070 choice is still open. Both work with the pinned build below:

| Card | Architecture | Compute capability | Notes |
|---|---|---|---|
| RTX 5090 | Blackwell | sm_120 | Needs a CUDA 12.8+ torch build; older builds fail with "no kernel image". 32 GB VRAM. |
| RTX 3070 | Ampere | sm_86 | Supported by every current CUDA build. 8 GB VRAM: keep `micro_batch` small and use `accumulation_steps`. |

`resolve_device` checks that the card's `sm_XX` is in `torch.cuda.get_arch_list()` and refuses to start
otherwise, so a wrong wheel fails before any data is read.

## Pinned environment

- Python 3.13 (the build box version; 3.11+ is the package minimum).
- `requirements-gpu-cu132.txt`: torch 2.15.0+cu132 and torchvision 0.29.2+cu132 (the CUDA builds of the
  locked CPU versions). On 2026-11-25 download.pytorch.org lists these wheels for Linux x86_64 and
  Windows amd64. cu134 builds exist too.
- NVIDIA driver: a CUDA 13.x runtime needs the R580 driver branch or newer. Check the "CUDA Version" in
  `nvidia-smi` is at least 13.2 before installing.
- No CUDA toolkit is needed; the wheels bundle the runtime.

## Check a host (no training)

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); \
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())"
python -m whalemix_training train --config run.json   # with training.max_updates = 0: preflight only
```

The second command writes `train-plan.json` with the `device` block and the preflight counts and
changes no weights.

## Resume and hardware changes

- `train/finetune --resume` restores the model, optimizer, sampler/replay state, host RNGs, the CUDA RNG
  of the selected device and, for `fp16`, the gradient-scaler state. A same-device resume is recorded
  as `exact_resume: true` in `<stage>-plan.json`.
- A checkpoint from a different card, compute capability or precision is refused. Rerun with
  `--allow-hardware-transition` to continue anyway: the plan records `hardware_transition` and
  `exact_resume: false`, the checkpoint appends to `hardware_transitions`, and the CUDA RNG and scaler
  restart. For a planned move such as RTX 3070 to RTX 5090, prefer `finetune --parent` from the 3070
  checkpoint.
- `tests/test_cuda_resume.py` has two `cuda`-marked tests (fp32 and fp16). They skip without a GPU, also
  under `WHALEMIX_REQUIRE_FIXTURE=1`, so run them on the training host.

## Verified on (2026-11-25)

RTX 3070 Laptop GPU (sm_86), torch 2.15.0+cu132 in a separate virtual environment:

- `tests/test_cuda_resume.py`: 7 passed. 2+2 interrupted updates vs 4 straight updates gave identical
  weights (max abs diff 0.0) and identical sample logs in fp32 and fp16.
- `train` with `max_updates = 0` on `cuda:0`: finite outputs and masked loss; no weights written.
- Synthetic random-tensor steps (not training), fp32, input `[1, 257, 100]`: about 490 windows/s at
  micro-batch 16 (0.8 GB VRAM) and 600 to 630 windows/s at 32 to 64 (1.6 to 3.2 GB). Single-process
  window reading is about 70 windows/s, so a real run needs `loader_workers` or cached features.

## Not done yet

- No real training run. The verified labels are far too few for a grouped train/validation/test run.
- The RTX 5090 has not been tested.
