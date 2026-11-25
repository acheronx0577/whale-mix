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

## Not done yet

- No GPU run has used these pins. Nothing was installed on the owner's PC.
- Rented or local GPU host, driver version and VRAM budget are still the owner's decision.
