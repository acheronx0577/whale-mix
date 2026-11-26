"""Explicit training device and precision.

`training.device` is `cpu` (default) or `cuda`; nothing falls back silently. Asking for CUDA when this
torch build cannot use the GPU is an error that names the cause (no CUDA build, no visible device, or no
kernels for the card's architecture, e.g. sm_120 on RTX 50-series needs a cu128+ build). Every batch
reaches the model through `move_batch`, and the resolved device is written to the run ledger.
"""

from __future__ import annotations

import contextlib
import platform
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

DEVICES = ("cpu", "cuda")
PRECISIONS = ("fp32", "bf16", "fp16")


class DeviceError(RuntimeError):
    """The requested device or precision cannot be used; nothing was trained."""


@dataclass(frozen=True)
class DeviceInfo:
    requested: str
    device: torch.device
    precision: str
    name: str
    capability: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "actual": str(self.device),
            "precision": self.precision,
            "name": self.name,
            "capability": self.capability,
            "torch": str(torch.__version__),
            "torch_cuda_build": None if torch.version.cuda is None else str(torch.version.cuda),
        }

    @property
    def autocast_dtype(self) -> torch.dtype | None:
        return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(self.precision)


def resolve_device(training: dict[str, Any]) -> DeviceInfo:
    requested = str(training.get("device", "cpu"))
    precision = str(training.get("precision", "fp32"))
    if requested not in DEVICES:
        raise DeviceError(f"training.device must be one of {', '.join(DEVICES)}")
    if precision not in PRECISIONS:
        raise DeviceError(f"training.precision must be one of {', '.join(PRECISIONS)}")
    if requested == "cpu":
        if precision != "fp32":
            raise DeviceError("cpu training runs in fp32; set training.precision to fp32")
        return DeviceInfo("cpu", torch.device("cpu"), "fp32", platform.processor() or "cpu", None)
    if not torch.cuda.is_available():
        raise DeviceError(
            f"training.device is 'cuda' but CUDA is unavailable (torch {torch.__version__}, CUDA build "
            f"{torch.version.cuda or 'none'}); install a CUDA build of torch on the GPU host "
            "(app/training/GPU-Setup.md) or set training.device to 'cpu'"
        )
    index = int(training.get("cuda_index", 0))
    if not 0 <= index < torch.cuda.device_count():
        raise DeviceError(f"training.cuda_index {index} is not a visible CUDA device")
    major, minor = torch.cuda.get_device_capability(index)
    arch = f"sm_{major}{minor}"
    name = torch.cuda.get_device_name(index)
    if arch not in torch.cuda.get_arch_list():
        raise DeviceError(
            f"{name} is {arch}, but this torch build has kernels only for "
            f"{' '.join(torch.cuda.get_arch_list())}; install a torch build that includes {arch}"
        )
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise DeviceError(f"{name} does not support bf16; use fp16 or fp32")
    return DeviceInfo("cuda", torch.device(f"cuda:{index}"), precision, name, arch)


def move_batch(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    """The single host-to-device path for training and setup batches."""
    non_blocking = device.type == "cuda"
    return tuple(t.to(device, non_blocking=non_blocking) for t in batch)


@contextlib.contextmanager
def autocast(info: DeviceInfo) -> Iterator[None]:
    dtype = info.autocast_dtype
    if info.device.type == "cuda" and dtype is not None:
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield
    else:
        yield
