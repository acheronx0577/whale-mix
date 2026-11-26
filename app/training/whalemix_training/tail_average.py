"""Opt-in late-training schedule and tail weight averaging (both off unless configured).

`training.lr_schedule` = {"kind": "cosine", "start_update": S, "end_update": E, "final_factor": f} sets the
learning rate of every update from its global index alone: the base rate before S, a cosine from the base
rate down to `f` x base between S and E, and `f` x base from E on. Because it depends only on the global
update, a `--resume` at any update continues the same schedule exactly.

`training.tail_average` = {"start_update": S, "every_updates": k, "bn_refresh_batches": n} keeps a uniform
running mean of the model state after updates S, S + k, S + 2k, ... (stored in every training-state
checkpoint, so resuming continues the same mean). When the run reaches `max_updates`, the averaged weights
get fresh BatchNorm statistics from `n` seeded micro-batches of reviewed TRAIN windows only (other layers in
eval mode, no gradients) and are saved as their own scorable checkpoint; the live training state is untouched.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

LR_SCHEDULE_KINDS = ("cosine",)


def validate_lr_schedule(schedule: dict[str, Any] | None) -> dict[str, Any] | None:
    if schedule is None:
        return None
    kind = schedule.get("kind")
    if kind not in LR_SCHEDULE_KINDS:
        raise ValueError(f"training.lr_schedule.kind must be one of {LR_SCHEDULE_KINDS}")
    start, end = int(schedule["start_update"]), int(schedule["end_update"])
    final = float(schedule["final_factor"])
    if not 0 <= start < end:
        raise ValueError("training.lr_schedule needs 0 <= start_update < end_update")
    if not 0.0 < final <= 1.0:
        raise ValueError("training.lr_schedule.final_factor must be in (0, 1]")
    return {"kind": kind, "start_update": start, "end_update": end, "final_factor": final}


def lr_factor(update_index: int, schedule: dict[str, Any] | None) -> float:
    """Multiplier of the base rate for the update with 0-based global index `update_index` (the update
    that produces global update `update_index + 1`)."""
    if schedule is None:
        return 1.0
    start, end, final = schedule["start_update"], schedule["end_update"], schedule["final_factor"]
    if update_index < start:
        return 1.0
    if update_index >= end:
        return final
    t = (update_index - start) / (end - start)
    return final + (1.0 - final) * 0.5 * (1.0 + math.cos(math.pi * t))


def lr_function(base_lr: float, schedule: dict[str, Any] | None) -> Callable[[int], float] | None:
    if schedule is None:
        return None
    return lambda update_index: base_lr * lr_factor(update_index, schedule)


class TailAverage:
    """Uniform running mean of the model state at the scheduled tail updates."""

    def __init__(self, start_update: int, every_updates: int, bn_refresh_batches: int) -> None:
        if start_update < 1 or every_updates < 1 or bn_refresh_batches < 0:
            raise ValueError(
                "training.tail_average needs start_update >= 1, every_updates >= 1, bn_refresh_batches >= 0"
            )
        self.start_update = start_update
        self.every_updates = every_updates
        self.bn_refresh_batches = bn_refresh_batches
        self.count = 0
        self.last_update = 0
        self.state: dict[str, torch.Tensor] = {}

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> TailAverage | None:
        if cfg is None:
            return None
        return cls(int(cfg["start_update"]), int(cfg["every_updates"]), int(cfg.get("bn_refresh_batches", 0)))

    def config(self) -> dict[str, int]:
        return {
            "start_update": self.start_update,
            "every_updates": self.every_updates,
            "bn_refresh_batches": self.bn_refresh_batches,
        }

    def due(self, update: int) -> bool:
        return update >= self.start_update and (update - self.start_update) % self.every_updates == 0

    def expected_count(self, max_updates: int) -> int:
        if max_updates < self.start_update:
            return 0
        return (max_updates - self.start_update) // self.every_updates + 1

    def observe(self, update: int, model: nn.Module) -> bool:
        """Fold the current model state into the mean when `update` is a scheduled snapshot."""
        if not self.due(update) or update <= self.last_update:
            return False
        self.count += 1
        for key, value in model.state_dict().items():
            current = value.detach().to("cpu")
            if not current.is_floating_point():
                self.state[key] = current.clone()  # integer buffers (BN batch counters): latest value
            elif key not in self.state:
                self.state[key] = current.to(torch.float64).clone()
            else:
                self.state[key] += (current.to(torch.float64) - self.state[key]) / self.count
        self.last_update = update
        return True

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": self.config(),
            "count": self.count,
            "last_update": self.last_update,
            "state": self.state,
        }

    def load_state(self, saved: dict[str, Any] | None, resume_update: int) -> None:
        if saved is None:
            if resume_update >= self.start_update:
                raise ValueError("checkpoint has no tail-average state but the tail window has started")
            return
        if saved["config"] != self.config():
            raise ValueError("tail_average settings differ from the checkpoint")
        if int(saved["last_update"]) > resume_update:
            raise ValueError("tail-average state is ahead of the checkpoint")
        self.count = int(saved["count"])
        self.last_update = int(saved["last_update"])
        self.state = {k: v.clone() for k, v in saved["state"].items()}

    def averaged_state(self, reference: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.count == 0:
            raise ValueError("no tail snapshots were taken")
        return {key: self.state[key].to(dtype=value.dtype) for key, value in reference.items()}


def bn_refresh_ids(train_ids: list[str], batches: int, micro_batch: int, seed: int) -> list[list[str]]:
    """Seeded, sorted-input choice of reviewed train windows for the BatchNorm refresh."""
    pool = sorted(train_ids)
    n = min(len(pool), batches * micro_batch)
    if n == 0:
        return []
    rng = np.random.default_rng([int(seed), 0xB4])
    chosen = [pool[i] for i in rng.choice(len(pool), size=n, replace=False)]
    return [chosen[i : i + micro_batch] for i in range(0, n, micro_batch)]


def has_batchnorm(model: nn.Module) -> bool:
    return any(isinstance(m, _BatchNorm) for m in model.modules())


@torch.no_grad()
def refresh_batchnorm(
    model: nn.Module, batches: list[list[str]], load: Callable[[list[str]], torch.Tensor]
) -> int:
    """Recompute BatchNorm running statistics as an exact cumulative mean over `batches`, with every other
    layer in eval mode (no dropout or stochastic depth). Returns the number of batches used."""
    norms = [m for m in model.modules() if isinstance(m, _BatchNorm)]
    if not norms or not batches:
        return 0
    model.eval()
    momenta = [m.momentum for m in norms]
    for m in norms:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    try:
        for ids in batches:
            model(load(ids))
    finally:
        for m, momentum in zip(norms, momenta, strict=True):
            m.momentum = momentum
        model.eval()
    return len(batches)
