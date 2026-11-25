"""Training/fine-tuning session loop with explicit update and time budgets.

The loop checkpoints at completed optimizer-update boundaries (every N updates or T seconds, at session
end and on pause) and advances the sampler cursor only for consumed micro-batches. A budget of zero updates
is a supported, explicit setup-only mode: it builds and checks everything and changes no weights.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from whalemix_training.device import DeviceInfo, autocast, move_batch, resolve_device
from whalemix_training.losses import masked_bce
from whalemix_training.replay import DeterministicSampler

Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class BatchSource(Protocol):
    """DeterministicSampler or ReplayMixer: draw IDs, then consume them once the micro-batch is used."""

    def draw(self, n: int) -> list[str]: ...
    def consume(self) -> None: ...
    def state_dict(self) -> dict[str, Any]: ...
    def stats(self) -> dict[str, Any]: ...


@dataclass
class SessionBudget:
    max_updates: int
    session_max_updates: int
    session_max_seconds: float
    checkpoint_every_updates: int
    checkpoint_every_seconds: float
    accumulation_steps: int = 1

    @classmethod
    def from_config(cls, training: dict[str, Any]) -> SessionBudget:
        return cls(
            max_updates=int(training["max_updates"]),
            session_max_updates=int(training["session_max_updates"]),
            session_max_seconds=float(training["session_max_seconds"]),
            checkpoint_every_updates=int(training["checkpoint_every_updates"]),
            checkpoint_every_seconds=float(training["checkpoint_every_seconds"]),
            accumulation_steps=int(training["accumulation_steps"]),
        )

    def updates_allowed(self, done: int) -> int:
        """Updates this session may run after `done` global updates: the per-session cap applies to the
        remaining global budget, so resumed sessions keep going until `max_updates` is reached."""
        return max(0, min(self.max_updates - done, self.session_max_updates))


def build_optimizer(model: torch.nn.Module, training: dict[str, Any]) -> torch.optim.Optimizer:
    if training["optimizer"] != "adamw":
        raise ValueError("only adamw is configured for v1")
    return torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )


def run_session(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler: BatchSource | DeterministicSampler,
    load_batch: Callable[[list[str]], Batch],
    budget: SessionBudget,
    micro_batch: int,
    on_checkpoint: Callable[[str, int], None],
    should_pause: Callable[[], bool] = lambda: False,
    clock: Callable[[], float] = time.monotonic,
    device: DeviceInfo | None = None,
    start_update: int = 0,
    on_update: Callable[[int, list[str]], None] | None = None,
    scaler: Any = None,
) -> dict[str, Any]:
    """Run up to the remaining update budget. `on_checkpoint(reason, global_update)` is called only at
    completed, finite updates, so a non-finite loss stops the session and leaves the last good checkpoint
    in place."""
    allowed = budget.updates_allowed(start_update)
    if allowed == 0:
        return {
            "optimizer_updates": 0,
            "stopped": "zero_update_budget",
            "cursor": sampler.state_dict(),
            "mix": sampler.stats(),
        }
    info = device or resolve_device({})
    model.to(info.device)
    if scaler is None and info.precision == "fp16":
        scaler = torch.amp.GradScaler("cuda")
    model.train()
    start = last_ckpt_time = clock()
    updates = since_ckpt = 0
    stopped = "update_budget"
    while updates < allowed:
        if should_pause():
            stopped = "paused"
            break
        if clock() - start >= budget.session_max_seconds:
            stopped = "session_time"
            break
        optimizer.zero_grad(set_to_none=True)
        finite = True
        update_ids: list[str] = []
        for _ in range(budget.accumulation_steps):
            ids = sampler.draw(micro_batch)
            update_ids.extend(ids)
            x, y, mask = move_batch(load_batch(ids), info.device)
            with autocast(info):
                logits = model(x)
            loss = masked_bce(logits.float(), y, mask) / budget.accumulation_steps
            if not bool(torch.isfinite(loss)):
                finite = False
                break
            (scaler.scale(loss) if scaler else loss).backward()
            sampler.consume()
        if not finite:
            optimizer.zero_grad(set_to_none=True)
            return {
                "optimizer_updates": updates,
                "stopped": "non_finite_loss",
                "cursor": sampler.state_dict(),
                "mix": sampler.stats(),
            }
        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        updates += 1
        since_ckpt += 1
        if on_update is not None:
            on_update(start_update + updates, update_ids)
        if (
            since_ckpt >= budget.checkpoint_every_updates
            or clock() - last_ckpt_time >= budget.checkpoint_every_seconds
        ):
            on_checkpoint("periodic", start_update + updates)
            since_ckpt, last_ckpt_time = 0, clock()
    on_checkpoint(stopped, start_update + updates)
    return {
        "optimizer_updates": updates,
        "stopped": stopped,
        "cursor": sampler.state_dict(),
        "mix": sampler.stats(),
    }
