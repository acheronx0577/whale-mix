"""Training/fine-tuning session loop with explicit update and time budgets.

The loop checkpoints at completed optimizer-update boundaries (every N updates or T seconds, at session
end and on pause) and advances the sampler cursor only for consumed micro-batches. A budget of zero updates
is a supported, explicit setup-only mode: it builds and checks everything and changes no weights.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from whalemix_training.losses import masked_bce
from whalemix_training.replay import DeterministicSampler

Batch = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


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
        return max(0, min(self.max_updates, self.session_max_updates) - done)


def build_optimizer(model: torch.nn.Module, training: dict[str, Any]) -> torch.optim.Optimizer:
    if training["optimizer"] != "adamw":
        raise ValueError("only adamw is configured for v1")
    return torch.optim.AdamW(
        model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"])
    )


def run_session(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler: DeterministicSampler,
    load_batch: Callable[[list[str]], Batch],
    budget: SessionBudget,
    micro_batch: int,
    on_checkpoint: Callable[[str, int], None],
    should_pause: Callable[[], bool] = lambda: False,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    allowed = budget.updates_allowed(0)
    if allowed == 0:
        return {"optimizer_updates": 0, "stopped": "zero_update_budget", "cursor": sampler.state_dict()}
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
        for _ in range(budget.accumulation_steps):
            ids = sampler.peek(micro_batch)
            x, y, mask = load_batch(ids)
            loss = masked_bce(model(x), y, mask) / budget.accumulation_steps
            loss.backward()
            sampler.mark_consumed(len(ids))
        optimizer.step()
        updates += 1
        since_ckpt += 1
        if (
            since_ckpt >= budget.checkpoint_every_updates
            or clock() - last_ckpt_time >= budget.checkpoint_every_seconds
        ):
            on_checkpoint("periodic", updates)
            since_ckpt, last_ckpt_time = 0, clock()
    on_checkpoint(stopped, updates)
    return {"optimizer_updates": updates, "stopped": stopped, "cursor": sampler.state_dict()}
