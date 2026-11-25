"""Masked multi-label binary cross-entropy. Masked (unassessed/unreviewed) targets contribute nothing."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.shape != targets.shape or logits.shape != mask.shape:
        raise ValueError("logits, targets and mask must share a shape")
    per_item = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    supervised = mask.sum()
    if supervised == 0:
        return (per_item * 0.0).sum()
    return (per_item * mask).sum() / supervised


def training_pos_weight(targets: torch.Tensor, mask: torch.Tensor, cap: float = 50.0) -> torch.Tensor:
    """Per-output positive weights derived from supervised training targets only (recorded in the config)."""
    pos = (targets * mask).sum(dim=0)
    neg = ((1 - targets) * mask).sum(dim=0)
    weight = torch.where(pos > 0, neg / pos.clamp(min=1), torch.ones_like(pos))
    return weight.clamp(min=1.0, max=cap)
