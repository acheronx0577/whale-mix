"""Model construction: EfficientNet-B0 from random initialisation with one shared multi-label head.

The architecture is the published EfficientNet-B0 (Tan and Le, 2019) from torchvision, instantiated with
`weights=None`. Only the stem convolution (to accept 1-channel spectrograms) and the classifier head (one
sigmoid output per whale target and broad sound-context label) are replaced. No parameter is frozen.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models import efficientnet_b0

from whalemix_training.labels import LabelMap, load_profiles

ARCHITECTURE = "efficientnet_b0"


class AcousticNet(nn.Module):
    def __init__(self, label_map: LabelMap, in_channels: int = 1) -> None:
        super().__init__()
        backbone = efficientnet_b0(weights=None)
        stem = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            in_channels,
            stem.out_channels,
            kernel_size=stem.kernel_size,
            stride=stem.stride,
            padding=stem.padding,
            bias=False,
        )
        head = backbone.classifier[1]
        backbone.classifier[1] = nn.Linear(head.in_features, len(label_map.outputs))
        self.backbone = backbone
        self.label_map = label_map
        self.in_channels = in_channels

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Return logits [N, outputs]."""
        return self.backbone(spectrogram)


class ProbabilityHead(nn.Module):
    """Export wrapper: the deployed graph emits sigmoid probabilities in label-map order."""

    def __init__(self, net: AcousticNet) -> None:
        super().__init__()
        self.net = net

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(spectrogram))


def build_model(label_map: LabelMap, in_channels: int = 1, initialization: str = "random") -> AcousticNet:
    if initialization != "random":
        raise ValueError(
            "the primary lineage starts from random initialisation; pretrained runs are separate"
        )
    model = AcousticNet(label_map, in_channels)
    frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
    if frozen:
        raise AssertionError(f"randomly initialised parameters must not be frozen: {frozen[:3]}")
    return model


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def profile_scores(
    probabilities: torch.Tensor, label_map: LabelMap, profile_id: str
) -> dict[str, torch.Tensor]:
    """Select one profile's outputs from the shared head. Planned profiles have no operational outputs."""
    for profile in load_profiles()["profiles"]:
        if profile["id"] == profile_id:
            if profile["status"] != "required" or not profile["outputs"]:
                raise ValueError(
                    f"profile {profile_id!r} is {profile['status']} and has no operational outputs"
                )
            return {name: probabilities[:, label_map.index(name)] for name in profile["outputs"]}
    raise KeyError(f"unknown profile {profile_id!r}")


def describe(model: AcousticNet) -> dict[str, Any]:
    return {
        "architecture": ARCHITECTURE,
        "architecture_citation": "Tan & Le, EfficientNet, ICML 2019 (torchvision efficientnet_b0)",
        "in_channels": model.in_channels,
        "outputs": list(model.label_map.outputs),
        "parameters": parameter_count(model),
        "initialization": "random",
    }
