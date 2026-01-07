from __future__ import annotations

import torch
from torch import nn


class DBNet(nn.Module):
    """Minimal DBNet skeleton. Replace with real implementation."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        self.backbone = backbone
        self.pretrained = pretrained
        self.dummy = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dummy(x)
