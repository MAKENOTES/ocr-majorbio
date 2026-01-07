from __future__ import annotations

import torch
from torch import nn


class DBNet(nn.Module):
    """Minimal DBNet skeleton. Replace with real implementation."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True):
        """初始化 DBNet 占位模型。

        Args:
            backbone: 主干网络名称。
            pretrained: 是否使用预训练权重。
        """
        super().__init__()
        self.backbone = backbone
        self.pretrained = pretrained
        self.dummy = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向计算，返回概率图（占位）。

        Args:
            x: 输入张量（B,C,H,W）。

        Returns:
            概率图张量。
        """
        return self.dummy(x)
