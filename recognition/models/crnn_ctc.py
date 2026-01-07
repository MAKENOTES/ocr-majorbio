from __future__ import annotations

import torch
from torch import nn


class CRNNCTC(nn.Module):
    """Minimal CRNN-CTC skeleton. Replace with real implementation."""

    def __init__(self, num_classes: int, hidden_size: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.rnn = nn.LSTM(64, hidden_size, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.cnn(x)
        b, c, h, w = feats.size()
        feats = feats.view(b, c * h, w).permute(2, 0, 1)
        seq, _ = self.rnn(feats)
        out = self.fc(seq)
        return out
