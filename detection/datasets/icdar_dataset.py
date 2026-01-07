from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset


@dataclass
class ICDARBox:
    points: List[Tuple[int, int]]
    text: str


def _parse_icdar_label(label_path: str) -> List[ICDARBox]:
    """解析 ICDAR 标注文件，返回文本框列表。

    Args:
        label_path: 标注文件路径。

    Returns:
        ICDARBox 列表。
    """
    boxes: List[ICDARBox] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 9:
                continue
            coords = list(map(int, parts[:8]))
            text = ",".join(parts[8:])
            points = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
            boxes.append(ICDARBox(points=points, text=text))
    return boxes


class ICDARDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str, label_prefix: str = ""):
        """构建 ICDAR 数据集对象。

        Args:
            images_dir: 图像目录路径。
            labels_dir: 标注目录路径。
            label_prefix: 标注文件名前缀。
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.label_prefix = label_prefix
        self.images = sorted([p for p in self.images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])

    def __len__(self) -> int:
        """返回数据集长度。"""
        return len(self.images)

    def __getitem__(self, idx: int):
        """返回指定索引的数据样本（图像、标注、路径）。

        Args:
            idx: 数据索引。

        Returns:
            (image, boxes, image_path)。
        """
        image_path = self.images[idx]
        label_name = f"{self.label_prefix}{image_path.stem}.txt"
        label_path = self.labels_dir / label_name
        image = Image.open(image_path).convert("RGB")
        boxes = _parse_icdar_label(str(label_path)) if label_path.exists() else []
        return image, boxes, str(image_path)
