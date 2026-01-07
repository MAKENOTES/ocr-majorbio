from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset


def _load_labels(label_path: str) -> List[Tuple[str, str]]:
    """加载识别标注文件，返回 (图片名, 文本) 列表。

    Args:
        label_path: 标注文件路径。

    Returns:
        (图片名, 文本) 列表。
    """
    pairs: List[Tuple[str, str]] = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            img_name, text = line.split("\t", 1)
            pairs.append((img_name, text))
    return pairs


class RecDataset(Dataset):
    def __init__(self, images_dir: str, label_path: str):
        """构建识别数据集对象。

        Args:
            images_dir: 图片目录。
            label_path: 标注文件路径。
        """
        self.images_dir = Path(images_dir)
        self.pairs = _load_labels(label_path)

    def __len__(self) -> int:
        """返回数据集长度。"""
        return len(self.pairs)

    def __getitem__(self, idx: int):
        """返回指定索引的数据样本（图像、文本、路径）。

        Args:
            idx: 数据索引。

        Returns:
            (image, text, image_path)。
        """
        img_name, text = self.pairs[idx]
        img_path = self.images_dir / img_name
        image = Image.open(img_path).convert("L")
        return image, text, str(img_path)
