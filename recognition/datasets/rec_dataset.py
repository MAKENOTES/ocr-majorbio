from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset


def _load_labels(label_path: str) -> List[Tuple[str, str]]:
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
        self.images_dir = Path(images_dir)
        self.pairs = _load_labels(label_path)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_name, text = self.pairs[idx]
        img_path = self.images_dir / img_name
        image = Image.open(img_path).convert("L")
        return image, text, str(img_path)
