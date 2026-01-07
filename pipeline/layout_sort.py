from __future__ import annotations

from typing import List, Dict


def sort_by_reading_order(items: List[Dict]) -> List[Dict]:
    """按阅读顺序排序（先 y 后 x）。

    Args:
        items: 包含 bbox 的结果列表。

    Returns:
        排序后的列表。
    """
    def key_fn(item: Dict):
        bbox = item.get("bbox", [0, 0, 0, 0])
        return (bbox[1], bbox[0])

    return sorted(items, key=key_fn)
