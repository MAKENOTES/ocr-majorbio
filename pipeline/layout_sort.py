from __future__ import annotations

from typing import List, Dict


def sort_by_reading_order(items: List[Dict]) -> List[Dict]:
    def key_fn(item: Dict):
        bbox = item.get("bbox", [0, 0, 0, 0])
        return (bbox[1], bbox[0])

    return sorted(items, key=key_fn)
