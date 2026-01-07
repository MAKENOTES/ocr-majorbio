from __future__ import annotations

from typing import List, Dict


def postprocess_dummy(image_w: int, image_h: int) -> List[Dict]:
    """占位后处理：返回空检测结果。

    Args:
        image_w: 图像宽度。
        image_h: 图像高度。

    Returns:
        空的检测结果列表。
    """
    return []
