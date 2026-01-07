import json
from typing import List, Dict


def export_json(items: List[Dict], out_path: str) -> None:
    """将结构化 OCR 结果写入 JSON 文件。

    Args:
        items: 结构化 OCR 结果列表。
        out_path: 输出 JSON 路径。
    """
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
