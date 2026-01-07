import json
from typing import List, Dict


def export_json(items: List[Dict], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
