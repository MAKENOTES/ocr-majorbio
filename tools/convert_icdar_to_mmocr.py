import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

from PIL import Image


def _parse_icdar_label(
    label_path: Path,
    ignore_tokens: List[str],
    assume_bbox_only: bool,
) -> List[Dict]:
    """解析 ICDAR 标注文本，输出 MMOCR 所需的 instances 列表。

    Args:
        label_path: 标注文件路径。
        ignore_tokens: 忽略文本标记列表（如 ###）。
        assume_bbox_only: 是否仅按 x1 y1 x2 y2 解析。

    Returns:
        MMOCR 需要的 instances 列表。
    """
    instances: List[Dict] = []
    if not label_path.exists():
        return instances

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not assume_bbox_only and "," in line:
                # ICDAR Localization: x1,y1,...,x4,y4,text
                parts = line.split(",")
                if len(parts) < 9:
                    continue
                coords = list(map(float, parts[:8]))
                text = ",".join(parts[8:]).strip()
                polygon = coords
                xs = coords[0::2]
                ys = coords[1::2]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                ignore = text in ignore_tokens
            else:
                # 兼容简单四点框：x1 y1 x2 y2（无文字）
                parts = line.split()
                if len(parts) < 4:
                    continue
                x1, y1, x2, y2 = map(float, parts[:4])
                polygon = [x1, y1, x2, y1, x2, y2, x1, y2]
                text = ""
                ignore = False
            instances.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "bbox_label": 0,
                    "polygon": polygon,
                    "ignore": ignore,
                    "text": text,
                }
            )
    return instances


def _build_data_list(
    images: List[Path],
    labels_dir: Path,
    label_prefix: str,
    ignore_tokens: List[str],
    assume_bbox_only: bool,
) -> List[Dict]:
    """构建 MMOCR 的 data_list 结构。

    Args:
        images: 图像路径列表。
        labels_dir: 标注目录。
        label_prefix: 标注文件前缀。
        ignore_tokens: 忽略文本标记列表。
        assume_bbox_only: 是否仅按矩形框解析。

    Returns:
        MMOCR 的 data_list 列表。
    """
    data_list: List[Dict] = []
    for image_path in images:
        # 读取图像尺寸，写入到 annotation 中
        with Image.open(image_path) as im:
            width, height = im.size
        label_name = f"{label_prefix}{image_path.stem}.txt"
        instances = _parse_icdar_label(
            labels_dir / label_name,
            ignore_tokens,
            assume_bbox_only=assume_bbox_only,
        )
        data_list.append(
            {
                "img_path": str(image_path.as_posix()),
                "height": height,
                "width": width,
                "instances": instances,
            }
        )
    return data_list


def _write_json(out_path: Path, data_list: List[Dict]) -> None:
    """写出 MMOCR JSON 文件（含 metainfo + data_list）。

    Args:
        out_path: 输出 JSON 路径。
        data_list: MMOCR data_list 数据。
    """
    out = {
        "metainfo": {
            "dataset_type": "custom_icdar",
            "task_name": "textdet",
            "classes": ["text"],
        },
        "data_list": data_list,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def main() -> None:
    """命令行入口：转换 ICDAR 标注为 MMOCR JSON。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--labels_dir", required=True)
    parser.add_argument("--label_prefix", default="gt_")
    parser.add_argument("--out_train_json", required=True)
    parser.add_argument("--out_val_json", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    # 如果标注仅为 x1 y1 x2 y2（无 polygon），启用此开关
    parser.add_argument("--bbox_only", action="store_true")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])

    # 按固定随机种子打乱，便于稳定划分训练/验证
    random.seed(args.seed)
    random.shuffle(images)

    split_idx = int(len(images) * (1 - args.val_ratio))
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # ICDAR 常用的忽略标记
    ignore_tokens = ["###"]

    train_list = _build_data_list(
        train_images,
        labels_dir,
        args.label_prefix,
        ignore_tokens,
        assume_bbox_only=args.bbox_only,
    )
    val_list = _build_data_list(
        val_images,
        labels_dir,
        args.label_prefix,
        ignore_tokens,
        assume_bbox_only=args.bbox_only,
    )

    # 输出 JSON 文件
    _write_json(Path(args.out_train_json), train_list)
    _write_json(Path(args.out_val_json), val_list)

    print(f"Train images: {len(train_images)}, Val images: {len(val_images)}")


if __name__ == "__main__":
    main()
