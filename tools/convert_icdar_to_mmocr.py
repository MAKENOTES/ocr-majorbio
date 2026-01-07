import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

from PIL import Image


def _parse_icdar_label(label_path: Path, ignore_tokens: List[str]) -> List[Dict]:
    instances: List[Dict] = []
    if not label_path.exists():
        return instances

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "," in line:
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


def _build_data_list(images: List[Path], labels_dir: Path, label_prefix: str, ignore_tokens: List[str]) -> List[Dict]:
    data_list: List[Dict] = []
    for image_path in images:
        with Image.open(image_path) as im:
            width, height = im.size
        label_name = f"{label_prefix}{image_path.stem}.txt"
        instances = _parse_icdar_label(labels_dir / label_name, ignore_tokens)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--labels_dir", required=True)
    parser.add_argument("--label_prefix", default="gt_")
    parser.add_argument("--out_train_json", required=True)
    parser.add_argument("--out_val_json", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])

    random.seed(args.seed)
    random.shuffle(images)

    split_idx = int(len(images) * (1 - args.val_ratio))
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    ignore_tokens = ["###"]

    train_list = _build_data_list(train_images, labels_dir, args.label_prefix, ignore_tokens)
    val_list = _build_data_list(val_images, labels_dir, args.label_prefix, ignore_tokens)

    _write_json(Path(args.out_train_json), train_list)
    _write_json(Path(args.out_val_json), val_list)

    print(f"Train images: {len(train_images)}, Val images: {len(val_images)}")


if __name__ == "__main__":
    main()
