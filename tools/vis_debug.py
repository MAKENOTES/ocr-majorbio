import argparse
from pathlib import Path
from typing import List, Dict
import json

from PIL import Image, ImageDraw


def draw_boxes(image_path: str, dets: List[Dict], out_path: str) -> None:
    """在图像上绘制检测框并保存到指定路径。

    Args:
        image_path: 原图路径。
        dets: 检测结果列表（含 bbox）。
        out_path: 输出图像路径。
    """
    # 在原图上绘制检测框，便于快速检查检测质量
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for det in dets:
        bbox = det.get("bbox")
        if bbox:
            draw.rectangle(bbox, outline="red", width=2)
    image.save(out_path)


def main() -> None:
    """命令行入口：可视化检测 JSON 到图像。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    # 读取检测结果并可视化
    # 读取 JSON 检测结果
    dets = json.load(open(args.json, "r", encoding="utf-8"))
    draw_boxes(args.image, dets, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
