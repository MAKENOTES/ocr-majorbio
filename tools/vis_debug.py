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
    """脚本入口：按代码内默认参数可视化检测结果。"""
    # 原图路径
    image_path = "data/images/000001.jpg"
    # 检测 JSON 路径
    det_json = "outputs/det_preds/sample_det.json"
    # 输出路径
    out_path = "outputs/det_preds/vis_000001.jpg"

    # 读取检测结果并可视化
    dets = json.load(open(det_json, "r", encoding="utf-8"))
    draw_boxes(image_path, dets, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
