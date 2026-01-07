import argparse
import json
from pathlib import Path

from PIL import Image


def main() -> None:
    """命令行入口：根据检测框裁剪识别训练样本。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--det_json", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    # 加载原始页面图像
    image = Image.open(args.image)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取检测结果（包含 bbox）
    dets = json.load(open(args.det_json, "r", encoding="utf-8"))
    for i, det in enumerate(dets):
        bbox = det.get("bbox")
        if not bbox:
            continue
        # 按检测框裁剪，生成识别训练样本
        crop = image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        crop.save(out_dir / f"crop_{i:06d}.png")

    print(f"Cropped {len(dets)} items to {out_dir}")


if __name__ == "__main__":
    main()
