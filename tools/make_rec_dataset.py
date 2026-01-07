import json
from pathlib import Path

from PIL import Image


def main() -> None:
    """脚本入口：按代码内默认参数裁剪识别训练样本。"""
    # 原图路径
    image_path = "data/images/000001.jpg"
    # 检测结果 JSON 路径
    det_json = "outputs/det_preds/sample_det.json"
    # 裁剪输出目录
    out_dir = Path("data/crops")

    # 加载原始页面图像
    image = Image.open(image_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取检测结果（包含 bbox）
    dets = json.load(open(det_json, "r", encoding="utf-8"))
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
