from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import yaml
from PIL import Image

from tools.pdf_to_images import pdf_to_images
from detection.infer import detect_text
from recognition.infer import recognize_text
from pipeline.layout_sort import sort_by_reading_order
from pipeline.export import export_json


def _crop_image(image_path: str, bbox: List[int], out_dir: str, idx: int) -> str:
    image = Image.open(image_path)
    x1, y1, x2, y2 = bbox
    crop = image.crop((x1, y1, x2, y2))
    out_path = Path(out_dir) / f"crop_{idx:06d}.png"
    crop.save(out_path)
    return str(out_path)


def run_ocr_on_pdf(pdf_path: str, config_path: str) -> List[Dict]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dpi = cfg.get("pdf_dpi", 300)
    output_dir = Path(cfg.get("output_dir", "outputs/final"))
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = output_dir / "images"
    image_paths = pdf_to_images(pdf_path, str(image_dir), dpi=dpi)

    det_cfg = cfg.get("detection", {})
    rec_cfg = cfg.get("recognition", {})

    results: List[Dict] = []
    crop_dir = output_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    for page_idx, image_path in enumerate(image_paths, start=1):
        dets = detect_text(image_path, det_cfg.get("model_path", ""), det_cfg.get("conf_thresh", 0.3))
        for det_idx, det in enumerate(dets):
            bbox = det.get("bbox")
            if not bbox:
                continue
            crop_path = _crop_image(image_path, bbox, str(crop_dir), idx=len(results))
            rec = recognize_text(
                crop_path,
                rec_cfg.get("model_path", ""),
                rec_cfg.get("vocab_path", ""),
                img_height=rec_cfg.get("img_height", 32),
                max_width=rec_cfg.get("max_width", 1000),
            )
            results.append(
                {
                    "page": page_idx,
                    "bbox": bbox,
                    "text": rec.get("text", ""),
                    "score": rec.get("score", 0.0),
                }
            )

    results = sort_by_reading_order(results)
    export_json(results, str(output_dir / "result.json"))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    out = run_ocr_on_pdf(args.pdf, args.config)
    print(f"OCR done. Items: {len(out)}")


if __name__ == "__main__":
    main()
