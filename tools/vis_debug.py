import argparse
from pathlib import Path
from typing import List, Dict
import json

from PIL import Image, ImageDraw


def draw_boxes(image_path: str, dets: List[Dict], out_path: str) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for det in dets:
        bbox = det.get("bbox")
        if bbox:
            draw.rectangle(bbox, outline="red", width=2)
    image.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dets = json.load(open(args.json, "r", encoding="utf-8"))
    draw_boxes(args.image, dets, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
