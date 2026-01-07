from __future__ import annotations

import argparse
from typing import List, Dict

import torch
from PIL import Image
from torchvision import transforms

from detection.models.dbnet import DBNet
from detection.postprocess import postprocess_dummy


def detect_text(image_path: str, model_path: str, conf_thresh: float = 0.3) -> List[Dict]:
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    model = DBNet()
    if model_path and torch.cuda.is_available():
        state = torch.load(model_path, map_location="cpu")
        if isinstance(state, dict):
            model.load_state_dict(state, strict=False)
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    x = transform(image).unsqueeze(0)
    with torch.no_grad():
        _ = model(x)

    return postprocess_dummy(w, h)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--conf", type=float, default=0.3)
    args = parser.parse_args()

    dets = detect_text(args.image, args.model, conf_thresh=args.conf)
    print(dets)


if __name__ == "__main__":
    main()
