from __future__ import annotations

import argparse
from typing import Dict

import torch
from PIL import Image
from torchvision import transforms

from recognition.decode import load_vocab, ctc_greedy_decode
from recognition.models.crnn_ctc import CRNNCTC


def _prepare_image(image: Image.Image, img_height: int, max_width: int) -> torch.Tensor:
    w, h = image.size
    new_w = min(int(w * (img_height / h)), max_width)
    image = image.resize((new_w, img_height))
    tensor = transforms.ToTensor()(image)
    if tensor.size(0) == 3:
        tensor = tensor.mean(dim=0, keepdim=True)
    return tensor


def recognize_text(crop_path: str, model_path: str, vocab_path: str, img_height: int = 32, max_width: int = 1000) -> Dict:
    image = Image.open(crop_path).convert("L")
    vocab = load_vocab(vocab_path)

    model = CRNNCTC(num_classes=len(vocab))
    if model_path and torch.cuda.is_available():
        state = torch.load(model_path, map_location="cpu")
        if isinstance(state, dict):
            model.load_state_dict(state, strict=False)
    model.eval()

    x = _prepare_image(image, img_height, max_width).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
    text = ctc_greedy_decode(logits, vocab)
    return {"text": text, "score": 1.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--img_height", type=int, default=32)
    parser.add_argument("--max_width", type=int, default=1000)
    args = parser.parse_args()

    out = recognize_text(args.image, args.model, args.vocab, args.img_height, args.max_width)
    print(out)


if __name__ == "__main__":
    main()
