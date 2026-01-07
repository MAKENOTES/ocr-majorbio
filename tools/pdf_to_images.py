import argparse
import os
from pathlib import Path

import fitz  # PyMuPDF


def pdf_to_images(pdf_path: str, out_dir: str, dpi: int = 300) -> list[str]:
    """Render PDF pages to images; return image paths."""
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    out_paths = []

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = out_dir_path / f"page_{page_index + 1:04d}.png"
        pix.save(out_path)
        out_paths.append(str(out_path))

    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    paths = pdf_to_images(args.pdf, args.out, dpi=args.dpi)
    print(f"Rendered {len(paths)} pages to {args.out}")


if __name__ == "__main__":
    main()
