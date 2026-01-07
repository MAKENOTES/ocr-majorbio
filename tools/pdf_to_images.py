import argparse
import os
from pathlib import Path

import fitz  # PyMuPDF


def pdf_to_images(pdf_path: str, out_dir: str, dpi: int = 300) -> list[str]:
    """将 PDF 渲染为页面图像并返回输出路径列表。

    Args:
        pdf_path: PDF 文件路径。
        out_dir: 输出图像目录。
        dpi: 渲染分辨率（默认 300）。

    Returns:
        渲染后的图像路径列表。
    """
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # 打开 PDF 文档
    doc = fitz.open(pdf_path)
    # PyMuPDF 默认 72 DPI，这里按目标 DPI 放大
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    out_paths = []

    for page_index in range(len(doc)):
        # 逐页渲染为像素图
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out_path = out_dir_path / f"page_{page_index + 1:04d}.png"
        pix.save(out_path)
        out_paths.append(str(out_path))

    return out_paths


def main() -> None:
    """命令行入口：将指定 PDF 渲染为图像。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    # 执行渲染并输出结果路径
    paths = pdf_to_images(args.pdf, args.out, dpi=args.dpi)
    print(f"Rendered {len(paths)} pages to {args.out}")


if __name__ == "__main__":
    main()
