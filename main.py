import argparse

from pipeline.run_ocr import run_ocr_on_pdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    run_ocr_on_pdf(args.pdf, args.config)


if __name__ == "__main__":
    main()
