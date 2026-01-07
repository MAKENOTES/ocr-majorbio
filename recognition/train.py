import argparse

def main() -> None:
    """命令行入口：识别训练占位脚本。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    raise SystemExit("Training stub. Integrate full CRNN-CTC training here.")


if __name__ == "__main__":
    main()
