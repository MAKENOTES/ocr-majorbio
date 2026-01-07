import argparse

def main() -> None:
    """命令行入口：检测训练占位脚本。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    raise SystemExit("Training stub. Integrate full DBNet training here.")


if __name__ == "__main__":
    main()
