import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mmocr_dbnet_resnet18_contract.py")
    parser.add_argument("--work_dir", default="outputs/det_preds/mmocr_dbnet")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "third_party/mmocr/tools/train.py",
        args.config,
        "--work-dir",
        args.work_dir,
    ]
    if args.resume:
        cmd.append("--resume")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
