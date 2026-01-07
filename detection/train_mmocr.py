import argparse
import subprocess
import sys


def main() -> None:
    """命令行入口：调用 MMOCR 训练脚本进行检测训练。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mmocr_dbnet_resnet18_contract.py")
    parser.add_argument("--work_dir", default="outputs/det_preds/mmocr_dbnet")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    # 使用 MMOCR 自带训练入口，保持和官方流程一致
    cmd = [
        sys.executable,
        "third_party/mmocr/tools/train.py",
        args.config,
        "--work-dir",
        args.work_dir,
    ]
    # 支持断点恢复训练
    if args.resume:
        cmd.append("--resume")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
