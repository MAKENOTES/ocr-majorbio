import argparse
import os
import re
from pathlib import Path


def _find_latest_log(work_dir: Path) -> Path:
    """查找工作目录下最新的训练日志文件路径。

    Args:
        work_dir: 训练工作目录。

    Returns:
        最新日志文件路径。
    """
    # 在工作目录下寻找最新的日志文件
    log_paths = sorted(work_dir.rglob("*.log"))
    if not log_paths:
        raise FileNotFoundError(f"No log files found under {work_dir}")
    return log_paths[-1]


def _parse_best_epoch(log_path: Path) -> tuple[int, float]:
    """解析日志中的验证集指标，返回最佳 epoch 与 hmean。

    Args:
        log_path: 训练日志路径。

    Returns:
        (最佳 epoch, 最佳 hmean)。
    """
    # 解析日志中的验证集 hmean，选择最佳 epoch
    best_epoch = -1
    best_hmean = -1.0
    pattern = re.compile(r"Epoch\(val\) \[(\d+)\]\[\d+/\d+\]\s+icdar/precision: ([0-9.]+)\s+icdar/recall: ([0-9.]+)\s+icdar/hmean: ([0-9.]+)")

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if not match:
                continue
            epoch = int(match.group(1))
            hmean = float(match.group(4))
            if hmean > best_hmean:
                best_hmean = hmean
                best_epoch = epoch

    if best_epoch < 0:
        raise ValueError(f"No validation metrics found in {log_path}")
    return best_epoch, best_hmean


def main() -> None:
    """命令行入口：从日志中选出最佳 epoch 并生成 best.pth。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--log", default=None)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    # 未指定日志时，默认取最新日志
    log_path = Path(args.log) if args.log else _find_latest_log(work_dir)

    best_epoch, best_hmean = _parse_best_epoch(log_path)
    ckpt_name = f"epoch_{best_epoch}.pth"
    ckpt_path = work_dir / ckpt_name

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # 创建 best.pth 软链接，方便后续推理/导出
    out_path = work_dir / "best.pth"
    if out_path.exists() or out_path.is_symlink():
        out_path.unlink()
    os.symlink(ckpt_path.name, out_path)

    print(f"Best epoch: {best_epoch}, hmean: {best_hmean:.4f}")
    print(f"Best checkpoint: {out_path}")


if __name__ == "__main__":
    main()
