import argparse
import os
import re
from pathlib import Path


def _find_latest_log(work_dir: Path) -> Path:
    log_paths = sorted(work_dir.rglob("*.log"))
    if not log_paths:
        raise FileNotFoundError(f"No log files found under {work_dir}")
    return log_paths[-1]


def _parse_best_epoch(log_path: Path) -> tuple[int, float]:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--log", default=None)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    log_path = Path(args.log) if args.log else _find_latest_log(work_dir)

    best_epoch, best_hmean = _parse_best_epoch(log_path)
    ckpt_name = f"epoch_{best_epoch}.pth"
    ckpt_path = work_dir / ckpt_name

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_path = work_dir / "best.pth"
    if out_path.exists() or out_path.is_symlink():
        out_path.unlink()
    os.symlink(ckpt_path.name, out_path)

    print(f"Best epoch: {best_epoch}, hmean: {best_hmean:.4f}")
    print(f"Best checkpoint: {out_path}")


if __name__ == "__main__":
    main()
