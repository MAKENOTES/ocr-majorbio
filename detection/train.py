import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    # 便于以脚本方式运行时找到项目内模块
    sys.path.append(str(ROOT_DIR))

from doctr.models import detection as doctr_detection


@dataclass
class TrainConfig:
    """训练配置参数定义。"""

    # 训练图像目录
    images_dir: str
    # 训练标注目录
    labels_dir: str
    # 验证图像目录
    val_images_dir: str
    # 验证标注目录
    val_labels_dir: str
    # 标注文件名前缀
    label_prefix: str
    # 模型名称
    model_name: str
    # 批次大小
    batch_size: int
    # 数据加载线程数
    num_workers: int
    # 训练轮数
    epochs: int
    # 学习率
    lr: float
    # 训练设备
    device: str
    # 模型保存目录
    save_dir: str
    # 保存频率（每多少个 epoch 保存一次）
    save_every: int
    # 每个 epoch 最大步数（<=0 表示不限制）
    max_steps: int


class DocTRDetDataset(Dataset):
    """读取检测标注并转换为 docTR 所需格式。

    Args:
        images_dir: 图像目录。
        labels_dir: 标注目录。
        label_prefix: 标注文件名前缀。
    """

    def __init__(self, images_dir: str, labels_dir: str, label_prefix: str = "") -> None:
        """初始化检测数据集。

        Args:
            images_dir: 图像目录。
            labels_dir: 标注目录。
            label_prefix: 标注文件名前缀。
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.label_prefix = label_prefix
        self.images = sorted([p for p in self.images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])

    def __len__(self) -> int:
        """返回数据集大小。

        Returns:
            数据集长度。
        """
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """获取单条样本。

        Args:
            idx: 样本索引。

        Returns:
            图像数组与目标字典。
        """
        image_path = self.images[idx]
        label_path = self.labels_dir / f"{self.label_prefix}{image_path.stem}.txt"
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        boxes = _parse_label_file(label_path)
        polys = _boxes_to_polys(boxes, width, height)
        # docTR 期望 target 为类别名到多边形数组的映射
        target = {"text": polys}
        return np.array(image), target


def _parse_label_file(label_path: Path) -> List[List[Tuple[float, float]]]:
    """解析检测标注文件。

    Args:
        label_path: 标注文件路径。

    Returns:
        多边形点列表。
    """
    polys: List[List[Tuple[float, float]]] = []
    if not label_path.exists():
        return polys
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 统一空格/逗号分隔格式
            parts = [p for p in line.replace(",", " ").split() if p]
            if len(parts) < 4:
                continue
            nums = [float(p) for p in parts]
            if len(nums) >= 8:
                # 4 点多边形
                coords = nums[:8]
                points = [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]
            else:
                # 轴对齐矩形 (x1, y1, x2, y2)
                x1, y1, x2, y2 = nums[:4]
                points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            polys.append(points)
    return polys


def _boxes_to_polys(boxes: List[List[Tuple[float, float]]], width: int, height: int) -> np.ndarray:
    """将检测框转换为归一化多边形数组。

    Args:
        boxes: 多边形点列表。
        width: 图像宽度。
        height: 图像高度。

    Returns:
        归一化后的多边形数组。
    """
    polys: List[np.ndarray] = []
    for points in boxes:
        if len(points) != 4:
            continue
        poly = np.array(points, dtype=np.float32)
        # 按图像尺寸归一化到 [0, 1]
        poly[:, 0] = np.clip(poly[:, 0] / max(width, 1), 0.0, 1.0)
        poly[:, 1] = np.clip(poly[:, 1] / max(height, 1), 0.0, 1.0)
        polys.append(poly)
    if not polys:
        return np.zeros((0, 4, 2), dtype=np.float32)
    return np.stack(polys).astype(np.float32)


def _collate_fn(batch: List[Tuple[np.ndarray, Dict[str, np.ndarray]]]):
    """自定义批次拼接函数。

    Args:
        batch: 样本列表。

    Returns:
        图像列表与目标列表。
    """
    # 保持原始 list 结构，便于 docTR 内部构建目标
    images, targets = zip(*batch)
    return list(images), list(targets)


def _build_model(name: str):
    """构建 docTR 检测模型。

    Args:
        name: 模型名称。

    Returns:
        docTR 检测模型实例。
    """
    if not hasattr(doctr_detection, name):
        raise ValueError(f"不支持的模型名称: {name}")
    # 不加载预训练权重，保持从零训练
    return getattr(doctr_detection, name)(pretrained=False)


def _build_transform(cfg: Dict[str, Tuple[float, float, float]]):
    """构建输入预处理。

    Args:
        cfg: 模型配置字典。

    Returns:
        预处理组合变换。
    """
    _, input_h, input_w = cfg["input_shape"]
    mean = cfg["mean"]
    std = cfg["std"]
    # 训练时使用 docTR 配置中的尺寸与归一化参数
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((input_h, input_w), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    transform,
    max_steps: int,
    epoch: int,
) -> None:
    """训练一个 epoch。

    Args:
        model: 检测模型。
        loader: 训练数据加载器。
        optimizer: 优化器。
        device: 训练设备。
        transform: 预处理函数。
        max_steps: 每个 epoch 最大步数，<=0 表示不限制。
        epoch: 当前 epoch 序号。
    """
    model.train()
    step = 0
    running_loss = 0.0
    valid_steps = 0
    for images, targets in loader:
        if all(t["text"].shape[0] == 0 for t in targets):
            # 空标注批次跳过，避免 docTR 内部损失计算异常
            continue
        batch = torch.stack([transform(img) for img in images]).to(device)
        output = model(batch, targets)
        loss = output["loss"]
        running_loss += loss.item()
        valid_steps += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Epoch {epoch} Step {step} Loss {loss.item():.4f}")

        step += 1
        if max_steps > 0 and step >= max_steps:
            break
    if valid_steps > 0:
        avg_loss = running_loss / valid_steps
        print(f"Epoch {epoch} 训练平均损失 {avg_loss:.4f}")


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    transform,
    epoch: int,
) -> None:
    """在验证集上评估模型。

    Args:
        model: 检测模型。
        loader: 验证数据加载器。
        device: 评估设备。
        transform: 预处理函数。
        epoch: 当前 epoch 序号。
    """
    model.eval()
    running_loss = 0.0
    valid_steps = 0
    with torch.no_grad():
        for images, targets in loader:
            if all(t["text"].shape[0] == 0 for t in targets):
                continue
            batch = torch.stack([transform(img) for img in images]).to(device)
            output = model(batch, targets)
            loss = output["loss"]
            running_loss += loss.item()
            valid_steps += 1
    if valid_steps > 0:
        avg_loss = running_loss / valid_steps
        print(f"Epoch {epoch} 验证平均损失 {avg_loss:.4f}")
    else:
        print(f"Epoch {epoch} 验证集无有效样本")


def _save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, save_dir: str) -> str:
    """保存模型与优化器状态。

    Args:
        model: 检测模型。
        optimizer: 优化器。
        epoch: 当前 epoch 序号。
        save_dir: 保存目录。

    Returns:
        保存路径。
    """
    os.makedirs(save_dir, exist_ok=True)
    # 按 epoch 保存模型与优化器状态
    path = os.path.join(save_dir, f"doctr_det_epoch_{epoch}.pth")
    torch.save(
        {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, path
    )
    return path


def _build_config() -> TrainConfig:
    """构建训练配置。

    Returns:
        训练配置对象。
    """
    return TrainConfig(
        images_dir="data/images",
        labels_dir="data/det_labels",
        val_images_dir="data/images",
        val_labels_dir="data/det_labels",
        label_prefix="gt_",
        model_name="db_resnet50",
        batch_size=4,
        num_workers=4,
        epochs=200,
        lr=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu",
        save_dir="outputs/doctr_det",
        save_every=20,
        max_steps=10,
    )


def main() -> None:
    """训练入口。

    Returns:
        无返回值。
    """
    cfg = _build_config()
    device = torch.device(cfg.device)

    # 构建数据集与 DataLoader
    dataset = DocTRDetDataset(cfg.images_dir, cfg.labels_dir, cfg.label_prefix)
    val_dataset = DocTRDetDataset(cfg.val_images_dir, cfg.val_labels_dir, cfg.label_prefix)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=_collate_fn,
    )

    # 初始化模型与优化器
    model = _build_model(cfg.model_name).to(device)
    transform = _build_transform(model.cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(1, cfg.epochs + 1):
        _train_one_epoch(model, loader, optimizer, device, transform, cfg.max_steps, epoch)
        _evaluate(model, val_loader, device, transform, epoch)
        if cfg.save_every > 0 and epoch % cfg.save_every == 0:
            ckpt_path = _save_checkpoint(model, optimizer, epoch, cfg.save_dir)
            print(f"已保存模型: {ckpt_path}")


if __name__ == "__main__":
    main()
