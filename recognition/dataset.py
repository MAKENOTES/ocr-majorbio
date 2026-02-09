"""
用于文本识别的自定义数据集和数据加载器 (通用封装版)。
包含:
1. BucketBatchSampler: 桶采样 (训练用)
2. dynamic_padding_collate_fn: 动态 Padding (训练/推理通用)
3. TextRecognitionDataset: 带标签的数据集 (训练用)
4. InferenceDataset: 纯图片数据集 (推理用)
"""
import os
import math
import torch
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms

# 防止截断报错 (针对超大图)
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ==========================================
# 1. 采样与整理工具 (无需修改，完全通用)
# ==========================================

class BucketBatchSampler(Sampler):
    """
    桶采样器：将长宽比相近的图片放在同一个 Batch 中。
    """

    def __init__(self, dataset, batch_size, drop_last=False, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        print("正在预扫描数据集长宽比以优化 Padding (Bucket Sampling)...")
        self.aspect_ratios = []
        for path in dataset.image_paths:
            try:
                with Image.open(path) as img:
                    w, h = img.size
                    h = max(1, h)
                    self.aspect_ratios.append(w / h)
            except Exception as e:
                self.aspect_ratios.append(1.0)

        self.aspect_ratios = np.array(self.aspect_ratios)
        self.num_samples = len(self.dataset)
        print(f"扫描完成，样本数: {self.num_samples}")

    def __iter__(self):
        indices = np.argsort(self.aspect_ratios)
        batches = []
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i: i + self.batch_size]
            if len(batch_indices) < self.batch_size and self.drop_last:
                continue
            batches.append(batch_indices.tolist())

        if self.shuffle:
            np.random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size


def dynamic_padding_collate_fn(batch):
    """
    自定义 collate_fn，实现按批次动态 Padding。
    支持训练模式 (img, label) 和 推理模式 (img, path)。
    """
    images, labels_or_paths = zip(*batch)

    # 找到当前 Batch 中最宽的宽度
    max_width = 0
    for img in images:
        max_width = max(max_width, img.shape[2])

    # 宽度补齐到 8 的倍数
    max_width = ((max_width + 7) // 8) * 8

    num_channels = images[0].shape[0]
    height = images[0].shape[1]
    batch_size = len(images)

    # 填充背景 (Normalize mean=0.5 -> 1.0 为白色)
    padded_imgs = torch.full((batch_size, num_channels, height, max_width), 1.0)

    for i, img in enumerate(images):
        w = img.shape[2]
        padded_imgs[i, :, :, :w] = img

    return padded_imgs, labels_or_paths


# ==========================================
# 2. 基础数据集类 (封装通用逻辑)
# ==========================================

class BaseTextDataset(Dataset):
    """
    文本识别数据集基类。
    封装了核心的图片加载、预处理、增强逻辑。
    """

    def __init__(self, img_height, num_channels, is_train=False):
        self.img_height = img_height
        self.num_channels = num_channels
        self.image_paths = []  # 子类填充此列表

        # 定义 Transform
        if is_train:
            self.transform = transforms.Compose([
                # 训练集：加入颜色抖动
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transform = transforms.Compose([
                # 验证/推理集：仅归一化
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_and_process_image(self, index: int) -> torch.Tensor:
        """
        通用的单张图片处理逻辑：读取 -> 转通道 -> Resize -> Transform
        """
        image_path = self.image_paths[index]
        try:
            # 1. 统一转为 RGB 读取
            image = Image.open(image_path).convert('RGB')

            # 2. 通道处理 (保留红章逻辑)
            if self.num_channels == 1:
                image = image.convert('L')

            # 3. Resize (只固定高度，宽度按比例缩放)
            w, h = image.size
            target_h = self.img_height
            scale = target_h / h
            new_w = int(w * scale)
            # 必须使用 LANCZOS 以保证缩小时的清晰度
            image = image.resize((new_w, target_h), Image.Resampling.LANCZOS)

            # 4. Apply Transforms
            image_tensor = self.transform(image)
            return image_tensor

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回 dummy tensor
            return torch.ones((self.num_channels, self.img_height, 8))


# ==========================================
# 3. 具体实现类 (训练 vs 推理)
# ==========================================

class TextRecognitionDataset(BaseTextDataset):
    """
    [训练用] 读取标注文件，返回 (Image, Label)
    """

    def __init__(self, annotation_path: str, img_height: int, img_width: int, num_channels: int, is_train=True):
        # 初始化基类
        super().__init__(img_height, num_channels, is_train)

        # img_width 参数仅作为兼容保留，实际由 dynamic_padding 控制
        self.labels = []

        annotation_dir = os.path.dirname(annotation_path)
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        path_part = parts[0].strip()
                        label_part = ','.join(parts[1:]).strip().replace('"', '')

                        full_path = os.path.join(annotation_dir, path_part)
                        self.image_paths.append(full_path)
                        self.labels.append(label_part)
        else:
            print(f"Warning: Annotation file not found at {annotation_path}")

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        # 调用基类方法处理图片
        image = self._load_and_process_image(index)
        label = self.labels[index]
        return image, label


class InferenceDataset(BaseTextDataset):
    """
    [推理用] 直接读取图片列表，返回 (Image, ImagePath)
    """

    def __init__(self, image_paths: list, img_height: int, num_channels: int):
        # 推理模式 is_train=False
        super().__init__(img_height, num_channels, is_train=False)
        self.image_paths = image_paths

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        # 调用基类方法处理图片
        image = self._load_and_process_image(index)
        # 推理时没有 Label，返回路径作为 ID，方便对应结果
        path = self.image_paths[index]
        return image, path


# ==========================================
# 4. DataLoader 创建工厂 (保持不变，仅适配新类)
# ==========================================

def create_dataloader(annotation_path: str, img_height: int, img_width: int, num_channels: int, batch_size: int,
                      num_workers: int, is_train=True) -> DataLoader:
    """
    创建训练/验证用的 DataLoader
    """
    dataset = TextRecognitionDataset(
        annotation_path=annotation_path,
        img_height=img_height,
        img_width=img_width,
        num_channels=num_channels,
        is_train=is_train
    )

    if is_train:
        sampler = BucketBatchSampler(dataset, batch_size=batch_size, drop_last=False, shuffle=True)
        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=dynamic_padding_collate_fn
        )
    else:
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=dynamic_padding_collate_fn
        )

    return dataloader