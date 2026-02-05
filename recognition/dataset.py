"""
用于文本识别的自定义数据集和数据加载器 (终极优化版)。
集成特性:
1. Bucket Sampling: 按长宽比聚类采样，极大减少 Padding 浪费。
2. Dynamic Padding: Batch 内动态对齐宽度。
3. Robust Image Loading: 修复红章丢失问题，增强读取稳定性。
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

class BucketBatchSampler(Sampler):
    """
    桶采样器：将长宽比相近的图片放在同一个 Batch 中。
    优化了训练速度，解决了短图被迫 Pad 到长图宽度的问题。
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
                # Lazy loading: 只读头部信息，不加载像素，速度快
                with Image.open(path) as img:
                    w, h = img.size
                    # 防止 h 为 0 的极端情况
                    h = max(1, h)
                    self.aspect_ratios.append(w / h)
            except Exception as e:
                # 异常图片给个默认比例
                self.aspect_ratios.append(1.0)

        self.aspect_ratios = np.array(self.aspect_ratios)
        self.num_samples = len(self.dataset)
        print(f"扫描完成，样本数: {self.num_samples}")

    def __iter__(self):
        # 1. 生成索引并按照长宽比排序 (从小到大)
        indices = np.argsort(self.aspect_ratios)

        # 2. 生成 Batches
        batches = []
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]

            # 处理最后一个不足 batch_size 的情况   drop_last为false不丢弃
            if len(batch_indices) < self.batch_size and self.drop_last:
                continue

            batches.append(batch_indices.tolist())

        # 3. 关键步骤：打乱 Batch 之间的顺序
        #    Batch 内部是按长宽比排序的，但 Batch 的提交顺序必须随机
        if self.shuffle:
            np.random.shuffle(batches)

        # 4. Yield
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
    """
    # 1. 分离图片和标签
    # batch 是一个 list，每个元素是 (image_tensor, label_str)
    images, labels = zip(*batch)

    # 2. 找到当前 Batch 中最宽的宽度
    max_width = 0
    for img in images:
        # img shape is (C, H, W)
        max_width = max(max_width, img.shape[2])

    # 3. 宽度补齐到 8 的倍数 (适配 ResNet 下采样)
    max_width = ((max_width + 7) // 8) * 8

    # 4. 创建容器 (Batch, C, H, Max_W)
    num_channels = images[0].shape[0]
    height = images[0].shape[1]
    batch_size = len(images)

    # 初始化填充值
    # Normalize(mean=0.5) -> 原始255(白) 变为 1.0。
    # 这里用 1.0 填充背景（白色），如果背景是黑的，建议改为 -1.0
    padded_imgs = torch.full((batch_size, num_channels, height, max_width), 1.0)

    # 5. 填入数据 (左对齐)
    for i, img in enumerate(images):
        w = img.shape[2]
        padded_imgs[i, :, :, :w] = img

    return padded_imgs, labels

class TextRecognitionDataset(Dataset):
    def __init__(self, annotation_path: str, img_height: int, img_width: int, num_channels: int, is_train=True):
        """
        注意: img_width 参数在这里不再强制用于 resize，仅作为兼容性参数保留。
        宽度将由 dynamic_padding_collate_fn 动态决定。
        """
        self.img_height = img_height
        self.img_width = img_width # 保留但不强制 resize 到此宽度
        self.num_channels = num_channels
        self.is_train = is_train
        self.image_paths = []
        self.labels = []

        annotation_dir = os.path.dirname(annotation_path)

        if os.path.exists(annotation_path):
            with open(annotation_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        path_part = parts[0].strip()
                        # 处理标签中可能包含逗号的情况
                        label_part = ','.join(parts[1:]).strip().replace('"', '')

                        full_path = os.path.join(annotation_dir, path_part)
                        self.image_paths.append(full_path)
                        self.labels.append(label_part)
        else:
            print(f"Warning: Annotation file not found at {annotation_path}")

        # 定义增强流程
        if self.is_train:
            self.transform = transforms.Compose([
                # 颜色抖动 (增强鲁棒性)
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
                # 必须先转 Tensor
                transforms.ToTensor(),
                # 归一化 [-1, 1]
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image_path = self.image_paths[index]
        label = self.labels[index]

        try:
            # 1. 统一转为 RGB 读取
            image = Image.open(image_path).convert('RGB')

            # 2. 通道处理 (修复红章问题)
            if self.num_channels == 1:
                # 使用标准灰度转换，保留红色印章信息
                image = image.convert('L')

            # 3. Resize (只固定高度，宽度按比例缩放)
            w, h = image.size
            target_h = self.img_height # 通常是 32

            scale = target_h / h
            new_w = int(w * scale)

            # 核心修改: 只 Resize，不 Pad 到固定宽度，Padding 交给 collate_fn
            image = image.resize((new_w, target_h), Image.Resampling.LANCZOS)

            # 4. Augmentation & Normalize
            image = self.transform(image)

            return image, label

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回一个极小的合法 Tensor 防止 DataLoader 崩溃
            return torch.ones((self.num_channels, self.img_height, 8)), label

def create_dataloader(annotation_path: str, img_height: int, img_width: int, num_channels: int, batch_size: int, num_workers: int, is_train=True) -> DataLoader:
    """
    创建 DataLoader。
    训练集使用 BucketBatchSampler，验证集使用普通 Shuffle。
    """
    dataset = TextRecognitionDataset(
        annotation_path=annotation_path,
        img_height=img_height,
        img_width=img_width, # 此参数仅传参，Dataset 内部已改为动态宽
        num_channels=num_channels,
        is_train=is_train
    )

    if is_train:
        # 训练时：使用 Bucket Sampler
        sampler = BucketBatchSampler(
            dataset,
            batch_size=batch_size,
            drop_last=False,
            shuffle=True
        )
        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler, # 使用自定义采样器 (互斥参数: 不能再传 batch_size, shuffle, drop_last)
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=dynamic_padding_collate_fn # 关键：动态 Padding
        )
    else:
        # 验证/测试时：普通加载
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False, # 验证集不需要 shuffle
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=dynamic_padding_collate_fn
        )

    return dataloader