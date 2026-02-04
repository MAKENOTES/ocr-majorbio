"""
用于文本识别的自定义数据集和数据加载器 (优化版)。
"""
import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 尝试导入增强库，如果没有则使用基础变换
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

class ResizeAndPad:
    """
    保持长宽比缩放，并进行 Padding 到固定尺寸。
    避免直接 Resize 导致的文字变形。
    """
    def __init__(self, target_h, target_w, pad_value=0):
        self.target_h = target_h
        self.target_w = target_w
        self.pad_value = pad_value

    def __call__(self, img):
        # img: PIL Image
        w, h = img.size

        # 1. 计算缩放比例，优先保证高度填满
        scale = self.target_h / h
        new_w = int(w * scale)

        # 如果缩放后宽度超过目标宽度，则需要截断。TODO给足够的宽度。
        if new_w > self.target_w:
            # 简单策略：强制 Resize 到目标尺寸 (针对超长文本)
            img = img.resize((self.target_w, self.target_h), Image.Resampling.LANCZOS)
            print(f"警告: 图像 {new_w} 调整后不匹配目标宽度，强制适应。")
        else:
            img = img.resize((new_w, self.target_h), Image.Resampling.LANCZOS)

        # 2. 创建画布并 Padding (右侧补零)
        # 如果是灰度图模式 'L'，如果是 RGB 模式 'RGB'
        mode = img.mode
        new_img = Image.new(mode, (self.target_w, self.target_h), self.pad_value)
        new_img.paste(img, (0, 0)) # 靠左对齐

        return new_img

class TextRecognitionDataset(Dataset):
    def __init__(self, annotation_path: str, img_height: int, img_width: int, num_channels: int, is_train=True):
        self.img_height = img_height
        self.img_width = img_width
        self.num_channels = num_channels
        self.is_train = is_train

        self.image_paths = []
        self.labels = []

        annotation_dir = os.path.dirname(annotation_path)

        # 读取标注
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2:
                        image_path = os.path.join(annotation_dir, parts[0].strip())
                        label = ','.join(parts[1:]).strip().replace('"', '')  # 处理标签中逗号
                        self.image_paths.append(image_path)
                        self.labels.append(label)
        else:
            print(f"警告: 标注文件不存在 {annotation_path}")

        # 定义变换流程
        self.resize_pad = ResizeAndPad(img_height, img_width, pad_value=255) # 填充白色  因为图片在填充前进行了 image = r

        # 标准化参数 (ImageNet 均值方差，加速收敛)
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image_path = self.image_paths[index]
        label = self.labels[index]

        try:
            # 读取图片
            image = Image.open(image_path).convert('RGB')

            # 如果是单通道训练，先转灰度
            if self.num_channels == 1:
                # --- [核心修改] ---
                # 不要用 convert('L')，那是标准灰度
                # 而是提取 R 通道 (Red Channel)
                r, g, b = image.split()
                image = r  # 只取 R 通道，红章会变白消失

            # 1. 保持比例缩放 + Padding
            image = self.resize_pad(image)

            # 2. 转 Tensor
            image = transforms.ToTensor()(image)

            # 3. 归一化
            image = self.normalize(image)

            # 4. (可选) 训练时增加更多增强，这里暂仅做基础处理
            # 实际项目中建议在此处加入 albumentations 对 numpy 数组进行增强

            return image, label

        except Exception as e:
            print(f"错误读取图片: {image_path}, {e}")
            return torch.zeros((self.num_channels, self.img_height, self.img_width)), label


def create_dataloader(annotation_path: str, img_height: int, img_width: int, num_channels: int, batch_size: int, num_workers: int, is_train=True) -> DataLoader:
    """
    为文本识别数据集创建DataLoader。
    """
    dataset = TextRecognitionDataset(
        annotation_path=annotation_path,
        img_height=img_height,
        img_width=img_width,
        num_channels=num_channels,
        is_train=is_train
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
    )
    return dataloader
