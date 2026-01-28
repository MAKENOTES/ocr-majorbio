import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
from .ground_truth import generate_ground_truth
from torchvision.transforms import Normalize

class ResizeAndPad:
    """
    一个自定义的transform，用于在保持宽高比的同时缩放和填充图像。
    它会将图像缩放到一个目标尺寸内，然后用灰色像素填充剩余部分，
    使得最终图像尺寸固定。同时，它也会相应地调整边界框的坐标。
    """
    def __init__(self, output_size):
        """
        初始化ResizeAndPad。
        :param output_size: int或tuple, 目标输出尺寸。如果是int，则宽高相同。
        """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        """
        对输入的样本（包含图像和四边形）进行变换。
        :param sample: dict, 包含'image'和'quads'的字典。
        :return: dict, 变换后的样本。
        """
        image, quads = sample['image'], sample['quads']
        
        w, h = image.size
        new_h, new_w = self.output_size
        
        # 计算缩放比例和新的尺寸
        scale = min(new_w / w, new_h / h)
        resized_w, resized_h = int(w * scale), int(h * scale)
        
        # 缩放图像
        resized_image = image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
        
        # 创建一个带灰色背景的新图像用于填充
        new_image = Image.new('RGB', (new_w, new_h), (128, 128, 128))
        new_image.paste(resized_image, ((new_w - resized_w) // 2, (new_h - resized_h) // 2))
        
        # 调整四边形坐标
        if quads.size > 0:
            quads *= scale
            quads[:, :, 0] += (new_w - resized_w) // 2
            quads[:, :, 1] += (new_h - resized_h) // 2
            
        sample['image'] = new_image
        sample['quads'] = quads
        
        return sample

class GenerateGT:
    """
    一个自定义的transform，用于生成EAST模型的真值（ground truth）图。
    它调用 `generate_ground_truth` 函数来创建分数图和几何图。
    """
    def __init__(self, input_size=512, score_map_scale=0.25):
        """
        初始化GenerateGT。
        :param input_size: int, 模型的输入图像尺寸。
        :param score_map_scale: float, 分数图相对于输入图像的缩放比例。
        """
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else input_size
        self.score_map_scale = score_map_scale

    def __call__(self, sample):
        """
        对输入的样本进行变换，添加'score_map'和'geo_map'。
        :param sample: dict, 包含'quads'的字典。
        :return: dict, 添加了真值图的样本。
        """
        quads = sample['quads']
        
        score_map, geo_map = generate_ground_truth(
            image_size=self.input_size,
            quads=quads,
            score_map_scale=self.score_map_scale
        )
        
        sample['score_map'] = score_map
        sample['geo_map'] = geo_map
        
        return sample

class ToTensor:
    def __init__(self):
        # ImageNet 的标准均值和方差
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

    def __call__(self, sample):
        """
        对输入的样本进行变换。
        :param sample: dict, 包含'image', 'score_map', 'geo_map'等键的字典。
        :return: dict, 只包含'image', 'score_map', 'geo_map'张量的字典。
        """
        # 将 image, score_map, 和 geo_map 转换为张量
        image = F.to_tensor(sample['image'])
        image = self.normalize(image)
        score_map = torch.from_numpy(sample['score_map']).float()
        geo_map = torch.from_numpy(sample['geo_map']).float()

        # 只返回训练所需的张量。
        # 这可以防止 collate 函数尝试堆叠 'quads' 和 'transcriptions'，
        # 因为它们的尺寸是可变的。
        return {
            'image': image,
            'score_map': score_map,
            'geo_map': geo_map
        }

def get_transforms(is_train=True, input_size=512):
    """
    获取一个组合好的数据转换流程 (pipeline)。
    :param is_train: bool, 是否为训练模式。如果是，未来可以加入数据增强。
    :param input_size: int, 模型的输入尺寸。
    :return: 一个 `torchvision.transforms.Compose` 对象。
    """
    # TODO: 为训练添加更多的数据增强 (例如，随机旋转、颜色抖动)
    if is_train:
        return transforms.Compose([
            ResizeAndPad(input_size),
            GenerateGT(input_size),
            ToTensor()
        ])
    else:
        return transforms.Compose([
            ResizeAndPad(input_size),
            GenerateGT(input_size),
            ToTensor()
        ])
