"""
使用训练好的 CRNN 模型进行单张或批量图片推理的脚本。
"""
import os
import sys
import torch
from PIL import Image
from torchvision import transforms

# -- 动态添加项目根目录到 sys.path --
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from recognition.model import CRNN
from recognition.converter import LabelConverter

# --- 配置区域 ---
KEYS_PATH = os.path.join(project_root, "recognition/keys.txt")
CHECKPOINT_PATH = os.path.join(project_root, "recognition/checkpoints/recognition_epoch_200.pth")
TEST_IMAGE_PATH = os.path.join(project_root, "recognition/data/test_images")

# 模型参数
IMG_HEIGHT = 32
NUM_CHANNELS = 1
HIDDEN_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class OCRPredictor:
    def __init__(self, keys_path, checkpoint_path, device):
        self.device = device

        if not os.path.exists(keys_path):
            raise FileNotFoundError(f"未找到字典文件: {keys_path}")
        self.converter = LabelConverter(keys_path)

        # 初始化模型
        self.model = CRNN(
            img_height=IMG_HEIGHT,
            nc=NUM_CHANNELS,
            n_class=self.converter.num_classes,
            nh=HIDDEN_SIZE
        ).to(self.device)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"未找到模型权重: {checkpoint_path}")

        print(f"正在加载模型权重: {checkpoint_path} ...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        print("模型加载完成!")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def preprocess_image(self, image_path):
        try:
            # 1. 读取图片
            img = Image.open(image_path).convert('RGB')

            # 2. 通道处理
            # 必须与 dataset.py 保持完全一致。
            # 当前 dataset.py 使用的是 convert('L')。
            if NUM_CHANNELS == 1:
                img = img.convert('L')

            # 3. Resize 与 Padding 对齐 (关键修复)
            w, h = img.size
            scale = IMG_HEIGHT / h
            new_w = int(w * scale)
            new_w = max(4, new_w)

            # [核心修复] 对齐宽度到 8 的倍数
            # 训练时的 dynamic_padding_collate_fn 做了这个操作。
            # 如果不对齐，ResNet 下采样可能会切掉边缘特征。
            aligned_w = ((new_w + 7) // 8) * 8

            # 使用 LANCZOS 缩放
            img = img.resize((new_w, IMG_HEIGHT), Image.Resampling.LANCZOS)

            # 如果 new_w != aligned_w，我们需要手动创建一个画布来模拟 Padding
            if new_w != aligned_w:
                # 创建一个白色背景 (255) 的画布，大小为对齐后的尺寸
                # 训练时的 pad_value=1.0 (归一化后)，对应原始像素 255
                padded_img = Image.new(img.mode, (aligned_w, IMG_HEIGHT), 255)
                padded_img.paste(img, (0, 0))
                img = padded_img

            # 4. 转 Tensor & 归一化
            img_tensor = self.transform(img)

            # 5. 增加 Batch 维度: [C, H, W] -> [1, C, H, W]
            return img_tensor.unsqueeze(0)

        except Exception as e:
            print(f"处理图片失败 {image_path}: {e}")
            return None

    def predict(self, image_path):
        img_tensor = self.preprocess_image(image_path)
        if img_tensor is None:
            return None

        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            preds = self.model(img_tensor)
            decoded_texts = self.converter.decode(preds.cpu())

        return decoded_texts[0]

if __name__ == "__main__":
    try:
        predictor = OCRPredictor(KEYS_PATH, CHECKPOINT_PATH, DEVICE)

        if os.path.isfile(TEST_IMAGE_PATH):
            print(f"\n正在识别: {TEST_IMAGE_PATH}")
            result = predictor.predict(TEST_IMAGE_PATH)
            print(f"识别结果: {result}")
        elif os.path.isdir(TEST_IMAGE_PATH):
            print(f"\n正在批量识别文件夹: {TEST_IMAGE_PATH}")
            image_files = [f for f in os.listdir(TEST_IMAGE_PATH) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
            image_files.sort()
            for img_file in image_files:
                full_path = os.path.join(TEST_IMAGE_PATH, img_file)
                result = predictor.predict(full_path)
                print(f"[{img_file}] -> {result}")
        else:
            print(f"错误: 路径不存在 {TEST_IMAGE_PATH}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"发生错误: {e}")