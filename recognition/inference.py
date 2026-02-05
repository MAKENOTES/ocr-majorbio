"""
使用训练好的 CRNN 模型进行单张或批量图片推理的脚本。
"""
import os
import sys
import torch
from PIL import Image
from torchvision import transforms

# -- 动态添加项目根目录到 sys.path，防止找不到模块 --
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from recognition.model import CRNN
from recognition.converter import LabelConverter

# --- 配置区域 ---
# 字典文件路径
KEYS_PATH = os.path.join(project_root, "recognition/keys.txt")
# 训练好的权重文件路径 (请修改为你最好的 checkpoint)
CHECKPOINT_PATH = os.path.join(project_root, "recognition/checkpoints/recognition_epoch_200.pth")
# 待测试的图片路径 (可以是单个文件，也可以是文件夹)
TEST_IMAGE_PATH = os.path.join(project_root, "recognition/data/test_images")

# 模型参数 (必须与训练时一致)
IMG_HEIGHT = 32
NUM_CHANNELS = 1  # 灰度输入
HIDDEN_SIZE = 512  # 训练时你改为了 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OCRPredictor:
    def __init__(self, keys_path, checkpoint_path, device):
        self.device = device

        # 1. 初始化转换器
        if not os.path.exists(keys_path):
            raise FileNotFoundError(f"未找到字典文件: {keys_path}")
        self.converter = LabelConverter(keys_path)

        # 2. 初始化模型
        # 注意: n_class 必须包含 blank/unk，由 converter 自动计算
        self.model = CRNN(
            img_height=IMG_HEIGHT,
            nc=NUM_CHANNELS,
            n_class=self.converter.num_classes,
            nh=HIDDEN_SIZE
        ).to(self.device)

        # 3. 加载权重
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"未找到模型权重: {checkpoint_path}")

        print(f"正在加载模型权重: {checkpoint_path} ...")
        # map_location 确保在 CPU 机器上也能加载 CUDA 训练的权重
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        print("模型加载完成!")

        # 4. 定义预处理 (与 dataset.py 保持一致)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def preprocess_image(self, image_path):
        """
        读取并预处理图片：转灰度 -> 等比例缩放高度到32 -> 归一化
        """
        try:
            # 读取图片
            img = Image.open(image_path).convert('RGB')

            # 转灰度 (策略与 dataset.py 一致: convert('L'))
            if NUM_CHANNELS == 1:
                img = img.convert('L')

            # Resize: 高度固定 32，宽度按比例缩放
            w, h = img.size
            scale = IMG_HEIGHT / h
            new_w = int(w * scale)
            # 限制最小宽度，防止报错
            new_w = max(4, new_w)

            img = img.resize((new_w, IMG_HEIGHT), Image.Resampling.LANCZOS)

            # 转换和归一化
            img_tensor = self.transform(img)

            # 增加 Batch 维度: [C, H, W] -> [1, C, H, W]
            return img_tensor.unsqueeze(0)

        except Exception as e:
            print(f"处理图片失败 {image_path}: {e}")
            return None

    def predict(self, image_path):
        """
        对单张图片进行推理
        """
        img_tensor = self.preprocess_image(image_path)
        if img_tensor is None:
            return None

        img_tensor = img_tensor.to(self.device)

        with torch.no_grad():
            # 模型输出: [SeqLen, Batch, NumClasses]
            preds = self.model(img_tensor)

            # 注意: 这里不要手动 transpose，直接交给 converter.decode
            # decode 内部逻辑: if shape[1] != num_classes -> transpose
            # 输入 (Seq, 1, C) -> 内部转为 (1, Seq, C) -> 正确解码
            decoded_texts = self.converter.decode(preds.cpu())

        return decoded_texts[0]


# --- 主程序入口 ---
if __name__ == "__main__":
    # 实例化预测器
    try:
        predictor = OCRPredictor(KEYS_PATH, CHECKPOINT_PATH, DEVICE)

        # 模式 1: 测试单张图片
        if os.path.isfile(TEST_IMAGE_PATH):
            print(f"\n正在识别: {TEST_IMAGE_PATH}")
            result = predictor.predict(TEST_IMAGE_PATH)
            print(f"识别结果: {result}")

        # 模式 2: 测试文件夹下的所有图片
        elif os.path.isdir(TEST_IMAGE_PATH):
            print(f"\n正在批量识别文件夹: {TEST_IMAGE_PATH}")
            image_files = [f for f in os.listdir(TEST_IMAGE_PATH) if
                           f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]

            for img_file in image_files:
                full_path = os.path.join(TEST_IMAGE_PATH, img_file)
                result = predictor.predict(full_path)
                print(f"[{img_file}] -> {result}")
        else:
            print(f"错误: 路径不存在 {TEST_IMAGE_PATH}")

    except Exception as e:
        print(f"发生错误: {e}")