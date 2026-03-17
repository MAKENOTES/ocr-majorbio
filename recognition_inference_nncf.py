"""
recognition_inference_nncf.py — OpenVINO 原生推理脚本 (配合 NNCF INT8 量化模型)
================================================================================
专门用于 recognition_quantize_nncf.py 生成的 OpenVINO IR 模型 (.xml + .bin)。
也兼容 FP32 ONNX 模型 (OpenVINO 可直接读取)。

与 recognition_inference.py 完全独立，不共享任何后端代码。

支持的模型格式:
  - .xml  OpenVINO IR (NNCF INT8 量化产物，推荐)
  - .onnx FP32 ONNX (OpenVINO 可直接加载，但不要用 ONNX Runtime INT8 的)

用法:
    # NNCF INT8 模型推理
    python recognition_inference_nncf.py \
        --model_path best_recognition_nncf_int8.xml \
        --image_path recognition/data/test_images

    # FP32 ONNX 模型推理 (也支持)
    python recognition_inference_nncf.py \
        --model_path best_recognition.onnx \
        --image_path recognition/data/test_images

依赖:
    pip install openvino torch torchvision numpy Pillow
"""

import os
import sys
import time
import glob
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

# -- 动态添加项目根目录到 sys.path --
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from recognition.components.converter import LabelConverter
from recognition.components.dataset import InferenceDataset, dynamic_padding_collate_fn

# ============================================================
# 默认配置
# ============================================================

DEFAULT_MODEL_PATH = "best_recognition_nncf_int8.xml"
DEFAULT_IMAGE_PATH = "recognition/data/test_images"
DEFAULT_KEYS_PATH  = "recognition/keys.txt"

IMG_HEIGHT   = 32
NUM_CHANNELS = 1


# ============================================================
# OpenVINO 推理引擎
# ============================================================

class OpenVINOEngine:
    """
    OpenVINO 推理引擎，支持 .xml (IR) 和 .onnx (FP32) 模型。
    """

    def __init__(self, model_path: str, device: str = "CPU"):
        from openvino import Core

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")

        ext = os.path.splitext(model_path)[1].lower()
        if ext not in (".xml", ".onnx"):
            raise ValueError(f"不支持的模型格式: {ext}，仅支持 .xml 和 .onnx")

        # 检查是否误传了 ONNX Runtime INT8 模型
        if ext == ".onnx" and "_int8_" in os.path.basename(model_path):
            print(f"  ⚠ 警告: 检测到 ONNX Runtime INT8 模型，OpenVINO 可能无法正确推理")
            print(f"    建议: 使用 recognition_quantize_nncf.py 生成 OpenVINO 原生 INT8 模型")

        self.core = Core()
        print(f"  加载模型: {model_path}")
        model = self.core.read_model(model=model_path)

        # 编译模型
        config = {
            "PERFORMANCE_HINT": "LATENCY",
        }
        self.compiled_model = self.core.compile_model(
            model=model, device_name=device, config=config
        )
        self.infer_request = self.compiled_model.create_infer_request()

        # 获取输入输出信息
        self.input_node = self.compiled_model.inputs[0]
        self.output_node = self.compiled_model.outputs[0]
        self.input_name = self.input_node.any_name

        print(f"  设备: {device}")
        print(f"  输入: {self.input_name} {self.input_node.partial_shape}")
        print(f"  输出: {self.output_node.any_name} {self.output_node.partial_shape}")

        # 预热
        self._warmup()
        print(f"  ✓ OpenVINO 引擎就绪")

    def _warmup(self, n: int = 3):
        dummy = np.random.randn(1, NUM_CHANNELS, IMG_HEIGHT, 256).astype(np.float32)
        for _ in range(n):
            self.infer_request.infer({self.input_name: dummy})

    def infer(self, images_np: np.ndarray) -> np.ndarray:
        """
        执行推理。

        Args:
            images_np: [B, C, H, W] numpy float32 数组
        Returns:
            output: [SeqLen, B, Classes] numpy float32 数组
        """
        result = self.infer_request.infer({self.input_name: images_np})
        out = np.array(result[self.output_node])

        # 确保输出维度是 [SeqLen, Batch, Classes]
        # 正常情况: shape = (SeqLen, Batch, Classes)，SeqLen >> Batch
        # 如果被转置: shape = (Batch, SeqLen, Classes)，Batch < SeqLen
        if out.ndim == 3 and out.shape[0] < out.shape[1]:
            out = np.transpose(out, (1, 0, 2))

        return out


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="CRNN OpenVINO 推理 (配合 NNCF INT8 量化模型)"
    )
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="模型路径 (.xml 或 .onnx)")
    parser.add_argument("--image_path", type=str, default=DEFAULT_IMAGE_PATH,
                        help="图片路径 (文件夹或单张图片)")
    parser.add_argument("--keys_path", type=str, default=DEFAULT_KEYS_PATH,
                        help="字典文件路径")
    parser.add_argument("--output_file", type=str, default=None,
                        help="结果保存路径 (默认打印到控制台)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批量大小")
    parser.add_argument("--device", type=str, default="CPU",
                        help="推理设备 (CPU / GPU / AUTO)")
    args = parser.parse_args()

    print("=" * 60)
    print("CRNN OpenVINO 推理 (NNCF INT8)")
    print("=" * 60)

    # 1. 收集图片
    image_paths = []
    if os.path.isdir(args.image_path):
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif"]:
            image_paths.extend(glob.glob(os.path.join(args.image_path, ext)))
            image_paths.extend(glob.glob(os.path.join(args.image_path, ext.upper())))
    elif os.path.isfile(args.image_path):
        image_paths = [args.image_path]

    if not image_paths:
        print("✗ 未找到图像文件")
        return

    print(f"\n  图片数量: {len(image_paths)}")

    # 2. 初始化组件
    converter = LabelConverter(args.keys_path)
    engine = OpenVINOEngine(args.model_path, device=args.device)

    # 3. 构建 DataLoader (复用项目 dataset.py，预处理 100% 一致)
    dataset = InferenceDataset(
        image_paths=image_paths,
        img_height=IMG_HEIGHT,
        num_channels=NUM_CHANNELS,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=dynamic_padding_collate_fn,
    )

    # 4. 推理
    results = []
    total_infer_ms = 0.0
    t_wall_start = time.perf_counter()

    for images, paths, img_widths in loader:
        images_np = images.numpy()

        t0 = time.perf_counter()
        output_np = engine.infer(images_np)
        infer_ms = (time.perf_counter() - t0) * 1000
        total_infer_ms += infer_ms

        # CTC 解码
        preds_tensor = torch.from_numpy(output_np)
        pred_texts = converter.decode(preds_tensor)

        for path, text in zip(paths, pred_texts):
            results.append((os.path.basename(path), text))

    t_wall = time.perf_counter() - t_wall_start

    # 5. 输出结果
    print(f"\n{'-'*60}")
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for filename, text in results:
                f.write(f"{filename}\t{text}\n")
        print(f"结果已保存至: {args.output_file}")
    else:
        print("识别结果:")
        for filename, text in results:
            print(f"  {filename}\t{text}")

    print(f"\n{'='*60}")
    print(f"推理完成")
    print(f"  图像数:           {len(image_paths)} 张")
    print(f"  纯推理耗时:       {total_infer_ms:.1f} ms")
    print(f"  总耗时(含预处理):  {t_wall*1000:.0f} ms")
    if t_wall > 0:
        print(f"  吞吐量:           {len(image_paths)/t_wall:.2f} 张/秒")
    if len(image_paths) > 0:
        print(f"  平均每张(纯推理):  {total_infer_ms/len(image_paths):.2f} ms")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()