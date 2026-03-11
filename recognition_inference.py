"""
recognition_inference.py — 多后端优化识别推理脚本
=====================================================================
支持三种后端（自动按速度选择）：
  1. ONNX Runtime FP32   （无损，2-3x 提速）
  2. ONNX Runtime INT8   （极小精度损失，3-5x 提速）
  3. OpenVINO            （Intel CPU 专属，5-10x 提速）
  4. PyTorch（原始，作为 fallback）

用法:
    # 自动选最优后端
    python recognition_inference.py \
        --model_path best_recognition_int8_dynamic.onnx \
        --image_path recognition/data/test_images

    # 指定 OpenVINO 后端
    python recognition_inference.py \
        --model_path best_recognition.onnx \
        --backend openvino \
        --image_path your_image.jpg

依赖:
    pip install onnxruntime onnxsim
    # OpenVINO（可选，Intel CPU 专属）:
    pip install openvino
"""

import os
import sys
import time
import glob
import argparse
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recognition.converter import LabelConverter

# ============================================================
# 参数配置
# ============================================================

DEFAULT_MODEL_PATH  = "best_recognition_int8_dynamic.onnx"
DEFAULT_IMAGE_PATH  = "recognition/data/test_images"
DEFAULT_KEYS_PATH   = "recognition/keys.txt"

IMG_HEIGHT          = 32
NUM_CHANNELS        = 1
MIN_WIDTH           = 48      # 最小宽度，与训练时一致


# ============================================================
# 后端加载器
# ============================================================

class ONNXBackend:
    """ONNX Runtime 后端（支持 FP32 和 INT8 模型）"""

    def __init__(self, model_path: str):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = os.cpu_count()
        opts.inter_op_num_threads = max(1, os.cpu_count() // 2)
        opts.enable_mem_pattern = True
        opts.enable_mem_reuse   = True

        self.session = ort.InferenceSession(
            model_path, opts,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self._warmup()
        print(f"  ✓ ONNX Runtime 后端加载完成")
        print(f"    线程数: {os.cpu_count()} / 推理线程: {opts.intra_op_num_threads}")

    def _warmup(self, n: int = 3):
        dummy = np.random.randn(1, 1, IMG_HEIGHT, 256).astype(np.float32)
        for _ in range(n):
            self.session.run(None, {self.input_name: dummy})

    def infer(self, img_np: np.ndarray):
        """
        img_np: (1, 1, H, W) float32 numpy array
        Returns: output (SeqLen, 1, NumClasses) numpy array
        """
        outputs = self.session.run(None, {self.input_name: img_np})
        return outputs[0]


class OpenVINOBackend:
    """OpenVINO 后端（Intel CPU 专属）"""

    def __init__(self, model_path: str):
        try:
            from openvino import Core
        except ImportError:
            raise ImportError(
                "OpenVINO 未安装。请运行: pip install openvino\n"
                "或使用 --backend onnx 改用 ONNX Runtime"
            )

        ie = Core()
        model = ie.read_model(model=model_path)
        config = {
            "PERFORMANCE_HINT": "LATENCY",
        }
        self.compiled_model = ie.compile_model(model=model, device_name="CPU", config=config)
        self.infer_request = self.compiled_model.create_infer_request()
        self._warmup()
        print(f"  ✓ OpenVINO 后端加载完成（Intel CPU 优化）")

    def _warmup(self, n: int = 3):
        dummy = np.random.randn(1, 1, IMG_HEIGHT, 256).astype(np.float32)
        for _ in range(n):
            self.infer_request.infer({"input": dummy})

    def infer(self, img_np: np.ndarray):
        result = self.infer_request.infer({"input": img_np})
        outputs = list(result.values())
        return outputs[0]


class PyTorchBackend:
    """原始 PyTorch 后端（fallback，无优化）"""

    def __init__(self, model_path: str, keys_path: str):
        import torch
        from recognition.model import CRNN

        converter = LabelConverter(keys_path)
        self.model = CRNN(img_height=IMG_HEIGHT, nc=1, n_class=converter.num_classes)
        checkpoint = torch.load(model_path, map_location="cpu")
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()
        print(f"  ✓ PyTorch 后端加载完成（无优化）")

    def infer(self, img_np: np.ndarray):
        import torch
        tensor = torch.from_numpy(img_np)
        with torch.no_grad():
            output = self.model(tensor)
        return output.numpy()


def load_backend(model_path: str, backend: str, keys_path: str):
    """根据指定后端或文件扩展名自动选择最优后端。"""
    ext = os.path.splitext(model_path)[1].lower()

    if backend == "auto":
        if ext == ".onnx":
            backend = "onnx"
        elif ext == ".pth":
            backend = "pytorch"
        else:
            backend = "onnx"

    print(f"\n[后端] 使用: {backend.upper()}")

    if backend == "onnx":
        return ONNXBackend(model_path)
    elif backend == "openvino":
        return OpenVINOBackend(model_path)
    elif backend == "pytorch":
        return PyTorchBackend(model_path, keys_path)
    else:
        raise ValueError(f"未知后端: {backend}")


# ============================================================
# 图像预处理（与训练/推理保持一致）
# ============================================================

def preprocess(image_path: str, img_height: int = IMG_HEIGHT):
    """
    加载 & 预处理图像。
    与 recognition/dataset.py 中 InferenceDataset 的处理逻辑一致：
      RGB → 去红章 → 灰度 → Resize(固定高度) → Normalize(0.5, 0.5)
    返回: (img_np, original_width)
    """
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"  ✗ 无法读取图像 {image_path}: {e}")
        return None, 0

    # 去红章
    try:
        from recognition.dataset import remove_red_seal
        image = remove_red_seal(image)
    except ImportError:
        pass

    # 转灰度
    image = image.convert('L')
    w, h = image.size

    # Resize：固定高度，宽度按比例
    scale = img_height / h
    new_w = int(w * scale)

    # 高图宽度保护（与训练时一致）
    min_w_from_aspect = int(w * 0.5)
    if new_w < min_w_from_aspect:
        new_w = min(min_w_from_aspect, 800)

    image = image.resize((new_w, img_height), Image.Resampling.LANCZOS)

    # 最小宽度保护
    if new_w < MIN_WIDTH:
        canvas = Image.new('L', (MIN_WIDTH, img_height), 255)
        offset_x = (MIN_WIDTH - new_w) // 2
        canvas.paste(image, (offset_x, 0))
        image = canvas
        new_w = MIN_WIDTH

    # → numpy, normalize
    arr = np.array(image).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = arr[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)

    return arr.astype(np.float32), new_w


# ============================================================
# CTC 解码（纯 numpy 实现，不依赖 PyTorch）
# ============================================================

def ctc_decode_numpy(output_np: np.ndarray, converter: LabelConverter) -> list[str]:
    """
    对 ONNX/OpenVINO 的输出进行 CTC 贪心解码。
    output_np: (SeqLen, Batch, NumClasses) — log_softmax 后的输出
    """
    # argmax 得到 token 序列
    tokens = np.argmax(output_np, axis=2)  # (SeqLen, Batch)
    batch_size = tokens.shape[1]

    decoded_texts = []
    for b in range(batch_size):
        token_seq = tokens[:, b]  # (SeqLen,)
        char_list = []
        for j in range(len(token_seq)):
            token = int(token_seq[j])
            if token != 0 and (j == 0 or token != int(token_seq[j - 1])):
                char_list.append(converter.idx2char.get(token, ''))
        decoded_texts.append("".join(char_list))

    return decoded_texts


# ============================================================
# 单图推理
# ============================================================

def recognize_one(backend, image_path: str, converter: LabelConverter,
                  timing: bool = False):
    """
    对单张图像执行文字识别。
    返回: (识别文本, 推理耗时ms)
    """
    img_np, real_w = preprocess(image_path)
    if img_np is None:
        return "", 0.0

    # 前向推理
    t0 = time.perf_counter()
    output = backend.infer(img_np)
    infer_ms = (time.perf_counter() - t0) * 1000

    # CTC 解码
    texts = ctc_decode_numpy(output, converter)
    text = texts[0] if texts else ""

    if timing:
        fname = os.path.basename(image_path)
        print(f"  [{fname}] \"{text}\"  推理: {infer_ms:.1f}ms")

    return text, infer_ms


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="CRNN 优化推理（ONNX/INT8/OpenVINO）")
    parser.add_argument("--model_path",  type=str,
                        default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image_path",  type=str,
                        default=DEFAULT_IMAGE_PATH,
                        help="待识别的图像文件或目录")
    parser.add_argument("--keys_path",   type=str,
                        default=DEFAULT_KEYS_PATH,
                        help="字典文件路径")
    parser.add_argument("--output_file", type=str,
                        default=None,
                        help="结果保存路径（默认打印到控制台）")
    parser.add_argument("--backend",     type=str,
                        choices=["auto", "onnx", "openvino", "pytorch"],
                        default="auto",
                        help="推理后端（auto=自动选择）")
    parser.add_argument("--benchmark",   action="store_true",
                        help="显示每张图的推理耗时")
    args = parser.parse_args()

    print("=" * 60)
    print("CRNN 优化推理")
    print("=" * 60)
    print(f"  模型:    {args.model_path}")
    print(f"  输入:    {args.image_path}")
    print(f"  字典:    {args.keys_path}")
    print(f"  后端:    {args.backend}")
    print(f"  CPU核心: {os.cpu_count()}")
    print()

    if not os.path.exists(args.model_path):
        print(f"✗ 模型文件不存在: {args.model_path}")
        return

    if not os.path.exists(args.keys_path):
        print(f"✗ 字典文件不存在: {args.keys_path}")
        return

    # 加载字典
    converter = LabelConverter(args.keys_path)

    # 加载后端
    backend = load_backend(args.model_path, args.backend, args.keys_path)

    # 收集图像文件
    if os.path.isdir(args.image_path):
        images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif"]:
            images.extend(glob.glob(os.path.join(args.image_path, ext)))
            images.extend(glob.glob(os.path.join(args.image_path, ext.upper())))
        images.sort()
    elif os.path.isfile(args.image_path):
        images = [args.image_path]
    else:
        print(f"✗ 路径不存在: {args.image_path}")
        return

    if not images:
        print("✗ 未找到图像文件")
        return

    print(f"\n共 {len(images)} 张图像，开始推理...\n")

    # 批量推理
    results = []
    total_ms = 0.0
    t_wall_start = time.perf_counter()

    for img_path in images:
        text, ms = recognize_one(
            backend, img_path, converter,
            timing=args.benchmark
        )
        results.append((os.path.basename(img_path), text))
        total_ms += ms

    t_wall = time.perf_counter() - t_wall_start

    # 输出结果
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

    # 汇总统计
    print(f"\n{'='*60}")
    print(f"推理完成")
    print(f"  图像数:         {len(images)} 张")
    print(f"  纯推理均值:     {total_ms/len(images):.1f} ms/张")
    print(f"  总耗时(含预处理): {t_wall*1000:.0f} ms")
    print(f"  吞吐量:         {len(images)/t_wall:.2f} 张/秒")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
