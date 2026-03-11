"""
quantize_int8.py — 对 ONNX 模型做 INT8 后训练量化（PTQ）
=====================================================================
第二步：INT8 量化（在 ONNX 基础上再提速 3-4x）

两种量化模式：
  - 动态量化（无需校准数据，快速，推荐优先尝试）
  - 静态量化（需要校准数据集，精度更好，速度更快）

用法（动态量化，最快上手）:
    python quantize_int8.py \
        --onnx_path detection/checkpoints/east_model.onnx \
        --mode dynamic

用法（静态量化，精度更优）:
    python quantize_int8.py \
        --onnx_path detection/checkpoints/east_model.onnx \
        --mode static \
        --calib_image_dir detection/data/val_images \
        --num_calib 100

依赖:
    pip install onnxruntime onnx onnxruntime-tools
    # 静态量化还需要：
    pip install Pillow torchvision
"""

import os
import sys
import argparse
import numpy as np
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 工具函数
# ============================================================

def preprocess_image_for_calib(image_path: str, input_size: int = 1024) -> np.ndarray:
    """
    预处理单张图像用于校准（与训练时保持一致的归一化）。
    ResizeAndPad + ImageNet normalize
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # ResizeAndPad：保持宽高比，padding 灰色
    scale = min(input_size / w, input_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    padded = Image.new("RGB", (input_size, input_size), (128, 128, 128))
    padded.paste(img, ((input_size - new_w) // 2, (input_size - new_h) // 2))

    # → numpy, normalize
    arr = np.array(padded).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)           # HWC → CHW
    arr = np.expand_dims(arr, axis=0)      # → (1, 3, H, W)
    return arr.astype(np.float32)


class CalibrationDataReader:
    """
    给 ONNX Runtime 静态量化提供校准数据。
    从 val_images 目录随机采样 num_calib 张图像。
    """
    def __init__(self, image_dir: str, num_calib: int = 100, input_size: int = 1024):
        import glob
        all_images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            all_images.extend(glob.glob(os.path.join(image_dir, ext)))

        if not all_images:
            raise ValueError(f"在 {image_dir} 下未找到图像文件")

        np.random.shuffle(all_images)
        self.image_paths = all_images[:num_calib]
        self.input_size = input_size
        self.index = 0
        print(f"    校准数据集: {len(self.image_paths)} 张图像")

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None
        batch = preprocess_image_for_calib(self.image_paths[self.index], self.input_size)
        self.index += 1
        # 每10张打印一次进度
        if self.index % 10 == 0:
            print(f"    校准进度: {self.index}/{len(self.image_paths)}")
        return {"input": batch}


# ============================================================
# 动态量化（最简单，无需校准数据）
# ============================================================

def quantize_dynamic(onnx_path: str, output_path: str) -> str:
    """
    动态量化：权重量化为 INT8，激活在推理时动态量化。
    无需校准数据集，速度提升约 2-3x，精度损失极小。
    """
    print("\n[动态量化] 开始...")
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_input=onnx_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,        # 权重量化为 INT8
        optimize_model=True,                # 自动做算子融合
        extra_options={
            "WeightSymmetric": True,        # 对称量化，精度更好
            "ActivationSymmetric": False,
        }
    )
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"    ✓ 动态量化完成: {output_path} ({size_mb:.1f} MB)")
    return output_path


# ============================================================
# 静态量化（需要校准数据，精度 & 速度最优）
# ============================================================

def quantize_static(onnx_path: str, output_path: str,
                    calib_image_dir: str, num_calib: int = 100,
                    input_size: int = 1024) -> str:
    """
    静态量化：通过校准数据集确定激活值分布，权重和激活都量化为 INT8。
    速度提升约 3-4x，精度损失比动态量化更小。
    """
    print("\n[静态量化] 开始，这可能需要几分钟...")
    from onnxruntime.quantization import (
        quantize_static, QuantType, QuantFormat,
        CalibrationMethod
    )
    from onnxruntime.quantization import quant_pre_process

    # Step 1: 预处理模型（形状推断 + 算子融合，提升量化质量）
    preprocessed_path = onnx_path.replace(".onnx", "_preprocessed.onnx")
    print(f"    Step 1/3: 预处理模型（形状推断）...")
    try:
        quant_pre_process(onnx_path, preprocessed_path, skip_optimization=False)
        source_model = preprocessed_path
        print(f"    ✓ 预处理完成")
    except Exception as e:
        print(f"    ⚠ 预处理失败({e})，使用原始模型继续")
        source_model = onnx_path

    # Step 2: 准备校准数据
    print(f"    Step 2/3: 准备校准数据...")
    calib_reader = CalibrationDataReader(calib_image_dir, num_calib, input_size)

    # Step 3: 静态量化
    print(f"    Step 3/3: 执行静态量化（MinMax 校准）...")
    quantize_static(
        model_input=source_model,
        model_output=output_path,
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,           # QDQ 格式，兼容性最好
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={
            "ActivationSymmetric": False,       # 激活非对称，精度更好
            "WeightSymmetric": True,
            "EnableSubgraph": False,
        }
    )

    # 清理预处理临时文件
    if os.path.exists(preprocessed_path) and preprocessed_path != onnx_path:
        os.remove(preprocessed_path)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"    ✓ 静态量化完成: {output_path} ({size_mb:.1f} MB)")
    return output_path


# ============================================================
# 速度 & 精度 Benchmark
# ============================================================

def benchmark_and_compare(fp32_path: str, quant_path: str, input_size: int = 1024,
                           n_warmup: int = 5, n_runs: int = 30):
    """
    对比 FP32 ONNX 与 INT8 量化模型的速度和精度差异。
    """
    import onnxruntime as ort

    print(f"\n{'='*60}")
    print(f"[Benchmark] FP32 vs INT8 对比")
    print(f"{'='*60}")

    dummy = np.random.randn(1, 3, input_size, input_size).astype(np.float32)

    def make_session(model_path):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = os.cpu_count()
        return ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])

    def run_benchmark(session, inp, n_warmup, n_runs):
        feeds = {"input": inp}
        for _ in range(n_warmup):
            session.run(None, feeds)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            out = session.run(None, feeds)
        elapsed = (time.perf_counter() - t0) / n_runs * 1000
        return elapsed, out

    # FP32
    print(f"\n  加载 FP32 模型...")
    sess_fp32 = make_session(fp32_path)
    t_fp32, out_fp32 = run_benchmark(sess_fp32, dummy, n_warmup, n_runs)
    fp32_size = os.path.getsize(fp32_path) / 1024 / 1024

    # INT8
    print(f"  加载 INT8 模型...")
    sess_int8 = make_session(quant_path)
    t_int8, out_int8 = run_benchmark(sess_int8, dummy, n_warmup, n_runs)
    int8_size = os.path.getsize(quant_path) / 1024 / 1024

    # 精度对比
    score_diff = np.max(np.abs(out_fp32[0] - out_int8[0]))
    geo_diff   = np.max(np.abs(out_fp32[1] - out_int8[1]))

    print(f"\n  {'指标':<20} {'FP32':>12} {'INT8':>12} {'对比':>12}")
    print(f"  {'-'*56}")
    print(f"  {'推理时间 (ms)':<20} {t_fp32:>11.1f} {t_int8:>11.1f} {t_fp32/t_int8:>10.2f}x")
    print(f"  {'模型大小 (MB)':<20} {fp32_size:>11.1f} {int8_size:>11.1f} {fp32_size/int8_size:>10.2f}x")
    print(f"  {'score_map 最大误差':<20} {'—':>12} {score_diff:>11.2e} {'':>12}")
    print(f"  {'geo_map 最大误差':<20} {'—':>12} {geo_diff:>11.2e} {'':>12}")
    print(f"\n  ★ INT8 量化加速比: {t_fp32/t_int8:.2f}x")
    print(f"  ★ 模型压缩比:      {fp32_size/int8_size:.2f}x")

    if score_diff < 0.05:
        print(f"\n  ✓ 精度损失在可接受范围内（score_map 误差 < 0.05）")
    else:
        print(f"\n  ⚠ score_map 误差较大，建议使用静态量化或增加校准数据")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="EAST ONNX 模型 INT8 量化工具")
    parser.add_argument("--onnx_path", type=str,
                        default="best_detection.onnx",
                        help="输入 FP32 ONNX 模型路径")
    parser.add_argument("--mode", type=str, choices=["dynamic", "static"],
                        default="static",
                        help="量化模式：dynamic（快速）或 static（更精确，需要校准集）")
    parser.add_argument("--output_path", type=str,
                        default=None,
                        help="输出量化模型路径（默认在原文件名加 _int8 后缀）")
    parser.add_argument("--calib_image_dir", type=str,
                        default="detection/data/val_images",
                        help="静态量化的校准图像目录（仅 static 模式需要）")
    parser.add_argument("--num_calib", type=int,
                        default=100,
                        help="校准图像数量（建议 50-200 张）")
    parser.add_argument("--input_size", type=int,
                        default=640,
                        help="模型输入尺寸")
    parser.add_argument("--no_benchmark", action="store_true",
                        help="跳过 benchmark 测试")
    args = parser.parse_args()

    # 确定输出路径
    if args.output_path is None:
        base = args.onnx_path.replace(".onnx", "")
        args.output_path = f"{base}_int8_{args.mode}.onnx"

    print("=" * 60)
    print(f"EAST ONNX → INT8 量化工具 (模式: {args.mode})")
    print("=" * 60)
    print(f"  输入模型:  {args.onnx_path}")
    print(f"  输出模型:  {args.output_path}")
    print(f"  量化模式:  {args.mode}")

    # 执行量化
    if args.mode == "dynamic":
        quantize_dynamic(args.onnx_path, args.output_path)
    else:
        quantize_static(
            args.onnx_path, args.output_path,
            args.calib_image_dir, args.num_calib, args.input_size
        )

    # Benchmark 对比
    if not args.no_benchmark:
        benchmark_and_compare(args.onnx_path, args.output_path,
                               input_size=args.input_size)

    print(f"\n{'='*60}")
    print(f"✓ 量化完成！")
    print(f"  下一步: 使用 inference_optimized.py 进行推理")
    print(f"  命令示例:")
    print(f"    python inference_optimized.py \\")
    print(f"      --model_path {args.output_path} \\")
    print(f"      --image_path your_image.jpg \\")
    print(f"      --backend onnx")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()