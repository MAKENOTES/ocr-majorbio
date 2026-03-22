"""
detection_quantize_int8.py — EAST ONNX 模型 INT8 静态量化 (v2)
=====================================================================
在 ONNX FP32 模型基础上，通过校准数据集进行 INT8 静态量化。

两种量化格式：
  - QOperator (默认)：权重直接存为 INT8 → 文件体积缩小 ~4x
    Conv → QLinearConv, 权重物理存储为 int8
  - QDQ：插入 QuantizeLinear/DeQuantizeLinear 节点 → 文件体积不变
    但 ONNX Runtime 加载时自动融合，兼容性更好

精度保护策略：
  - 自动排除输出头 (score_conv / geo_conv) 不做量化
    score_map 经过 sigmoid 映射到 [0,1]，对量化误差极敏感
    geo_map 是像素级距离回归，量化噪声会直接传播到检测框坐标
  - 支持 per-channel 量化提升精度
  - 支持 Entropy / Percentile 校准方法

用法:
    # 推荐用法：QOperator + 真实校准图像
    python detection_quantize_int8.py \
        --input_model best_detection.onnx \
        --output_model best_detection_int8.onnx \
        --calib_dir detection/data/val_images \
        --calib_num 100

    # 高精度配置
    python detection_quantize_int8.py \
        --input_model best_detection.onnx \
        --output_model best_detection_int8.onnx \
        --calib_dir detection/data/val_images \
        --per_channel \
        --calibrate_method entropy

    # 使用 QDQ 格式（兼容性更好，体积不缩小但速度一样快）
    python detection_quantize_int8.py \
        --input_model best_detection.onnx \
        --output_model best_detection_int8_qdq.onnx \
        --calib_dir detection/data/val_images \
        --quant_format qdq

依赖:
    pip install onnx onnxruntime numpy Pillow
"""

import os
import sys
import glob
import time
import argparse
import numpy as np
from pathlib import Path

# ============================================================
# 常量（与训练 / 推理保持一致）
# ============================================================

INPUT_SIZE = 640
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ============================================================
# 图像预处理（与 detection_inference.py 完全一致）
# ============================================================

def preprocess_image(image_path: str, input_size: int = INPUT_SIZE) -> np.ndarray:
    """
    加载并预处理单张图像，返回 (1, 3, H, W) float32 numpy array。
    预处理逻辑与 detection_inference.py 中 preprocess() 完全一致：
      1. 保持宽高比 resize
      2. 灰色 (128,128,128) padding 到正方形
      3. /255 → ImageNet normalize → HWC→CHW
    """
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size

    scale = min(input_size / orig_w, input_size / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    resized = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    padded = Image.new("RGB", (input_size, input_size), (128, 128, 128))
    padded.paste(resized, ((input_size - new_w) // 2, (input_size - new_h) // 2))

    arr = np.array(padded).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)
    return arr


# ============================================================
# 校准数据读取器
# ============================================================

class EASTCalibrationDataReader:
    """
    校准数据读取器，支持两种模式：
      1. 真实图像：从指定目录读取图像并预处理
      2. 随机数据：生成符合 ImageNet normalize 分布的随机张量
    """

    def __init__(self, input_name: str, calib_dir: str = None,
                 calib_num: int = 50, input_size: int = INPUT_SIZE):
        self.input_name = input_name
        self.input_size = input_size
        self.data_list = []
        self.idx = 0

        if calib_dir and os.path.isdir(calib_dir):
            image_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                image_files.extend(glob.glob(os.path.join(calib_dir, ext)))
                image_files.extend(glob.glob(os.path.join(calib_dir, "**", ext), recursive=True))
            image_files = sorted(set(image_files))

            if len(image_files) == 0:
                print(f"  ⚠ 校准目录 {calib_dir} 中未找到图像，回退到随机数据")
                self._generate_random(calib_num)
            else:
                if len(image_files) > calib_num:
                    indices = np.linspace(0, len(image_files) - 1, calib_num, dtype=int)
                    image_files = [image_files[i] for i in indices]

                print(f"  校准图像: {len(image_files)} 张 (来自 {calib_dir})")
                for fpath in image_files:
                    try:
                        arr = preprocess_image(fpath, input_size)
                        self.data_list.append(arr)
                    except Exception as e:
                        print(f"    跳过 {os.path.basename(fpath)}: {e}")

                if len(self.data_list) == 0:
                    print(f"  ⚠ 所有图像读取失败，回退到随机数据")
                    self._generate_random(calib_num)
                else:
                    print(f"  成功加载 {len(self.data_list)} 张校准图像")
        else:
            if calib_dir:
                print(f"  ⚠ 校准目录 {calib_dir} 不存在，使用随机数据")
            else:
                print(f"  使用随机校准数据 ({calib_num} 条)")
            self._generate_random(calib_num)

    def _generate_random(self, n: int):
        for _ in range(n):
            fake_img = np.random.rand(1, 3, self.input_size, self.input_size).astype(np.float32)
            for c in range(3):
                fake_img[0, c] = (fake_img[0, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
            self.data_list.append(fake_img)

    def get_next(self):
        if self.idx >= len(self.data_list):
            return None
        data = self.data_list[self.idx]
        self.idx += 1
        return {self.input_name: data}

    def rewind(self):
        self.idx = 0


# ============================================================
# 自动发现输出头节点（排除量化以保护精度）
# ============================================================

def find_output_head_nodes(model_path: str) -> list:
    """
    自动扫描 ONNX 图，找到直接连接到模型输出的 Conv 节点。

    EAST 模型结构：
      ... → fuse1 → score_conv (Conv 32→1) → Sigmoid → score_map
      ... → fuse1 → geo_conv (Conv 32→4) → geo_map

    score_conv 输出经过 sigmoid 映射到 [0,1]，量化会引入较大误差。
    geo_conv 输出是像素级距离回归值，量化噪声会直接影响检测框坐标。
    这两个头部 Conv 应该保留 FP32 精度。
    """
    import onnx

    model = onnx.load(model_path)
    graph = model.graph

    # 收集所有输出张量名
    output_names = {o.name for o in graph.output}

    # 构建 tensor_name → consumer_nodes 映射
    tensor_consumers = {}
    for node in graph.node:
        for inp in node.input:
            if inp not in tensor_consumers:
                tensor_consumers[inp] = []
            tensor_consumers[inp].append(node)

    exclude_nodes = []

    for node in graph.node:
        for out in node.output:
            # 直接连接到模型输出（geo_conv 的情况）
            if out in output_names:
                exclude_nodes.append(node.name)
                continue

            # 经过 Sigmoid 后连接到模型输出（score_conv 的情况）
            if out in tensor_consumers:
                for consumer in tensor_consumers[out]:
                    if consumer.op_type == "Sigmoid":
                        for consumer_out in consumer.output:
                            if consumer_out in output_names:
                                exclude_nodes.append(node.name)

    if exclude_nodes:
        print(f"  自动排除输出头节点: {exclude_nodes}")
        print(f"    → 这些节点保留 FP32 精度，避免 score/geo 量化误差")
    else:
        print(f"  ⚠ 未自动发现输出头节点")

    return exclude_nodes


# ============================================================
# 模型预处理
# ============================================================

def preprocess_model(input_path: str) -> str:
    """ONNX Runtime 官方推荐的量化前预处理"""
    from onnxruntime.quantization.shape_inference import quant_pre_process

    output_path = input_path.replace(".onnx", "_prep.onnx")
    print(f"  预处理模型: {input_path} → {output_path}")
    try:
        quant_pre_process(
            input_model_path=input_path,
            output_model_path=output_path,
            skip_symbolic_shape=False,
            skip_optimization=False,
            skip_onnx_shape=False,
            auto_merge=True,
            verbose=0,
        )
        print(f"  ✓ 预处理完成")
        return output_path
    except Exception as e:
        print(f"  ⚠ 预处理失败 ({e})，将直接使用原始模型")
        return input_path


# ============================================================
# 主量化流程
# ============================================================

def quantize_int8(
    input_model: str,
    output_model: str,
    calib_dir: str = None,
    calib_num: int = 50,
    per_channel: bool = False,
    calibrate_method: str = "MinMax",
    input_size: int = INPUT_SIZE,
    quant_format: str = "qoperator",
    reduce_range: bool = False,
):
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import (
        quantize_static,
        CalibrationMethod,
        QuantFormat,
        QuantType,
    )

    print("=" * 60)
    print("EAST 模型 INT8 静态量化 v2")
    print("=" * 60)

    fmt_str = quant_format.upper()
    if quant_format.lower() in ("qoperator", "qop"):
        q_format = QuantFormat.QOperator
        # QOperator 用 U8S8：权重物理存为 int8，文件体积缩小
        act_type = QuantType.QUInt8
        wgt_type = QuantType.QInt8
        fmt_str = "QOperator (U8S8) → 文件体积缩小"
    else:
        q_format = QuantFormat.QDQ
        # QDQ 用 S8S8：ONNX Runtime 默认推荐
        act_type = QuantType.QInt8
        wgt_type = QuantType.QInt8
        fmt_str = "QDQ (S8S8) → 文件体积不变，运行时融合"

    print(f"  输入模型:     {input_model}")
    print(f"  输出模型:     {output_model}")
    print(f"  量化格式:     {fmt_str}")
    print(f"  校准目录:     {calib_dir or '(随机数据)'}")
    print(f"  校准样本数:   {calib_num}")
    print(f"  Per-channel:  {per_channel}")
    print(f"  Reduce-range: {reduce_range}")
    print(f"  校准方法:     {calibrate_method}")
    print(f"  输入尺寸:     {input_size}x{input_size}")
    print()

    if not os.path.exists(input_model):
        print(f"✗ 输入模型不存在: {input_model}")
        return

    orig_size = os.path.getsize(input_model) / 1024 / 1024
    print(f"[1/6] 原始 FP32 模型大小: {orig_size:.1f} MB")

    # ---- Step 1: 发现需要排除的输出头节点 ----
    print(f"\n[2/6] 分析模型结构，定位敏感输出节点...")
    exclude_nodes = find_output_head_nodes(input_model)

    # ---- Step 2: 预处理 ----
    print(f"\n[3/6] 模型预处理 (shape inference + optimization)...")
    prep_model_path = preprocess_model(input_model)

    # 预处理后节点名可能变化，重新扫描
    if prep_model_path != input_model:
        exclude_nodes_prep = find_output_head_nodes(prep_model_path)
        if exclude_nodes_prep:
            exclude_nodes = exclude_nodes_prep

    # ---- Step 3: 准备校准数据 ----
    print(f"\n[4/6] 准备校准数据...")
    sess = ort.InferenceSession(prep_model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    del sess

    calib_reader = EASTCalibrationDataReader(
        input_name=input_name,
        calib_dir=calib_dir,
        calib_num=calib_num,
        input_size=input_size,
    )

    # ---- Step 4: 执行静态量化 ----
    print(f"\n[5/6] 执行 INT8 静态量化 ({q_format.name})...")

    if calibrate_method.lower() == "entropy":
        cal_method = CalibrationMethod.Entropy
    elif calibrate_method.lower() == "percentile":
        cal_method = CalibrationMethod.Percentile
    else:
        cal_method = CalibrationMethod.MinMax

    os.makedirs(os.path.dirname(output_model) if os.path.dirname(output_model) else ".", exist_ok=True)

    # 构建 extra_options
    extra_opts = {
        "WeightSymmetric": True,
        "ActivationSymmetric": False,
    }

    # QDQ 特有选项
    if q_format == QuantFormat.QDQ:
        extra_opts["AddQDQPairToWeight"] = True
        extra_opts["QuantizeBias"] = True
        extra_opts["ForceQuantizeNoInputCheck"] = True

    t0 = time.perf_counter()

    quantize_static(
        model_input=prep_model_path,
        model_output=output_model,
        calibration_data_reader=calib_reader,
        quant_format=q_format,
        activation_type=act_type,
        weight_type=wgt_type,
        per_channel=per_channel,
        reduce_range=reduce_range,
        calibrate_method=cal_method,
        # 量化所有支持的算子
        op_types_to_quantize=["Conv", "ConvTranspose", "MatMul", "Gemm"],
        # 排除输出头节点（保护 score/geo 精度）
        nodes_to_exclude=exclude_nodes if exclude_nodes else None,
        extra_options=extra_opts,
    )

    quant_time = time.perf_counter() - t0
    quant_size = os.path.getsize(output_model) / 1024 / 1024

    print(f"  ✓ 量化完成！ (耗时 {quant_time:.1f}s)")
    print(f"  FP32 模型:  {orig_size:.1f} MB")
    print(f"  INT8 模型:  {quant_size:.1f} MB")
    print(f"  压缩比:     {orig_size / quant_size:.2f}x")

    if q_format == QuantFormat.QDQ and abs(orig_size - quant_size) < 1.0:
        print(f"  说明: QDQ 格式文件体积不缩小是正常的（权重仍存为 FP32）")
        print(f"        ONNX Runtime 加载时自动融合 QDQ 节点并用 INT8 计算")
        print(f"        若需文件体积也缩小，请使用 --quant_format qoperator")

    # ---- 清理 ----
    if prep_model_path != input_model and os.path.exists(prep_model_path):
        os.remove(prep_model_path)

    # ---- Step 5: 验证精度 ----
    print(f"\n[6/6] 验证 FP32 vs INT8 输出一致性...")
    try:
        _verify_accuracy(input_model, output_model, input_size)
    except Exception as e:
        print(f"  ⚠ 验证失败: {e}")

    # ---- Benchmark ----
    print(f"\n[Benchmark] 速度对比 (CPU, 5次预热 + 20次计时)...")
    try:
        _benchmark(input_model, output_model, input_size)
    except Exception as e:
        print(f"  ⚠ Benchmark 失败: {e}")

    print(f"\n{'='*60}")
    print(f"✓ INT8 静态量化完成: {output_model}")
    print(f"  模型大小: {orig_size:.1f} MB → {quant_size:.1f} MB ({orig_size/quant_size:.1f}x 压缩)")
    print(f"  用法: 直接替换推理脚本中的模型路径")
    print(f"  示例: python detection_inference.py --model_path {output_model}")
    print(f"{'='*60}")


# ============================================================
# 验证：FP32 vs INT8
# ============================================================

def _verify_accuracy(fp32_model: str, int8_model: str, input_size: int):
    import onnxruntime as ort

    dummy = np.random.rand(1, 3, input_size, input_size).astype(np.float32)
    for c in range(3):
        dummy[0, c] = (dummy[0, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess_fp32 = ort.InferenceSession(fp32_model, opts, providers=["CPUExecutionProvider"])
    input_name = sess_fp32.get_inputs()[0].name
    fp32_outs = sess_fp32.run(None, {input_name: dummy})

    sess_int8 = ort.InferenceSession(int8_model, opts, providers=["CPUExecutionProvider"])
    input_name_int8 = sess_int8.get_inputs()[0].name
    int8_outs = sess_int8.run(None, {input_name_int8: dummy})

    # 按 shape 匹配输出（通道数少的是 score，多的是 geo）
    fp32_sorted = sorted(fp32_outs, key=lambda x: x.shape[1])
    int8_sorted = sorted(int8_outs, key=lambda x: x.shape[1])

    fp32_score, fp32_geo = fp32_sorted[0], fp32_sorted[1]
    int8_score, int8_geo = int8_sorted[0], int8_sorted[1]

    score_max_err = np.max(np.abs(fp32_score - int8_score))
    score_mae = np.mean(np.abs(fp32_score - int8_score))
    geo_max_err = np.max(np.abs(fp32_geo - int8_geo))
    geo_mae = np.mean(np.abs(fp32_geo - int8_geo))

    # score_map 相关性
    fp32_flat = fp32_score.flatten()
    int8_flat = int8_score.flatten()
    if np.std(fp32_flat) > 1e-8 and np.std(int8_flat) > 1e-8:
        score_corr = np.corrcoef(fp32_flat, int8_flat)[0, 1]
    else:
        score_corr = 1.0

    print(f"  score_map 最大误差:   {score_max_err:.4e}")
    print(f"  score_map 平均误差:   {score_mae:.4e}")
    print(f"  score_map 相关系数:   {score_corr:.6f}")
    print(f"  geo_map   最大误差:   {geo_max_err:.4e}")
    print(f"  geo_map   平均误差:   {geo_mae:.4e}")

    if score_corr > 0.99:
        print(f"  ✓ 精度优秀，量化几乎无损")
    elif score_corr > 0.97:
        print(f"  ✓ 精度良好，适合生产使用")
    elif score_corr > 0.95:
        print(f"  △ 精度可接受，建议在实际图像上验证检测效果")
    else:
        print(f"  ⚠ 精度损失较大，建议：")
        print(f"    1. 尝试 per-channel 量化 (--per_channel)")
        print(f"    2. 尝试 Entropy 校准 (--calibrate_method entropy)")
        print(f"    3. 增加校准样本 (--calib_num 200)")
        print(f"    4. 非 VNNI CPU 尝试 --reduce_range")


# ============================================================
# Benchmark
# ============================================================

def _benchmark(fp32_model: str, int8_model: str, input_size: int):
    import onnxruntime as ort

    dummy = np.random.rand(1, 3, input_size, input_size).astype(np.float32)

    def bench_model(model_path, label, warmup=5, repeat=20):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = os.cpu_count()
        opts.inter_op_num_threads = max(1, os.cpu_count() // 2)
        opts.enable_mem_pattern = True
        opts.enable_mem_reuse = True

        sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        feed = {input_name: dummy}

        for _ in range(warmup):
            sess.run(None, feed)
        t0 = time.perf_counter()
        for _ in range(repeat):
            sess.run(None, feed)
        avg_ms = (time.perf_counter() - t0) / repeat * 1000
        print(f"  {label:20s} {avg_ms:.1f} ms/张")
        return avg_ms

    fp32_ms = bench_model(fp32_model, "ONNX FP32:")
    int8_ms = bench_model(int8_model, "ONNX INT8:")
    print(f"  {'INT8 加速比:':<20s} {fp32_ms / int8_ms:.2f}x")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EAST ONNX 模型 INT8 静态量化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # QOperator 格式（文件缩小 + 加速）— 推荐
  python detection_quantize_int8.py \\
      --input_model best_detection.onnx \\
      --output_model best_detection_int8.onnx \\
      --calib_dir detection/data/val_images

  # QDQ 格式（兼容性好，体积不变但速度一样）
  python detection_quantize_int8.py \\
      --input_model best_detection.onnx \\
      --output_model best_detection_int8_qdq.onnx \\
      --calib_dir detection/data/val_images \\
      --quant_format qdq

  # 高精度配置
  python detection_quantize_int8.py \\
      --input_model best_detection.onnx \\
      --output_model best_detection_int8.onnx \\
      --calib_dir detection/data/val_images \\
      --per_channel --calibrate_method entropy
        """,
    )

    parser.add_argument("--input_model", type=str, default="best_detection.onnx")
    parser.add_argument("--output_model", type=str, default="best_detection_int8_static.onnx")
    parser.add_argument("--calib_dir", type=str, default="detection/data/val_images",
                        help="校准图像目录")
    parser.add_argument("--calib_num", type=int, default=100,
                        help="校准样本数量（默认 100）")
    parser.add_argument("--quant_format", type=str, default="qoperator",
                        choices=["qoperator", "qop", "qdq"],
                        help="量化格式：qoperator(默认,体积缩小) | qdq(兼容性好)")
    parser.add_argument("--per_channel", action="store_true",
                        help="per-channel 量化（精度更高）")
    parser.add_argument("--reduce_range", action="store_true",
                        help="7-bit 量化范围（非 VNNI CPU 可改善精度）")
    parser.add_argument("--calibrate_method", type=str, default="MinMax",
                        choices=["MinMax", "Entropy", "Percentile"])
    parser.add_argument("--input_size", type=int, default=INPUT_SIZE)

    args = parser.parse_args()

    quantize_int8(
        input_model=args.input_model,
        output_model=args.output_model,
        calib_dir=args.calib_dir,
        calib_num=args.calib_num,
        per_channel=args.per_channel,
        calibrate_method=args.calibrate_method,
        input_size=args.input_size,
        quant_format=args.quant_format,
        reduce_range=args.reduce_range,
    )