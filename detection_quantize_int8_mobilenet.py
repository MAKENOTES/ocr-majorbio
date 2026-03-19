"""
detection_quantize_int8.py — 对 ONNX 模型做 INT8 后训练量化（PTQ）
=====================================================================
第二步：INT8 量化（在 ONNX 基础上再提速 2-4x）

★ 修复版：增加了对预测头的强力逆向追溯保护，并切换至 Entropy 校准。
"""

import os
import sys
import argparse
import numpy as np
import time
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# 敏感层名称列表 — 这些层量化后误差极大，必须跳过
# ============================================================

def get_sensitive_op_names(onnx_path: str) -> list:
    """
    自动扫描 ONNX 图，找出对量化敏感的层。

    ★ 核心修复：基于图遍历的逆向追溯保护。
    不仅仅依赖算子类型，而是直接从输出节点往回找，将预测头的所有关键层
    （Sigmoid, 1x1 Conv, 3x3 Conv）全部强制保留为 FP32。
    """
    import onnx

    model = onnx.load(onnx_path)
    graph = model.graph
    nodes_to_exclude = []

    output_names = {o.name for o in graph.output}

    # 构建张量名称到生产该张量的节点映射
    tensor_to_producer = {}
    for node in graph.node:
        for out in node.output:
            tensor_to_producer[out] = node

    # 向前追溯函数
    def trace_back(tensor_name, depth):
        if depth == 0 or tensor_name not in tensor_to_producer:
            return []
        node = tensor_to_producer[tensor_name]
        node_name = node.name or node.op_type
        res = [node_name]
        for inp in node.input:
            res.extend(trace_back(inp, depth - 1))
        return res

    # 1) 强力保护：从最终输出向前追溯 4 层
    # 足够覆盖：Sigmoid -> Conv(1x1) -> Relu/BN -> Conv(3x3)
    for out_name in output_names:
        protected_nodes = trace_back(out_name, depth=4)
        nodes_to_exclude.extend(protected_nodes)

    # 2) 保护其他容易造成精度坍塌的骨干层
    for node in graph.node:
        name = node.name or node.op_type
        op = node.op_type
        # 保护反卷积（避免棋盘格伪影）和残差相加（误差叠加）
        if op in ["ConvTranspose", "Add"]:
            nodes_to_exclude.append(name)

    # 去重
    nodes_to_exclude = list(set(nodes_to_exclude))
    print(f"    检测到 {len(nodes_to_exclude)} 个敏感层将跳过量化 (包含预测头追溯):")
    for n in sorted(nodes_to_exclude)[:15]:
        print(f"      - {n}")
    if len(nodes_to_exclude) > 15:
        print(f"      ... 及其他 {len(nodes_to_exclude) - 15} 个")

    return nodes_to_exclude


# ============================================================
# 工具函数
# ============================================================

def preprocess_image_for_calib(image_path: str, input_size: int = 1024) -> np.ndarray:
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    scale = min(input_size / w, input_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    padded = Image.new("RGB", (input_size, input_size), (128, 128, 128))
    padded.paste(img, ((input_size - new_w) // 2, (input_size - new_h) // 2))

    arr = np.array(padded).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)
    return arr.astype(np.float32)


class CalibrationDataReader:
    def __init__(self, image_dir: str, num_calib: int = 100, input_size: int = 1024):
        import glob
        all_images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            all_images.extend(glob.glob(os.path.join(image_dir, ext)))
            all_images.extend(glob.glob(os.path.join(image_dir, ext.upper())))

        if not all_images:
            raise ValueError(f"在 {image_dir} 下未找到图像文件")

        all_images = list(set(all_images))
        np.random.shuffle(all_images)
        self.image_paths = all_images[:num_calib]
        self.input_size = input_size
        self.index = 0
        self.total = len(self.image_paths)
        print(f"    校准数据集: {self.total} 张图像")

    def get_next(self):
        if self.index >= self.total:
            return None
        try:
            batch = preprocess_image_for_calib(
                self.image_paths[self.index], self.input_size
            )
        except Exception as e:
            print(f"    ⚠ 跳过损坏图像: {self.image_paths[self.index]} ({e})")
            self.index += 1
            return self.get_next()

        self.index += 1
        if self.index % 10 == 0 or self.index == self.total:
            print(f"    校准进度: {self.index}/{self.total}")
        return {"input": batch}

    def rewind(self):
        self.index = 0


# ============================================================
# 动态量化
# ============================================================

def do_quantize_dynamic(onnx_path: str, output_path: str) -> str:
    print("\n[动态量化] 开始...")
    from onnxruntime.quantization import quantize_dynamic, QuantType

    sensitive_nodes = get_sensitive_op_names(onnx_path)

    quantize_dynamic(
        model_input=onnx_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=sensitive_nodes,
        extra_options={
            "WeightSymmetric": True,
            "ActivationSymmetric": False,
        }
    )
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"    ✓ 动态量化完成: {output_path} ({size_mb:.1f} MB)")
    return output_path


# ============================================================
# 静态量化
# ============================================================

def do_quantize_static(onnx_path: str, output_path: str,
                       calib_image_dir: str, num_calib: int = 100,
                       input_size: int = 1024) -> str:
    print("\n[静态量化] 开始，这可能需要几分钟...")
    from onnxruntime.quantization import (
        quantize_static as _quantize_static,
        QuantType, QuantFormat, CalibrationMethod
    )
    from onnxruntime.quantization import quant_pre_process

    preprocessed_path = onnx_path.replace(".onnx", "_preprocessed.onnx")
    print(f"    Step 1/4: 预处理模型（形状推断 + 算子融合）...")
    try:
        quant_pre_process(
            input_model=onnx_path,
            output_model_path=preprocessed_path,
            skip_optimization=False,
        )
        source_model = preprocessed_path
        print(f"    ✓ 预处理完成")
    except Exception as e:
        print(f"    ⚠ 预处理失败({e})，使用原始模型继续")
        source_model = onnx_path

    print(f"    Step 2/4: 分析模型结构，检测量化敏感层...")
    sensitive_nodes = get_sensitive_op_names(source_model)

    print(f"    Step 3/4: 准备校准数据...")
    calib_reader = CalibrationDataReader(calib_image_dir, num_calib, input_size)

    print(f"    Step 4/4: 执行静态量化...")

    # ★ 核心修复：将 MinMax 切换为 Entropy (KL-Divergence)
    calib_method = CalibrationMethod.Entropy
    print(f"    校准方法: Entropy (KL散度) - 对目标检测特征更友好")

    # ★ 针对 MobileNetV3 的终极防崩塌配置
    extra_opts = {
        "ActivationSymmetric": True,  # 强制激活值对称量化
        "WeightSymmetric": True,  # 强制权重对称量化
        "EnableSubgraph": False,
        "ForceQuantizeNoInputCheck": False,
    }

    _quantize_static(
        model_input=source_model,
        model_output=output_path,
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ,  # ★ 1. 换成对 CNN 更友好的 QDQ 格式
        per_channel=True,  # ★ 2. 开启 per-channel 量化（MobileNet 精度救星！）
        reduce_range=False,
        activation_type=QuantType.QInt8,  # ★ 3. 激活值使用带符号的 INT8
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.Entropy,
        nodes_to_exclude=sensitive_nodes,
        extra_options=extra_opts,
    )

    if os.path.exists(preprocessed_path) and preprocessed_path != onnx_path:
        os.remove(preprocessed_path)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"    ✓ 静态量化完成: {output_path} ({size_mb:.1f} MB)")
    return output_path


# ============================================================
# Benchmark
# ============================================================

def benchmark_and_compare(fp32_path: str, quant_path: str, input_size: int = 1024,
                          n_warmup: int = 5, n_runs: int = 30):
    import onnxruntime as ort

    print(f"\n{'='*60}")
    print(f"[Benchmark] FP32 vs INT8 对比")
    print(f"{'='*60}")

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1).astype(np.float32)
    std  = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1).astype(np.float32)
    dummy_img = np.random.rand(1, 3, input_size, input_size).astype(np.float32)
    dummy = ((dummy_img - mean) / std).astype(np.float32)

    def make_session(model_path):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = os.cpu_count() or 1
        return ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])

    def run_benchmark(session, inp, n_warmup, n_runs):
        feeds = {"input": inp}
        for _ in range(n_warmup):
            session.run(None, feeds)
        t0 = time.perf_counter()
        outputs = None
        for _ in range(n_runs):
            outputs = session.run(None, feeds)
        elapsed = (time.perf_counter() - t0) / n_runs * 1000
        return elapsed, outputs

    print(f"\n  加载 FP32 模型...")
    sess_fp32 = make_session(fp32_path)
    t_fp32, out_fp32 = run_benchmark(sess_fp32, dummy, n_warmup, n_runs)
    fp32_size = os.path.getsize(fp32_path) / 1024 / 1024

    print(f"  加载 INT8 模型...")
    sess_int8 = make_session(quant_path)
    t_int8, out_int8 = run_benchmark(sess_int8, dummy, n_warmup, n_runs)
    int8_size = os.path.getsize(quant_path) / 1024 / 1024

    score_diff = np.max(np.abs(out_fp32[0] - out_int8[0]))
    geo_diff   = np.max(np.abs(out_fp32[1] - out_int8[1]))
    score_mean_diff = np.mean(np.abs(out_fp32[0] - out_int8[0]))
    geo_mean_diff   = np.mean(np.abs(out_fp32[1] - out_int8[1]))

    speedup = t_fp32 / t_int8 if t_int8 > 0 else float("inf")
    compress = fp32_size / int8_size if int8_size > 0 else float("inf")

    print(f"\n  {'指标':<24} {'FP32':>12} {'INT8':>12} {'对比':>12}")
    print(f"  {'-'*60}")
    print(f"  {'推理时间 (ms)':<24} {t_fp32:>11.1f} {t_int8:>11.1f} {speedup:>10.2f}x")
    print(f"  {'模型大小 (MB)':<24} {fp32_size:>11.1f} {int8_size:>11.1f} {compress:>10.2f}x")
    print(f"  {'score_map 最大误差':<24} {'—':>12} {score_diff:>11.2e} {'':>12}")
    print(f"  {'score_map 平均误差':<24} {'—':>12} {score_mean_diff:>11.2e} {'':>12}")
    print(f"  {'geo_map 最大误差':<24} {'—':>12} {geo_diff:>11.2e} {'':>12}")
    print(f"  {'geo_map 平均误差':<24} {'—':>12} {geo_mean_diff:>11.2e} {'':>12}")
    print(f"\n  ★ INT8 量化加速比: {speedup:.2f}x")
    print(f"  ★ 模型压缩比:      {compress:.2f}x")

    if score_diff < 0.01:
        print(f"\n  ✓ 精度优秀（score_map 最大误差 < 0.01）")
    elif score_diff < 0.05:
        print(f"\n  ✓ 精度良好（score_map 最大误差 < 0.05）")
    elif score_diff < 0.1:
        print(f"\n  ⚠ 精度可接受（score_map 最大误差 < 0.1），建议在实际数据上验证")
    else:
        print(f"\n  ✗ 精度损失偏大（score_map 最大误差 = {score_diff:.3f}）")
        print(f"    通常通过逆向追溯保护预测头后，误差会在 0.05 以内。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EAST ONNX 模型 INT8 量化工具")
    parser.add_argument("--onnx_path", type=str, default="best_detection.onnx")
    parser.add_argument("--mode", type=str, choices=["dynamic", "static"], default="static")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--calib_image_dir", type=str, default="detection/data/val_images")
    parser.add_argument("--num_calib", type=int, default=30)
    parser.add_argument("--input_size", type=int, default=1024)
    parser.add_argument("--no_benchmark", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.onnx_path):
        raise FileNotFoundError(f"ONNX 模型文件不存在: {args.onnx_path}")

    if args.output_path is None:
        base = args.onnx_path.replace(".onnx", "")
        args.output_path = f"{base}_int8_{args.mode}.onnx"

    print("=" * 60)
    print(f"EAST ONNX → INT8 量化工具 (模式: {args.mode})")
    print("=" * 60)

    if args.mode == "dynamic":
        do_quantize_dynamic(args.onnx_path, args.output_path)
    else:
        if not os.path.isdir(args.calib_image_dir):
            raise FileNotFoundError(f"校准图像目录不存在: {args.calib_image_dir}")
        do_quantize_static(args.onnx_path, args.output_path, args.calib_image_dir, args.num_calib, args.input_size)

    if not args.no_benchmark:
        benchmark_and_compare(args.onnx_path, args.output_path, input_size=args.input_size)