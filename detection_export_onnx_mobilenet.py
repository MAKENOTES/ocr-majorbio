"""
detection_export_onnx_mobilenet.py
==================================
将训练好的 EAST (MobileNetV3-Large) PyTorch 模型导出为 ONNX 格式。

用法:
    python detection_export_onnx_mobilenet.py \
        --model_path best_detection.pth \
        --output_path best_detection.onnx \
        --input_size 1024

依赖:
    pip install onnx onnxruntime onnxsim
"""

import torch
import argparse
import os
import sys
import time
import numpy as np

# ---- 路径修正：让脚本在任意位置都能找到模块 ----
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.models.east_model import EASTModel


def load_model(model_path: str, device: torch.device) -> EASTModel:
    """
    智能加载模型：自动判断 .pth 文件是 state_dict 还是完整模型。
    """
    model = EASTModel(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        # 情况 1: state_dict 字典（可能带 'model_state_dict' 等 key）
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            # 尝试直接作为 state_dict 加载
            try:
                model.load_state_dict(checkpoint)
            except RuntimeError:
                # 可能是带额外 key 的 checkpoint（如 optimizer 等）
                # 尝试过滤出模型参数
                state = {k: v for k, v in checkpoint.items()
                         if isinstance(v, torch.Tensor)}
                model.load_state_dict(state)
    elif isinstance(checkpoint, nn.Module):
        # 情况 2: torch.save(model, ...) 保存的完整模型
        model = checkpoint
    else:
        raise ValueError(
            f"无法识别的 checkpoint 格式: {type(checkpoint)}。"
            f"期望 state_dict (dict) 或完整模型 (nn.Module)。"
        )

    model.eval()
    model.to(device)
    return model


def export_onnx(model_path: str, output_path: str, input_size: int = 1024):
    """
    将 PyTorch EAST 模型导出为 ONNX 格式，并可选地使用 onnx-simplifier 优化。

    Args:
        model_path:  .pth 权重文件路径
        output_path: 输出的 .onnx 文件路径
        input_size:  推理时的输入图像尺寸（正方形边长）
    """
    print("=" * 60)
    print("EAST (MobileNetV3-Large) → ONNX 导出工具")
    print("=" * 60)

    # ---------- 1. 加载模型 ----------
    print(f"\n[1/4] 加载 PyTorch 模型: {model_path}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    device = torch.device("cpu")
    model = load_model(model_path, device)
    print(f"    ✓ 模型加载成功")

    # ---------- 2. 准备虚拟输入 ----------
    print(f"[2/4] 准备虚拟输入 (1, 3, {input_size}, {input_size})")
    dummy_input = torch.randn(1, 3, input_size, input_size, device=device)

    # ---------- 3. 导出 ONNX ----------
    print(f"[3/4] 导出 ONNX 到: {output_path}")
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["score_map", "geo_map"],
        dynamic_axes={
            "input":     {0: "batch_size"},
            "score_map": {0: "batch_size"},
            "geo_map":   {0: "batch_size"},
        },
        verbose=False,
    )

    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"    ✓ 基础 ONNX 导出完成 ({file_size_mb:.1f} MB)")

    # ---------- 4. onnxsim 简化 ----------
    print(f"[4/4] 尝试使用 onnx-simplifier 进行图优化...")
    try:
        import onnx
        from onnxsim import simplify

        model_onnx = onnx.load(output_path)
        onnx.checker.check_model(model_onnx)

        model_simplified, check = simplify(model_onnx)
        if check:
            onnx.save(model_simplified, output_path)
            new_size = os.path.getsize(output_path) / 1024 / 1024
            print(f"    ✓ 简化成功！模型大小: {new_size:.1f} MB")
        else:
            print("    ⚠ 简化校验失败，保留原始 ONNX")
    except ImportError:
        print("    ⚠ onnxsim 未安装，跳过简化步骤")
        print("    → 安装命令: pip install onnxsim")

    # ---------- 5. 验证 + Benchmark（复用同一 session） ----------
    print(f"\n[验证] 对比 PyTorch 与 ONNX 输出...")
    try:
        import onnxruntime as ort

        # 创建 ONNX Runtime session（仅创建一次，后续复用）
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = os.cpu_count() or 1
        ort_session = ort.InferenceSession(
            output_path, sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        ort_inputs = {"input": dummy_input.numpy()}

        # --- PyTorch 推理 ---
        with torch.no_grad():
            pt_score, pt_geo = model(dummy_input)
        pt_score_np = pt_score.numpy()
        pt_geo_np = pt_geo.numpy()

        # --- ONNX 推理 ---
        ort_score, ort_geo = ort_session.run(None, ort_inputs)

        # --- 比较最大绝对误差 ---
        score_err = np.max(np.abs(pt_score_np - ort_score))
        geo_err = np.max(np.abs(pt_geo_np - ort_geo))
        print(f"    score_map 最大误差: {score_err:.2e}")
        print(f"    geo_map   最大误差: {geo_err:.2e}")

        if score_err < 1e-4 and geo_err < 1e-4:
            print("    ✓ 输出一致，导出成功！")
        elif score_err < 1e-3 and geo_err < 1e-3:
            print("    ⚠ 误差在可接受范围内 (<1e-3)，可能是浮点精度差异")
        else:
            print("    ✗ 误差偏大，请检查模型结构是否包含不支持的算子")

        # --- Benchmark ---
        print(f"\n[Benchmark] 速度对比 (CPU, 5次预热 + 20次计时)...")
        n_warmup, n_runs = 5, 20

        # PyTorch 预热 + 计时
        for _ in range(n_warmup):
            with torch.no_grad():
                model(dummy_input)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            with torch.no_grad():
                model(dummy_input)
        pt_time = (time.perf_counter() - t0) / n_runs * 1000

        # ONNX 预热 + 计时
        for _ in range(n_warmup):
            ort_session.run(None, ort_inputs)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            ort_session.run(None, ort_inputs)
        ort_time = (time.perf_counter() - t0) / n_runs * 1000

        speedup = pt_time / ort_time if ort_time > 0 else float("inf")
        print(f"    PyTorch (CPU):    {pt_time:.1f} ms/张")
        print(f"    ONNX Runtime:     {ort_time:.1f} ms/张")
        print(f"    加速比:           {speedup:.2f}x")

    except ImportError:
        print("    ⚠ onnxruntime 未安装，跳过验证和 benchmark")
        print("    → 安装命令: pip install onnxruntime")
    except Exception as e:
        print(f"    ✗ 验证/Benchmark 失败: {e}")

    # ---------- 总结 ----------
    print(f"\n{'=' * 60}")
    print(f"✓ 导出完成: {output_path}")
    final_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"  文件大小: {final_size:.1f} MB")
    print(f"  下一步: 运行 quantize_int8.py 进一步提速 3-4x")
    print(f"{'=' * 60}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="EAST (MobileNetV3) PyTorch → ONNX 导出工具"
    )
    parser.add_argument(
        "--model_path", type=str, default="best_detection.pth",
        help="PyTorch 模型权重路径 (.pth)"
    )
    parser.add_argument(
        "--output_path", type=str, default="best_detection.onnx",
        help="输出 ONNX 模型路径 (.onnx)"
    )
    parser.add_argument(
        "--input_size", type=int, default=640,
        help="输入图像尺寸（正方形边长，默认 1024）"
    )
    return parser.parse_args()


if __name__ == "__main__":
    import torch.nn as nn  # 用于 load_model 中的 nn.Module 类型检查

    args = parse_args()
    export_onnx(args.model_path, args.output_path, args.input_size)