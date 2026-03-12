"""
recognition_export_onnx.py — 将训练好的 CRNN PyTorch 模型导出为 ONNX 格式
=====================================================================
第一步：导出 ONNX（无损，推理速度 2-3x 提升）

用法:
    python recognition_export_onnx.py \
        --model_path best_recognition.pth \
        --output_path best_recognition.onnx \
        --img_width 256

依赖:
    pip install onnx onnxruntime onnxsim onnxscript
"""

import torch
import os
import numpy as np

# ---- 路径修正 ----
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recognition.models.model import CRNN
from recognition.components.converter import LabelConverter


def export_onnx(model_path: str, output_path: str, keys_path: str,
                img_height: int = 32, img_width: int = 256):
    """
    将 PyTorch CRNN 模型导出为 ONNX 格式。

    Args:
        model_path:  .pth 权重文件路径
        output_path: 输出的 .onnx 文件路径
        keys_path:   字典文件路径
        img_height:  输入图像高度（固定 32）
        img_width:   导出时使用的参考宽度（实际推理支持动态宽度）
    """
    print("=" * 60)
    print("CRNN 模型 → ONNX 导出工具")
    print("=" * 60)

    # 1. 加载模型
    print(f"\n[1/4] 加载 PyTorch 模型: {model_path}")
    device = torch.device("cpu")

    converter = LabelConverter(keys_path)
    num_classes = converter.num_classes

    model = CRNN(img_height=img_height, nc=1, n_class=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    print(f"    类别数: {num_classes}")

    # 2. 准备虚拟输入（不传 img_widths，ONNX 中不使用 padding mask）
    print(f"[2/4] 准备虚拟输入 (1, 1, {img_height}, {img_width})")
    dummy_input = torch.randn(1, 1, img_height, img_width, device=device)

    # 3. 导出 ONNX
    print(f"[3/4] 导出 ONNX 到: {output_path}")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    torch.onnx.export(
        model,
        (dummy_input,),             # 只传图像，不传 img_widths
        output_path,
        export_params=True,
        opset_version=14,           # LSTM 需要 opset >= 14
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size", 3: "width"},   # 动态 batch 和宽度
            "output": {0: "seq_len", 1: "batch_size"},
        },
        verbose=False,
        dynamo=False, #GRU专用  ，resnet50  不需要
    )
    print(f"    ✓ 基础 ONNX 导出完成")

    # 4. 使用 onnxsim 简化模型
    print(f"[4/4] 尝试使用 onnx-simplifier 进行图优化...")
    try:
        import onnx
        from onnxsim import simplify

        model_onnx = onnx.load(output_path)
        onnx.checker.check_model(model_onnx)

        model_simplified, check = simplify(model_onnx)
        if check:
            onnx.save(model_simplified, output_path)
            orig_size = os.path.getsize(output_path) / 1024 / 1024
            print(f"    ✓ 简化成功！模型大小: {orig_size:.1f} MB")
        else:
            print("    ⚠ 简化校验失败，保留原始 ONNX")
    except ImportError:
        print("    ⚠ onnxsim 未安装，跳过简化步骤")
        print("    → 安装命令: pip install onnxsim")

    # 5. 验证 ONNX 输出与 PyTorch 输出一致性
    print(f"\n[验证] 对比 PyTorch 与 ONNX 输出...")
    try:
        import onnxruntime as ort

        # PyTorch 推理
        with torch.no_grad():
            pt_output = model(dummy_input)
        pt_output_np = pt_output.numpy()

        # ONNX Runtime 推理
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        ort_session = ort.InferenceSession(output_path, sess_options=sess_options,
                                           providers=["CPUExecutionProvider"])
        ort_inputs = {"input": dummy_input.numpy()}
        ort_output = ort_session.run(None, ort_inputs)[0]

        # 比较最大绝对误差
        max_err = np.max(np.abs(pt_output_np - ort_output))
        print(f"    output 最大误差: {max_err:.2e}")

        if max_err < 1e-4:
            print("    ✓ 输出一致，导出成功！")
        else:
            print("    ⚠ 误差偏大，请检查模型结构是否包含不支持的算子")

    except ImportError:
        print("    ⚠ onnxruntime 未安装，跳过验证")
        print("    → 安装命令: pip install onnxruntime")

    # 6. 速度对比 benchmark
    print(f"\n[Benchmark] 速度对比 (CPU, 5次预热 + 20次计时)...")
    try:
        import onnxruntime as ort
        import time

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = os.cpu_count()
        ort_session = ort.InferenceSession(output_path, sess_options=sess_options,
                                           providers=["CPUExecutionProvider"])
        ort_inputs = {"input": dummy_input.numpy()}

        # PyTorch 预热
        for _ in range(5):
            with torch.no_grad():
                model(dummy_input)

        # PyTorch 计时
        t0 = time.perf_counter()
        for _ in range(20):
            with torch.no_grad():
                model(dummy_input)
        pt_time = (time.perf_counter() - t0) / 20 * 1000

        # ONNX 预热
        for _ in range(5):
            ort_session.run(None, ort_inputs)

        # ONNX 计时
        t0 = time.perf_counter()
        for _ in range(20):
            ort_session.run(None, ort_inputs)
        ort_time = (time.perf_counter() - t0) / 20 * 1000

        speedup = pt_time / ort_time
        print(f"    PyTorch (CPU):    {pt_time:.1f} ms/张")
        print(f"    ONNX Runtime:     {ort_time:.1f} ms/张")
        print(f"    加速比:           {speedup:.2f}x")

    except Exception as e:
        print(f"    Benchmark 失败: {e}")

    print(f"\n{'='*60}")
    print(f"✓ 导出完成: {output_path}")
    print(f"  下一步: 运行 recognition_quantize_int8.py 进一步提速")
    print(f"{'='*60}")


if __name__ == "__main__":
    model_path = "best_recognition.pth"
    output_path = "best_recognition.onnx"
    keys_path = "recognition/keys.txt"
    img_height = 32
    img_width = 256

    export_onnx(model_path, output_path, keys_path, img_height, img_width)
