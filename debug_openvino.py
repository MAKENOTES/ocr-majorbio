"""
诊断脚本 v2：对比 ONNX Runtime vs OpenVINO 的输出差异
修复 dynamic shape 报错
用法: python debug_openvino.py
"""
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ====== 配置 ======
ONNX_MODEL = "best_recognition.onnx"  # 改成你的 onnx 路径
IMG_HEIGHT = 32
IMG_WIDTH = 256

# ====== 构造相同输入 ======
np.random.seed(42)
dummy = np.random.randn(1, 1, IMG_HEIGHT, IMG_WIDTH).astype(np.float32)

# ====== ONNX Runtime ======
import onnxruntime as ort
sess = ort.InferenceSession(ONNX_MODEL, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
onnx_outputs = sess.run(None, {input_name: dummy})

print("=" * 60)
print("ONNX Runtime")
print("=" * 60)
print(f"  输出数量: {len(onnx_outputs)}")
for i, o in enumerate(onnx_outputs):
    print(f"  output[{i}]: shape={o.shape}, dtype={o.dtype}")
    print(f"    min={o.min():.4f}, max={o.max():.4f}, mean={o.mean():.4f}")

onnx_main = onnx_outputs[0]

# ====== OpenVINO ======
from openvino import Core
ie = Core()
model = ie.read_model(model=ONNX_MODEL)

print(f"\n{'=' * 60}")
print("OpenVINO 模型信息")
print("=" * 60)

# 用 partial_shape 代替 shape，避免 dynamic shape 报错
print(f"  输入节点数: {len(model.inputs)}")
for i, inp in enumerate(model.inputs):
    print(f"  input[{i}]: name='{inp.any_name}', partial_shape={inp.partial_shape}, dtype={inp.element_type}")
print(f"  输出节点数: {len(model.outputs)}")
for i, out in enumerate(model.outputs):
    print(f"  output[{i}]: name='{out.any_name}', partial_shape={out.partial_shape}, dtype={out.element_type}")

# 编译并推理
compiled = ie.compile_model(model=model, device_name="CPU")
infer_request = compiled.create_infer_request()
result = infer_request.infer({"input": dummy})

print(f"\n{'=' * 60}")
print("OpenVINO 推理结果")
print("=" * 60)

ov_all_outputs = []
for i, out_node in enumerate(compiled.outputs):
    ov_out = np.array(result[out_node])
    ov_all_outputs.append(ov_out)
    print(f"  output[{i}] (name='{out_node.any_name}'):")
    print(f"    shape={ov_out.shape}, dtype={ov_out.dtype}")
    print(f"    min={ov_out.min():.4f}, max={ov_out.max():.4f}, mean={ov_out.mean():.4f}")

# ====== 和 ONNX 逐个对比 ======
print(f"\n{'=' * 60}")
print("ONNX vs OpenVINO 数值对比")
print("=" * 60)

for i, ov_out in enumerate(ov_all_outputs):
    print(f"\n  --- OpenVINO output[{i}] (shape={ov_out.shape}) ---")

    # 直接和每个 ONNX 输出对比
    for j, onnx_out in enumerate(onnx_outputs):
        if ov_out.shape == onnx_out.shape:
            diff = np.abs(ov_out - onnx_out)
            print(f"  直接对比 ONNX output[{j}]: max_diff={diff.max():.6e}, mean_diff={diff.mean():.6e}")
        else:
            print(f"  vs ONNX output[{j}]: shape 不匹配 (ov={ov_out.shape} vs onnx={onnx_out.shape})")

    # 如果 shape 不匹配，尝试各种转置
    for j, onnx_out in enumerate(onnx_outputs):
        if ov_out.shape != onnx_out.shape and ov_out.ndim == 3:
            for perm in [(1,0,2), (2,0,1), (0,2,1), (1,2,0), (2,1,0)]:
                try:
                    transposed = np.transpose(ov_out, perm)
                    if transposed.shape == onnx_out.shape:
                        diff = np.abs(transposed - onnx_out)
                        print(f"  转置{perm}后匹配 ONNX output[{j}]: max_diff={diff.max():.6e}")
                except:
                    pass

# ====== 取值方式对比 ======
print(f"\n{'=' * 60}")
print("取值方式对比")
print("=" * 60)

by_values = np.array(list(result.values())[0])
by_node = np.array(result[compiled.outputs[0]])

print(f"  list(values())[0] shape: {by_values.shape}")
print(f"  outputs[0] shape:        {by_node.shape}")
print(f"  两者是否相同: {np.array_equal(by_values, by_node)}")

# ====== Decode 对比 ======
print(f"\n{'=' * 60}")
print("解码对比")
print("=" * 60)

import torch
from recognition.components.converter import LabelConverter
converter = LabelConverter("recognition/keys.txt")

onnx_preds = torch.from_numpy(onnx_main)
print(f"  ONNX preds shape: {onnx_preds.shape}")
onnx_text = converter.decode(onnx_preds)
print(f"  ONNX decode: {onnx_text}")

# 对 OpenVINO 每个输出都尝试 decode
for i, ov_out in enumerate(ov_all_outputs):
    ov_preds = torch.from_numpy(ov_out)
    print(f"\n  OV output[{i}] shape: {ov_preds.shape}")
    try:
        ov_text = converter.decode(ov_preds)
        print(f"  OV output[{i}] decode: {ov_text}")
    except Exception as e:
        print(f"  OV output[{i}] decode 失败: {e}")

    # 如果 shape 不同，尝试转置后 decode
    if ov_preds.shape != onnx_preds.shape and ov_preds.ndim == 3:
        for perm in [(1,0,2), (2,0,1), (0,2,1)]:
            try:
                ov_t = torch.from_numpy(np.transpose(ov_out, perm))
                text = converter.decode(ov_t)
                print(f"  OV output[{i}] 转置{perm} shape={ov_t.shape} decode: {text}")
            except:
                pass

print(f"\n{'=' * 60}")
print("诊断完成")
print("=" * 60)