"""
recognition_quantize_nncf.py — 使用 NNCF 对 CRNN 模型做 OpenVINO 原生 INT8 量化
==================================================================================
解决问题：ONNX Runtime 的 QDQ 量化格式与 OpenVINO 对 Depthwise Conv / GRU 的
         解析不兼容，导致 OpenVINO 加载 INT8 ONNX 模型时结果错误。

方案：使用 OpenVINO 自己的量化框架 NNCF，生成 OpenVINO 原生 IR 格式的 INT8 模型，
     100% 兼容 OpenVINO 推理。

流程：
  FP32 ONNX → OpenVINO IR (FP32) → NNCF PTQ → OpenVINO IR (INT8)

用法:
    python recognition_quantize_nncf.py \
        --onnx_path best_recognition.onnx \
        --calib_image_dir recognition/data/val_images \
        --num_calib 300

    # 推理时使用 OpenVINO 后端:
    python recognition_inference.py \
        --model_path best_recognition_nncf_int8.xml \
        --backend openvino

依赖:
    pip install nncf openvino
"""

import os
import sys
import argparse
import time
import glob
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# 配置
# ============================================================

IMG_HEIGHT = 32
NUM_CHANNELS = 1


# ============================================================
# 校准数据集
# ============================================================

def build_calibration_dataset(image_dir: str, num_calib: int = 300,
                               img_height: int = 32, num_channels: int = 1):
    """
    构建 NNCF 校准数据集。
    复用 dataset.py 的预处理逻辑，确保与训练/推理完全一致。
    """
    import nncf
    import torch
    from torch.utils.data import DataLoader

    # 复用项目的 InferenceDataset，保证预处理 100% 一致
    from recognition.components.dataset import InferenceDataset, dynamic_padding_collate_fn

    # 收集图片
    all_images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        all_images.extend(glob.glob(os.path.join(image_dir, ext)))
        all_images.extend(glob.glob(os.path.join(image_dir, ext.upper())))

    if not all_images:
        raise ValueError(f"在 {image_dir} 下未找到图像文件")

    np.random.shuffle(all_images)
    selected = all_images[:num_calib]
    print(f"  校准图片: {len(selected)} 张 (从 {len(all_images)} 张中采样)")

    dataset = InferenceDataset(
        image_paths=selected,
        img_height=img_height,
        num_channels=num_channels,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # NNCF 校准用单线程更稳定
        collate_fn=dynamic_padding_collate_fn,
    )

    # NNCF 的 transform_fn: 从 DataLoader 的输出中提取模型输入
    # dynamic_padding_collate_fn 返回 (padded_imgs, paths, img_widths)
    # OpenVINO 模型只需要 padded_imgs (numpy)
    def transform_fn(data_item):
        images, paths, img_widths = data_item
        return images.numpy()  # [B, C, H, W] numpy array

    calibration_dataset = nncf.Dataset(loader, transform_fn)
    return calibration_dataset


# ============================================================
# 量化主流程
# ============================================================

def quantize_with_nncf(onnx_path: str, output_dir: str,
                       calib_image_dir: str, num_calib: int = 300,
                       subset_size: int = 300):
    """
    使用 NNCF 对 FP32 ONNX 模型做 OpenVINO 原生 INT8 量化。

    Args:
        onnx_path:      FP32 ONNX 模型路径
        output_dir:     输出目录 (会生成 .xml + .bin 文件)
        calib_image_dir: 校准图像目录
        num_calib:      校准图像数量
        subset_size:    NNCF 使用的校准子集大小
    """
    import nncf
    import openvino as ov

    print("=" * 60)
    print("NNCF OpenVINO 原生 INT8 量化")
    print("=" * 60)

    # Step 1: 加载 FP32 ONNX → OpenVINO IR
    print(f"\n[1/4] 加载 FP32 模型: {onnx_path}")
    core = ov.Core()
    ov_model = core.read_model(model=onnx_path)

    # 打印模型信息
    print(f"  输入: {[(inp.any_name, inp.partial_shape) for inp in ov_model.inputs]}")
    print(f"  输出: {[(out.any_name, out.partial_shape) for out in ov_model.outputs]}")

    # Step 2: 构建校准数据集
    print(f"\n[2/4] 构建校准数据集...")
    calibration_dataset = build_calibration_dataset(
        calib_image_dir, num_calib, IMG_HEIGHT, NUM_CHANNELS
    )

    # Step 3: NNCF 量化
    print(f"\n[3/4] 执行 NNCF INT8 量化 (subset_size={subset_size})...")
    print(f"  这可能需要几分钟，请耐心等待...")

    t0 = time.perf_counter()
    quantized_model = nncf.quantize(
        model=ov_model,
        calibration_dataset=calibration_dataset,
        subset_size=subset_size,
        # MIXED 模式: 对精度敏感的层 (如 GRU) 自动保留 FP16/FP32
        preset=nncf.QuantizationPreset.MIXED,
        # 更精确的 bias 校正
        fast_bias_correction=False,
        # 高级选项: 如果精度不够，可以取消注释下面的 ignored_scope
        # 排除 GRU 层不量化 (GRU 对量化最敏感)
        # ignored_scope=nncf.IgnoredScope(types=["GRU"]),
    )
    quant_time = time.perf_counter() - t0
    print(f"  ✓ 量化完成 (耗时 {quant_time:.1f} 秒)")

    # Step 4: 保存量化后的 OpenVINO IR 模型
    print(f"\n[4/4] 保存量化模型...")
    os.makedirs(output_dir, exist_ok=True)

    model_name = os.path.splitext(os.path.basename(onnx_path))[0] + "_nncf_int8"
    output_xml = os.path.join(output_dir, model_name + ".xml")
    output_bin = os.path.join(output_dir, model_name + ".bin")

    ov.save_model(quantized_model, output_xml, compress_to_fp16=False)

    xml_size = os.path.getsize(output_xml) / 1024 / 1024
    bin_size = os.path.getsize(output_bin) / 1024 / 1024
    print(f"  ✓ 保存完成:")
    print(f"    {output_xml} ({xml_size:.1f} MB)")
    print(f"    {output_bin} ({bin_size:.1f} MB)")
    print(f"    总大小: {xml_size + bin_size:.1f} MB")

    return output_xml, quantized_model


# ============================================================
# Benchmark
# ============================================================

def benchmark(onnx_path: str, int8_xml_path: str,
              img_height: int = 32, img_width: int = 256,
              n_warmup: int = 5, n_runs: int = 50):
    """
    对比 FP32 ONNX (OpenVINO) vs NNCF INT8 (OpenVINO) 的推理速度。
    """
    import openvino as ov

    print(f"\n{'='*60}")
    print(f"[Benchmark] FP32 vs NNCF-INT8 (OpenVINO)")
    print(f"{'='*60}")

    core = ov.Core()
    dummy = np.random.randn(1, 1, img_height, img_width).astype(np.float32)

    def run_bench(model_path, label):
        model = core.read_model(model=model_path)
        compiled = core.compile_model(model, "CPU",
                                      config={"PERFORMANCE_HINT": "LATENCY"})
        req = compiled.create_infer_request()

        # warmup
        for _ in range(n_warmup):
            req.infer({"input": dummy})

        # benchmark
        t0 = time.perf_counter()
        for _ in range(n_runs):
            result = req.infer({"input": dummy})
        elapsed = (time.perf_counter() - t0) / n_runs * 1000
        print(f"  {label:<25} {elapsed:.2f} ms/张")
        return elapsed, result

    t_fp32, out_fp32 = run_bench(onnx_path, "FP32 (ONNX → OpenVINO)")
    t_int8, out_int8 = run_bench(int8_xml_path, "NNCF INT8 (OpenVINO IR)")

    # 数值对比
    fp32_out = np.array(list(out_fp32.values())[0])
    int8_out = np.array(list(out_int8.values())[0])

    if fp32_out.shape == int8_out.shape:
        max_diff = np.max(np.abs(fp32_out - int8_out))
        print(f"\n  output 最大误差: {max_diff:.2e}")
    else:
        print(f"\n  output shape 不同: FP32={fp32_out.shape}, INT8={int8_out.shape}")

    speedup = t_fp32 / t_int8
    print(f"\n  ★ NNCF INT8 加速比: {speedup:.2f}x")

    # 与 FP32 ONNX 模型大小对比
    onnx_size = os.path.getsize(onnx_path) / 1024 / 1024
    bin_path = int8_xml_path.replace(".xml", ".bin")
    xml_size = os.path.getsize(int8_xml_path) / 1024 / 1024
    bin_size = os.path.getsize(bin_path) / 1024 / 1024 if os.path.exists(bin_path) else 0
    int8_total = xml_size + bin_size
    print(f"  ★ 模型压缩比: {onnx_size / int8_total:.2f}x ({onnx_size:.1f}MB → {int8_total:.1f}MB)")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="NNCF OpenVINO 原生 INT8 量化工具")
    parser.add_argument("--onnx_path", type=str,
                        default="best_recognition.onnx",
                        help="输入 FP32 ONNX 模型路径")
    parser.add_argument("--output_dir", type=str,
                        default=".",
                        help="输出目录 (默认当前目录)")
    parser.add_argument("--calib_image_dir", type=str,
                        default="recognition/data/val_images",
                        help="校准图像目录")
    parser.add_argument("--num_calib", type=int,
                        default=300,
                        help="校准图像数量 (建议 200-500)")
    parser.add_argument("--subset_size", type=int,
                        default=300,
                        help="NNCF 校准子集大小")
    parser.add_argument("--no_benchmark", action="store_true",
                        help="跳过 benchmark 测试")
    args = parser.parse_args()

    # 执行量化
    output_xml, _ = quantize_with_nncf(
        onnx_path=args.onnx_path,
        output_dir=args.output_dir,
        calib_image_dir=args.calib_image_dir,
        num_calib=args.num_calib,
        subset_size=args.subset_size,
    )

    # Benchmark
    if not args.no_benchmark:
        benchmark(args.onnx_path, output_xml)

    print(f"\n{'='*60}")
    print(f"✓ NNCF 量化完成！")
    print(f"")
    print(f"  生成的模型文件:")
    print(f"    {output_xml}")
    print(f"    {output_xml.replace('.xml', '.bin')}")
    print(f"")
    print(f"  推理命令:")
    print(f"    python recognition_inference.py \\")
    print(f"      --model_path {output_xml} \\")
    print(f"      --backend openvino")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()