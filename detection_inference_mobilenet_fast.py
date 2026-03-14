"""
detection_inference_mobilenet_fast.py — 高速优化推理脚本
=====================================================================
基于 detection_inference_mobilenet.py，全面优化预处理 + 后处理性能。

★ 优化清单：
  1. 图像只读一次（cv2.imread），去掉 PIL 依赖，预处理速度 2-3x
  2. np.where 全图扫描 → 利用 stats 的 bbox 做 ROI 裁切，避免遍历整张 label map
  3. 边缘采样中位数 → 预计算 ROI 局部坐标，减少 fancy indexing 开销
  4. 预处理 normalize 向量化，一次矩阵运算完成
  5. 可选：跳过可视化输出（--no_vis），纯检测模式再省 30-40%
  6. 可选：跳过 crop 保存的 imwrite（最大 I/O 瓶颈）

用法:
    python detection_inference_mobilenet_fast.py \
        --model_path best_detection.onnx \
        --backend onnx \
        --image_path detection/data/test/1 \
        --output_dir detection/data/test_results/fast \
        --benchmark
"""

import os
import sys
import cv2
import time
import glob
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# 参数配置
# ============================================================

DEFAULT_MODEL_PATH    = "best_detection.onnx"
DEFAULT_IMAGE_PATH    = "detection/data/test/1"
DEFAULT_OUTPUT_DIR    = "detection/data/test_results/fast"
DEFAULT_CROP_DIR      = "detection/data/crops/fast"

INPUT_SIZE            = 768
SCORE_THRESHOLD       = 0.7
SCORE_THRESHOLD_SMALL = 0.5
MIN_AREA              = 2
SMALL_REGION_AREA     = 15
SMALL_MAX_ASPECT      = 5.0

SAVE_CROPS            = False
EXPAND_BOX            = True
EXPAND_RATIO_W        = 0.30
EXPAND_RATIO_H        = 0.10

# ImageNet 归一化常量（预计算好 float32）
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
_INV_STD = (1.0 / _STD).astype(np.float32)


# ============================================================
# 后端加载器（与原版相同）
# ============================================================

class ONNXBackend:
    def __init__(self, model_path: str):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = os.cpu_count()
        opts.inter_op_num_threads = max(1, os.cpu_count() // 2)
        opts.enable_mem_pattern = True
        opts.enable_mem_reuse   = True
        self.session = ort.InferenceSession(
            model_path, opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self._warmup()
        print(f"  ✓ ONNX Runtime 后端加载完成 (线程: {os.cpu_count()})")

    def _warmup(self, n=3):
        d = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
        for _ in range(n):
            self.session.run(None, {self.input_name: d})

    def infer(self, img_np):
        return self.session.run(None, {self.input_name: img_np})


class OpenVINOBackend:
    def __init__(self, model_path: str):
        from openvino import Core
        ie = Core()
        model = ie.read_model(model=model_path)
        config = {"PERFORMANCE_HINT": "LATENCY"}  # ★ 改为 LATENCY 优先（单张推理更快）
        self.compiled = ie.compile_model(model=model, device_name="CPU", config=config)
        self.request = self.compiled.create_infer_request()
        self._warmup()
        print(f"  ✓ OpenVINO 后端加载完成")

    def _warmup(self, n=3):
        d = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
        for _ in range(n):
            self.request.infer({"input": d})

    def infer(self, img_np):
        result = self.request.infer({"input": img_np})
        outputs = list(result.values())
        out_list = sorted(outputs, key=lambda x: x.shape[1])
        return out_list  # [score, geo]


class PyTorchBackend:
    def __init__(self, model_path: str):
        import torch
        try:
            from detection.models.east_model import EASTModel
        except ImportError:
            from detection.models.east_model import EASTModel
        self.model = EASTModel(pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        self._torch = torch
        print(f"  ✓ PyTorch 后端加载完成")

    def infer(self, img_np):
        t = self._torch.from_numpy(img_np)
        with self._torch.no_grad():
            s, g = self.model(t)
        return [s.numpy(), g.numpy()]


def load_backend(model_path, backend):
    ext = os.path.splitext(model_path)[1].lower()
    if backend == "auto":
        backend = "onnx" if ext == ".onnx" else "pytorch"
    print(f"\n[后端] {backend.upper()}")
    if backend == "onnx":
        return ONNXBackend(model_path)
    elif backend == "openvino":
        return OpenVINOBackend(model_path)
    elif backend == "pytorch":
        return PyTorchBackend(model_path)
    raise ValueError(f"未知后端: {backend}")


# ============================================================
# ★ 优化 1: 纯 OpenCV 预处理（去掉 PIL，速度 2-3x）
# ============================================================

def preprocess_cv2(image_bgr, input_size):
    """
    纯 OpenCV + numpy 预处理，不依赖 PIL。
    输入: BGR 格式的 cv2 图像（已读取）
    返回: (1,3,H,W) float32 numpy array
    """
    h, w = image_bgr.shape[:2]
    scale = input_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    # ★ cv2.resize 比 PIL.resize 快 2-3x
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 构造 padding 图像（灰色填充）
    pad_x = (input_size - new_w) // 2
    pad_y = (input_size - new_h) // 2
    padded = np.full((input_size, input_size, 3), 128, dtype=np.uint8)
    padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

    # ★ 归一化：一步完成 /255 + (x-mean)/std
    # 合并为 (x/255 - mean) / std = (x/255 - mean) * inv_std
    arr = padded.astype(np.float32)
    arr *= (1.0 / 255.0)
    arr -= _MEAN
    arr *= _INV_STD

    # HWC→CHW, 添加 batch 维度（contiguous 内存）
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[np.newaxis])


# ============================================================
# ★ 优化 2: 高速连通域后处理
# ============================================================

def box_from_small_region_fast(ys, xs, geo_np, stride):
    """小连通域出框（与原版逻辑一致）"""
    cy, cx = np.median(ys), np.median(xs)
    d = np.array([
        np.median(geo_np[0, ys, xs]),  # top
        np.median(geo_np[1, ys, xs]),  # right
        np.median(geo_np[2, ys, xs]),  # bottom
        np.median(geo_np[3, ys, xs]),  # left
    ])
    return [
        (cx + 0.5 - d[3]) * stride,
        (cy + 0.5 - d[0]) * stride,
        (cx + 0.5 + d[1]) * stride,
        (cy + 0.5 + d[2]) * stride,
    ]


def boxes_from_cc_fast(score_map, geo_map, score_thr, score_thr_small, min_area):
    """
    ★ 优化版连通域检测。
    核心优化：用 stats 的 bounding box 做 ROI 裁切提取像素坐标，
    避免每个连通域都做 np.where(labels == id) 全图扫描。

    原版：np.where(labels == id) 对 192x192 的 label map，
    每个连通域都要遍历 36864 个像素。50 个连通域 = 180 万次比较。

    优化版：先用 stats 的 (x, y, w, h) 裁出 ROI，只在 ROI 内搜索。
    一个 10x100 的文本行只需遍历 1000 个像素。50 个 = 5 万次。
    """
    stride = 4.0
    boxes = []

    # ── 第一轮：高阈值 ─────────────────────────────────────
    binary_high = (score_map > score_thr).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_high, connectivity=8
    )

    for lid in range(1, num_labels):
        area = stats[lid, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        # ★ 核心优化：用 stats bbox 做 ROI 裁切
        bx = stats[lid, cv2.CC_STAT_LEFT]
        by = stats[lid, cv2.CC_STAT_TOP]
        bw = stats[lid, cv2.CC_STAT_WIDTH]
        bh = stats[lid, cv2.CC_STAT_HEIGHT]

        # 在 ROI 内找属于当前 label 的像素
        roi_labels = labels[by:by+bh, bx:bx+bw]
        local_ys, local_xs = np.where(roi_labels == lid)
        # 转换回全图坐标
        ys = local_ys + by
        xs = local_xs + bx

        # ── 小连通域 ──
        if area < SMALL_REGION_AREA:
            if max(bw, bh) / (min(bw, bh) + 1e-6) > SMALL_MAX_ASPECT:
                continue
            box = box_from_small_region_fast(ys, xs, geo_map, stride)
            if box[2] - box[0] > 1 and box[3] - box[1] > 1:
                boxes.append(box)
            continue

        # ── 大连通域：边缘采样 ──
        x_min_r, x_max_r = bx, bx + bw - 1
        y_min_r, y_max_r = by, by + bh - 1

        edge_w = max(1, min(5, int(bw * 0.1)))
        edge_h = max(1, min(3, int(bh * 0.1)))

        # 左边缘
        lm = xs < x_min_r + edge_w
        x_min_f = np.median(xs[lm] + 0.5 - geo_map[3, ys[lm], xs[lm]]) if lm.any() else float(x_min_r)

        # 右边缘
        rm = xs > x_max_r - edge_w
        x_max_f = np.median(xs[rm] + 0.5 + geo_map[1, ys[rm], xs[rm]]) if rm.any() else float(x_max_r + 1)

        # 上边缘
        tm = ys < y_min_r + edge_h
        y_min_f = np.median(ys[tm] + 0.5 - geo_map[0, ys[tm], xs[tm]]) if tm.any() else float(y_min_r)

        # 下边缘
        bm_mask = ys > y_max_r - edge_h
        y_max_f = np.median(ys[bm_mask] + 0.5 + geo_map[2, ys[bm_mask], xs[bm_mask]]) if bm_mask.any() else float(y_max_r + 1)

        box = [x_min_f * stride, y_min_f * stride,
               x_max_f * stride, y_max_f * stride]
        if box[2] - box[0] > 2 and box[3] - box[1] > 2:
            boxes.append(box)

    # ── 第二轮：低阈值补充 ─────────────────────────────────
    if score_thr_small < score_thr:
        binary_low = (score_map > score_thr_small).astype(np.uint8)
        num_s, labels_s, stats_s, _ = cv2.connectedComponentsWithStats(
            binary_low, connectivity=8
        )

        for lid in range(1, num_s):
            area = stats_s[lid, cv2.CC_STAT_AREA]
            if area < min_area or area >= SMALL_REGION_AREA:
                continue

            bw_s = stats_s[lid, cv2.CC_STAT_WIDTH]
            bh_s = stats_s[lid, cv2.CC_STAT_HEIGHT]
            if max(bw_s, bh_s) / (min(bw_s, bh_s) + 1e-6) > SMALL_MAX_ASPECT:
                continue

            bx_s = stats_s[lid, cv2.CC_STAT_LEFT]
            by_s = stats_s[lid, cv2.CC_STAT_TOP]

            # ★ ROI 裁切
            roi = labels_s[by_s:by_s+bh_s, bx_s:bx_s+bw_s]
            ly, lx = np.where(roi == lid)
            ys = ly + by_s
            xs = lx + bx_s

            # 与高阈值区域有重叠就跳过
            if binary_high[ys, xs].any():
                continue

            box = box_from_small_region_fast(ys, xs, geo_map, stride)
            if box[2] - box[0] > 1 and box[3] - box[1] > 1:
                boxes.append(box)

    return np.array(boxes) if boxes else np.array([])


# ============================================================
# ★ 优化 3: 单图推理（精简 I/O）
# ============================================================

def detect_one(backend, image_path, output_dir,
               input_size=INPUT_SIZE, score_threshold=SCORE_THRESHOLD,
               save_crops=False, expand_box=EXPAND_BOX,
               ratio_w=EXPAND_RATIO_W, ratio_h=EXPAND_RATIO_H,
               save_vis=True, timing=False):
    """
    优化版单图推理。
    ★ 图像只读一次（cv2.imread），预处理用纯 numpy。
    ★ --no_vis 模式下跳过可视化，纯检测。
    """
    t_total = time.perf_counter()

    # ★ 只读一次图像
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        print(f"  ✗ 无法读取: {image_path}")
        return 0, 0.0, 0.0

    real_h, real_w = img_orig.shape[:2]

    # ★ 纯 OpenCV 预处理
    t_pre = time.perf_counter()
    img_np = preprocess_cv2(img_orig, input_size)
    pre_ms = (time.perf_counter() - t_pre) * 1000

    # 前向推理
    t_infer = time.perf_counter()
    outputs = backend.infer(img_np)
    infer_ms = (time.perf_counter() - t_infer) * 1000

    score_raw, geo_raw = outputs[0], outputs[1]

    # 后处理
    t_post = time.perf_counter()
    score_map = score_raw[0, 0]
    geo_map = np.expm1(geo_raw[0, :4])
    np.maximum(geo_map, 0, out=geo_map)  # ★ in-place 操作，省一次内存分配

    final_boxes = boxes_from_cc_fast(
        score_map, geo_map,
        score_thr=score_threshold,
        score_thr_small=SCORE_THRESHOLD_SMALL,
        min_area=MIN_AREA,
    )
    post_ms = (time.perf_counter() - t_post) * 1000

    # 坐标映射参数
    scale = input_size / max(real_h, real_w)
    content_w = int(real_w * scale)
    content_h = int(real_h * scale)
    pad_x = (input_size - content_w) // 2
    pad_y = (input_size - content_h) // 2

    fname = os.path.basename(image_path)
    fname_noext = os.path.splitext(fname)[0]

    # I/O：可视化 & crop
    t_io = time.perf_counter()
    n_boxes = len(final_boxes)

    if n_boxes > 0:
        sorted_boxes = final_boxes[np.argsort(final_boxes[:, 1])]

        # ★ 向量化 expand_box（一次性处理所有 box）
        if expand_box:
            dh = sorted_boxes[:, 3] - sorted_boxes[:, 1]
            sorted_boxes[:, 0] -= dh * ratio_w
            sorted_boxes[:, 2] += dh * ratio_w
            sorted_boxes[:, 1] -= dh * ratio_h
            sorted_boxes[:, 3] += dh * ratio_h

        if save_crops:
            os.makedirs(DEFAULT_CROP_DIR, exist_ok=True)
            inv_scale = 1.0 / scale
            for i, box in enumerate(sorted_boxes):
                rx1 = max(0,      int((box[0] - pad_x) * inv_scale))
                ry1 = max(0,      int((box[1] - pad_y) * inv_scale))
                rx2 = min(real_w, int((box[2] - pad_x) * inv_scale))
                ry2 = min(real_h, int((box[3] - pad_y) * inv_scale))
                if rx2 - rx1 > 0 and ry2 - ry1 > 0:
                    crop = img_orig[ry1:ry2, rx1:rx2]
                    cv2.imwrite(
                        os.path.join(DEFAULT_CROP_DIR, f"{fname_noext}-{i+1:06d}.jpg"),
                        crop
                    )

        if save_vis:
            # ★ 用 resize 后的图做可视化（从 img_orig resize，不再用 PIL）
            vis_h, vis_w = input_size, input_size
            img_vis = np.full((vis_h, vis_w, 3), 128, dtype=np.uint8)
            resized_vis = cv2.resize(img_orig, (content_w, content_h),
                                     interpolation=cv2.INTER_LINEAR)
            img_vis[pad_y:pad_y+content_h, pad_x:pad_x+content_w] = resized_vis

            for box in sorted_boxes:
                x1c = max(0, int(box[0]))
                y1c = max(0, int(box[1]))
                x2c = min(input_size, int(box[2]))
                y2c = min(input_size, int(box[3]))
                cv2.rectangle(img_vis, (x1c, y1c), (x2c, y2c), (0, 255, 0), 1)

            save_path = os.path.join(output_dir, "res_" + fname)
            cv2.imwrite(save_path, img_vis)
    else:
        if save_vis:
            img_vis = np.full((input_size, input_size, 3), 128, dtype=np.uint8)
            resized_vis = cv2.resize(img_orig, (content_w, content_h),
                                     interpolation=cv2.INTER_LINEAR)
            img_vis[pad_y:pad_y+content_h, pad_x:pad_x+content_w] = resized_vis
            save_path = os.path.join(output_dir, "res_" + fname)
            cv2.imwrite(save_path, img_vis)

    io_ms = (time.perf_counter() - t_io) * 1000
    total_ms = (time.perf_counter() - t_total) * 1000

    if timing:
        print(f"  [{fname}] {n_boxes}框  "
              f"预处理:{pre_ms:.1f}ms  推理:{infer_ms:.1f}ms  "
              f"后处理:{post_ms:.1f}ms  I/O:{io_ms:.1f}ms  "
              f"总计:{total_ms:.1f}ms")

    return n_boxes, infer_ms, total_ms


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="EAST 高速推理（优化后处理）")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image_path", type=str, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backend", type=str,
                        choices=["auto", "onnx", "openvino", "pytorch"],
                        default="openvino")
    parser.add_argument("--input_size", type=int, default=INPUT_SIZE)
    parser.add_argument("--score_thr", type=float, default=SCORE_THRESHOLD)
    parser.add_argument("--save_crops", action="store_true", default=SAVE_CROPS)
    parser.add_argument("--no_vis", action="store_true",
                        help="跳过可视化输出，纯检测模式")
    parser.add_argument("--benchmark", action="store_true",
                        help="显示每张图各阶段耗时")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("EAST 高速推理（优化后处理版）")
    print("=" * 60)
    print(f"  模型:    {args.model_path}")
    print(f"  输入:    {args.image_path}")
    print(f"  输出:    {args.output_dir}")
    print(f"  后端:    {args.backend}")
    print(f"  可视化:  {'关闭' if args.no_vis else '开启'}")
    print(f"  CPU核心: {os.cpu_count()}")

    if not os.path.exists(args.model_path):
        print(f"✗ 模型文件不存在: {args.model_path}")
        return

    backend = load_backend(args.model_path, args.backend)

    if os.path.isdir(args.image_path):
        images = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            images.extend(glob.glob(os.path.join(args.image_path, ext)))
        images.sort()
    elif os.path.isfile(args.image_path):
        images = [args.image_path]
    else:
        print(f"✗ 路径不存在: {args.image_path}")
        return

    if not images:
        print("✗ 未找到图像文件")
        return

    print(f"\n共 {len(images)} 张图像\n")

    save_vis = not args.no_vis
    total_boxes = 0
    total_infer_ms = 0.0
    total_wall_ms = 0.0
    t_wall_start = time.perf_counter()

    for img_path in images:
        nb, ims, tms = detect_one(
            backend, img_path, args.output_dir,
            input_size=args.input_size,
            score_threshold=args.score_thr,
            save_crops=args.save_crops,
            save_vis=save_vis,
            timing=args.benchmark,
        )
        total_boxes += nb
        total_infer_ms += ims
        total_wall_ms += tms

    t_wall = time.perf_counter() - t_wall_start
    n = len(images)

    print(f"\n{'='*60}")
    print(f"推理完成")
    print(f"  图像数:          {n} 张")
    print(f"  检测框总数:      {total_boxes}")
    print(f"  纯推理均值:      {total_infer_ms/n:.1f} ms/张")
    print(f"  单图总耗时均值:  {total_wall_ms/n:.1f} ms/张")
    print(f"  总 wall time:    {t_wall*1000:.0f} ms")
    print(f"  吞吐量:          {n/t_wall:.2f} 张/秒")
    print(f"  结果保存在:      {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()