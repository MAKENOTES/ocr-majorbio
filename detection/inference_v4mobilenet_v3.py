"""
inference_v5_mobilenet.py — 改进版推理脚本 v5
=======================================
核心改动（基于诊断结果）：
  诊断发现：
    - score map 上相邻行有 0-1px 的极细粘连
    - geo map 的 d_top + d_bottom 准确预测了单行高度（2-5px score map 尺度）
    - 3x1 腐蚀太猛，会杀死正常的 1-2px 细行

  解决方案：
    ★ 对纵向过高的连通域，用 geo 预测的行高做「自适应水平切割」
    - 不依赖投影谷值（粘连区域没有谷）
    - 不依赖形态学腐蚀（会误杀正常行）
    - 直接用 geo map 的 d_top + d_bottom 中位数作为步长，
      从上到下等间隔切割，每段独立出框
"""

import torch
import cv2
import numpy as np
import os
from PIL import Image
import sys
from torchvision.transforms import Normalize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.east_model import EASTModel
from components.transforms import ResizeAndPad

# --- 配置参数 ---
MODEL_PATH = "detection/checkpoints/best_detection_epoch_247_loss_0.02701.pth"
IMAGE_PATH = "detection/data/test/1"
OUTPUT_DIR = "detection/data/test_results/6"

SAVE_CROPS = True
CROP_DIR_NAME = "detection/data/crops/20260313_v5"

EXPAND_BOX = True
EXPAND_RATIO_W = 0.30
EXPAND_RATIO_H = 0.10

INPUT_SIZE = 640
SCORE_THRESHOLD       = 0.7
SCORE_THRESHOLD_SMALL = 0.5
MIN_AREA              = 2
SMALL_REGION_AREA     = 15
SMALL_MAX_ASPECT      = 5.0

# ★ v5 行分割参数
# 当连通域高度 > geo 预测行高的 N 倍时，触发切割
ROW_SPLIT_HEIGHT_RATIO = 1.6


def load_model(model_path, device):
    model = EASTModel(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, input_size):
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"无法读取图像: {image_path}")
        return None, None, None

    original_w, original_h = image.size
    transform = ResizeAndPad(input_size)
    sample = {'image': image, 'quads': np.array([])}
    sample = transform(sample)
    resized_image = sample['image']

    img_tensor = torch.from_numpy(np.array(resized_image)).permute(2, 0, 1).float() / 255.0
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_tensor = normalize(img_tensor)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor, resized_image, (original_w, original_h)


def box_from_small_region(ys, xs, geo_map_np, stride):
    """针对小连通域的出框策略：全像素中位数。"""
    d_top    = np.median(geo_map_np[0, ys, xs])
    d_right  = np.median(geo_map_np[1, ys, xs])
    d_bottom = np.median(geo_map_np[2, ys, xs])
    d_left   = np.median(geo_map_np[3, ys, xs])

    cy = np.median(ys)
    cx = np.median(xs)

    box = [
        (cx + 0.5 - d_left)   * stride,
        (cy + 0.5 - d_top)    * stride,
        (cx + 0.5 + d_right)  * stride,
        (cy + 0.5 + d_bottom) * stride,
    ]
    return box


def box_from_edge_sampling(ys, xs, geo_map_np, stride):
    """大连通域的边缘采样出框。"""
    x_min_r = xs.min();  x_max_r = xs.max()
    y_min_r = ys.min();  y_max_r = ys.max()
    rw = x_max_r - x_min_r + 1
    rh = y_max_r - y_min_r + 1

    edge_w = max(1, min(5, int(rw * 0.1)))
    edge_h = max(1, min(3, int(rh * 0.1)))

    lm = (xs >= x_min_r) & (xs < x_min_r + edge_w)
    x_min_final = np.median(xs[lm] + 0.5 - geo_map_np[3, ys[lm], xs[lm]]) if lm.sum() > 0 else float(x_min_r)

    rm = (xs > x_max_r - edge_w) & (xs <= x_max_r)
    x_max_final = np.median(xs[rm] + 0.5 + geo_map_np[1, ys[rm], xs[rm]]) if rm.sum() > 0 else float(x_max_r + 1)

    tm = (ys >= y_min_r) & (ys < y_min_r + edge_h)
    y_min_final = np.median(ys[tm] + 0.5 - geo_map_np[0, ys[tm], xs[tm]]) if tm.sum() > 0 else float(y_min_r)

    bm = (ys > y_max_r - edge_h) & (ys <= y_max_r)
    y_max_final = np.median(ys[bm] + 0.5 + geo_map_np[2, ys[bm], xs[bm]]) if bm.sum() > 0 else float(y_max_r + 1)

    box = [x_min_final * stride, y_min_final * stride,
           x_max_final * stride, y_max_final * stride]
    return box


def geo_guided_row_split(ys, xs, geo_map_np, stride):
    """
    ★ v5 核心函数：基于 geo map 的行高信息做自适应行切割。

    原理：
    1. 计算连通域内 geo 预测的中位行高 (d_top + d_bottom 的中位数)
    2. 如果连通域实际高度 > 行高 * 1.6，说明多行粘连
    3. 从连通域顶部开始，以行高为步长，逐行切割
    4. 每个切割段内的像素独立用边缘采样出框

    为什么这样做有效：
    - 诊断发现 geo map 的 d_top/d_bottom 是准确的（即使 score map 粘连）
    - 不需要投影谷值（粘连区域投影是满的，没有谷）
    - 不需要形态学操作（腐蚀会杀死正常细行）
    """
    y_min_r = ys.min()
    y_max_r = ys.max()
    cc_height = y_max_r - y_min_r + 1

    # 计算 geo 预测的中位行高（score map 尺度）
    row_heights = geo_map_np[0, ys, xs] + geo_map_np[2, ys, xs]
    median_row_h = np.median(row_heights)

    # 行高太小（< 1px）或连通域不够高，不需要分割
    if median_row_h < 1.0 or cc_height < median_row_h * ROW_SPLIT_HEIGHT_RATIO:
        return None

    # 估计行数（向上取整）
    est_num_rows = max(2, round(cc_height / median_row_h))

    # 计算切割步长
    step = cc_height / est_num_rows

    boxes = []
    for row_idx in range(est_num_rows):
        seg_y_lo = y_min_r + row_idx * step
        seg_y_hi = y_min_r + (row_idx + 1) * step

        # 选出落在这个 Y 区间内的像素
        mask_seg = (ys >= seg_y_lo) & (ys < seg_y_hi)
        seg_ys = ys[mask_seg]
        seg_xs = xs[mask_seg]

        if len(seg_ys) < 2:
            continue

        # 对每个子行用边缘采样出框
        box = box_from_edge_sampling(seg_ys, seg_xs, geo_map_np, stride)
        if box[2] - box[0] > 2 and box[3] - box[1] > 2:
            boxes.append(box)

    if len(boxes) >= 2:
        return boxes
    return None


def boxes_from_connected_components(score_map_np, geo_map_np, score_threshold,
                                     score_threshold_small, min_area):
    """
    两轮连通域检测 + geo 引导的行分割。
    """
    stride = 4.0
    boxes = []

    # ════════════════════════════════════════════════════════
    # 第一轮：高阈值
    # ════════════════════════════════════════════════════════
    binary_high = (score_map_np > score_threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_high, connectivity=8
    )

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        ys, xs = np.where(labels == label_id)

        # ── 小连通域 ──
        if area < SMALL_REGION_AREA:
            w_cc = stats[label_id, cv2.CC_STAT_WIDTH]
            h_cc = stats[label_id, cv2.CC_STAT_HEIGHT]
            if max(w_cc, h_cc) / (min(w_cc, h_cc) + 1e-6) > SMALL_MAX_ASPECT:
                continue
            box = box_from_small_region(ys, xs, geo_map_np, stride)
            if box[2] - box[0] > 1 and box[3] - box[1] > 1:
                boxes.append(box)
            continue

        # ── 大连通域 ──
        # ★ 先尝试 geo 引导的行分割
        split_boxes = geo_guided_row_split(ys, xs, geo_map_np, stride)
        if split_boxes is not None:
            boxes.extend(split_boxes)
            continue

        # 无法分割，正常边缘采样出框
        box = box_from_edge_sampling(ys, xs, geo_map_np, stride)
        if box[2] - box[0] > 2 and box[3] - box[1] > 2:
            boxes.append(box)

    # ════════════════════════════════════════════════════════
    # 第二轮：低阈值补充
    # ════════════════════════════════════════════════════════
    if score_threshold_small < score_threshold:
        binary_low = (score_map_np > score_threshold_small).astype(np.uint8)
        num_labels_s, labels_s, stats_s, _ = cv2.connectedComponentsWithStats(
            binary_low, connectivity=8
        )

        for label_id in range(1, num_labels_s):
            area = stats_s[label_id, cv2.CC_STAT_AREA]
            if area < min_area or area >= SMALL_REGION_AREA:
                continue

            w_cc = stats_s[label_id, cv2.CC_STAT_WIDTH]
            h_cc = stats_s[label_id, cv2.CC_STAT_HEIGHT]
            if max(w_cc, h_cc) / (min(w_cc, h_cc) + 1e-6) > SMALL_MAX_ASPECT:
                continue

            ys, xs = np.where(labels_s == label_id)

            if binary_high[ys, xs].any():
                continue

            box = box_from_small_region(ys, xs, geo_map_np, stride)
            if box[2] - box[0] > 1 and box[3] - box[1] > 1:
                boxes.append(box)

    return np.array(boxes) if boxes else np.array([])


def detect(model, image_path, output_dir, input_size=640, score_threshold=0.7,
           save_crops=False, expand_box=False, ratio_w=0.0, ratio_h=0.0):
    device = next(model.parameters()).device

    img_tensor, resized_image, (orig_w, orig_h) = preprocess_image(image_path, input_size)
    if img_tensor is None:
        return

    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        score_map, geo_map = model(img_tensor)

    score_map_np = score_map[0, 0].cpu().numpy()
    geo_map_np   = geo_map[0, :4, :, :].cpu().numpy()

    # 从 log 空间还原
    geo_map_np = np.expm1(geo_map_np)
    geo_map_np = np.maximum(geo_map_np, 0)

    final_boxes = boxes_from_connected_components(
        score_map_np, geo_map_np,
        score_threshold=score_threshold,
        score_threshold_small=SCORE_THRESHOLD_SMALL,
        min_area=MIN_AREA,
    )

    # 准备图像
    img_vis      = cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR)
    img_original = cv2.imread(image_path)
    if img_original is None:
        print(f"原图读取失败: {image_path}")
        return

    real_h, real_w  = img_original.shape[:2]
    filename_full   = os.path.basename(image_path)
    filename_no_ext = os.path.splitext(filename_full)[0]

    # 坐标映射
    scale             = input_size / max(orig_h, orig_w)
    current_content_w = int(orig_w * scale)
    current_content_h = int(orig_h * scale)
    pad_x             = (input_size - current_content_w) // 2
    pad_y             = (input_size - current_content_h) // 2

    if len(final_boxes) > 0:
        sorted_indices = np.argsort(final_boxes[:, 1])
        sorted_boxes   = final_boxes[sorted_indices]

        if save_crops and not os.path.exists(CROP_DIR_NAME):
            os.makedirs(CROP_DIR_NAME)

        for i, box in enumerate(sorted_boxes):
            x_min, y_min, x_max, y_max = box

            if expand_box:
                h_box  = y_max - y_min
                dx     = h_box * ratio_w
                dy     = h_box * ratio_h
                x_min -= dx
                x_max += dx
                y_min -= dy
                y_max += dy

            real_x_min = max(0,      int((x_min - pad_x) / scale))
            real_y_min = max(0,      int((y_min - pad_y) / scale))
            real_x_max = min(real_w, int((x_max - pad_x) / scale))
            real_y_max = min(real_h, int((y_max - pad_y) / scale))

            if real_x_max - real_x_min <= 0 or real_y_max - real_y_min <= 0:
                continue

            if save_crops:
                crop_img       = img_original[real_y_min:real_y_max, real_x_min:real_x_max]
                crop_filename  = f"{filename_no_ext}-{i + 1:06d}.jpg"
                crop_save_path = os.path.join(CROP_DIR_NAME, crop_filename)
                cv2.imwrite(crop_save_path, crop_img)

            draw_x_min = max(0,          int(x_min))
            draw_y_min = max(0,          int(y_min))
            draw_x_max = min(input_size, int(x_max))
            draw_y_max = min(input_size, int(y_max))
            cv2.rectangle(img_vis,
                          (draw_x_min, draw_y_min),
                          (draw_x_max, draw_y_max),
                          (0, 255, 0), 1)

    save_path = os.path.join(output_dir, "res_" + filename_full)
    cv2.imwrite(save_path, img_vis)
    print(f"[{filename_full}] (原图: {real_w}x{real_h}) 检测到 {len(final_boxes)} 个文本框")


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print("-" * 30)
    print(f"模型路径:       {MODEL_PATH}")
    print(f"推理模式:       两轮连通域 + geo引导行分割 (v5)")
    print(f"第一轮阈值:     {SCORE_THRESHOLD}")
    print(f"第二轮阈值:     {SCORE_THRESHOLD_SMALL}")
    print(f"分割触发比例:   连通域高度 > geo行高 × {ROW_SPLIT_HEIGHT_RATIO}")
    print("-" * 30)

    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在 -> {MODEL_PATH}")
        return

    model = load_model(MODEL_PATH, device)

    if os.path.isdir(IMAGE_PATH):
        import glob
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(IMAGE_PATH, ext)))

        if not image_files:
            print(f"警告: 在目录 {IMAGE_PATH} 中未找到图像。")
            return

        print(f"共找到 {len(image_files)} 张图像，开始推理...")
        for img_file in sorted(image_files):
            detect(model, img_file, OUTPUT_DIR, INPUT_SIZE, SCORE_THRESHOLD,
                   save_crops=SAVE_CROPS, expand_box=EXPAND_BOX,
                   ratio_w=EXPAND_RATIO_W, ratio_h=EXPAND_RATIO_H)

    elif os.path.isfile(IMAGE_PATH):
        detect(model, IMAGE_PATH, OUTPUT_DIR, INPUT_SIZE, SCORE_THRESHOLD,
               save_crops=SAVE_CROPS, expand_box=EXPAND_BOX,
               ratio_w=EXPAND_RATIO_W, ratio_h=EXPAND_RATIO_H)
    else:
        print(f"错误: 路径不存在 -> {IMAGE_PATH}")


if __name__ == '__main__':
    main()