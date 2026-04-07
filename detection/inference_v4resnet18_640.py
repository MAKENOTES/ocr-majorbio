"""
inference_v4_final.py
=====================
vs v3 的唯一区别：大连通域的左右边界不再用 geo_map 边缘采样，
改为直接在 CC bounding box 上加 d（二次方程反推训练收缩距离）。
上下边界保持 geo_map 采样不变。

不需要 pyclipper / shapely。不需要重新训练。
"""

import torch
import cv2
import numpy as np
import os
import math
from PIL import Image
import sys
from torchvision.transforms import Normalize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.east_model import EASTModel
from components.transforms import ResizeAndPad

# --- 配置 ---
MODEL_PATH = "detection/checkpoints/detection_epoch_642_97_resnet18_640.pth"
IMAGE_PATH = "detection/data/test/5"
OUTPUT_DIR = "detection/data/test_results/resnet18_v4final/5"

SAVE_CROPS = True
CROP_DIR_NAME = "detection/data/crops/resnet18_v4final/5"

SAVE_LABELS = False
LABEL_DIR = "detection/data/test_results/resnet18_v4final/complement"

EXPAND_BOX = False
EXPAND_RATIO_W = 0.10
EXPAND_RATIO_H = 0.10

INPUT_SIZE = 640
SCORE_THRESHOLD = 0.7
MIN_AREA = 2
SMALL_REGION_AREA = round(15 * (INPUT_SIZE / 1024) ** 2)
SMALL_MAX_ASPECT = 5

# ★ 必须与 ground_truth.py 中 shrink_polygon 调用一致
TRAIN_SHRINK_R = 0.2
TRAIN_MAX_D_RATIO = 0.45


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


def solve_shrink_distance(cc_w, cc_h, r=0.2, max_d_ratio=0.45):
    """
    已知 CC 的 bounding box 尺寸 (cc_w, cc_h)，
    反推训练时 shrink_polygon 施加的收缩距离 d。

    推导：
      训练时 d = A_orig*(1-r²)/L_orig
      其中 W_orig=cc_w+2d, H_orig=cc_h+2d
      代入展开得二次方程：
        (8-4c)d² + 2(W'+H')(1-c)d - cW'H' = 0,  c=1-r²
    """
    if cc_w <= 0 or cc_h <= 0:
        return 0.0

    c = 1 - r * r
    a = 4 * (2 - c)
    b = 2 * (cc_w + cc_h) * (1 - c)
    cc_term = -c * cc_w * cc_h

    disc = b * b - 4 * a * cc_term
    if disc < 0:
        return 0.0

    d = (-b + math.sqrt(disc)) / (2 * a)
    d = max(d, 0.0)

    # clamp：训练时 d ≤ min_side_orig × max_d_ratio
    min_side_orig = min(cc_w + 2 * d, cc_h + 2 * d)
    d = min(d, min_side_orig * max_d_ratio)

    return d


def box_from_small_region(ys, xs, geo_map_np, stride):
    d_top    = np.median(geo_map_np[0, ys, xs])
    d_right  = np.median(geo_map_np[1, ys, xs])
    d_bottom = np.median(geo_map_np[2, ys, xs])
    d_left   = np.median(geo_map_np[3, ys, xs])
    cy = np.median(ys)
    cx = np.median(xs)
    return [
        (cx + 0.5 - d_left)   * stride,
        (cy + 0.5 - d_top)    * stride,
        (cx + 0.5 + d_right)  * stride,
        (cy + 0.5 + d_bottom) * stride,
    ]


def boxes_from_connected_components(score_map_np, geo_map_np,
                                     score_threshold, min_area):
    stride = 4.0
    boxes = []
    sm_h, sm_w = score_map_np.shape

    binary = (score_map_np > score_threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=4
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
        x_min_r = xs.min();  x_max_r = xs.max()
        y_min_r = ys.min();  y_max_r = ys.max()
        rw = x_max_r - x_min_r + 1
        rh = y_max_r - y_min_r + 1

        # 上下：geo_map 边缘采样（与 v3 完全相同）
        edge_h = max(1, min(3, int(rh * 0.1)))

        tm = (ys >= y_min_r) & (ys < y_min_r + edge_h)
        y_min_final = np.median(ys[tm] + 0.5 - geo_map_np[0, ys[tm], xs[tm]]) \
            if tm.sum() > 0 else float(y_min_r)

        bm = (ys > y_max_r - edge_h) & (ys <= y_max_r)
        y_max_final = np.median(ys[bm] + 0.5 + geo_map_np[2, ys[bm], xs[bm]]) \
            if bm.sum() > 0 else float(y_max_r + 1)

        # ★ 左右：bbox + d（替代 v3 的 geo_map 采样）
        d = solve_shrink_distance(float(rw), float(rh),
                                  r=TRAIN_SHRINK_R,
                                  max_d_ratio=TRAIN_MAX_D_RATIO)

        x_min_final = max(0.0,       (x_min_r + 0.5) - d)
        x_max_final = min(float(sm_w), (x_max_r + 0.5) + d)

        box = [x_min_final * stride, y_min_final * stride,
               x_max_final * stride, y_max_final * stride]

        if box[2] - box[0] > 2 and box[3] - box[1] > 2:
            boxes.append(box)

    return np.array(boxes) if boxes else np.array([])


# ─── 以下 detect / main 与 v3 完全相同 ───

def detect(model, image_path, output_dir, input_size=640, score_threshold=0.7,
           save_crops=False, expand_box=False, ratio_w=0.0, ratio_h=0.0,
           save_labels=False):
    device = next(model.parameters()).device

    img_tensor, resized_image, (orig_w, orig_h) = preprocess_image(image_path, input_size)
    if img_tensor is None:
        return

    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        score_map, geo_map = model(img_tensor)

    score_map_np = score_map[0, 0].cpu().numpy()
    geo_map_np = geo_map[0, :4, :, :].cpu().numpy()
    geo_map_np = np.expm1(geo_map_np)
    geo_map_np = np.maximum(geo_map_np, 0)

    final_boxes = boxes_from_connected_components(
        score_map_np, geo_map_np,
        score_threshold=score_threshold, min_area=MIN_AREA)

    img_vis = cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR)
    img_original = cv2.imread(image_path)
    if img_original is None:
        return

    real_h, real_w = img_original.shape[:2]
    filename_full = os.path.basename(image_path)
    filename_no_ext = os.path.splitext(filename_full)[0]

    scale = input_size / max(orig_h, orig_w)
    pad_x = (input_size - int(orig_w * scale)) // 2
    pad_y = (input_size - int(orig_h * scale)) // 2

    if len(final_boxes) > 0:
        sorted_indices = np.argsort(final_boxes[:, 1])
        sorted_boxes = final_boxes[sorted_indices]

        if save_crops and not os.path.exists(CROP_DIR_NAME):
            os.makedirs(CROP_DIR_NAME)

        for i, box in enumerate(sorted_boxes):
            x_min, y_min, x_max, y_max = box

            if expand_box:
                h_box = y_max - y_min
                x_min -= h_box * ratio_w
                x_max += h_box * ratio_w
                y_min -= h_box * ratio_h
                y_max += h_box * ratio_h

            real_x_min = max(0,      int((x_min - pad_x) / scale))
            real_y_min = max(0,      int((y_min - pad_y) / scale))
            real_x_max = min(real_w, int((x_max - pad_x) / scale))
            real_y_max = min(real_h, int((y_max - pad_y) / scale))

            if real_x_max - real_x_min <= 0 or real_y_max - real_y_min <= 0:
                continue

            if save_crops:
                crop = img_original[real_y_min:real_y_max, real_x_min:real_x_max]
                cv2.imwrite(os.path.join(CROP_DIR_NAME,
                            f"{filename_no_ext}-{i+1:06d}.jpg"), crop)

            cv2.rectangle(img_vis,
                          (max(0, int(x_min)), max(0, int(y_min))),
                          (min(input_size, int(x_max)), min(input_size, int(y_max))),
                          (0, 255, 0), 1)

    cv2.imwrite(os.path.join(output_dir, "res_" + filename_full), img_vis)

    if save_labels and len(final_boxes) > 0:
        if not os.path.exists(LABEL_DIR):
            os.makedirs(LABEL_DIR)
        lines = []
        for box in final_boxes[np.argsort(final_boxes[:, 1])]:
            x1, y1, x2, y2 = box
            if expand_box:
                h = y2 - y1
                x1 -= h*ratio_w; x2 += h*ratio_w
                y1 -= h*ratio_h; y2 += h*ratio_h
            rx1 = max(0.0, (x1-pad_x)/scale)
            ry1 = max(0.0, (y1-pad_y)/scale)
            rx2 = min(float(real_w), (x2-pad_x)/scale)
            ry2 = min(float(real_h), (y2-pad_y)/scale)
            if rx2>rx1 and ry2>ry1:
                lines.append(f"{rx1:.1f} {ry1:.1f} {rx2:.1f} {ry2:.1f}")
        with open(os.path.join(LABEL_DIR, f"gt_{filename_no_ext}.txt"), 'w') as f:
            f.write('\n'.join(lines) + '\n')

    print(f"[{filename_full}] ({real_w}x{real_h}) 检测到 {len(final_boxes)} 个文本框")


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"策略: geo_map(↕) + bbox+d(↔), r={TRAIN_SHRINK_R}")

    if not os.path.exists(MODEL_PATH):
        print(f"错误: {MODEL_PATH} 不存在")
        return

    model = load_model(MODEL_PATH, device)

    if os.path.isdir(IMAGE_PATH):
        import glob
        files = []
        for ext in ['*.jpg','*.jpeg','*.png','*.bmp']:
            files.extend(glob.glob(os.path.join(IMAGE_PATH, ext)))
        for f in sorted(files):
            detect(model, f, OUTPUT_DIR, INPUT_SIZE, SCORE_THRESHOLD,
                   save_crops=SAVE_CROPS, expand_box=EXPAND_BOX,
                   ratio_w=EXPAND_RATIO_W, ratio_h=EXPAND_RATIO_H,
                   save_labels=SAVE_LABELS)
    elif os.path.isfile(IMAGE_PATH):
        detect(model, IMAGE_PATH, OUTPUT_DIR, INPUT_SIZE, SCORE_THRESHOLD,
               save_crops=SAVE_CROPS, expand_box=EXPAND_BOX,
               ratio_w=EXPAND_RATIO_W, ratio_h=EXPAND_RATIO_H,
               save_labels=SAVE_LABELS)

if __name__ == '__main__':
    main()