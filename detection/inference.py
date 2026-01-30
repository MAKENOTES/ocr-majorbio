import torch
import cv2
import numpy as np
import os
from PIL import Image
import sys
from torchvision.transforms import Normalize

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.east_model import EASTModel
from utils.transforms import ResizeAndPad

# --- 配置参数 ---
MODEL_PATH = "detection/checkpoints/best_detection_epoch_981_loss_0.06583.pth"
IMAGE_PATH = "detection/data/train_images"
OUTPUT_DIR = "detection/data/test_results/5"

# 1. 裁剪保存开关
SAVE_CROPS = True
CROP_DIR_NAME = "detection/data/crops/20260130"

# 2. [新增] 边框扩大开关 (Padding)
EXPAND_BOX = True  # 是否开启扩大
EXPAND_RATIO_W = 0.05  # 宽度扩大比例 (例如 0.05 = 左右各增加宽度的 5%)
EXPAND_RATIO_H = 0.10  # 高度扩大比例 (例如 0.10 = 上下各增加高度的 10%)

INPUT_SIZE = 1024
SCORE_THRESHOLD = 0.8
NMS_THRESHOLD = 0.2


def load_model(model_path, device):
    """加载模型"""
    model = EASTModel(pretrained=False)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        print("权重加载失败，请检查模型结构是否与权重文件匹配。")
        raise e
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, input_size):
    """预处理"""
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


def detect(model, image_path, output_dir, input_size=1024, score_threshold=0.8, nms_threshold=0.2,
           save_crops=False, expand_box=False, ratio_w=0.0, ratio_h=0.0):
    """
    执行检测，支持从原图裁剪高清图片
    """
    device = next(model.parameters()).device

    # 1. 预处理 (获取 resized_image 用于推理, orig_wh 用于坐标还原)
    img_tensor, resized_image, (orig_w, orig_h) = preprocess_image(image_path, input_size)
    if img_tensor is None:
        return

    img_tensor = img_tensor.to(device)

    # 2. 推理
    with torch.no_grad():
        score_map, geo_map = model(img_tensor)

    # 3. 后处理
    score_map = score_map[0, 0]
    geo_map_data = geo_map[0, :4, :, :]

    score_map_np = score_map.cpu().numpy()
    geo_map_np = geo_map_data.cpu().numpy()

    # 3.1 阈值过滤
    xy_text = np.argwhere(score_map_np > score_threshold)

    if len(xy_text) == 0:
        print(f"[{os.path.basename(image_path)}] 未检测到文本。")
        return

    # 3.2 恢复边界框
    boxes = []
    scores = []

    for y, x in xy_text:
        score = score_map_np[y, x]
        dist_top = geo_map_np[0, y, x]
        dist_right = geo_map_np[1, y, x]
        dist_bottom = geo_map_np[2, y, x]
        dist_left = geo_map_np[3, y, x]

        pixel_center_x = x + 0.5
        pixel_center_y = y + 0.5

        x_min_feature = pixel_center_x - dist_left
        x_max_feature = pixel_center_x + dist_right
        y_min_feature = pixel_center_y - dist_top
        y_max_feature = pixel_center_y + dist_bottom

        stride = 4.0
        x_min = x_min_feature * stride
        x_max = x_max_feature * stride
        y_min = y_min_feature * stride
        y_max = y_max_feature * stride

        boxes.append([x_min, y_min, x_max, y_max])
        scores.append(score)

    boxes = np.array(boxes)
    scores = np.array(scores)

    # 3.3 NMS
    boxes_xywh = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        w = x_max - x_min
        h = y_max - y_min
        boxes_xywh.append([int(x_min), int(y_min), int(w), int(h)])

    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), score_threshold, nms_threshold)

    final_boxes = []
    if len(indices) > 0:
        indices = indices.flatten()
        final_boxes = boxes[indices]

    # 准备图像：
    # 1. img_vis: 用于画绿框和显示结果 (resize 后的 1024 版，画图方便)
    img_vis = cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR)

    # 2. [关键] img_original: 读取原图，用于高清裁剪
    # 注意：cv2.imread 读取的是 BGR，如果后面要用 PIL 处理可能需要转 RGB，但这里直接保存 jpg 无所谓
    img_original = cv2.imread(image_path)
    if img_original is None:
        print(f"原图读取失败: {image_path}")
        return

    # 获取原图尺寸
    real_h, real_w = img_original.shape[:2]

    filename_full = os.path.basename(image_path)
    filename_no_ext = os.path.splitext(filename_full)[0]

    # --- 计算 ResizeAndPad 的缩放比例和偏移量，用于坐标映射 ---
    # 假设 ResizeAndPad 的逻辑是：保持长宽比，长边缩放到 input_size，短边居中补零
    scale = input_size / max(orig_h, orig_w)

    # 计算 resize 后的实际内容宽高
    current_content_w = int(orig_w * scale)
    current_content_h = int(orig_h * scale)

    # 计算 Padding 的偏移量 (假设是居中 padding)
    pad_x = (input_size - current_content_w) // 2
    pad_y = (input_size - current_content_h) // 2

    # --- 处理所有框 ---
    if len(final_boxes) > 0:
        sorted_indices = np.argsort(final_boxes[:, 1])
        sorted_boxes = final_boxes[sorted_indices]

        # 确保裁剪目录存在
        if save_crops and not os.path.exists(CROP_DIR_NAME):
            os.makedirs(CROP_DIR_NAME)

        for i, box in enumerate(sorted_boxes):
            # box 是在 1024x1024 图片上的坐标
            x_min, y_min, x_max, y_max = box

            # --- 1. 计算扩边 (在 1024 尺度上算，或者映射回去算都可以，这里先算方便) ---
            if expand_box:
                w_box = x_max - x_min
                h_box = y_max - y_min
                dx = w_box * ratio_w
                dy = h_box * ratio_h
                x_min -= dx
                x_max += dx
                y_min -= dy
                y_max += dy

            # --- 2. 坐标映射回原图 (关键步骤) ---
            # 还原公式：(坐标 - padding) / 缩放比例
            real_x_min = (x_min - pad_x) / scale
            real_x_max = (x_max - pad_x) / scale
            real_y_min = (y_min - pad_y) / scale
            real_y_max = (y_max - pad_y) / scale

            # --- 3. 边界检查 (针对原图尺寸) ---
            real_x_min = max(0, int(real_x_min))
            real_y_min = max(0, int(real_y_min))
            real_x_max = min(real_w, int(real_x_max))
            real_y_max = min(real_h, int(real_y_max))

            # 校验有效性
            if real_x_max - real_x_min <= 0 or real_y_max - real_y_min <= 0:
                continue

            # --- 4. 高清裁剪 ---
            if save_crops:
                crop_img = img_original[real_y_min:real_y_max, real_x_min:real_x_max]

                # [可选优化] 如果裁剪出来的图依然很小（比如原图本文就很小），可以用插值放大一点点
                # 例如高度小于 20 时，放大 2 倍
                # if crop_img.shape[0] < 20:
                #     crop_img = cv2.resize(crop_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

                crop_filename = f"{filename_no_ext}-{i + 1:06d}.jpg"
                crop_save_path = os.path.join(CROP_DIR_NAME, crop_filename)
                cv2.imwrite(crop_save_path, crop_img)

            # --- 5. 在预览图(1024版)上画框 ---
            # 为了画图，我们还是要用回 1024 尺度下的整数坐标
            # 这里简单 clamp 一下用于画图
            draw_x_min = max(0, int(x_min))
            draw_y_min = max(0, int(y_min))
            draw_x_max = min(input_size, int(x_max))
            draw_y_max = min(input_size, int(y_max))

            cv2.rectangle(img_vis, (draw_x_min, draw_y_min), (draw_x_max, draw_y_max), (0, 255, 0), 1)

    # 保存检测结果全图 (依然保存带框的 1024 图，方便预览，文件体积小)
    # 如果你也想保存高清带框原图，可以在 img_original 上画矩形并保存
    save_path = os.path.join(output_dir, "res_" + filename_full)
    cv2.imwrite(save_path, img_vis)

    # 打印日志时加上原图尺寸信息
    print(f"[{filename_full}] (原图: {real_w}x{real_h}) 检测到 {len(final_boxes)} 个文本框")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 打印当前配置
    print("-" * 30)
    print(f"模型路径: {MODEL_PATH}")
    print(f"裁剪保存: {SAVE_CROPS} -> {CROP_DIR_NAME}")
    print(f"边框扩大: {EXPAND_BOX}")
    if EXPAND_BOX:
        print(f"  - 宽度扩大: {EXPAND_RATIO_W * 100}%")
        print(f"  - 高度扩大: {EXPAND_RATIO_H * 100}%")
    print("-" * 30)

    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在。")
        return

    model = load_model(MODEL_PATH, device)

    if os.path.isdir(IMAGE_PATH):
        import glob
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(IMAGE_PATH, ext)))

        if not image_files:
            print(f"警告: 在目录中未找到图像。")

        print(f"共找到 {len(image_files)} 张图像，开始推理...")
        for img_file in image_files:
            detect(model, img_file, OUTPUT_DIR, INPUT_SIZE, SCORE_THRESHOLD, NMS_THRESHOLD,
                   save_crops=SAVE_CROPS,
                   expand_box=EXPAND_BOX,
                   ratio_w=EXPAND_RATIO_W,
                   ratio_h=EXPAND_RATIO_H)

    elif os.path.isfile(IMAGE_PATH):
        detect(model, IMAGE_PATH, OUTPUT_DIR, INPUT_SIZE, SCORE_THRESHOLD, NMS_THRESHOLD,
               save_crops=SAVE_CROPS,
               expand_box=EXPAND_BOX,
               ratio_w=EXPAND_RATIO_W,
               ratio_h=EXPAND_RATIO_H)
    else:
        print(f"错误: 路径不存在 -> {IMAGE_PATH}")


if __name__ == '__main__':
    main()