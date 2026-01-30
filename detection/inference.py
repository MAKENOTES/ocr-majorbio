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
# 确保这里的路径是你最新训练出来的模型 (ReLU版本的权重)
MODEL_PATH = "detection/checkpoints/best_detection_epoch_981_loss_0.06583.pth"
IMAGE_PATH = "detection/data/test"
OUTPUT_DIR = "detection/data/test_results/4"

# [新增] 裁剪保存开关及路径
SAVE_CROPS = True  # 是否保存裁剪后的文本小图
CROP_DIR_NAME = "detection/data/crops"  # 裁剪图片存放的子目录名称（建议分开存放，以免和原图混淆）

INPUT_SIZE = 1024  # 保持与训练一致
SCORE_THRESHOLD = 0.8  # 置信度阈值
NMS_THRESHOLD = 0.2  # NMS 阈值


def load_model(model_path, device):
    """
    加载模型。
    """
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
    """
    预处理：ResizeAndPad
    """
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


def detect(model, image_path, output_dir, input_size=1024, score_threshold=0.8, nms_threshold=0.2, save_crops=False):
    """
    执行纯水平文本检测，并支持裁剪
    """
    device = next(model.parameters()).device

    # 1. 预处理
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

    # 转换图像格式用于 OpenCV 处理
    img_cv = cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR)

    # 获取原始文件名（不含扩展名），用于命名
    filename_full = os.path.basename(image_path)
    filename_no_ext = os.path.splitext(filename_full)[0]

    # --- [新增功能] 裁剪并保存文本区域 ---
    if save_crops and len(final_boxes) > 0:
        if not os.path.exists(CROP_DIR_NAME):
            os.makedirs(CROP_DIR_NAME)

        # 按照从上到下、从左到右排序框，这样编号更符合阅读习惯
        # 这里简单按 y_min 排序
        sorted_indices = np.argsort(final_boxes[:, 1])
        sorted_boxes = final_boxes[sorted_indices]

        h_img, w_img = img_cv.shape[:2]

        for i, box in enumerate(sorted_boxes):
            x_min, y_min, x_max, y_max = box.astype(int)

            # 1. 边界检查 (Clamp)，防止坐标超出图像范围导致报错
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w_img, x_max)
            y_max = min(h_img, y_max)

            # 2. 检查有效性
            if x_max - x_min <= 0 or y_max - y_min <= 0:
                continue

            # 3. 裁剪 (注意：先是在干净的 img_cv 上裁剪，所以没有绿框)
            crop_img = img_cv[y_min:y_max, x_min:x_max]

            # 4. 命名：原名-000001.jpg
            # i+1 表示从 1 开始计数，:06d 表示补零至6位
            crop_filename = f"{filename_no_ext}-{i + 1:06d}.jpg"
            crop_save_path = os.path.join(CROP_DIR_NAME, crop_filename)

            cv2.imwrite(crop_save_path, crop_img)

    # --- 4. 绘图并保存整张结果图 ---
    for box in final_boxes:
        x_min, y_min, x_max, y_max = box.astype(int)
        cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

    # 保存检测结果全图
    save_path = os.path.join(output_dir, "res_" + filename_full)
    cv2.imwrite(save_path, img_cv)
    print(f"[{filename_full}] 检测到 {len(final_boxes)} 个文本框")


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"加载模型: {MODEL_PATH}")
    print(f"裁剪开关: {SAVE_CROPS}")

    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在。")
        return

    # 加载模型
    model = load_model(MODEL_PATH, device)

    # 处理输入路径
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
            detect(model, img_file, OUTPUT_DIR, INPUT_SIZE, SCORE_THRESHOLD, NMS_THRESHOLD, save_crops=SAVE_CROPS)

    elif os.path.isfile(IMAGE_PATH):
        detect(model, IMAGE_PATH, OUTPUT_DIR, INPUT_SIZE, SCORE_THRESHOLD, NMS_THRESHOLD, save_crops=SAVE_CROPS)
    else:
        print(f"错误: 路径不存在 -> {IMAGE_PATH}")


if __name__ == '__main__':
    main()