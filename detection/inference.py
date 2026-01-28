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
MODEL_PATH = "detection/checkpoints/detection_epoch_491.pth"
IMAGE_PATH = "detection/data/test"
OUTPUT_DIR = "detection/data/test_results/1"
INPUT_SIZE = 1024  # 保持与训练一致
SCORE_THRESHOLD = 0.8  # 置信度阈值
NMS_THRESHOLD = 0.2  # NMS 阈值


def load_model(model_path, device):
    """
    加载模型。
    注意：确保 models/east_model.py 中的类定义已经更新为 4 通道 + ReLU 的版本。
    """
    model = EASTModel(pretrained=False)
    # 加载权重
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        print("权重加载失败，请检查模型结构是否与权重文件匹配（例如通道数 5 vs 4）。")
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
    # 推理时不需要 quads，传入空数组占位
    sample = {'image': image, 'quads': np.array([])}
    sample = transform(sample)

    resized_image = sample['image']

    # 转 Tensor 并归一化
    img_tensor = torch.from_numpy(np.array(resized_image)).permute(2, 0, 1).float() / 255.0
    # --- [新增] 同步应用 ImageNet 归一化 ---
    normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_tensor = normalize(img_tensor)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor, resized_image, (original_w, original_h)


def detect(model, image_path, output_dir, input_size=1024, score_threshold=0.8, nms_threshold=0.2):
    """
    执行纯水平文本检测
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
    # score_map: (1, 1, H/4, W/4) -> (H/4, W/4)
    score_map = score_map[0, 0]

    # geo_map: (1, 4, H/4, W/4)
    # --- [关键修改 1] ---
    # 我们的模型现在输出的是 ReLU 后的绝对值，不需要 Sigmoid，也不需要乘以 map_size
    # 直接取出来即可
    geo_map_data = geo_map[0, :4, :, :]  # shape (4, H/4, W/4)

    # 转 numpy
    score_map_np = score_map.cpu().numpy()
    geo_map_np = geo_map_data.cpu().numpy()

    # 3.1 阈值过滤
    # xy_text 是满足阈值的特征图坐标 (y, x)
    xy_text = np.argwhere(score_map_np > score_threshold)

    if len(xy_text) == 0:
        print(f"[{os.path.basename(image_path)}] 未检测到文本。")
        return

    # 3.2 恢复边界框
    boxes = []
    scores = []

    # 按照行优先排序可能会让结果更有序，但不是必须的
    # xy_text = xy_text[np.argsort(xy_text[:, 0])]

    for y, x in xy_text:
        score = score_map_np[y, x]

        # 获取预测的 4 个距离
        dist_top = geo_map_np[0, y, x]
        dist_right = geo_map_np[1, y, x]
        dist_bottom = geo_map_np[2, y, x]
        dist_left = geo_map_np[3, y, x]

        # --- [关键修改 2] ---
        # 像素中心对齐：训练时我们假设 anchor 是像素中心 (x+0.5, y+0.5)
        # 所以恢复时也要加上 0.5
        pixel_center_x = x + 0.5
        pixel_center_y = y + 0.5

        # 计算特征图上的边界 (Feature Map Scale)
        x_min_feature = pixel_center_x - dist_left
        x_max_feature = pixel_center_x + dist_right
        y_min_feature = pixel_center_y - dist_top
        y_max_feature = pixel_center_y + dist_bottom

        # 映射回 输入图像 (Input Image Scale, 1024x1024)
        stride = 4.0
        x_min = x_min_feature * stride
        x_max = x_max_feature * stride
        y_min = y_min_feature * stride
        y_max = y_max_feature * stride

        boxes.append([x_min, y_min, x_max, y_max])
        scores.append(score)

    boxes = np.array(boxes)
    scores = np.array(scores)

    # 3.3 NMS (非极大值抑制)
    # 既然是纯水平框，标准的 OpenCV NMSBoxes 效果就很好且很快

    # 转换格式为 [x, y, w, h] 用于 NMS
    boxes_xywh = []
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        w = x_max - x_min
        h = y_max - y_min
        boxes_xywh.append([int(x_min), int(y_min), int(w), int(h)])

    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), score_threshold, nms_threshold)

    if len(indices) > 0:
        indices = indices.flatten()
        final_boxes = boxes[indices]
    else:
        final_boxes = []

    # 4. 绘图并保存
    img_cv = cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR)

    for box in final_boxes:
        x_min, y_min, x_max, y_max = box.astype(int)

        # 绘制矩形
        cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

        # 可选：如果你想画出非常精确的浮点坐标，可以用 polylines
        # pts = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], np.int32)
        # cv2.polylines(img_cv, [pts], True, (0, 0, 255), 2)

    # 保存结果
    filename = os.path.basename(image_path)
    save_path = os.path.join(output_dir, "res_" + filename)
    cv2.imwrite(save_path, img_cv)
    print(f"[{filename}] 检测到 {len(final_boxes)} 个文本框，结果已保存。")


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"加载模型: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件不存在。")
        return

    # 加载模型
    model = load_model(MODEL_PATH, device)

    # 处理输入路径
    if os.path.isdir(IMAGE_PATH):
        import glob
        # 支持 jpg, png, jpeg, bmp
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(IMAGE_PATH, ext)))

        if not image_files:
            print(f"警告: 在目录中未找到图像。")

        print(f"共找到 {len(image_files)} 张图像，开始推理...")
        for img_file in image_files:
            detect(model, img_file, OUTPUT_DIR, INPUT_SIZE, SCORE_THRESHOLD, NMS_THRESHOLD)

    elif os.path.isfile(IMAGE_PATH):
        detect(model, IMAGE_PATH, OUTPUT_DIR, INPUT_SIZE, SCORE_THRESHOLD, NMS_THRESHOLD)
    else:
        print(f"错误: 路径不存在 -> {IMAGE_PATH}")


if __name__ == '__main__':
    main()