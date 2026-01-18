import torch
import cv2
import numpy as np
import os
from PIL import Image
import sys
from pathlib import Path

# 添加项目根目录到 sys.path，以便可以导入 models 和 utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.east_model import EASTModel
from utils.transforms import ResizeAndPad

# --- 配置参数 ---
IMAGE_PATH = "data/train_images/000201.jpg" # 输入图像路径或目录 Path("../data/train_images/000201.jpg")
MODEL_PATH = "east_model_epoch_100.pth" # 模型权重路径
OUTPUT_DIR = "test_results" # 结果保存目录
INPUT_SIZE = 1024 # 输入图像尺寸
SCORE_THRESHOLD = 0.5 # 分数阈值
NMS_THRESHOLD = 0.2 # NMS 阈值

def load_model(model_path, device):
    """
    加载训练好的 EAST 模型。
    """
    model = EASTModel(pretrained=False) # 推理时不需要加载预训练的 ImageNet 权重，因为我们会加载自己的权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, input_size):
    """
    预处理图像：读取、调整大小并转换为张量。
    """
    image = Image.open(image_path).convert('RGB')
    original_w, original_h = image.size
    
    # 使用 ResizeAndPad 进行预处理，保持宽高比
    transform = ResizeAndPad(input_size)
    # ResizeAndPad 需要 'quads' 键，即使我们在推理时没有它
    sample = {'image': image, 'quads': np.array([])}
    sample = transform(sample)
    
    resized_image = sample['image']
    
    # 转换为张量并归一化 (0-1)
    img_tensor = torch.from_numpy(np.array(resized_image)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0) # 添加 batch 维度
    
    return img_tensor, resized_image, (original_w, original_h)

def detect(model, image_path, output_dir, input_size=512, score_threshold=0.8, nms_threshold=0.2):
    """
    执行文本检测并保存结果。
    """
    device = next(model.parameters()).device
    
    # 1. 预处理
    img_tensor, resized_image, (orig_w, orig_h) = preprocess_image(image_path, input_size)
    img_tensor = img_tensor.to(device)
    
    # 2. 推理
    with torch.no_grad():
        score_map, geo_map = model(img_tensor)
    
    # 3. 后处理
    # 3.1 处理 Score Map
    score_map = score_map[0, 0] # (H/4, W/4)
    
    # 3.2 处理 Geo Map
    # 这里的 geo_map 是 (1, 5, H/4, W/4)
    # 根据 loss.py 中的逻辑: pred_aabb = torch.sigmoid(y_pred_geo[:, 0:4, :, :]) * feature_map_size
    feature_map_h, feature_map_w = score_map.shape
    
    # 恢复 AABB 距离
    geo_map_aabb = torch.sigmoid(geo_map[0, :4, :, :]) * feature_map_h 
    
    # 转换为 numpy
    score_map_np = score_map.cpu().numpy()
    geo_map_np = geo_map_aabb.cpu().numpy()
    
    # 3.3 阈值过滤
    xy_text = np.argwhere(score_map_np > score_threshold)
    
    if len(xy_text) == 0:
        print(f"未在 {image_path} 中检测到文本。")
        return

    # 3.4 恢复边界框
    boxes = []
    scores = []
    for y, x in xy_text:
        score = score_map_np[y, x]
        
        top = geo_map_np[0, y, x]
        right = geo_map_np[1, y, x]
        bottom = geo_map_np[2, y, x]
        left = geo_map_np[3, y, x]
        
        # 计算特征图上的坐标
        # 这里的 x, y 是特征图上的索引
        # 边界框坐标:
        x_min = x - left
        x_max = x + right
        y_min = y - top
        y_max = y + bottom
        
        # 映射回 resized_image (512x512) 的尺度
        # 特征图是 128x128，输入是 512x512，所以 stride 是 4
        stride = 4
        x_min *= stride
        x_max *= stride
        y_min *= stride
        y_max *= stride
        
        boxes.append([x_min, y_min, x_max, y_max])
        scores.append(score)
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # 3.5 NMS (非极大值抑制)
    # 使用 OpenCV 的 NMS
    # boxes 格式需要是 [x, y, w, h]
    # 我们的 boxes 是 [x_min, y_min, x_max, y_max]
    
    # 转换格式用于 cv2.dnn.NMSBoxes
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
    # 将 PIL 图像转换为 OpenCV 格式 (BGR)
    img_cv = cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR)
    
    for box in final_boxes:
        x_min, y_min, x_max, y_max = box.astype(int)
        cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        
    # 保存结果
    filename = os.path.basename(image_path)
    save_path = os.path.join(output_dir, "res_" + filename)
    cv2.imwrite(save_path, img_cv)
    print(f"结果已保存至: {save_path}")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件 '{MODEL_PATH}' 不存在。请修改 MODEL_PATH 变量。")
        return

    model = load_model(MODEL_PATH, device)
    
    # 支持单个文件或目录
    if os.path.isdir(IMAGE_PATH):
        import glob
        image_files = glob.glob(os.path.join(IMAGE_PATH, "*.jpg")) + \
                      glob.glob(os.path.join(IMAGE_PATH, "*.png"))
        if not image_files:
            print(f"警告: 在 '{IMAGE_PATH}' 中未找到 .jpg 或 .png 图像。")
            
        for img_file in image_files:
            detect(model, img_file, OUTPUT_DIR, INPUT_SIZE, SCORE_THRESHOLD, NMS_THRESHOLD)
    elif os.path.isfile(IMAGE_PATH):
        detect(model, IMAGE_PATH, OUTPUT_DIR, INPUT_SIZE, SCORE_THRESHOLD, NMS_THRESHOLD)
    else:
        print(f"错误: 图像路径 '{IMAGE_PATH}' 不存在。")

if __name__ == '__main__':
    main()
