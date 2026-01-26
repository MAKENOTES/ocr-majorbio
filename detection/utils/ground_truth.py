import numpy as np
import cv2
import math

def shrink_polygon(poly, r=0.4):
    """
    修正后的多边形收缩函数。
    1. 修正了顶点顺序错位的问题。
    2. 增加了防止过度收缩的保护机制（特别是针对细长文本）。
    """
    area = cv2.contourArea(poly)
    perimeter = cv2.arcLength(poly, True)

    if perimeter == 0:
        return poly

    # --- 核心修正 1: 收缩公式与限制 ---
    # 计算理论收缩距离
    d = area * (1 - r * r) / perimeter

    # [保护机制]：如果文本非常细长（例如高度只有 4px），d 可能会导致两条长边交叉
    # 我们计算多边形的“短边”长度近似值
    # 简单的近似：面积 / (周长/2) = 平均厚度
    # 或者直接计算最短边长
    edge_lengths = [np.linalg.norm(poly[i] - poly[(i + 1) % 4]) for i in range(4)]
    min_side = min(edge_lengths)

    # 限制 d 不能超过最短边的一半，否则多边形会消失或翻转
    # 留 1 个像素的余地，保证至少有 mask 存在
    max_d = (min_side - 1.0) / 2.0

    # 如果 max_d <= 0，说明框太小了，甚至不足1px，此时强行保留极小厚度或不收缩
    if max_d < 0:
        max_d = 0  # 不收缩，或者极小收缩

    d = min(d, max_d)

    pco = []
    lines = []
    for i in range(4):
        p1 = poly[i]
        p2 = poly[(i + 1) % 4]
        edge = p2 - p1
        length = np.linalg.norm(edge)
        if length < 1e-6:
            normal = np.array([0, 0])
        else:
            # 顺时针顺序，向内法向量
            normal = np.array([-edge[1], edge[0]]) / length

        # 偏移后的直线点和方向
        start_pt = p1 + d * normal
        lines.append((start_pt, edge))  # 存方向向量即可，不需要归一化

    new_poly = []
    for i in range(4):
        # --- 核心修正 2: 顶点对齐逻辑 ---
        # 我们想要计算对应原 poly[i] 的新顶点
        # 原 poly[i] 是 Line[i-1] (前一条边) 和 Line[i] (当前边) 的交点

        # Line 1: 上一条边 (i-1)
        p1_prev, v1_prev = lines[(i - 1) % 4]
        # Line 2: 当前边 (i)
        p1_curr, v1_curr = lines[i]

        # 计算交点
        det = v1_prev[0] * v1_curr[1] - v1_prev[1] * v1_curr[0]

        if abs(det) < 1e-6:
            new_poly.append(poly[i])
            continue

        diff = p1_curr - p1_prev
        t = (diff[0] * v1_curr[1] - diff[1] * v1_curr[0]) / det
        intersection = p1_prev + t * v1_prev
        new_poly.append(intersection)

    return np.array(new_poly, dtype=np.float32)

def generate_ground_truth(image_size, quads, score_map_scale=0.25):
    """
    为EAST模型生成真值（ground truth）分数图和几何图。
    """
    img_h, img_w = image_size
    score_map_h, score_map_w = int(img_h * score_map_scale), int(img_w * score_map_scale)
    
    score_map = np.zeros((score_map_h, score_map_w), dtype=np.float32)
    geo_map = np.zeros((5, score_map_h, score_map_w), dtype=np.float32)
    
    if quads.size == 0:
        return score_map, geo_map

    for quad in quads:
        # 1. 坐标缩放
        scaled_quad = quad * score_map_scale
        
        # 2. 生成收缩后的多边形用于 Score Map
        # EAST 建议收缩比例约为 0.3 (即保留 0.7 的区域? 不，是收缩边界)
        # 这里的参数 0.3 是 EAST 论文中的 shrink_ratio
        shrunk_quad = shrink_polygon(scaled_quad, r=0.4)
        
        # 3. 绘制 Score Map (使用收缩后的框)
        mask = np.zeros((score_map_h, score_map_w), dtype=np.uint8)
        cv2.fillPoly(mask, [shrunk_quad.astype(np.int32)], 1)
        score_map[mask == 1] = 1
        
        # 4. 计算 Geo Map
        # 注意：Geo Map 的距离是基于“未收缩”的原始框计算的，
        # 但是只在“收缩后”的 Score Map 区域内进行赋值。
        # 这样模型学会了：在文本中心区域预测出完整的文本框大小。
        
        # 计算原始框的 AABB (或者使用更复杂的点到边距离)
        x_min, y_min = np.min(scaled_quad, axis=0)
        x_max, y_max = np.max(scaled_quad, axis=0)
        
        # 获取收缩后区域内的像素
        positive_pixels = np.argwhere(mask == 1)
        
        for y, x in positive_pixels:
            # 计算像素点到原始(未收缩)边界框四边的距离
            dist_top = y - y_min
            dist_right = x_max - x
            dist_bottom = y_max - y
            dist_left = x - x_min
            
            # 只有当像素在原始框内时距离才有效 (理论上收缩后的肯定在原始内)
            # 加上一个小的 epsilon 防止除零或负数
            
            # 更新几何图
            if geo_map[0, y, x] == 0 or (dist_top * dist_bottom < geo_map[0, y, x] * geo_map[2, y, x]):
                geo_map[0, y, x] = dist_top
                geo_map[1, y, x] = dist_right
                geo_map[2, y, x] = dist_bottom
                geo_map[3, y, x] = dist_left
                geo_map[4, y, x] = 0 # 角度暂设为0
                
    return score_map, geo_map
