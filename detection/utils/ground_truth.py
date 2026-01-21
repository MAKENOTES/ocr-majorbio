import numpy as np
import cv2
import math

def shrink_polygon(poly, r):
    """
    收缩多边形。
    参考 EAST 论文公式。
    :param poly: 形状为 (4, 2) 的多边形坐标
    :param r: 收缩比例
    :return: 收缩后的多边形
    """
    # 计算多边形面积
    area = cv2.contourArea(poly)
    # 计算多边形周长
    perimeter = cv2.arcLength(poly, True)
    
    if perimeter == 0:
        return poly

    # 计算收缩距离 d = Area * (1 - r^2) / Perimeter
    # 这里的 r 是保留比例，例如 0.3 表示收缩掉边界
    # EAST 论文中 R = 0.3
    d = area * (1 - r * r) / perimeter
    
    # 使用 Clipper 库或者简单的几何偏移进行收缩
    # 这里使用 shapely 或者 pyclipper 会更精确，但为了不引入新依赖，
    # 我们使用简单的向心收缩或 cv2 的多边形偏移（如果可用）
    # 或者使用简单的几何计算：向内移动边
    
    # 简单实现：使用 pco (Polygon Clipper Offset) 逻辑的简化版
    # 计算每条边的向内法向量并移动
    
    pco = []
    for i in range(4):
        p1 = poly[i]
        p2 = poly[(i + 1) % 4]
        
        edge = p2 - p1
        length = np.linalg.norm(edge)
        if length == 0:
            continue
            
        # 法向量 (旋转90度)
        normal = np.array([-edge[1], edge[0]]) / length
        
        # 向内移动 d
        offset = normal * d
        
        # 这里只是简单示意，真正的多边形收缩需要计算交点
        # 为了稳健性，我们使用一个更简单的近似：
        # 计算重心，然后向重心缩放
    
    # --- 简化的重心收缩法 ---
    # 这对于凸四边形（大多数文本框）通常足够有效
    center = np.mean(poly, axis=0)
    # 简单的线性插值收缩可能不完全符合 d 的定义，但在工程上常用
    # 我们尝试用 shrink_ratio 来近似
    # 实际上 EAST 推荐的是基于距离 d 的收缩
    
    # 让我们尝试一种更符合 EAST 的方式：
    # 使用 Vatti 裁剪算法的库通常最好，但这里我们手动计算偏移后的顶点
    # 由于实现复杂，我们退回到一个鲁棒的近似：
    # 找出每条边向内平移 d 后的直线，计算它们的交点。
    
    lines = []
    for i in range(4):
        p1 = poly[i]
        p2 = poly[(i + 1) % 4]
        
        edge = p2 - p1
        length = np.linalg.norm(edge)
        if length < 1e-6:
             # 避免除零
            normal = np.array([0, 0])
        else:
            # 假设点是顺时针或逆时针，我们需要确定向内的方向
            # 文本框通常是凸的。
            # 我们可以检查中心点是否在法向量的一侧
            normal = np.array([-edge[1], edge[0]]) / length
            mid_point = (p1 + p2) / 2
            if np.dot(normal, center - mid_point) < 0:
                normal = -normal
        
        # 直线方程: point + t * direction
        # 偏移后的直线经过点: p1 + d * normal
        start_pt = p1 + d * normal
        lines.append((start_pt, edge / (length + 1e-6))) # (点, 方向向量)

    new_poly = []
    for i in range(4):
        p1, v1 = lines[i]
        p2, v2 = lines[(i + 1) % 4]
        
        # 求两直线交点
        # L1: P1 + t * V1
        # L2: P2 + u * V2
        # Cross product approach
        det = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(det) < 1e-6:
            # 平行线，直接取中点或其他
            new_poly.append(poly[i]) 
            continue
            
        diff = p2 - p1
        t = (diff[0] * v2[1] - diff[1] * v2[0]) / det
        intersection = p1 + t * v1
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
