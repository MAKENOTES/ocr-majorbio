import numpy as np
import cv2


def shrink_polygon_directional(poly, r_long=0.3, r_short=0.6):
    """
    方向性收缩：对文本行的长边（水平）和短边（垂直）使用不同的收缩距离。

    核心改进：
      原版 shrink_polygon 按面积/周长算一个统一的 d，
      对于宽扁的文本行，上下收缩量极小，导致相邻行在 score map 上粘连。

      本版本：
      - 识别文本行的长边方向和短边方向
      - 短边方向（通常是上下）用更大的收缩比例，确保行间分离
      - 长边方向（通常是左右）用较小的收缩比例，保留足够的正样本

    参数:
      poly: (4, 2) float32 四边形
      r_long:  长边方向收缩比例 (相对于该方向边长的比例)
      r_short: 短边方向收缩比例 (相对于该方向边长的比例)
    """
    # 计算四条边的长度
    edge_lengths = []
    for i in range(4):
        length = np.linalg.norm(poly[(i + 1) % 4] - poly[i])
        edge_lengths.append(length)

    # 判断哪对边是长边、哪对边是短边
    # 边 0-1 和 2-3 是一对（通常是上边和下边）
    # 边 1-2 和 3-0 是一对（通常是右边和左边）
    pair_a = (edge_lengths[0] + edge_lengths[2]) / 2  # 上下边平均长度
    pair_b = (edge_lengths[1] + edge_lengths[3]) / 2  # 左右边平均长度

    # 为每条边分配收缩距离
    d_per_edge = []
    for i in range(4):
        length = edge_lengths[i]
        if length < 1e-6:
            d_per_edge.append(0)
            continue

        # 判断这条边属于长边还是短边
        if i in [0, 2]:  # 上边和下边
            if pair_a >= pair_b:
                # 上下是长边 → 用小收缩
                d = length * r_long * 0.5  # 除以2是因为两端各收缩一半
            else:
                # 上下是短边 → 用大收缩
                d = length * r_short * 0.5
        else:  # 左边和右边
            if pair_b >= pair_a:
                d = length * r_long * 0.5
            else:
                d = length * r_short * 0.5

        # 安全限制：不能超过边长的 45%
        d = min(d, length * 0.45)
        d_per_edge.append(d)

    # 用各边自己的 d 值做偏移
    lines = []
    for i in range(4):
        p1 = poly[i]
        p2 = poly[(i + 1) % 4]
        edge = p2 - p1
        length = np.linalg.norm(edge)
        if length < 1e-6:
            normal = np.array([0.0, 0.0])
        else:
            normal = np.array([-edge[1], edge[0]]) / length

        start_pt = p1 + d_per_edge[i] * normal
        lines.append((start_pt, edge))

    # 计算相邻偏移线的交点
    new_poly = []
    for i in range(4):
        p1_prev, v1_prev = lines[(i - 1) % 4]
        p1_curr, v1_curr = lines[i]

        det = v1_prev[0] * v1_curr[1] - v1_prev[1] * v1_curr[0]
        if abs(det) < 1e-6:
            new_poly.append(poly[i])
            continue

        diff = p1_curr - p1_prev
        t = (diff[0] * v1_curr[1] - diff[1] * v1_curr[0]) / det
        intersection = p1_prev + t * v1_prev
        new_poly.append(intersection)

    result = np.array(new_poly, dtype=np.float32)

    # 验证收缩后的多边形面积 > 0
    if cv2.contourArea(result) < 1.0:
        # 收缩过度，退回原始多边形
        return poly

    return result

def generate_ground_truth(image_size, quads, score_map_scale=0.25):
    img_h, img_w = image_size
    score_map_h, score_map_w = int(img_h * score_map_scale), int(img_w * score_map_scale)

    score_map = np.zeros((score_map_h, score_map_w), dtype=np.float32)
    geo_map = np.zeros((4, score_map_h, score_map_w), dtype=np.float32)

    if quads.size == 0:
        return score_map, geo_map

    for quad in quads:
        scaled_quad = quad * score_map_scale

        # ★ 核心改动：使用方向性收缩
        # 短边（上下）收缩 60%，长边（左右）收缩 30%
        # 这样密集文本行之间在 score map 上就不会粘连
        shrunk_quad = shrink_polygon_directional(scaled_quad, r_long=0.3, r_short=0.6)

        mask = np.zeros((score_map_h, score_map_w), dtype=np.uint8)
        cv2.fillPoly(mask, [shrunk_quad.astype(np.int32)], 1)
        score_map[mask == 1] = 1

        x_min, y_min = np.min(scaled_quad, axis=0)
        x_max, y_max = np.max(scaled_quad, axis=0)

        positive_pixels = np.argwhere(mask == 1)

        for y, x in positive_pixels:
            pixel_center_x = x + 0.5
            pixel_center_y = y + 0.5
            dist_top = max(0, pixel_center_y - y_min)
            dist_right = max(0, x_max - pixel_center_x)
            dist_bottom = max(0, y_max - pixel_center_y)
            dist_left = max(0, pixel_center_x - x_min)

            new_area = (dist_top + dist_bottom) * (dist_left + dist_right)
            old_area = (geo_map[0, y, x] + geo_map[2, y, x]) * (geo_map[1, y, x] + geo_map[3, y, x])

            if geo_map[0, y, x] == 0 or new_area < old_area:
                geo_map[0, y, x] = dist_top
                geo_map[1, y, x] = dist_right
                geo_map[2, y, x] = dist_bottom
                geo_map[3, y, x] = dist_left

    geo_map = np.log1p(geo_map)

    return score_map, geo_map