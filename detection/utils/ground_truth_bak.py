import numpy as np
import cv2


def generate_ground_truth(image_size, quads, score_map_scale=0.25):
    """
    为EAST模型生成真值（ground truth）分数图和几何图。

    此函数接收原始的四边形标注，并为模型训练生成两个目标图：
    1. 分数图 (score_map): 一个二值图，其中文本区域的像素为1，背景为0。
    2. 几何图 (geo_map): 一个多通道图，对于每个文本区域的像素，存储其到边界框
       四个边界的距离和边界框的旋转角度。

    :param image_size: tuple, 输入图像的尺寸 (高, 宽)。
    :param quads: np.array, 一个形状为 (N, 4, 2) 的numpy数组，代表N个四边形。
    :param score_map_scale: float, 分数图尺寸相对于原始图像尺寸的比例。
    :return: tuple, 一个包含以下内容的元组:
             - score_map (np.array): 真值分数图，形状为 (高', 宽')。
             - geo_map (np.array): 真值几何图，形状为 (5, 高', 宽')。
    """
    img_h, img_w = image_size
    score_map_h, score_map_w = int(img_h * score_map_scale), int(img_w * score_map_scale)
    # 分数图：作用: 这个图谱回答一个问题：“图像中的哪些像素是文本？”
    score_map = np.zeros((score_map_h, score_map_w), dtype=np.float32)
    # 对于 AABB (轴对齐边界框): 4个通道表示到上、右、下、左的距离
    # 对于 RBOX (旋转边界框): 增加1个通道表示角度。为简单起见，我们使用 AABB。
    # 几何图：这个图谱更加复杂，它回答一个问题：“如果一个像素是文本，那么它所在的完整文本框是什么形状和大小？”
    geo_map = np.zeros((5, score_map_h, score_map_w), dtype=np.float32)

    if quads.size == 0:
        return score_map, geo_map

    for quad in quads:
        # 将四边形坐标缩放到分数图的尺寸
        scaled_quad = quad * score_map_scale

        # 为当前多边形创建一个掩码 (mask)
        mask = np.zeros((score_map_h, score_map_w), dtype=np.uint8)
        # fillPoly 函数，在一个图像上（mask）填充一个或多个多边形
        cv2.fillPoly(mask, [scaled_quad.astype(np.int32)], 1)

        # 更新分数图
        # 1. 拆解 mask == 1 （这个比较的结果会生成一个新的、同样是 128x128 大小的数组，但这个新数组里存放的不再是数字，而是布尔值：True 或 False）
        # // 假设 mask 是这样(值为1的区域是文本框)
        # [[0, 0, 0, 0],
        #  [0, 1, 1, 0],
        #  [0, 1, 1, 0],
        #  [0, 0, 0, 0]]
        # // 那么 `mask == 1` 的结果就是
        # [[False, False, False, False],
        #  [False, True, True, False],
        #  [False, True, True, False],
        #  [False, False, False, False]]

        # 2. 拆解 score_map[...] = 1
        # // 假设score_map 在操作前是全黑的
        # [[0, 0, 0, 0],
        #  [0, 0, 0, 0],
        #  [0, 0, 0, 0],
        #  [0, 0, 0, 0]]
        #
        # // 1.用`mask == 1`的结果选中 score_map中的元素
        # // (选中了中间 2x2 的区域)
        # // 2. 将这些选中的元素赋值为 1
        # // 操作后的 score_map变成了
        # [[0, 0, 0, 0],
        #  [0, 1, 1, 0],
        #  [0, 1, 1, 0],
        #  [0, 0, 0, 0]]

        # 更新分数图 每次循环都会 利用mask更新score_map中的一块文本的区域
        score_map[mask == 1] = 1

        # 对于多边形内的每个像素，计算其到4个边界的距离
        # 这个计算过程非常耗时。一个常见的近似方法是使用收缩后多边形的
        # 最小轴对齐边界框。

        # 为简单起见，我们只计算四边形的轴对齐边界框的距离
        x_min, y_min = np.min(scaled_quad, axis=0)
        x_max, y_max = np.max(scaled_quad, axis=0)

        # 获取分数图上所有在多边形内部的像素坐标
        positive_pixels = np.argwhere(mask == 1)  # 返回 (行, 列)，即 (y, x)

        for y, x in positive_pixels:
            # 到上、右、下、左边界的距离
            dist_top = y - y_min
            dist_right = x_max - x
            dist_bottom = y_max - y
            dist_left = x - x_min

            # 更新几何图 (仅当新的四边形更大时才更新)
            # 这通过偏爱较大区域的几何信息来处理重叠的多边形
            if geo_map[0, y, x] == 0 or (dist_top * dist_bottom < geo_map[0, y, x] * geo_map[2, y, x]):
                geo_map[0, y, x] = dist_top
                geo_map[1, y, x] = dist_right
                geo_map[2, y, x] = dist_bottom
                geo_map[3, y, x] = dist_left
                # 在这个简化的 AABB 版本中，角度保持为 0
                geo_map[4, y, x] = 0

    return score_map, geo_map
